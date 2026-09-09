# SPDX-License-Identifier: GPL-3.0-only
"""Research-tool accounting, with fixed outcomes and bounded engine steps."""
from types import SimpleNamespace

import pytest

from hexset.arena import ClearedTrade, Entrant
from hexset.bench import trade_census
from hexset.bench.profile_heximax import TimedBot, play_one_game


def test_census_maps_rotated_seats_and_preserves_resource_direction(monkeypatch):
    trade = ClearedTrade(
        step=12, turn=3, phase="MAIN", a=0, b=1,
        received=(-2, 1, 0, 0, 0), hand_a=8, hand_b=4,
        gain_a=0.1, gain_b=0.2,
    )
    def compete(*args, **kwargs):
        return SimpleNamespace(
            winners=(1,), turns=(10,), points=((8, 10),), records=("record",),
            cleared=((trade,),), seating=((1, 0),),
        )

    monkeypatch.setattr(trade_census, "compete", compete)
    result = trade_census.run_census(
        [Entrant("first"), Entrant("second")], 1, records=True,
    )
    row, = result.trades
    assert (row.name_a, row.name_b) == ("second", "first")
    assert row.given_a == (2, 0, 0, 0, 0)
    assert row.given_b == (0, 1, 0, 0, 0)
    assert result.records == ["record"]
    summary = trade_census.summarize(result)
    assert summary["second"].large_hand_share == 1.0
    assert summary["first"].mean_net_cards == pytest.approx(1.0)
    assert summary["second"].trade_sides_per_seat_game_turn == pytest.approx(0.1)


def test_profiling_preserves_the_trade_gate_and_obeys_the_action_cap():
    class Gate:
        trade_floor = 0.03

        def gains_many(self, view, received, counterparties):
            return [0.1] * len(received)

    wrapped = TimedBot(Gate(), [])
    assert wrapped.trade_floor == 0.03
    assert wrapped.gains_many(None, [(1, 0, 0, 0, 0)], [1]) == [0.1]
    game, times = play_one_game("random", seed=3, action_cap=12)
    assert len(times) == 12
    assert all(t >= 0 for t in times)
    assert len(game.gates) == 4


def test_census_normalizes_repeated_labels_by_seat_exposure():
    row = trade_census.TradeRecord(
        game=0, turn=2, phase="MAIN", seat_a=0, seat_b=1,
        name_a="a", name_b="a", given_a=(2, 0, 0, 0, 0),
        given_b=(0, 1, 0, 0, 0), hand_before_a=8, hand_before_b=2,
        gain_a=100.0, gain_b=0.01,
    )
    result = trade_census.CensusResult(
        games=2, entrant_names=("a", "a", "b"),
        trades=[row], turns=[10, 30], winners=[0, None],
    )
    summary = trade_census.summarize(result)
    assert summary["a"].trade_sides == 2
    assert summary["a"].seat_games == 4
    assert summary["a"].seat_game_turns == 80
    assert summary["a"].trade_sides_per_seat_game_turn == pytest.approx(2 / 80)
    assert summary["a"].mean_net_cards == 0
    assert summary["a"].large_hand_share == 0.5
    assert summary["b"].trade_sides == 0
    assert summary["b"].seat_game_turns == 40
    assert result.to_json()["unfinished"] == 1
    assert "larger_gain" not in result.to_json()["trades"][0]


def test_board_metrics_keep_unfinished_games_and_do_not_count_pairs_twice():
    from hexset.bench.metrics import side_metrics

    tournament = SimpleNamespace(games=4, unfinished=2, winners=(0, None, 1, None))
    metrics = side_metrics(tournament, (0, 1))
    assert metrics["wins"] == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["decided"] == 2
    assert metrics["boards"] == 2
    assert metrics["interval_95"] == [0.5, 0.5]
    one_board = side_metrics(SimpleNamespace(games=2, unfinished=1, winners=(0, None)), (0,))
    assert one_board["interval_95"] == [0.0, 1.0]


def test_ablation_pairs_opposing_sides_and_counts_unfinished(monkeypatch):
    from hexset.bench import ablate

    def compete(lineup, games, **kwargs):
        assert [e.name for e in lineup] == ["challenger", "challenger", "incumbent", "incumbent"]
        return SimpleNamespace(games=4, unfinished=1, winners=(0, 2, 1, None))

    monkeypatch.setattr(ablate, "compete", compete)
    result = ablate.ablate("production", 4, seed=0, depth=1, width=1, workers=1)
    assert result["wins"] == 2
    assert result["win_rate"] == 0.5
    assert result["unfinished"] == 1


def test_baselines_json_keeps_raw_games_and_null_small_sample_bounds(monkeypatch, capsys):
    import json
    from hexset.arena import Standing, Tournament
    from hexset.bench import baselines

    def compete(lineup, games, **kwargs):
        return Tournament(
            standings=tuple(Standing(e.name, int(i == 0), games) for i, e in enumerate(lineup)),
            games=2, unfinished=1, mean_turns=10, seconds=0,
            winners=(0, None), points=((10, 4), (3, 5)), turns=(10, 10),
            seating=((0, 1), (1, 0)),
        )

    monkeypatch.setattr(baselines, "compete", compete)
    assert baselines.main(["--lineup", "random", "random", "--games", "2",
                           "--workers", "1", "--against", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    json.dumps(payload, allow_nan=False)
    assert payload["paired_points"][1]["mean"] == -2
    assert payload["paired_points"][1]["lower"] is None
    assert len(payload["experiment"]["outcomes"]) == 2
    assert payload["pooled"][0]["win_rate"] == 0.5


def test_weight_cell_preserves_unfinished_and_board_measurements(monkeypatch):
    from hexset.bench import weight_sweep

    monkeypatch.setattr(weight_sweep, "compete", lambda *args, **kwargs: SimpleNamespace(
        games=4, unfinished=2, winners=(0, None, 2, None), roads=((4, 2, 1, 1),) * 4,
    ))
    result = weight_sweep.run_cell(4, seed=0, workers=1,
                                   challenger=Entrant("a"), baseline=Entrant("b"))
    assert result["win_rate"] == 0.25
    assert result["unfinished"] == 2
    assert result["boards"] == 2
    assert result["challenger_roads_per_game"] == 3
