# SPDX-License-Identifier: GPL-3.0-only
"""Research-tool accounting, with fixed outcomes and bounded engine steps."""
from types import SimpleNamespace

import pytest

from hexset.arena import ClearedTrade, Entrant
from hexset.bench import trade_census


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
        [Entrant("first", "random"), Entrant("second", "random")], 1, records=True,
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
    assert summary["a"].mean_given == summary["a"].mean_received == 1.5
    assert summary["a"].mean_imbalance == pytest.approx(1 / 3)
    assert summary["a"].bundle_distribution == {"2:1": 2}
    assert summary["a"].mean_net_cards == 0
    assert summary["a"].large_hand_share == 0.5
    assert summary["b"].trade_sides == 0
    assert summary["b"].seat_game_turns == 40
    assert result.to_json()["unfinished"] == 1
    assert "larger_gain" not in result.to_json()["trades"][0]


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




def test_throughput_plays_random_games_under_the_unstructured_cap(monkeypatch):
    """Random play takes several times the turns real play does; under
    `MAX_TURNS` the benchmark aborted on its first long game."""
    from hexset.bench import throughput
    from hexset.game import UNSTRUCTURED_TURN_CAP

    seen = {}

    def compete(lineup, games, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(seconds=1.0, mean_turns=10.0, unfinished=0)

    monkeypatch.setattr(throughput, "compete", compete)
    throughput.run(4, 4, seed=0, workers=1)
    assert seen["turn_cap"] == UNSTRUCTURED_TURN_CAP


class _Stop(Exception):
    pass


@pytest.mark.parametrize("module,argv", [
    ("baselines", ["--lineup", "random", "random", "--games", "2", "--workers", "1"]),
    ("generate", ["--out", "unused.jsonl", "--bot", "random", "--players", "2",
                  "--games", "2", "--workers", "1"]),
    ("trade_census", ["random", "random", "--games", "2"]),
])
def test_every_runner_names_the_table_it_plays(monkeypatch, tmp_path, module, argv):
    """`--game-type`, `--turn-cap` and `--trade-mode` reach `compete` from
    every command that runs games, as they do from the duel."""
    import importlib

    from hexset.rules import DUEL_VARIANT_GAME

    runner = importlib.import_module(f"hexset.bench.{module}")
    seen = {}

    def compete(lineup, games, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(runner, "compete", compete)
    with pytest.raises(_Stop):
        runner.main([*argv, "--journal", str(tmp_path / "j.jsonl"), "--turn-cap", "3000",
                     "--game-type", "duel-variant", "--trade-mode", "auto"])
    assert (seen["turn_cap"], seen["game_type"], seen["trade_mode"]) == (
        3000, DUEL_VARIANT_GAME, "auto",
    )
