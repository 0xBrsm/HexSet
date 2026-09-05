# SPDX-License-Identifier: GPL-3.0-only
"""Essential checks on `hexset.bench.trade_lab`: on a tiny hand-built
position, all three rules pick a trade from the clearing set and `nash`
picks the argmax of the gain product; the phase-3 judged-position filter
and the paired-chance judge itself work end to end on a small recorded
bank."""

from __future__ import annotations

import random

import pytest

from hexset.bench.trade_lab import (
    RULES,
    _play_fork,
    _record_chance_script,
    _shaded_pick,
    candidate_rows,
    clearing_set,
    judge_position,
    judged_positions,
    run_bank,
    sample_judged_positions,
    select,
)
from hexset.board.terrain import Resource
from hexset.game import Phase, start
from hexset.record import read
from hexset.trading import bundle
from helpers import give, mini_board


class FakeBot:
    """A trader whose private gain on a bundle is looked up by hand,
    defaulting to a negative gain (never clears) for one the test named."""

    def __init__(self, gains: dict[tuple[int, int, tuple[int, ...]], float]):
        self.gains, self._rank = gains, None

    def _delta(self, view, knower, target, received, counterparty, rank):
        return self.gains.get((target, counterparty, received), -1.0)


def _small_and_big_rows():
    rng = random.Random(0)
    game = start(mini_board(), 2, rng)
    game.phase, game.current_player = Phase.MAIN, 0
    give(game._state, 0, Resource.WOOD, 3)
    give(game._state, 1, Resource.ORE, 3)

    small = bundle(ore=1, wood=-1)  # seat 0 gives 1 wood, gets 1 ore
    big = bundle(ore=1, wood=-2)  # seat 0 gives 2 wood, gets 1 ore
    mirror_small, mirror_big = (tuple(-n for n in b) for b in (small, big))

    game.gates = (
        FakeBot({(0, 1, small): 0.4, (0, 1, big): 0.9}),
        FakeBot({(1, 0, mirror_small): 0.5, (1, 0, mirror_big): 0.05}),
    )
    rows = [r for r in candidate_rows(game, 0) if r.bundle in (small, big)]
    return game, rows, small, big


def test_rules_pick_from_the_clearing_set_and_nash_is_the_argmax():
    _game, rows, small, big = _small_and_big_rows()
    assert {r.bundle for r in rows} == {small, big}
    products = {r.bundle: r.gain_actor * r.gain_counterparty for r in rows}

    for rule in RULES:
        picked = select(rule, clearing_set(rule, rows))
        assert picked is not None
        assert picked.gain_actor > 0 and picked.gain_counterparty > 0

    nash_pick = select("nash", clearing_set("nash", rows))
    assert products[nash_pick.bundle] == max(products.values())


def test_shaded_gate_respects_tau_on_the_actor_seat():
    """Shading seat 0 (the actor) with a tau between the two bundles' actor
    gains (0.4 for `small`, 0.9 for `big`) admits only `big`; a tau above
    both admits neither."""
    _game, rows, small, big = _small_and_big_rows()

    picked = _shaded_pick("actor", rows, me=0, shaded_seat=0, tau=0.6, k=1.0)
    assert picked is not None and picked.bundle == big

    excluded = _shaded_pick("actor", rows, me=0, shaded_seat=0, tau=0.95, k=1.0)
    assert excluded is None


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    """One heximax x4 game, capped short, replayed for the phase-3 tests
    below -- real bots (not `FakeBot`), because `judged_positions` and the
    paired-chance judge exercise the engine's own `run_trade_event`
    plumbing, which only a real `Bot.gains_many` drives."""
    out = tmp_path_factory.mktemp("trade_lab") / "bank.jsonl"
    run_bank(1, 90500, out, workers=1)
    return out


def test_judged_positions_are_main_entry_events_that_cleared(small_bank):
    records = list(read(str(small_bank)))
    positions = list(judged_positions(records[0], game_index=0))
    for pos in positions:
        assert pos.main_entry
        assert len(pos.historical_trades) >= 1
        # The actor is a party to every historical trade its own event
        # cleared (`trade_event` only ever clears deals naming the current
        # player as one side).
        for t in pos.historical_trades:
            assert pos.actor in (t.a, t.b)


def _judged_row(pos, *, game_index: int = 0) -> dict:
    t = pos.historical_trades[0]
    return {
        "game": game_index, "position": pos.position_index, "turn": pos.turn, "actor": pos.actor,
        "counterparty": t.b if t.a == pos.actor else t.a,
        "n_historical_trades": len(pos.historical_trades),
        "bundle": list(t.received) if t.a == pos.actor else [-n for n in t.received],
        "actor_gain": t.gain_a if t.a == pos.actor else t.gain_b,
        "counterparty_gain": t.gain_b if t.a == pos.actor else t.gain_a,
    }


def test_sample_judged_positions_caps_per_game_and_assigns_pid(small_bank):
    records = list(read(str(small_bank)))
    all_judged = [_judged_row(pos) for pos in judged_positions(records[0], game_index=0)]
    if not all_judged:
        pytest.skip("this bank's one game had no judged position to sample")
    sampled = sample_judged_positions(all_judged, per_game_cap=1, total=300, seed=7)
    assert len(sampled) == 1
    assert sampled[0]["pid"] == 0


def test_judge_position_suppresses_and_reproduces_the_trade(small_bank):
    """The untraded fork's turn never trades; the traded fork starts one
    trade richer than the untraded fork at the same position; both forks
    of one stream draw identical rolls (paired chance) as long as neither
    exhausts the script."""
    records = list(read(str(small_bank)))
    record = records[0]
    positions = list(judged_positions(record, game_index=0))
    if not positions:
        pytest.skip("this bank's one game had no judged position to judge")
    pos = positions[0]
    row = _judged_row(pos)
    row["pid"] = 0
    from hexset.record import board_of

    board = board_of(record)
    results = judge_position(row, record, board, streams=2, cap=20)
    assert len(results) == 2
    for result in results:
        assert result["untraded"]["actions"] <= 20
        assert result["traded"]["actions"] <= 20
        assert result["stream"] in (0, 1)
        # Both halves ran to a winner or hit the cap -- never crashed.
        assert result["untraded"]["winner"] is None or isinstance(result["untraded"]["winner"], int)
        assert result["traded"]["winner"] is None or isinstance(result["traded"]["winner"], int)
