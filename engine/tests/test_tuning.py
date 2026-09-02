# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import fields

import pytest

from hexset.arena import compete, spawn
from hexset.board.board import random_base_board
from hexset.evaluate import Weights
from hexset.evaluate_tiered import Weights as TieredWeights
from hexset.heximax import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, Heximax
from hexset.tuning import (
    ANCHOR,
    WEIGHTS,
    as_source,
    climb,
    duel,
    entrant_for,
    perturb,
    tunable,
)


def changed_fields(before: Weights, after: Weights) -> set[str]:
    return {
        f.name
        for f in fields(Weights)
        if getattr(before, f.name) != getattr(after, f.name)
    }


def test_the_anchor_is_never_tuned():
    names = tunable(Weights())
    assert ANCHOR not in names
    assert set(names) | {ANCHOR} == {f.name for f in fields(Weights)}


def test_perturbing_moves_only_the_requested_number_of_weights():
    start = Weights()
    for seed in range(20):
        moved = changed_fields(start, perturb(start, random.Random(seed), sigma=0.4, count=2))
        assert len(moved) <= 2
        assert ANCHOR not in moved


def test_perturbing_can_revive_a_weight_sitting_at_zero():
    start = Weights(port=0.0)
    revived = any(
        perturb(start, random.Random(seed), sigma=0.4, count=len(tunable(start))).port != 0.0
        for seed in range(10)
    )
    assert revived


def test_the_tiered_weights_tune_too_and_keep_their_hierarchy():
    """A step scaled by the weight cannot move a term out of its tier."""
    start = TieredWeights()
    moved = perturb(start, random.Random(3), sigma=0.4, count=len(tunable(start)))

    assert moved != start
    assert moved.victory_point == start.victory_point
    assert moved.production > abs(moved.reachable_production_1)
    assert abs(moved.reachable_production_1) > abs(moved.hand_synergy)
    assert "reachable_production_1" in as_source(moved)


def test_perturbing_leaves_the_original_alone():
    start = Weights()
    perturb(start, random.Random(0), sigma=0.9, count=4)
    assert start == Weights()


def test_a_duel_returns_wins_out_of_the_games_that_finished():
    wins, decided = duel(
        Weights(), Weights(), 4, seed=0, depth=1, width=None
    )
    assert 0 <= wins <= decided <= 4


def test_a_zero_step_proposes_the_incumbent_unchanged():
    start = Weights()
    assert perturb(start, random.Random(0), sigma=0.0, count=4) == start


def test_a_stricter_bar_accepts_no_more_than_a_looser_one():
    """The whole point of `z`: it trades progress against drift monotonically.

    One round only. As soon as the two runs disagree about an acceptance their
    incumbents differ, and every later duel is a different experiment.
    """
    strict = climb(rounds=1, games=4, sigma=0.5, seed=3, z=1.96)[1][0]
    loose = climb(rounds=1, games=4, sigma=0.5, seed=3, z=0.0)[1][0]

    assert strict.wins == loose.wins
    assert strict.lower <= loose.lower
    assert strict.accepted <= loose.accepted


def test_a_climb_records_every_round_it_tried():
    _, history = climb(rounds=2, games=4, sigma=0.5, seed=1)
    assert [step.round for step in history] == [0, 1]
    assert all(step.decided <= 4 for step in history)


def test_the_climb_reports_each_step_as_it_goes():
    seen = []
    climb(rounds=2, games=4, sigma=0.5, seed=1, report=seen.append)
    assert len(seen) == 2


def test_weights_round_trip_through_their_source_form():
    original = Weights(production=1.234, surplus_card=-0.5)
    restored = eval(as_source(original), {"Weights": Weights})
    for f in fields(Weights):
        assert getattr(restored, f.name) == pytest.approx(getattr(original, f.name))


# --- heximax fits ---------------------------------------------------------
#
# P3's harness gap (`agents/reference/heximax.md` §7): before this, a
# "heximax" fit silently built `kind="search"` entrants over `evaluate2`,
# never a heximax bot at all. `evaluator="heximax-trading"` / "heximax-notrade"
# close that gap by making `entrant_for` build `kind="heximax"` entrants whose
# weights actually reach the bot's evaluator.


@pytest.mark.parametrize(
    "evaluator, mode, max_offers",
    [("heximax-trading", "honest", 3), ("heximax-notrade", "notrade", 0)],
)
def test_a_heximax_fit_builds_heximax_entrants_carrying_the_candidate(
    evaluator, mode, max_offers
):
    candidate = perturb(WEIGHTS[evaluator](), random.Random(0), sigma=0.4, count=2)
    entrant = entrant_for("challenger", candidate, depth=2, width=6, evaluator=evaluator)

    assert entrant.kind == "heximax"
    assert entrant.mode == mode
    assert entrant.max_offers == max_offers
    assert entrant.k == 1
    assert entrant.weights == candidate

    board = random_base_board(random.Random(0))
    bot = spawn(entrant, board, random.Random(0))
    assert isinstance(bot, Heximax)
    assert bot.mode == mode
    assert bot.max_offers == max_offers
    assert bot.evaluator.weights == candidate


def test_the_trading_profile_starts_from_todays_fit():
    assert WEIGHTS["heximax-trading"]() == TRADING_WEIGHTS


def test_the_notrade_profile_starts_from_its_own_table_not_the_bare_default():
    start = WEIGHTS["heximax-notrade"]()
    assert start == NO_TRADE_WEIGHTS
    assert start != Weights()


def test_default_and_tiered_fits_are_unaffected_by_the_heximax_registry():
    entrant = entrant_for("challenger", Weights(), depth=1, width=None)
    assert entrant.kind == "greedy"
    assert entrant.evaluator == "default"

    searched = entrant_for("challenger", Weights(), depth=2, width=6)
    assert searched.kind == "search"

    tiered = entrant_for(
        "challenger", TieredWeights(), depth=1, width=None, evaluator="tiered"
    )
    assert tiered.kind == "greedy"
    assert tiered.evaluator == "tiered"


def test_a_heximax_duel_runs_end_to_end():
    """Smoke: perturb the incumbent, then one 2-game duel, no-trade profile.

    `duel()` itself always plays a 4-seat lineup (`[a, b, a, b]`, rotated), so
    the 2-game shape here is built directly from `entrant_for` + `compete` --
    the same pieces `duel()` composes, kept to the smallest legal game count.
    """
    incumbent = WEIGHTS["heximax-notrade"]()
    challenger = perturb(incumbent, random.Random(1), sigma=0.4, count=2)

    a = entrant_for("challenger", challenger, 2, 6, evaluator="heximax-notrade")
    b = entrant_for("incumbent", incumbent, 2, 6, evaluator="heximax-notrade")
    result = compete([a, b], 2, seed=0)

    assert result.games == 2
    names = {s.name for s in result.standings}
    assert names == {"challenger", "incumbent"}
    assert sum(s.wins for s in result.standings) + result.unfinished == 2
