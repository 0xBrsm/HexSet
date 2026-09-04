# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import fields

import pytest

from hexset.arena import compete, spawn
from hexset.board.board import random_base_board
from hexset.bots.evaluate import Weights
from hexset.evaluate_tiered import Weights as TieredWeights
from hexset.bots.heximax import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, Heximax
from hexset.tuning import (
    ANCHOR,
    WEIGHTS,
    entrant_for,
    perturb,
    tunable,
)


def test_the_anchor_is_never_tuned():
    names = tunable(Weights())
    assert ANCHOR not in names
    assert set(names) | {ANCHOR} == {f.name for f in fields(Weights)}


# --- heximax fits ---------------------------------------------------------
#
# P3's harness gap (`agents/reference/heximax.md` §7): before this, a
# "heximax" fit silently built `kind="search"` entrants over `evaluate2`,
# never a heximax bot at all. `evaluator="heximax-trading"` / "heximax-notrade"
# close that gap by making `entrant_for` build `kind="heximax"` entrants whose
# weights actually reach the bot's evaluator.


@pytest.mark.parametrize(
    "evaluator, mode, max_trades",
    [("heximax-trading", "honest", None), ("heximax-notrade", "notrade", 0)],
)
def test_a_heximax_fit_builds_heximax_entrants_carrying_the_candidate(
    evaluator, mode, max_trades
):
    candidate = perturb(WEIGHTS[evaluator](), random.Random(0), sigma=0.4, count=2)
    entrant = entrant_for("challenger", candidate, depth=2, width=6, evaluator=evaluator)

    assert entrant.kind == "heximax"
    assert entrant.mode == mode
    assert entrant.max_trades == max_trades
    assert entrant.k == 1
    assert entrant.weights == candidate

    board = random_base_board(random.Random(0))
    bot = spawn(entrant, board, random.Random(0))
    assert isinstance(bot, Heximax)
    assert bot.mode == mode
    assert bot.max_trades == max_trades
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
