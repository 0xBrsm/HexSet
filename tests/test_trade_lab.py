# SPDX-License-Identifier: GPL-3.0-only
"""One essential check on `hexset.bench.trade_lab`'s selection rules: on a
tiny hand-built position, all four rules pick a trade from the clearing
set, and `nash` picks the argmax of the gain product."""

from __future__ import annotations

import random

from hexset.bench.trade_lab import RULES, _shaded_pick, candidate_rows, clearing_set, select
from hexset.board.terrain import Resource
from hexset.game import Phase, start
from hexset.trading import bundle
from helpers import give, mini_board


class FakeBot:
    """A trader whose private gain on a bundle is looked up by hand,
    defaulting to a negative gain (never clears) for one the test named."""

    def __init__(self, gains: dict[tuple[int, int, tuple[int, ...]], float]):
        self.gains, self._rank = gains, None

    def _delta(self, view, knower, target, received, counterparty, rank):
        return self.gains.get((target, counterparty, received), -1.0)


def test_rules_pick_from_the_clearing_set_and_nash_is_the_argmax():
    rng = random.Random(0)
    game = start(mini_board(), 2, rng)
    game.phase, game.current_player = Phase.MAIN, 0
    give(game._state, 0, Resource.WOOD, 3)
    give(game._state, 1, Resource.ORE, 3)
    game.valuations[0] = bundle(ore=1.0, wood=-1.0)
    game.valuations[1] = bundle(wood=1.0, ore=-1.0)

    small = bundle(ore=1, wood=-1)  # seat 0 gives 1 wood, gets 1 ore
    big = bundle(ore=1, wood=-2)  # seat 0 gives 2 wood, gets 1 ore
    mirror_small, mirror_big = (tuple(-n for n in b) for b in (small, big))

    game.gates = (
        FakeBot({(0, 1, small): 0.4, (0, 1, big): 0.9}),
        FakeBot({(1, 0, mirror_small): 0.5, (1, 0, mirror_big): 0.05}),
    )

    rows = [r for r in candidate_rows(game, 0) if r.bundle in (small, big)]
    assert {r.bundle for r in rows} == {small, big}
    products = {r.bundle: r.gain_actor * r.gain_counterparty for r in rows}

    for rule in RULES:
        picked = select(rule, clearing_set(rule, rows))
        assert picked is not None
        assert picked.gain_actor > 0 and picked.gain_counterparty > 0

    nash_pick = select("nash", clearing_set("nash", rows))
    assert products[nash_pick.bundle] == max(products.values())


def test_shaded_gate_respects_tau_on_the_actor_seat():
    """Same hand-built position: shading seat 0 (the actor) with a tau
    between the two bundles' actor gains (0.4 for `small`, 0.9 for `big`)
    admits only `big`; a tau above both admits neither."""
    rng = random.Random(0)
    game = start(mini_board(), 2, rng)
    game.phase, game.current_player = Phase.MAIN, 0
    give(game._state, 0, Resource.WOOD, 3)
    give(game._state, 1, Resource.ORE, 3)
    game.valuations[0] = bundle(ore=1.0, wood=-1.0)
    game.valuations[1] = bundle(wood=1.0, ore=-1.0)

    small = bundle(ore=1, wood=-1)
    big = bundle(ore=1, wood=-2)
    mirror_small, mirror_big = (tuple(-n for n in b) for b in (small, big))

    game.gates = (
        FakeBot({(0, 1, small): 0.4, (0, 1, big): 0.9}),
        FakeBot({(1, 0, mirror_small): 0.5, (1, 0, mirror_big): 0.05}),
    )

    rows = [r for r in candidate_rows(game, 0) if r.bundle in (small, big)]
    assert {r.bundle for r in rows} == {small, big}

    picked = _shaded_pick("actor", rows, me=0, shaded_seat=0, tau=0.6, k=1.0)
    assert picked is not None and picked.bundle == big

    excluded = _shaded_pick("actor", rows, me=0, shaded_seat=0, tau=0.95, k=1.0)
    assert excluded is None
