# SPDX-License-Identifier: GPL-3.0-only
"""`action_mask` must equal `hexset.actions.legal_actions`, checked over 50
random positions, per `docs/gym-design.md` §5.

This used to have to check the mask against a *second*, honest enumeration
(`server.rules.fair_legal_actions`), because the engine's own
`PROPOSE_TRADE` sample read every opponent's true hand to decide which
`want` anyone could cover. Trading is no longer an action
(`hexset.trading`), so no action's legality depends on another seat's hand,
there is only one list, and the gap the second enumeration existed to close
is gone by construction rather than by patching.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset.actions import legal_actions  # noqa: E402
from hexset.gym.aec import HexSetAEC  # noqa: E402


def _expected_mask(env: HexSetAEC) -> np.ndarray:
    mask = np.zeros(env._space.size, dtype=np.int8)
    for action in legal_actions(env._game):
        mask[env._space.index(action)] = 1
    return mask


def test_mask_matches_legal_actions_over_fifty_positions():
    env = HexSetAEC()
    rng = random.Random(0)
    checked = 0
    seed = 0
    while checked < 50:
        env.reset(seed=seed)
        seed += 1
        steps = 0
        while env.agents and checked < 50 and steps < 2000:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            observed = env.observe(agent)["action_mask"]
            assert np.array_equal(observed, _expected_mask(env)), (
                f"mask mismatch at seed {seed - 1}, step {steps}"
            )
            checked += 1

            legal = np.flatnonzero(observed)
            env.step(int(rng.choice(legal)))
            steps += 1


def test_no_action_legality_depends_on_another_seats_hand():
    """The property the second enumeration used to enforce, stated directly:
    permuting every opponent's hidden cards cannot change what the mover may
    do. Driven over real positions rather than asserted about the source."""
    env = HexSetAEC()
    rng = random.Random(1)
    for seed in range(5):
        env.reset(seed=seed)
        steps = 0
        while env.agents and steps < 300:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            game = env._game
            mover = env.possible_agents.index(agent)
            before = list(legal_actions(game))

            state = game.state(mover, hidden=False)
            keep = [hand[:] for hand in state.hands]
            pool = [
                card
                for seat, hand in enumerate(state.hands)
                if seat != mover
                for card, n in enumerate(hand)
                for _ in range(n)
            ]
            rng.shuffle(pool)
            cursor = 0
            for seat, hand in enumerate(state.hands):
                if seat == mover:
                    continue
                size = sum(hand)
                redealt = [0] * len(hand)
                for card in pool[cursor : cursor + size]:
                    redealt[card] += 1
                cursor += size
                state.hands[seat] = redealt
            assert list(legal_actions(game)) == before
            state.hands[:] = keep

            observed = env.observe(agent)["action_mask"]
            env.step(int(rng.choice(np.flatnonzero(observed))))
            steps += 1
