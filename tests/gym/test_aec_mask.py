# SPDX-License-Identifier: GPL-3.0-only
"""`action_mask` must equal `hexset.server.rules.fair_legal_actions`'s sample
-- never `hexset.actions.legal_actions`'s own, which reads opponents' true
hands to decide which `PROPOSE_TRADE` pairs they could cover
(`hexset.server.rules`'s module docstring). Checked over 50 random positions,
per `docs/gym-design.md` §5.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset.actions import ActionType, legal_actions  # noqa: E402
from hexset.gym.aec import HexSetAEC  # noqa: E402
from hexset.server.rules import fair_legal_actions  # noqa: E402


def _expected_mask(env: HexSetAEC) -> np.ndarray:
    mask = np.zeros(env._space.size, dtype=np.int8)
    for action in fair_legal_actions(env._game):
        mask[env._space.index(action)] = 1
    return mask


def test_mask_matches_fair_legal_actions_over_fifty_positions():
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


def test_mask_never_reflects_the_omniscient_propose_trade_sample():
    """The honest sample can be *wider* than the omniscient one for
    `PROPOSE_TRADE` -- `fair_legal_actions` offers "proposing is available"
    whenever the mover holds anything, regardless of whether any opponent
    could currently cover it, while `legal_actions` skips pairs nobody could
    cover. The one bit the flat space carries must come from the honest
    sample even when the two disagree, so this drives until they actually do
    and checks the direction of the disagreement rather than only their
    intersection."""
    env = HexSetAEC()
    rng = random.Random(1)
    propose_trade_slot = env._space.offsets[ActionType.PROPOSE_TRADE]
    for seed in range(30):
        env.reset(seed=seed)
        steps = 0
        while env.agents and steps < 500:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            game = env._game
            omniscient_has_trade = any(a.type is ActionType.PROPOSE_TRADE for a in legal_actions(game))
            honest_has_trade = any(a.type is ActionType.PROPOSE_TRADE for a in fair_legal_actions(game))
            observed = env.observe(agent)["action_mask"]
            observed_has_trade = bool(observed[propose_trade_slot])

            assert observed_has_trade == honest_has_trade
            if honest_has_trade and not omniscient_has_trade:
                # Found the disagreement this test exists to exercise; the
                # assertion above already confirmed the mask took the honest
                # side of it.
                return

            legal = np.flatnonzero(observed)
            env.step(int(rng.choice(legal)))
            steps += 1

    pytest.skip("no position in this sample separated the honest and omniscient samples")
