# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers for `tests/gym/`: a mask-respecting random rollout."""

from __future__ import annotations

import random

import numpy as np

from hexset.gym.aec import HexSetAEC


def play_randomly(env: HexSetAEC, *, seed: int, max_steps: int = 20_000) -> int:
    """Drive `env` (already `reset()`) with a mask-respecting random policy
    until it terminates or `max_steps` AEC steps have been taken.

    Returns the number of steps taken. Never hands a live agent an action
    outside `action_mask` -- unlike `gymnasium.utils.env_checker.check_env`'s
    own blind `action_space.sample()`, which `HexSetAEC` itself is not asked
    to tolerate (only `HexSetEnv` is, see its own `step`).
    """
    rng = random.Random(seed)
    steps = 0
    while env.agents and steps < max_steps:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        mask = env.observe(agent)["action_mask"]
        legal = np.flatnonzero(mask)
        assert legal.size > 0, f"empty action mask for a live agent ({agent})"
        env.step(int(rng.choice(legal)))
        steps += 1
    return steps
