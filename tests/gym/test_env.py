# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetEnv`: Gymnasium conformance and one episode against a bot the
consumer registered (`tests/traders.py`)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from hexset.arena import MAX_ACTIONS  # noqa: E402
from hexset.gym.env import HexSetEnv  # noqa: E402


def _first_legal(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def test_check_env():
    env = HexSetEnv(opponents=("random", "random"))
    check_env(env, skip_render_check=True)


def test_episode_vs_a_registered_bot_ends():
    """One opponent seat, not three: the episode's end is the contract, a
    win or the turn cap's truncation, well inside the arena's action cap."""
    env = HexSetEnv(opponents=("test-trader",), learner_seat=0)
    obs, info = env.reset(seed=3)
    steps = 0
    while steps < MAX_ACTIONS:
        action = _first_legal(info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert terminated or truncated, "episode did not end within MAX_ACTIONS learner steps"
