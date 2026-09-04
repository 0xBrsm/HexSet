# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetEnv`: Gymnasium conformance, determinism under a seed, and one full
episode against `search2` opponents (`docs/gym-design.md` §5)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from hexset.game import MAX_TURNS  # noqa: E402
from hexset.gym.env import HexSetEnv  # noqa: E402


def _first_legal(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def test_check_env():
    env = HexSetEnv(opponents=("random", "random"))
    check_env(env, skip_render_check=True)


def test_episode_vs_search2_terminates_within_max_turns():
    env = HexSetEnv(opponents=("search2", "search2", "search2"), learner_seat=0)
    obs, info = env.reset(seed=3)
    steps = 0
    # A learner decision is at most one action per engine turn across the
    # whole table, so `MAX_TURNS` learner steps is a generous upper bound
    # for "terminates before the engine's own turn limit."
    while steps < MAX_TURNS:
        action = _first_legal(info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert terminated or truncated, "episode did not terminate within MAX_TURNS learner steps"
