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


def test_determinism_under_seed():
    """Two envs, same seed, same (mask-respecting, deterministic) policy:
    identical trajectories."""

    def make():
        return HexSetEnv(opponents=("heximax", "heximax", "heximax"), learner_seat=0)

    env1, env2 = make(), make()
    obs1, info1 = env1.reset(seed=11)
    obs2, info2 = env2.reset(seed=11)
    assert np.array_equal(obs1, obs2)
    assert np.array_equal(info1["action_mask"], info2["action_mask"])

    for _ in range(20):
        action = _first_legal(info1["action_mask"])
        obs1, r1, term1, trunc1, info1 = env1.step(action)
        obs2, r2, term2, trunc2, info2 = env2.step(action)

        assert np.array_equal(obs1, obs2)
        assert np.array_equal(info1["action_mask"], info2["action_mask"])
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        if term1 or trunc1:
            break


def test_flatten_false_returns_dict_observation():
    env = HexSetEnv(opponents=("random", "random"), flatten=False)
    obs, info = env.reset(seed=1)
    assert set(obs) == {"hexes", "vertices", "edges", "globals"}
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape
        assert obs[key].dtype == space.dtype


def test_action_masks_matches_info():
    env = HexSetEnv(opponents=("random", "random"))
    _, info = env.reset(seed=2)
    assert np.array_equal(env.action_masks(), info["action_mask"].astype(bool))


def test_rotate_learner_seat_varies_across_resets():
    env = HexSetEnv(opponents=("random", "random", "random"), learner_seat="rotate")
    seats = set()
    for seed in range(20):
        env.reset(seed=seed)
        seats.add(env._learner_seat)
    assert len(seats) > 1, "rotate never actually rotated across 20 reset seeds"


def test_fixed_learner_seat_is_stable_across_resets():
    env = HexSetEnv(opponents=("random", "random", "random"), learner_seat=2)
    for seed in range(5):
        env.reset(seed=seed)
        assert env._learner_seat == 2
