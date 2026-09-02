# SPDX-License-Identifier: GPL-3.0-only
"""PettingZoo's own conformance suite against `HexSetAEC`, plus the one
full-episode smoke test the design (`docs/gym-design.md` §5) calls for."""

from __future__ import annotations

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from pettingzoo.test import api_test, seed_test  # noqa: E402

from hexset.game import MAX_TURNS  # noqa: E402
from hexset.gym.aec import HexSetAEC  # noqa: E402

from gym_helpers import play_randomly  # noqa: E402


def test_pettingzoo_api_test():
    api_test(HexSetAEC(), num_cycles=2000)


def test_pettingzoo_seed_test():
    seed_test(lambda: HexSetAEC(), num_cycles=300)


def test_four_seat_random_episode_terminates():
    """One full 4-seat random-agent episode reaches `is_over`, with no
    exception and no live seat ever handed an empty mask."""
    env = HexSetAEC()
    env.reset(seed=7)
    steps = play_randomly(env, seed=7, max_steps=MAX_TURNS * 40)
    assert steps < MAX_TURNS * 40, "episode did not terminate within the step budget"
    assert not env.agents or all(
        env.terminations[a] or env.truncations[a] for a in env.agents
    )


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_player_counts_construct_and_reset(num_players):
    """`num_players` is configurable (`docs/gym-design.md`'s own "agents
    seat_0..seat_3 (configurable players)"); every count the engine itself
    supports (`hexset.state.new_game`, 2..6) should too."""
    env = HexSetAEC(num_players=num_players)
    env.reset(seed=0)
    assert env.agents == [f"seat_{s}" for s in range(num_players)]
    assert env.action_space(env.agents[0]).n > 0
