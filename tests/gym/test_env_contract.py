# SPDX-License-Identifier: GPL-3.0-only
"""What the two PettingZoo/Gymnasium environments promise around the game:
an action outside the mask never reaches it, an unseeded reset is drawn
from the environment's own seeded stream, a game that runs out of turns is
a truncation worth nothing, and the opponents a `HexSetEnv` can name."""

from __future__ import annotations

import gc
import os
import random
import subprocess
import sys
import weakref

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset import arena  # noqa: E402
from hexset.actions import Action, ActionType, legal_actions  # noqa: E402
from hexset.game import Phase, is_over, to_move  # noqa: E402
from hexset.gym.aec import HexSetAEC  # noqa: E402
from hexset.gym.env import HexSetEnv  # noqa: E402
from hexset.victory import victory_points  # noqa: E402


def _snapshot(game) -> tuple:
    state = game._state
    return (
        [hand[:] for hand in state.hands],
        state.bank[:],
        [cards[:] for cards in state.dev_cards],
        state.edge_owner[:],
        state.vertex_building[:],
        game.phase,
        game.current_player,
        game.turns,
        game.free_roads,
        game.dev_card_played,
    )


def _main_with_cards(env: HexSetAEC, seat: int | None = None) -> int:
    """Play first-legal moves until `seat` (any, by default) is in its main
    phase, then hand it five of every resource: an illegal build is then
    one it can pay for, the case an engine paying before it validates would
    leave changed."""
    game = env._game
    for _ in range(5000):
        mover = to_move(game)
        if game.phase is Phase.MAIN and (seat is None or mover == seat):
            game._state.hands[mover] = [5] * 5
            return mover
        agent = env.agent_selection
        env.step(env._space.index(legal_actions(game, env.possible_agents.index(agent))[0]))
    raise AssertionError("no main phase came up")


def _illegal_road(env: HexSetAEC, seat: int) -> int:
    game = env._game
    legal = set(legal_actions(game, seat))
    for edge in range(len(game._state.edge_owner)):
        action = Action(ActionType.BUILD_ROAD, edge)
        if action not in legal:
            return env._space.index(action)
    raise AssertionError("every edge is a legal road")


# --- an action outside the mask ---------------------------------------------


def test_the_aec_refuses_an_action_outside_the_mask_before_applying_it():
    env = HexSetAEC()
    env.reset(seed=3)
    seat = _main_with_cards(env)
    index = _illegal_road(env, seat)
    assert not env.observe(env.agent_selection)["action_mask"][index]
    before = _snapshot(env._game)
    with pytest.raises(ValueError, match="action mask"):
        env.step(index)
    assert _snapshot(env._game) == before
    with pytest.raises(ValueError):
        env.step(env._space.size)  # not an action at all


def test_the_learner_env_treats_an_action_outside_the_mask_as_a_true_no_op():
    env = HexSetEnv(opponents=("random", "random", "random"), learner_seat=0)
    env.reset(seed=3)
    game = env._aec._game
    for _ in range(5000):
        if game.phase is Phase.MAIN and to_move(game) == 0:
            break
        _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert not (terminated or truncated)
    game._state.hands[0] = [5] * 5
    index = _illegal_road(env._aec, 0)
    before = _snapshot(game)
    _, reward, terminated, truncated, _ = env.step(index)
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert _snapshot(game) == before


# --- seeding ------------------------------------------------------------------


def _deal(game) -> tuple:
    board = game._state.board
    return tuple(int(t) for t in board.terrain), tuple(board.tokens)


def test_an_unseeded_aec_reset_follows_the_last_seeded_one():
    deals = []
    for _ in range(2):
        env = HexSetAEC()
        env.reset(seed=7)
        run = []
        for _ in range(3):
            env.reset()
            run.append(_deal(env._game))
        deals.append(run)
    assert deals[0] == deals[1]
    assert len(set(deals[0])) == 3, "every unseeded reset dealt the same board"


def test_an_unseeded_learner_env_reset_follows_the_last_seeded_one():
    deals = []
    for _ in range(2):
        env = HexSetEnv(opponents=("random", "random"))
        env.reset(seed=7)
        run = []
        for _ in range(3):
            env.reset()
            run.append((_deal(env._aec._game), env._learner_seat))
        deals.append(run)
    assert deals[0] == deals[1]


# --- the turn cap ---------------------------------------------------------------


@pytest.mark.parametrize("reward", ["terminal", "relative_points"])
def test_a_game_out_of_turns_is_truncated_with_nothing_scored(reward):
    """Including `relative_points`, where the seats' points at the cap would
    otherwise score an outcome nobody reached. A seed whose seats end the
    cap on different points is what makes that case bite. Read on the step
    that ends the game: PettingZoo drops a done agent's entries as it is
    stepped out."""
    rng = random.Random(0)
    for seed in range(20):
        env = HexSetAEC(num_players=2, reward=reward, turn_cap=30)
        env.reset(seed=seed)
        game = env._game
        while not is_over(game):
            agent = env.agent_selection
            legal = legal_actions(game, env.possible_agents.index(agent))
            env.step(env._space.index(rng.choice(legal)))
        assert game.won_by is None and game.turns == 30
        points = {victory_points(game.state(s, hidden=False), s) for s in range(2)}
        if len(points) > 1:
            break
    else:
        pytest.fail("no seed ended the cap with the seats on different points")
    assert all(env.truncations.values()) and not any(env.terminations.values())
    assert env.rewards == dict.fromkeys(env.possible_agents, 0.0)
    assert env._cumulative_rewards == dict.fromkeys(env.possible_agents, 0.0)


def test_the_learner_env_passes_its_turn_cap_to_the_game():
    env = HexSetEnv(opponents=("random",), turn_cap=12)
    env.reset(seed=0)
    assert env._aec._game.turn_cap == 12


def test_a_turn_cap_below_one_turn_is_refused():
    with pytest.raises(ValueError):
        HexSetAEC(turn_cap=0)


# --- opponents ------------------------------------------------------------------


def test_a_retired_seat_is_not_an_opponent():
    with pytest.raises(ValueError, match="retired"):
        HexSetEnv(opponents=("random", "retired"))


@pytest.fixture
def probe_runtime(tmp_path, monkeypatch):
    """A runtime module on the path that registers one preset, removed again
    with its registration afterwards."""
    name = "gym_runtime_probe"
    (tmp_path / f"{name}.py").write_text(
        "from hexset.arena import Entrant, register_preset\n"
        "register_preset('probe-random', Entrant('probe-random', kind='random'))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    presets = dict(arena.PRESETS)
    yield name
    sys.modules.pop(name, None)
    arena.PRESETS.clear()
    arena.PRESETS.update(presets)


def test_a_runtime_is_loaded_before_the_opponents_are_named(probe_runtime):
    with pytest.raises(ValueError):
        HexSetEnv(opponents=("probe-random",))
    env = HexSetEnv(opponents=("probe-random",), runtime=(probe_runtime,))
    env.reset(seed=0)
    assert env.runtime == (probe_runtime,)
    assert HexSetEnv(opponents=("probe-random",), runtime=probe_runtime).runtime == (
        probe_runtime,
    )


# --- the environment object ----------------------------------------------------


def test_an_env_is_not_kept_alive_by_its_spaces():
    env = HexSetAEC()
    env.observation_space("player_0")
    env.action_space("player_0")
    ref = weakref.ref(env)
    del env
    gc.collect()
    assert ref() is None


def test_star_import_on_a_base_install_names_only_what_imports():
    """`from hexset.gym import *` with neither gymnasium nor PettingZoo
    importable: `__all__` must not name the environments that need them."""
    code = (
        "import sys\n"
        "sys.modules['gymnasium'] = None\n"
        "sys.modules['pettingzoo'] = None\n"
        "from hexset.gym import *\n"
        "import hexset.gym as gym\n"
        "assert 'HexSetEnv' not in gym.__all__, gym.__all__\n"
        "assert LaneEnv and Outcome\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))
