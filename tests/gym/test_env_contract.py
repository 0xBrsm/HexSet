# SPDX-License-Identifier: GPL-3.0-only
"""What the two PettingZoo/Gymnasium environments promise around the game:
the turn cap they pass to it, and the opponents a `HexSetEnv` can name."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset import arena  # noqa: E402
from hexset.gym.aec import HexSetAEC  # noqa: E402
from hexset.gym.env import HexSetEnv  # noqa: E402


# --- the turn cap ---------------------------------------------------------------


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
