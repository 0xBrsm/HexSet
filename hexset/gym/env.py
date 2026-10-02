# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetEnv`: a single-agent Gymnasium wrapper around `HexSetAEC`.

One seat is the learner; every other seat is an `hexset.arena` entrant,
auto-played inside `step`/`reset` until the learner is next to move or the
episode ends.
"""

from __future__ import annotations

import random
from typing import Any, Literal, Sequence

import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import register as gym_register
from gymnasium.envs.registration import registry as gym_registry

from hexset import encoding
from hexset.actions import Action
from hexset.arena import RETIRED, Entrant, entrant_from_name, load_runtime, spawn
from hexset.bots import Bot
from hexset.game import MAX_TURNS, is_over, may_act, to_move

from .aec import TOPOLOGY, HexSetAEC, agent_name

__all__ = [
    "ENV_ID",
    "DEFAULT_OPPONENTS",
    "register",
    "HexSetEnv",
]


ENV_ID = "HexSet-v0"

# One seat short of the standard 4-seat table, so
# `len(opponents) + 1 == num_players`.
DEFAULT_OPPONENTS: tuple[str, ...] = ("random", "random", "random")


def register() -> None:
    """Register `HexSet-v0` with Gymnasium. Safe to call more than once: a
    second call is a no-op rather than an "id already registered" error."""
    if ENV_ID not in gym_registry:
        gym_register(id=ENV_ID, entry_point="hexset.gym:HexSetEnv")


def _flat_size(num_players: int) -> int:
    return (
        TOPOLOGY.num_hexes * encoding.HEX_FEATURES
        + TOPOLOGY.num_vertices * encoding.vertex_features(num_players)
        + TOPOLOGY.num_edges * encoding.edge_features(num_players)
        + encoding.global_features(num_players)
    )


def _flatten(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            observation["hexes"].ravel(),
            observation["vertices"].ravel(),
            observation["edges"].ravel(),
            observation["globals"].ravel(),
        ]
    ).astype(np.float32)


class HexSetEnv(Env):
    """`gymnasium.Env` with one learner seat; the rest are `hexset.arena` bots.

    `learner_seat` is a seat index or `"rotate"` (default), since seat is not
    neutral; `opponents` is one preset name or `<kind>:<checkpoint>` spec per
    non-learner seat, never `retired`; `runtime` names modules to import
    first (`hexset.arena.load_runtime`), so the bots they register can be
    named here, in this process and in every worker of a vectorised env;
    `flatten=True` (default) concatenates the four encoder arrays into one
    `Box`. `reward` and `turn_cap` are `HexSetAEC`'s; the default random
    opponents need `hexset.game.UNSTRUCTURED_TURN_CAP` to finish most games.

    The action mask is always `info["action_mask"]`, never in the
    observation, with `action_masks()` as the `sb3-contrib` hook, and
    `info["view"]` carries the seat's `hexset.view.View`. An action outside
    the mask is a no-op with reward 0, for a caller that deliberately ignores
    the mask such as `check_env`: nothing is applied. The learner has no
    gate, so it never trades and cannot be traded with -- use `LaneEnv`.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        learner_seat: int | Literal["rotate"] = "rotate",
        opponents: Sequence[str] = DEFAULT_OPPONENTS,
        *,
        reward: str = "terminal",
        discard_order: str = "random",
        flatten: bool = True,
        render_mode: str | None = None,
        turn_cap: int = MAX_TURNS,
        runtime: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if not opponents:
            raise ValueError("HexSetEnv needs at least one opponent seat")
        num_players = len(opponents) + 1
        if isinstance(learner_seat, int) and not 0 <= learner_seat < num_players:
            raise ValueError(f"learner_seat {learner_seat} out of range for {num_players} players")
        if isinstance(runtime, str):
            runtime = (runtime,)
        load_runtime(*runtime)
        entrants = [entrant_from_name(name) for name in opponents]
        # A retired seat never moves, and an episode cannot play around one:
        # the opponents fill the table, so a smaller table is fewer opponents.
        if any(entrant.kind == RETIRED for entrant in entrants):
            raise ValueError(
                f"{RETIRED!r} is not an opponent; a table of fewer seats is fewer opponents"
            )

        self._aec = HexSetAEC(
            num_players=num_players,
            reward=reward,
            discard_order=discard_order,
            render_mode=render_mode,
            turn_cap=turn_cap,
        )
        self.learner_seat_config = learner_seat
        self.opponent_names: tuple[str, ...] = tuple(opponents)
        self.runtime: tuple[str, ...] = tuple(runtime)
        self._entrants: list[Entrant] = entrants
        self.flatten = flatten
        self.render_mode = render_mode

        self._learner_seat = 0
        self._bots: dict[int, Bot] = {}
        self._last_mask = np.zeros(self._aec.action_space(agent_name(0)).n, dtype=np.int8)

        self.action_space = spaces.Discrete(self._aec.action_space(agent_name(0)).n)
        self.observation_space = self._build_observation_space(num_players)

    def _build_observation_space(self, num_players: int) -> spaces.Space:
        if self.flatten:
            size = _flat_size(num_players)
            return spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        return spaces.Dict(
            {
                "hexes": spaces.Box(
                    -np.inf, np.inf, (TOPOLOGY.num_hexes, encoding.HEX_FEATURES), dtype=np.float32
                ),
                "vertices": spaces.Box(
                    -np.inf,
                    np.inf,
                    (TOPOLOGY.num_vertices, encoding.vertex_features(num_players)),
                    dtype=np.float32,
                ),
                "edges": spaces.Box(
                    -np.inf, np.inf, (TOPOLOGY.num_edges, encoding.edge_features(num_players)), dtype=np.float32
                ),
                "globals": spaces.Box(
                    -np.inf, np.inf, (encoding.global_features(num_players),), dtype=np.float32
                ),
            }
        )

    # -- gymnasium.Env ------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is None:
            # Off `np_random`, which the seeded reset above reseeded, so the
            # resets after a seeded one are as reproducible as it is.
            seed = int(self.np_random.integers(2**31))
        episode_rng = random.Random(seed)

        if self.learner_seat_config == "rotate":
            self._learner_seat = episode_rng.randrange(self._aec.num_players)
        else:
            self._learner_seat = self.learner_seat_config

        self._aec.reset(seed=episode_rng.randrange(2**31))

        board = self._aec._game.state(0, hidden=False).board
        self._bots = {}
        for offset, entrant in enumerate(self._entrants, start=1):
            seat = (self._learner_seat + offset) % self._aec.num_players
            bot_rng = random.Random(episode_rng.randrange(2**31))
            self._bots[seat] = spawn(entrant, board, bot_rng)
        # The engine's trade event asks each seat's own `gains_many`; the
        # learner has no bot, so its gate is `None`.
        self._aec._game.gates = tuple(
            self._bots.get(seat) for seat in range(self._aec.num_players)
        )

        self._auto_play_opponents()
        observation, info = self._observe_learner()
        return observation, info

    def step(self, action: int | Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        learner = agent_name(self._learner_seat)
        if self._aec.agent_selection != learner:
            raise RuntimeError("HexSetEnv.step() called when it is not the learner's turn")

        if self._aec._legal(self._aec._game, self._learner_seat, action) is None:
            # Outside the mask: checked here, before the game is touched, so
            # it is a no-op rather than a crashed episode -- for a caller that
            # deliberately ignores the mask, such as `check_env`.
            observation, info = self._observe_learner()
            return observation, 0.0, False, False, info
        self._aec.step(action)

        self._auto_play_opponents()

        observation, info = self._observe_learner()
        reward = float(self._aec.rewards[learner])
        terminated = self._aec.terminations[learner]
        truncated = self._aec.truncations[learner]
        return observation, reward, terminated, truncated, info

    def render(self):
        return self._aec.render()

    def close(self) -> None:
        self._aec.close()

    def action_masks(self) -> np.ndarray:
        """`sb3_contrib.MaskablePPO`'s hook: the learner's legal-action mask
        for the current decision, as a boolean array."""
        return self._last_mask.astype(bool)

    # -- internals ------------------------------------------------------

    def _auto_play_opponents(self) -> None:
        """Play every non-learner decision until the learner is entitled to one
        -- entitled, not `to_move`, since several seats owe a discard at once.
        Bot seats are driven in `to_move` order because
        `hexset.bots.Bot.choose(game)` takes the position and nothing else."""
        aec = self._aec
        learner = agent_name(self._learner_seat)
        while True:
            if aec.terminations[learner] or aec.truncations[learner]:
                return
            if aec.agent_selection == learner:
                return
            game = aec._game
            if not is_over(game) and may_act(game, self._learner_seat):
                aec.select_agent(learner)
                return
            agent = aec.agent_selection
            if aec.terminations[agent] or aec.truncations[agent]:
                aec.step(None)
                continue
            seat = to_move(game)
            if agent != agent_name(seat):
                aec.select_agent(agent_name(seat))
            bot = self._bots[seat]
            action = bot.choose(game)
            aec.step(action)

    def _observe_learner(self) -> tuple[Any, dict[str, Any]]:
        learner = agent_name(self._learner_seat)
        obs = self._aec.observe(learner)
        self._last_mask = obs["action_mask"]
        observation = _flatten(obs["observation"]) if self.flatten else obs["observation"]
        info: dict[str, Any] = {
            "action_mask": obs["action_mask"],
            "view": self._aec._game.state(self._learner_seat, hidden=True),
        }
        return observation, info
