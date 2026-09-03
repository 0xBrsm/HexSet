# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetEnv`: a single-agent Gymnasium wrapper around `HexSetAEC`.

One seat is the learner; every other seat is an `hexset.arena` entrant,
auto-played inside `step`/`reset` until the learner is next to move or the
episode ends -- the same `_advance_until_p0_decision` shape Catanatron's own
`CatanatronEnv` uses (pinned catanatron `d3f4ad05bb7`,
`catanatron/gym/envs/catanatron_env.py`), and the pattern
`hexset.arena.play`'s own loop already follows, restricted to non-learner
seats.
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
from hexset.arena import Entrant, entrant_from_name, spawn
from hexset.bots import Bot
from hexset.trading import publish_valuation

from .aec import TOPOLOGY, HexSetAEC, agent_name

ENV_ID = "HexSet-v0"

# The default lineup: three honest `heximax` opponents (mode="honest", the
# live-deployment referent -- `hexset.arena.Entrant.mode`'s own default --
# never `heximax-omni`, the perfect-information evaluation ceiling), one seat
# short of the standard 4-seat table so `len(opponents) + 1 == num_players`.
DEFAULT_OPPONENTS: tuple[str, ...] = ("heximax", "heximax", "heximax")


def register() -> None:
    """Register `HexSet-v0` with Gymnasium, once.

    Safe to call more than once (including from a module already imported
    elsewhere in the same process) -- a second call is a no-op rather than
    `gymnasium.register`'s usual "id already registered" error.
    """
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

    `learner_seat`: a fixed seat index, or `"rotate"` (default) to draw a new
    one each `reset()` -- seat is not neutral (`docs/gym-design.md` §3 cites
    the seat-geometry duel result), and always learning from seat 0 would
    inherit that bias silently.

    `opponents`: one `hexset.arena` preset name (or `<kind>:<checkpoint>`
    spec, see `hexset.arena.entrant_from_name`) per non-learner seat.
    `reward`/`flatten` mirror `HexSetAEC`'s and this class's own `flatten`
    option: `flatten=True` (default) concatenates the four encoder arrays
    into one `Box`, matching what `sb3-contrib`'s `MaskablePPO` and most
    single-agent RL code expect (and `CatanatronEnv`'s own default "vector"
    representation); `flatten=False` returns the dict of arrays.

    The action mask never rides in the observation -- it is always
    `info["action_mask"]`, and `action_masks()` is the `sb3-contrib` hook.
    The learner's seat publishes no valuation vector, so it never trades;
    the opponent seats trade with each other and with the learner's cards
    only through what their own bots advertise and accept, each publishing
    right after its own auto-played action (`_auto_play_opponents`).
    `info["view"]` carries `hexset.view.View`, the seat's full information-set
    object (`known`/`unknown`/`sample`), for a caller that wants more than
    the encoder's arrays.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        learner_seat: int | Literal["rotate"] = "rotate",
        opponents: Sequence[str] = DEFAULT_OPPONENTS,
        *,
        reward: str = "terminal",
        flatten: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not opponents:
            raise ValueError("HexSetEnv needs at least one opponent seat")
        num_players = len(opponents) + 1
        if isinstance(learner_seat, int) and not 0 <= learner_seat < num_players:
            raise ValueError(f"learner_seat {learner_seat} out of range for {num_players} players")

        self._aec = HexSetAEC(num_players=num_players, reward=reward, render_mode=render_mode)
        self.learner_seat_config = learner_seat
        self.opponent_names: tuple[str, ...] = tuple(opponents)
        self._entrants: list[Entrant] = [entrant_from_name(name) for name in opponents]
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
        # The opponents bring their own trading to the table: their
        # `accepts` is what the engine's one trade event a turn asks
        # (`hexset.trading`); their `valuation` is published by
        # `_auto_play_opponents`, right after each one's own action. The
        # learner seat has no bot, so it publishes nothing and never trades
        # -- the deferred half of the mechanic's interface, not an omission
        # here.
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

        try:
            self._aec.step(action if isinstance(action, Action) else int(action))
        except (ValueError, IndexError):
            # Not legal for the current decision (wrong phase, cannot afford
            # it, cannot cover a want, nobody currently owes a discard, ...)
            # -- every engine check of this kind runs before any state
            # mutation (`hexset.game`'s handlers all `_require` the phase, or
            # equivalent, as their first line), so nothing here needs to be
            # undone. `IndexError` alongside `ValueError`: `actions.apply`'s
            # own `DISCARD` dispatch indexes `players_owing_discards(game)[0]`
            # with no phase guard of its own, so a `DISCARD` action decoded
            # outside `Phase.DISCARD` raises `IndexError` rather than
            # `ValueError` for the same "not legal right now" reason.
            # Rejected as a harmless no-op rather than crashing the episode: a
            # caller is expected to act through `action_masks()`/
            # `info["action_mask"]` (`gymnasium.spaces.Discrete.sample(mask)`,
            # what `MaskablePPO` and this package's own tests do) and never
            # offer an action outside it; this guard exists for the one
            # caller that deliberately doesn't --
            # `gymnasium.utils.env_checker.check_env`'s own unmasked
            # `action_space.sample()`.
            observation, info = self._observe_learner()
            return observation, 0.0, False, False, info

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
        """`sb3_contrib.MaskablePPO`'s expected hook: the legal-action mask
        for the learner's current decision, as a boolean array."""
        return self._last_mask.astype(bool)

    # -- internals ------------------------------------------------------

    def _auto_play_opponents(self) -> None:
        aec = self._aec
        learner = agent_name(self._learner_seat)
        while True:
            if aec.terminations[learner] or aec.truncations[learner]:
                return
            if aec.agent_selection == learner:
                return
            agent = aec.agent_selection
            if aec.terminations[agent] or aec.truncations[agent]:
                aec.step(None)
                continue
            seat = aec.possible_agents.index(agent)
            bot = self._bots[seat]
            # Once a turn, exactly like `arena.play`'s loop: only when the
            # engine says this seat is due (`Game.publish_due`, the
            # post-roll/robber point, before the turn's first trade event --
            # the PI amendment "publish points and the event trigger"), not
            # after every action.
            if aec._game.publish_due(seat):
                publish_valuation(aec._game, seat, bot)
            action = bot.choose(aec._game)
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
