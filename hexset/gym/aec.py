# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetAEC`: a PettingZoo AEC environment around the HexSet engine.

One agent per seat (`seat_0`..`seat_{n-1}`), `agent_selection` naming the seat
entitled to act next; `observe(agent)` never reads more than `agent`'s own
information set, and `action_mask` is `hexset.actions.legal_actions`.

`agent_selection` is `hexset.game.to_move` everywhere but `Phase.DISCARD`,
where every seat over the limit owes at the same instant
(`hexset.game.may_act`). AEC wants one active agent per `step()`, so
`discard_order` picks among them: `"random"` (default) uniformly, from a stream
seeded off `reset(seed)` alone rather than the game's rng, since ascending
order is biased; `"seat"` is that ascending serialization;
`select_agent(agent)` overrides either.

`step` refuses an action outside the acting agent's `action_mask` with a
`ValueError`, before anything is applied. `reset()` without a seed draws one
from `np_random`, which a seeded `reset` reseeds, so a run of resets after the
first seeded one is reproducible.

A seat trades by answering a private gate, not by taking an action, and this
environment seats none for its own agents, so nothing clears here.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import AECEnv

from hexset import encoding
from hexset.actions import Action, ActionSpace, apply, build_space, legal_actions
from hexset.board.board import random_base_board
from hexset.board.maps import BASE_LAYOUT
from hexset.board.topology import build as build_topology
from hexset.game import (
    MAX_TURNS, Game, Phase, is_over, may_act, players_owing_discards, start, to_move,
)
from hexset.rules import STANDARD_GAME, GameType
from hexset.victory import relative_points, victory_points

__all__ = [
    "REWARD_MODES",
    "DISCARD_ORDERS",
    "TOPOLOGY",
    "agent_name",
    "HexSetAEC",
]


REWARD_MODES = ("terminal", "relative_points")
# How `agent_selection` picks one of the seats owing a discard. The round is
# order-invariant: this decides only who is asked first.
DISCARD_ORDERS = ("random", "seat")

# The standard board's topology varies with nothing but
# `hexset.board.maps.BASE_LAYOUT`, so it is built once and shared with
# `hexset.gym.env` for its own space bookkeeping.
TOPOLOGY = build_topology(BASE_LAYOUT)


def agent_name(seat: int) -> str:
    """The PettingZoo agent name of `seat`: `"seat_0"` for seat 0."""
    return f"seat_{seat}"


class HexSetAEC(AECEnv):
    """One agent per seat on the standard board. `reward="terminal"` (default)
    gives +1 to the winning seat on the terminal step, 0 elsewhere;
    `"relative_points"` gives terminal victory points less the mean of the
    others, over the points the game is played to
    (`hexset.victory.relative_points`). A game that reaches `turn_cap` turns
    without a winner sets `truncations`, not `terminations`, with reward 0 for
    every seat in either mode. `turn_cap` defaults to `hexset.game.MAX_TURNS`;
    play far from a trained policy's -- random seats, an untrained learner --
    needs `hexset.game.UNSTRUCTURED_TURN_CAP` to finish."""

    metadata = {"render_modes": ["ansi", "human"], "name": "hexset_v0", "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 4,
        *,
        reward: str = "terminal",
        discard_order: str = "random",
        render_mode: str | None = None,
        game_type: GameType = STANDARD_GAME,
        turn_cap: int = MAX_TURNS,
    ) -> None:
        super().__init__()
        # The contract, not a bare seat range: the seat counts this env is
        # legal at are the ones its game type is played at. A format outside
        # the two shipped ones is a `GameType` the caller declares.
        game_type.check(num_players)
        self.game_type = game_type
        if reward not in REWARD_MODES:
            raise ValueError(f"unknown reward mode {reward!r}, expected one of {REWARD_MODES}")
        if discard_order not in DISCARD_ORDERS:
            raise ValueError(
                f"unknown discard order {discard_order!r}, expected one of {DISCARD_ORDERS}"
            )

        if turn_cap < 1:
            raise ValueError(f"a turn cap below one turn ends every game at the deal: {turn_cap}")

        self.num_players = num_players
        self.reward_mode = reward
        self.discard_order = discard_order
        self.render_mode = render_mode
        self.turn_cap = turn_cap
        # Where an unseeded `reset` takes its seed; a seeded one reseeds it.
        self.np_random, _ = seeding.np_random(None)

        self.possible_agents: list[str] = [agent_name(s) for s in range(num_players)]
        self.agents: list[str] = []

        self._space: ActionSpace = build_space(
            TOPOLOGY.num_vertices, TOPOLOGY.num_edges, TOPOLOGY.num_hexes, num_players
        )
        # Identical for every seat at a fixed player count, so built once.
        self._observation_space = self._build_observation_space()
        self._action_space = spaces.Discrete(self._space.size)
        self._game: Game | None = None
        # Its own stream, seeded off `reset(seed)` alone: drawing the discard
        # order from the game's rng would change the deal for the same seed.
        self._discard_rng = random.Random("discard-order:None")

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str | None = None

    # -- spaces ---------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        del agent  # identical for every seat at a fixed player count
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        del agent  # identical for every seat
        return self._action_space

    def _build_observation_space(self) -> spaces.Space:
        n = self.num_players
        return spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        "hexes": spaces.Box(
                            -np.inf, np.inf, (TOPOLOGY.num_hexes, encoding.HEX_FEATURES), dtype=np.float32
                        ),
                        "vertices": spaces.Box(
                            -np.inf,
                            np.inf,
                            (TOPOLOGY.num_vertices, encoding.vertex_features(n)),
                            dtype=np.float32,
                        ),
                        "edges": spaces.Box(
                            -np.inf, np.inf, (TOPOLOGY.num_edges, encoding.edge_features(n)), dtype=np.float32
                        ),
                        "globals": spaces.Box(
                            -np.inf, np.inf, (encoding.global_features(n),), dtype=np.float32
                        ),
                    }
                ),
                "action_mask": spaces.Box(0, 1, (self._space.size,), dtype=np.int8),
            }
        )

    # -- AECEnv -----------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        if seed is None:
            seed = int(self.np_random.integers(2**31))
        else:
            self.np_random, _ = seeding.np_random(seed)
        rng = random.Random(seed)
        board = random_base_board(rng)
        self._game = start(
            board, self.num_players, rng, game_type=self.game_type, turn_cap=self.turn_cap
        )
        self._discard_rng = random.Random(f"discard-order:{seed}")

        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self._next_agent(self._game)

    def select_agent(self, agent: str) -> None:
        """Name the seat that acts on the next `step()`, where more than one is
        entitled (only `Phase.DISCARD`). Refuses any seat the engine would, so
        it cannot act out of turn."""
        game = self._game
        assert game is not None, "select_agent() called before reset()"
        if agent not in self.possible_agents:
            raise ValueError(f"no such agent: {agent!r}")
        seat = self.possible_agents.index(agent)
        if not may_act(game, seat):
            raise ValueError(f"{agent} may not act in {game.phase.name}")
        self.agent_selection = agent

    def observe(self, agent: str) -> dict[str, Any]:
        game = self._game
        assert game is not None, "observe() called before reset()"
        seat = self.possible_agents.index(agent)
        obs = encoding.encode(game, perspective=seat)

        mask = np.zeros(self._space.size, dtype=np.int8)
        # Per PettingZoo convention, all zeros except for the agent about to
        # act, whose own options these are: during a discard round that is not
        # the seat bare `legal_actions` answers for, so it is passed.
        if agent == self.agent_selection:
            for action in legal_actions(game, seat):
                mask[self._space.index(action)] = 1

        return {
            "observation": {
                "hexes": obs.hexes,
                "vertices": obs.vertices,
                "edges": obs.edges,
                "globals": obs.globals,
            },
            "action_mask": mask,
        }

    def step(self, action: int | Action | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        game = self._game
        assert game is not None, "step() called before reset()"

        seat = self.possible_agents.index(agent)
        decoded = self._legal(game, seat, action)
        if decoded is None:
            raise ValueError(f"{action!r} is not in {agent}'s action mask")
        # The acting seat is dispatched with the action: a `DISCARD` resolves
        # against whoever `agent_selection` names, and every other action
        # belongs to `to_move` and ignores the argument.
        apply(game, decoded, seat)

        self._clear_rewards()
        if is_over(game):
            self._finish_episode(game, agent)
        else:
            self.agent_selection = self._next_agent(game)
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self) -> str | None:
        if self._game is None:
            return None
        text = self._summary(self._game)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        self._game = None

    # -- internals --------------------------------------------------------

    def _legal(self, game: Game, seat: int, action: int | Action) -> Action | None:
        """`action` as the `Action` it names, if it is one of `seat`'s legal
        actions (its `action_mask`), else `None`. Checked here, not left to
        the engine, so a refused action never reaches the game."""
        if isinstance(action, Action):
            decoded = action
        else:
            try:
                index = int(action)
            except (TypeError, ValueError):
                return None
            if not 0 <= index < self._space.size:
                return None
            decoded = self._space.decode(index)
        return decoded if decoded in legal_actions(game, seat) else None

    def _next_agent(self, game: Game) -> str:
        """Which single agent AEC hands the next `step()` to: `to_move`, except
        during discards where `discard_order` picks among the owing seats."""
        if self.discard_order == "random" and game.phase is Phase.DISCARD:
            owing = players_owing_discards(game)
            if owing:
                return agent_name(self._discard_rng.choice(owing))
        return agent_name(to_move(game))

    def _finish_episode(self, game: Game, acted_agent: str) -> None:
        won = game.won_by
        if won is None:
            # The turn cap, not an outcome: nothing to score in either mode.
            for agent in self.possible_agents:
                self.rewards[agent] = 0.0
        elif self.reward_mode == "relative_points":
            points = tuple(
                victory_points(game.state(seat, hidden=False), seat) for seat in range(self.num_players)
            )
            values = relative_points(
                points,
                winning_points=game.state(0, hidden=False).rules.winning_points,
            )
            for seat, agent in enumerate(self.possible_agents):
                self.rewards[agent] = float(values[seat])
        else:
            for seat, agent in enumerate(self.possible_agents):
                self.rewards[agent] = 1.0 if seat == won else 0.0

        if won is not None:
            for a in self.agents:
                self.terminations[a] = True
        else:
            for a in self.agents:
                self.truncations[a] = True

        # `to_move` is meaningless once the game is over, and every agent now
        # takes the dead-agent branch anyway; this just rotates.
        acted_seat = self.possible_agents.index(acted_agent)
        self.agent_selection = self.possible_agents[(acted_seat + 1) % self.num_players]

    def _summary(self, game: Game) -> str:
        # The acting agent, not `to_move`: during a discard round they differ.
        lines = [f"turn {game.turns}, phase {game.phase.name}, acting {self.agent_selection}"]
        if game.won_by is not None:
            lines.append(f"winner: {agent_name(game.won_by)}")
        for seat in range(self.num_players):
            points = victory_points(game.state(seat, hidden=False), seat)
            lines.append(f"{agent_name(seat)}: {points} VP")
        return "\n".join(lines)
