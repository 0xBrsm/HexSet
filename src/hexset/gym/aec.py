# SPDX-License-Identifier: GPL-3.0-only
"""`HexSetAEC`: a PettingZoo AEC environment around the HexSet engine.

See `docs/gym-design.md` for the ratified design this implements. One agent
per seat (`seat_0`..`seat_{n-1}`), `agent_selection` tracking
`hexset.game.to_move`; `observe(agent)` never reads more than `agent`'s own
information set.

**Honesty.** Every observation array comes from `hexset.encoding.encode`,
which is already information-set correct by construction (own hand and
own development cards exact, everyone else a count plus the ledger's
public-knowledge reconstruction -- see that module's docstring). The
`action_mask` is built from `hexset.server.rules.fair_legal_actions`, never
from `hexset.actions.legal_actions`'s own `PROPOSE_TRADE` sample, which
reads opponents' true hands to decide who could cover an offer -- exactly
the leak `fair_legal_actions` exists to close (see its module docstring).

**The `PROPOSE_TRADE` gap.** `hexset.actions.ActionSpace` gives `PROPOSE_TRADE`
a single flat slot meaning "proposing is available" -- an offer is ten
numbers and cannot be recovered from an index (`ActionSpace.decode`'s own
docstring). A caller with its own give/want heads (a trained policy) can
still name an exact offer by passing a `hexset.actions.Action` to `step`
directly instead of a bare index. A caller sampling the flat `Discrete`
space -- which is what `gymnasium.spaces.Discrete.sample(mask)` does, and so
what PettingZoo's own `api_test`/`seed_test` and a random policy do -- gets
one drawn uniformly, from the game's own seeded `rng`, among every
one-for-one pair `fair_legal_actions` would show this seat. This is a
necessary filling-in this implementation adds beneath the ratified design's
literal "decode the index, apply the action" (`docs/gym-design.md` §2): decoding
a bare `PROPOSE_TRADE` index yields empty `give`/`want` tuples, which
`hexset.trading.well_formed` rejects outright, so every random-policy episode
would otherwise crash the first time the sampler touched that slot.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from hexset import encoding
from hexset.actions import Action, ActionSpace, ActionType, apply, build_space
from hexset.board.board import random_base_board
from hexset.board.maps import BASE_LAYOUT
from hexset.board.topology import build as build_topology
from hexset.game import Game, is_over, start, to_move
from hexset.server.rules import fair_legal_actions
from hexset.victory import relative_points, victory_points

REWARD_MODES = ("terminal", "relative_points")

# The standard board's topology never varies with terrain, tokens, ports or
# seed -- only `hexset.board.maps.BASE_LAYOUT`'s hex coordinates decide vertex
# and edge counts -- so it is built once here rather than once per episode,
# and shared with `hexset.gym.env` for its own space bookkeeping.
TOPOLOGY = build_topology(BASE_LAYOUT)


def agent_name(seat: int) -> str:
    return f"seat_{seat}"


class HexSetAEC(AECEnv):
    """One agent per seat on the standard board.

    `reward="terminal"` (default): +1 to the winning seat, 0 to everyone
    else, on the step the game ends; 0 on every other step.
    `reward="relative_points"`: each seat's terminal victory points less the
    mean of the others, over 10 (`hexset.victory.relative_points`) -- zero-sum,
    read only on the terminal step.

    Hitting `hexset.game.MAX_TURNS` ends the game with `won_by is None`; every
    agent's `truncations` entry is set (not `terminations`), reward 0,
    mirroring the arena's own `exhausted` outcome.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "hexset_v0", "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 4,
        *,
        reward: str = "terminal",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 2 <= num_players <= 6:
            raise ValueError(f"unsupported player count: {num_players}")
        if reward not in REWARD_MODES:
            raise ValueError(f"unknown reward mode {reward!r}, expected one of {REWARD_MODES}")

        self.num_players = num_players
        self.reward_mode = reward
        self.render_mode = render_mode

        self.possible_agents: list[str] = [agent_name(s) for s in range(num_players)]
        self.agents: list[str] = []

        self._space: ActionSpace = build_space(
            TOPOLOGY.num_vertices, TOPOLOGY.num_edges, TOPOLOGY.num_hexes, num_players
        )
        self._game: Game | None = None

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str | None = None

    # -- spaces ---------------------------------------------------------

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        del agent  # identical for every seat at a fixed player count
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

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        del agent  # identical for every seat
        return spaces.Discrete(self._space.size)

    # -- AECEnv -----------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        rng = random.Random(seed)
        board = random_base_board(rng)
        self._game = start(board, self.num_players, rng)

        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = agent_name(to_move(self._game))

    def observe(self, agent: str) -> dict[str, Any]:
        game = self._game
        assert game is not None, "observe() called before reset()"
        seat = self.possible_agents.index(agent)
        obs = encoding.encode(game, perspective=seat)

        mask = np.zeros(self._space.size, dtype=np.int8)
        # Per PettingZoo convention (`pettingzoo.classic.tictactoe`), the mask
        # is all zeros for every agent except the one currently to move --
        # `fair_legal_actions` only ever answers for `to_move(game)` anyway.
        if agent == self.agent_selection:
            for action in fair_legal_actions(game):
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

        if isinstance(action, Action):
            decoded = action
        else:
            decoded = self._space.decode(int(action))
            if decoded.type is ActionType.PROPOSE_TRADE:
                decoded = self._fill_propose_trade(game)

        apply(game, decoded)

        self._clear_rewards()
        if is_over(game):
            self._finish_episode(game, agent)
        else:
            self.agent_selection = agent_name(to_move(game))
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

    def _fill_propose_trade(self, game: Game) -> Action:
        """A concrete, honest offer for a bare `PROPOSE_TRADE` index.

        Drawn from `game`'s own seeded `rng` (the same stream every other
        engine-internal choice -- responder order, robber ties -- already
        uses), so an episode replayed under the same `reset(seed=...)` picks
        the same offers. See the module docstring for why this exists.
        """
        options = [a for a in fair_legal_actions(game) if a.type is ActionType.PROPOSE_TRADE]
        if not options:
            raise ValueError("PROPOSE_TRADE is not currently legal")
        return game.rng.choice(options)

    def _finish_episode(self, game: Game, acted_agent: str) -> None:
        won = game.won_by
        if self.reward_mode == "relative_points":
            points = tuple(
                victory_points(game.state(seat, hidden=False), seat) for seat in range(self.num_players)
            )
            values = relative_points(points)
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

        # `to_move(game)` is meaningless once the game is over. Any live agent
        # works here -- every one of them is now terminated/truncated, so the
        # next `step()` on whichever one `agent_selection` names takes the
        # dead-agent branch above -- so this just rotates deterministically.
        acted_seat = self.possible_agents.index(acted_agent)
        self.agent_selection = self.possible_agents[(acted_seat + 1) % self.num_players]

    def _summary(self, game: Game) -> str:
        lines = [f"turn {game.turns}, phase {game.phase.name}, to_move {agent_name(to_move(game))}"]
        if game.won_by is not None:
            lines.append(f"winner: {agent_name(game.won_by)}")
        for seat in range(self.num_players):
            points = victory_points(game.state(seat, hidden=False), seat)
            lines.append(f"{agent_name(seat)}: {points} VP")
        return "\n".join(lines)
