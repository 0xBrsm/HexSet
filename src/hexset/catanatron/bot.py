# SPDX-License-Identifier: GPL-3.0-only
"""A `hexset.arena` bot whose brain is a catanatron `Player`.

The mirror image of `player.py`. Every decision rebuilds the catanatron `Game`
that mirrors this position (`state.to_catanatron`, off the map
`board.catanatron_map` builds from our own board), asks the catanatron player
for a move, and hands back the hexset `Action` it stands for.

What it is offered is not catanatron's own `playable_actions` but our
`legal_actions` run through `actions.to_catanatron` -- the same table
`player.py` uses in the other direction -- so the answer maps back by a dict
lookup and no move catanatron allows but hexset does not can be picked. That
filter is the whole resolution wherever the two engines split a decision
differently; the cases, one line each:

* Before the roll, catanatron's one `PLAY_TURN` prompt offers every dev card
  where hexset's `Phase.ROLL` offers only the roll and a knight; the rest drop.
* While free roads are owed catanatron offers roads alone where hexset also
  allows building and trading -- a subset, taken as offered.
* Catanatron's dev-card maturity is one boolean per type against hexset's
  per-copy count: one matured copy plus one fresh is offered one play, not two.
* Catanatron enforces piece caps hexset does not, likewise narrowing the offer.
* Catanatron's players have no notion of the one-event trade mechanic, so this
  seat defines neither `valuation` nor `accepts`, which `hexset.trading` reads
  as never publishing and declining every exchange (`published`/`judged`).
"""

from __future__ import annotations

import random
from typing import Callable

from hexset.actions import Action, legal_actions
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from hexset.game import Game, to_move

from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer

from .actions import to_catanatron
from .board import catanatron_map, translate_board
from .state import seating, to_catanatron as state_to_catanatron


def alpha_beta(depth: int) -> Callable[[Color], Player]:
    """`catanatron-play --players=AB:<depth>`, seat for seat.

    Its CLI splits the spec on `:` and passes the pieces positionally
    (`cli_players.parse_cli_string`), so `AB:2` is `AlphaBetaPlayer(color,
    "2")` -- depth two, no pruning, the default value function.
    """
    return lambda color: AlphaBetaPlayer(color, str(depth))


class CatanatronBot:
    """A catanatron `Player` playing a hexset seat. `player(color) -> Player`."""

    def __init__(self, player: Callable[[Color], Player] | None = None) -> None:
        self.player = player or alpha_beta(2)
        self._mapping = None
        self._seats = None
        self._players: dict[Color, Player] = {}

    def choose(self, game: Game) -> Action:
        # true state: `to_catanatron` mirrors the whole table, which is what a
        # catanatron player reads; the board alone is public either way.
        state = game.state(0, hidden=False)
        if self._mapping is None:
            self._mapping = translate_board(catanatron_map(state.board))
            self._seats = seating(tuple(list(Color)[: state.num_players]))

        # One catanatron `Player` per colour, kept across decisions: a
        # `Player` is built with the seat it plays, which is not known here
        # until this bot is first asked.
        color = self._seats.color_of[to_move(game)]
        if color not in self._players:
            self._players[color] = self.player(color)
        mirror = state_to_catanatron(game, self._mapping, self._seats)

        offered = self._offer(game, mirror)
        chosen = self._players[color].decide(mirror, mirror.playable_actions)
        return offered[chosen]

    def _offer(self, game: Game, mirror) -> dict:
        """The offer, keyed by the catanatron action standing for each of ours.

        Also installed on the mirror as its `playable_actions`: catanatron's
        own search reads that, not what it is handed
        (`AlphaBetaPlayer.get_actions`).
        """
        offered = {}
        for action in legal_actions(game):
            try:
                their = to_catanatron(
                    action, game, self._mapping, self._seats, mirror.playable_actions
                )
            except (ValueError, NotImplementedError):
                continue  # catanatron is not offering this one right now
            offered.setdefault(their, action)
        if not offered:
            raise ValueError(
                f"catanatron offered nothing hexset allows in {game.phase.name}: "
                f"{mirror.playable_actions}"
            )
        mirror.playable_actions = list(offered)
        return offered


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> CatanatronBot:
    return CatanatronBot(alpha_beta(entrant.depth))


register_entrant_kind("catanatron", _spawn)
register_preset("catanatron", Entrant("catanatron", kind="catanatron", depth=2))
