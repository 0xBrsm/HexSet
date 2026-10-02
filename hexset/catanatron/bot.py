# SPDX-License-Identifier: GPL-3.0-only
"""A `hexset.arena` bot whose brain is a catanatron `Player`.

Every decision rebuilds the catanatron `Game` mirroring this position
(`state.to_catanatron`), asks the catanatron player for a move, and hands back
the hexset `Action` it stands for. What it is offered is not catanatron's own
`playable_actions` but our `legal_actions` run through `actions.to_catanatron`,
so the answer maps back by a dict lookup and no move catanatron allows but
hexset does not can be picked. The rule sets agree on move generation, trading
excepted: a catanatron `Player` has no notion of the one-event mechanic, so
this seat defines neither `gains_many` nor `accepts`, which `hexset.trading`
reads as declining every exchange.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Callable
from weakref import WeakValueDictionary

from hexset.actions import Action, legal_actions
from hexset import arena
from hexset.arena import CATANATRON, Entrant
from hexset.board.board import Board
from hexset.bots.determinized import WorldKey, determinized, world_signature
from hexset.game import Game, to_move

from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer

from ._actions import to_catanatron
from ._board import catanatron_map, translate_board
from ._state import BoardMirrorCache, seating, to_catanatron as state_to_catanatron

__all__ = [
    "live_seats",
    "alpha_beta",
    "CatanatronBot",
    "PRESET",
    "parse",
]


class _TableMirror:
    def __init__(self, board: Board, live: tuple[int, ...]) -> None:
        self.board = board
        self.live = live
        self.mapping = translate_board(catanatron_map(board))
        self.seats = seating(tuple(list(Color)[: len(live)]), live)
        self.cache = BoardMirrorCache(self.mapping, self.seats)


def live_seats(game: Game, state) -> tuple[int, ...]:
    """The seats the mirror holds: every seat that can still act, plus any
    retired seat that left pieces behind.

    A seat retired before it placed anything is not a player -- a table with
    two of four seats closed is a two-player game, and mirroring the closed
    seats as colours that never move would have the reference engine search
    against opponents that do not exist. A seat that retires mid-game is
    different: its buildings and roads still occupy the board, so its colour
    has to stay.
    """
    locked = game.locked
    if not locked:
        return tuple(range(state.num_players))
    owning = set(state.vertex_owner) | set(state.edge_owner)
    return tuple(
        seat
        for seat in range(state.num_players)
        if seat not in locked or seat in owning
    )


# Keyed by identity because `Board` holds unhashable topology dicts; the live
# mirror holds the board, so its id cannot be recycled underneath us.
_TABLE_MIRRORS: WeakValueDictionary[tuple[int, tuple[int, ...]], _TableMirror] = WeakValueDictionary()


def alpha_beta(depth: int) -> Callable[[Color], Player]:
    """`catanatron-play --players=AB:<depth>`, seat for seat: catanatron's
    players construct as `(color, params)` with the tunables on a nested
    `Params` model."""
    return lambda color: AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=int(depth)))


class CatanatronBot:
    """A catanatron `Player` playing a hexset seat. `worlds=0` searches the
    supplied state once, as the reference does; positive `worlds` samples the
    mover's information set and votes, caching one answer per distinct sampled
    state. `world_key` may ignore deck order (see `hexset.bots.determinized`).

    `worlds=0` is the omniscient read: the mirror handed to the catanatron
    player holds every seat's true hand and development cards, so this seat
    plays on information a hexset seat may not see. `worlds>0` is the
    information-set read, where each mirror is a world sampled from the
    mover's own `View`. Zero is the default because it is how catanatron's
    own players play: the reference as published.

    `rng` is this seat's own stream, for the catanatron player's search and
    the world sampling; without one the seat draws a fresh, unseeded stream,
    never the live game's chance stream."""

    def __init__(
        self, player: Callable[[Color], Player] | None = None,
        *, rng: random.Random | None = None, worlds: int = 0,
        temperature: float = 0.0, select: str = "argmax",
        world_key: WorldKey = world_signature,
    ) -> None:
        self.player = player or alpha_beta(2)
        rng = random.Random() if rng is None else rng
        self._rng = rng
        self._mapping = None
        self._seats = None
        self._players: dict[Color, Player] = {}
        self._table = None
        # Sampling must not consume the reference player's search stream.
        sampling_rng = random.Random(0)
        sampling_rng.setstate(rng.getstate())
        self._choose = determinized(
            self._decide, worlds, sampling_rng, temperature=temperature,
            select=select, world_key=world_key,
        )

    def choose(self, game: Game) -> Action:
        action = self._choose(game)
        if action is None:
            raise ValueError("catanatron returned no action in any sampled world")
        return action

    def _decide(self, game: Game) -> Action:
        # true state: `to_catanatron` mirrors the whole table, which is what a
        # catanatron player reads.
        state = game.state(0, hidden=False)
        live = live_seats(game, state)
        key = (id(state.board), live)
        if (
            self._table is None or self._table.board is not state.board
            or self._table.live != live
        ):
            table = _TABLE_MIRRORS.get(key)
            if table is None:
                table = _TableMirror(state.board, live)
                _TABLE_MIRRORS[key] = table
            self._table = table
            self._mapping, self._seats = table.mapping, table.seats
            self._players.clear()

        # One `Player` per colour, kept across decisions: a `Player` is built
        # with the seat it plays, unknown until this bot is first asked.
        color = self._seats.color_of[to_move(game)]
        if color not in self._players:
            self._players[color] = self.player(color)
        mirror = state_to_catanatron(
            game, self._mapping, self._seats, board_cache=self._table.cache, rng=self._rng,
        )

        self._rng = mirror.random
        offered = self._offer(game, mirror)
        chosen = self._players[color].decide(mirror, mirror.playable_actions)
        return offered[chosen]

    def _offer(self, game: Game, mirror) -> dict:
        """The offer, keyed by the catanatron action standing for each of ours,
        and also installed on the mirror as its `playable_actions`: catanatron's
        own search reads that, not what it is handed."""
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
    return CatanatronBot(
        alpha_beta(entrant.depth), rng=rng,
        worlds=entrant.option("worlds", 0),
        temperature=entrant.option("temperature", 0.0),
        select=entrant.option("select", "argmax"),
    )


PRESET = Entrant("catanatron", kind="catanatron", depth=2)

_SPEC_FORM = (
    f"{CATANATRON}<key>=<value>[:...], keys depth=<n >= 1>, worlds=<n >= 0>, "
    "temperature=<x >= 0>, select=argmax|sample"
)


def _setting(key: str, value: str) -> object:
    """One `catanatron:` option's value, or `None` if it does not read."""
    if key in ("depth", "worlds"):
        if value.isdigit() and int(value) >= (1 if key == "depth" else 0):
            return int(value)
        return None
    if key == "temperature":
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) and number >= 0 else None
    if key == "select":
        return value if value in ("argmax", "sample") else None
    return None


def parse(name: str) -> Entrant:
    """`catanatron:depth=3`, `catanatron:worlds=8:temperature=1:select=sample`:
    the preset with its alpha-beta depth (`Entrant.depth`) or its world vote
    (`Entrant.options`: `worlds`, `temperature`, `select`, as
    `CatanatronBot` takes them) changed. The entrant is named by the spec."""
    settings: dict[str, object] = {}
    for option in name[len(CATANATRON):].split(":"):
        key, separator, value = option.partition("=")
        if not separator or key in settings:
            raise ValueError(f"{name!r}: invalid or repeated option {option!r}; use {_SPEC_FORM}")
        settings[key] = _setting(key, value)
        if settings[key] is None:
            raise ValueError(f"{name!r}: unknown option {option!r}; use {_SPEC_FORM}")
    depth = settings.pop("depth", PRESET.depth)
    return replace(PRESET, name=name, depth=depth, options=settings)


# Shipped names: registered here, where the extra is, and refused to runtimes.
arena._register(arena._ENTRANT_KIND_FACTORIES, "catanatron", _spawn, "entrant kind")
arena._register(arena.PRESETS, "catanatron", PRESET, "preset")
arena._register(arena._SPEC_PARSERS, CATANATRON, parse, "spec prefix")
