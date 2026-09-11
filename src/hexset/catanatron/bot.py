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
  seat defines neither `gains_many` nor `accepts`, which `hexset.trading`
  reads as declining every exchange (`valued`/`valued_many`).
"""

from __future__ import annotations

import random
from typing import Callable
from weakref import WeakValueDictionary

import numpy as np

from hexset.actions import Action, legal_actions
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from hexset.game import Game, imagine, to_move
from hexset.mcts import visit_policy

from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer

from .actions import to_catanatron
from .board import catanatron_map, translate_board
from .state import BoardMirrorCache, seating, to_catanatron as state_to_catanatron


class _TableMirror:
    def __init__(self, board: Board, num_players: int) -> None:
        self.board = board
        self.mapping = translate_board(catanatron_map(board))
        self.seats = seating(tuple(list(Color)[:num_players]))
        self.cache = BoardMirrorCache(self.mapping, self.seats)


# Bots retain their table; the registry does not retain finished games. Key by
# identity because Board contains unhashable topology dictionaries. The live
# mirror holds the board, preventing its id from being recycled underneath us.
_TABLE_MIRRORS: WeakValueDictionary[tuple[int, int], _TableMirror] = WeakValueDictionary()


def alpha_beta(depth: int) -> Callable[[Color], Player]:
    """`catanatron-play --players=AB:<depth>`, seat for seat.

    Its CLI splits the spec on `:` and passes the pieces positionally
    (`cli_players.parse_cli_string`), so `AB:2` was `AlphaBetaPlayer(color,
    "2")` -- depth two, no pruning, the default value function.

    Catanatron has since replaced that positional string with a `Params`
    dataclass carrying the same fields in the same order. Which one this
    build has is asked of the class rather than assumed, because passing
    the wrong one does not fail at construction: the string is accepted and
    only raises later, inside the search, as
    `'str' object has no attribute 'prunning'`.
    """
    params = getattr(AlphaBetaPlayer, "Params", None)
    if params is None:
        return lambda color: AlphaBetaPlayer(color, str(depth))
    return lambda color: AlphaBetaPlayer(color, params(depth=depth))


def _world_signature(state, perspective: int) -> tuple:
    """What makes two sampled worlds the same one, for reuse purposes.

    Every other seat's hidden holdings, and nothing else. The perspective's
    own hand is not sampled and the board is shared, so neither can tell two
    draws apart. The deck's order is deliberately excluded: it is a chance
    stream rather than something hidden about an opponent, and it is
    reshuffled on every draw, so including it would make every draw unique
    and defeat the reuse entirely.
    """
    return tuple(
        (tuple(state.hands[seat]), tuple(state.dev_cards[seat]),
         tuple(state.new_dev_cards[seat]))
        for seat in range(len(state.hands)) if seat != perspective
    )


class CatanatronBot:
    """A catanatron `Player` playing a hexset seat. `player(color) -> Player`.

    A catanatron `Player` cannot hold a belief: its state is fully typed
    fields with no way to say "unknown", and `AlphaBetaPlayer` uses
    randomness only for epsilon exploration, so it does not determinize for
    itself. Handed a position whose hidden parts are a stand-in, it plays as
    though the stand-in were certain.

    So `worlds` draws that many determinizations of the mover's information
    set and takes the vote, the way `bots.heximax.search.worlds` already
    does: `View.sample` keeps every public count and redraws the identities
    from what is genuinely unseen. It is a **cap, not a quota** -- draws are
    deduplicated and the search runs once per distinct world, its answer
    reused for later draws landing there, so the vote stays weighted by how
    likely each world is while costing only the worlds that differ. Where
    the belief admits one world, which is the common case once a ledger is
    doing its job, a high cap costs nothing.

    `worlds=0` is the default and the stock behaviour: read the true state
    and search it once. That is right wherever the state really is true --
    self-play, the arena, a bench -- and wrong wherever it is a
    reconstruction of a game seen from one seat.

    The vote is read the way a search's visit counts are
    (`hexset.mcts.visit_policy`): `temperature` 1 is proportional to vote
    share and 0 is argmax with ties split evenly, and `select` then takes
    that distribution's best or draws from it.
    """

    def __init__(self, player: Callable[[Color], Player] | None = None, *,
                 worlds: int = 0, temperature: float = 0.0,
                 select: str = "argmax",
                 rng: random.Random | None = None) -> None:
        if select not in ("argmax", "sample"):
            raise ValueError(f"select is 'argmax' or 'sample', not {select!r}")
        self.player = player or alpha_beta(2)
        self.worlds = worlds
        self.temperature = temperature
        self.select = select
        self.rng = rng or random.Random(0)
        self._mapping = None
        self._seats = None
        self._players: dict[Color, Player] = {}
        self._table = None

    def choose(self, game: Game) -> Action:
        """The move, determinized over `worlds` if asked for."""
        if self.worlds <= 0:
            return self._decide(game)

        seat = to_move(game)
        belief = game.state(seat)
        answered: dict[tuple, Action] = {}
        votes: dict[tuple, int] = {}
        for _ in range(self.worlds):
            sampled = belief.sample(self.rng)
            key = _world_signature(sampled, seat)
            action = answered.get(key)
            if action is None:
                world = imagine(game, self.rng, randomize_deck=False)
                world.set_state(sampled)
                action = self._decide(world)
                answered[key] = action
            vote = (int(action.type), action.a, action.b)
            votes[vote] = votes.get(vote, 0) + 1

        # Sorted keys, never dict order, so a seed reproduces a game exactly.
        keys = sorted(votes)
        policy = visit_policy(np.array([votes[k] for k in keys], dtype=float),
                              self.temperature)
        if self.select == "sample":
            index = self.rng.choices(range(len(keys)), weights=list(policy))[0]
        else:
            index = int(np.argmax(policy))
        chosen = keys[index]
        for action in answered.values():
            if (int(action.type), action.a, action.b) == chosen:
                return action
        raise AssertionError("the winning vote came from no searched world")

    def _decide(self, game: Game) -> Action:
        # true state: `to_catanatron` mirrors the whole table, which is what a
        # catanatron player reads; the board alone is public either way.
        state = game.state(0, hidden=False)
        key = (id(state.board), state.num_players)
        if (
            self._table is None or self._table.board is not state.board
            or len(self._table.seats.color_of) != state.num_players
        ):
            table = _TABLE_MIRRORS.get(key)
            if table is None:
                table = _TableMirror(state.board, state.num_players)
                _TABLE_MIRRORS[key] = table
            self._table = table
            self._mapping, self._seats = table.mapping, table.seats
            self._players.clear()

        # One catanatron `Player` per colour, kept across decisions: a
        # `Player` is built with the seat it plays, which is not known here
        # until this bot is first asked.
        color = self._seats.color_of[to_move(game)]
        if color not in self._players:
            self._players[color] = self.player(color)
        mirror = state_to_catanatron(
            game, self._mapping, self._seats, board_cache=self._table.cache,
        )

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
    # The arena plays the true state, so determinization stays off unless an
    # entrant asks for it; `rng` is the arena's, so a seeded run reproduces.
    return CatanatronBot(
        alpha_beta(entrant.depth),
        worlds=int(getattr(entrant, "worlds", 0) or 0),
        temperature=float(getattr(entrant, "temperature", 0.0) or 0.0),
        rng=rng,
    )


register_entrant_kind("catanatron", _spawn)
register_preset("catanatron", Entrant("catanatron", kind="catanatron", depth=2))
