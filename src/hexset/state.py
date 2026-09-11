# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from operator import index as integer_index
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

from .board.board import Board
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Terrain
from .cards import NUM_DEV_CARDS, make_deck
from .rules import STANDARD, Rules

NO_OWNER = -1
BANK_PER_RESOURCE = 19

# The standard piece supply, per player. The rule is enforced here, in the
# placement legality, so every path that builds -- a bought road or
# settlement, a free road from the card, an initial placement -- reads it,
# and `legal_actions` simply stops offering a piece that is not in the box.
# Identical to hexset-ui's `state.py` (g4 checkout `dfdd3f8`), which is the
# rules reference: until 2026-08-29 this engine had no supply at all and every
# recorded run, duel and ladder was played with unlimited pieces, while the
# deployment and the bridge's engine both capped. Measured incidence before
# the fix: `runs/eval/piece-supply-incidence.txt`.
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


class Building(IntEnum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2


class HiddenRead(Exception):
    """A card *identity* was read off a pile that carries only a count.

    Raised by `Hidden` and its subclasses. It is deliberately not a
    `LookupError` or a `TypeError`: nothing in the engine catches it, so a
    path that needs to know what somebody holds fails loudly on an observed
    state instead of quietly reading a fabricated number.
    """


class Hidden:
    """`count` cards whose types the state's author cannot see.

    A `GameState` is normally written by the referee, who knows everything;
    then every hand, every development holding and the deck is a list of
    counts by type. A state written by a *seat* -- the live adapter for a
    table it is playing at, a replay of a public log -- knows its own hand
    exactly and knows the rest only as sizes. Those slots hold one of these
    instead of a list:

    * `HiddenHand(n)` in `hands[seat]` -- n resource cards, types unknown,
    * `HiddenCards(n)` in `dev_cards[seat]` / `new_dev_cards[seat]`,
    * `HiddenDeck(n)` in `deck` -- n cards left, order and types unknown.

    What works: `len(pile)` is the count, `pile[:]` copies it (the idiom
    `copy_state` and `game._snapshot_hands` use), equality and `repr`. A
    hidden hand or development holding is always truthy, like its fixed
    count slots; a hidden deck is truthy only while it has cards.

    What raises `HiddenRead`: indexing a type (`hand[SHEEP]`), iterating
    (so `sum`, `zip`, `tuple`, `max`, `random.shuffle` all raise), and
    assignment. Read a size through `hexset.economy.hand_size`,
    `hexset.devcards.dev_count` or `len` -- never by summing a pile the
    caller has not established is concrete.
    """

    __slots__ = ("count",)
    #: What the pile holds, for the message a refused identity read raises.
    kind = "cards"

    def __init__(self, count: int) -> None:
        count = integer_index(count)
        if count < 0:
            raise ValueError(f"a hidden pile cannot hold {count} cards")
        self.count = count

    def __len__(self) -> int:
        """How many cards, not how many type slots: the count is all there is."""
        return self.count

    def __bool__(self) -> bool:
        # Without this, `Hidden(0)` would be falsy while the `[0] * 5` it
        # replaces is truthy, and `if state.hands[seat]:` would change
        # meaning between a true and an observed state.
        return True

    def __getitem__(self, index: object) -> "Hidden":
        if isinstance(index, slice) and index == slice(None):
            return type(self)(self.count)
        raise HiddenRead(
            f"{self.count} {self.kind} of unknown type: this state's author "
            f"cannot see which, so entry {index!r} has no value to read"
        )

    def __setitem__(self, index: object, value: object) -> None:
        raise HiddenRead(
            f"cannot write entry {index!r} of {self.count} hidden {self.kind}"
        )

    def __iter__(self):
        raise HiddenRead(
            f"{self.count} {self.kind} of unknown type: this state's author "
            f"cannot see the composition, so it cannot be iterated or summed"
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.count == other.count

    # Counts may be updated by an adapter; like concrete piles, these are
    # mutable and must not be used as dictionary keys.
    __hash__ = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.count})"

    def copy(self) -> "Hidden":
        return type(self)(self.count)


class HiddenHand(Hidden):
    """Resource cards held by a seat this state's author cannot see."""

    kind = "resource cards"


class HiddenCards(Hidden):
    """Development cards held by a seat this state's author cannot see."""

    kind = "development cards"


class HiddenDeck(Hidden):
    """A development deck known only by its length."""

    kind = "deck cards"

    def __bool__(self) -> bool:
        # A concrete deck is a variable-length list, unlike a hand's fixed
        # five count slots. Empty decks must forbid development-card buys.
        return self.count > 0


def is_hidden(pile: object) -> bool:
    """Whether `pile` carries a count instead of a composition."""
    return isinstance(pile, Hidden)


def pile_size(pile: "Hidden | Sequence[int]") -> int:
    """How many cards `pile` holds, whether or not its types are visible.

    The one primitive every size-only reader goes through
    (`hexset.economy.hand_size`, `hexset.devcards.dev_count`), so those
    readers work on a referee's state and on a seat's alike.
    """
    return len(pile) if isinstance(pile, Hidden) else sum(pile)


@dataclass
class GameState:
    """Mutable occupancy of a board, plus what each player is holding."""

    board: Board
    num_players: int
    vertex_owner: list[int]
    vertex_building: list[int]
    edge_owner: list[int]
    robber: int
    # A seat the state's author cannot see holds a `Hidden` pile -- a count
    # with no composition -- rather than a list of counts; see `Hidden`.
    hands: list[list[int] | HiddenHand] = field(default_factory=list)
    bank: list[int] = field(default_factory=list)
    deck: list[int] | HiddenDeck = field(default_factory=list)
    dev_cards: list[list[int] | HiddenCards] = field(default_factory=list)
    new_dev_cards: list[list[int] | HiddenCards] = field(default_factory=list)
    knights_played: list[int] = field(default_factory=list)
    longest_road_holder: int = NO_OWNER
    largest_army_holder: int = NO_OWNER
    # The game type this position is played under. Read by the win check,
    # the discard rule and the bot evaluators rather than the module-level
    # standard constants, so one engine plays every game type. Immutable and
    # shared by reference: `copy_state` and `imagine` carry it over, they
    # never mutate it.
    rules: Rules = STANDARD


def new_game(
    board: Board,
    num_players: int,
    rng: random.Random | None = None,
    *,
    rules: Rules = STANDARD,
) -> GameState:
    if not 2 <= num_players <= 6:
        raise ValueError(f"unsupported player count: {num_players}")
    topology = board.topology
    deserts = board.desert_hexes()
    return GameState(
        board=board,
        num_players=num_players,
        vertex_owner=[NO_OWNER] * topology.num_vertices,
        vertex_building=[Building.NONE] * topology.num_vertices,
        edge_owner=[NO_OWNER] * topology.num_edges,
        robber=deserts[0] if deserts else 0,
        hands=[[0] * NUM_RESOURCES for _ in range(num_players)],
        bank=[BANK_PER_RESOURCE] * NUM_RESOURCES,
        deck=make_deck(rng),
        dev_cards=[[0] * NUM_DEV_CARDS for _ in range(num_players)],
        new_dev_cards=[[0] * NUM_DEV_CARDS for _ in range(num_players)],
        knights_played=[0] * num_players,
        rules=rules,
    )


def copy_state(state: GameState) -> GameState:
    """A state that can be mutated without touching the original.

    The board is shared rather than copied: it is frozen and never changes
    during a game, and it is by far the largest object here.

    Hidden piles copy through the same `[:]` idiom as concrete ones
    (`Hidden.__getitem__` answers a slice with an equal pile), so an
    observed state copies exactly like a true one.
    """
    return GameState(
        board=state.board,
        num_players=state.num_players,
        vertex_owner=state.vertex_owner[:],
        vertex_building=state.vertex_building[:],
        edge_owner=state.edge_owner[:],
        robber=state.robber,
        hands=[hand[:] for hand in state.hands],
        bank=state.bank[:],
        deck=state.deck[:],
        dev_cards=[held[:] for held in state.dev_cards],
        new_dev_cards=[held[:] for held in state.new_dev_cards],
        knights_played=state.knights_played[:],
        longest_road_holder=state.longest_road_holder,
        largest_army_holder=state.largest_army_holder,
        rules=state.rules,
    )


def observed_by(state: GameState, seat: int) -> GameState:
    """A copy of `state` holding only what `seat` can see.

    `seat`'s own hand and development cards stay exact; every other seat's
    become a `HiddenHand`/`HiddenCards` of the same size and the deck a
    `HiddenDeck` of the same length. Public fields -- the board, ownership,
    the robber, the bank, knights played, the awards -- are copied as they
    are, because they are public.

    Two uses. It downgrades a true state for a caller that has one, and it
    is the honesty probe: a path that claims to read nothing it should not
    must decide the same on the result as on `state` itself. An adapter
    sitting at a real table has no truth to downgrade and writes the hidden
    piles itself; that is the case this type exists for.
    """
    observed = copy_state(state)
    for other in range(state.num_players):
        if other == seat:
            continue
        observed.hands[other] = HiddenHand(pile_size(state.hands[other]))
        observed.dev_cards[other] = HiddenCards(pile_size(state.dev_cards[other]))
        observed.new_dev_cards[other] = HiddenCards(
            pile_size(state.new_dev_cards[other])
        )
    observed.deck = HiddenDeck(len(state.deck))
    return observed


def settlement_count(state: GameState, player: int) -> int:
    return sum(
        1
        for v, owner in enumerate(state.vertex_owner)
        if owner == player and state.vertex_building[v] == Building.SETTLEMENT
    )


def city_count(state: GameState, player: int) -> int:
    return sum(
        1
        for v, owner in enumerate(state.vertex_owner)
        if owner == player and state.vertex_building[v] == Building.CITY
    )


def road_count(state: GameState, player: int) -> int:
    return sum(1 for owner in state.edge_owner if owner == player)


def can_place_settlement(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> bool:
    """`connected` is False during initial placement, when roads are not required."""
    if state.vertex_building[vertex] != Building.NONE:
        return False
    if settlement_count(state, player) >= MAX_SETTLEMENTS:
        return False
    topology = state.board.topology
    if any(
        state.vertex_building[n] != Building.NONE
        for n in topology.vertex_neighbors[vertex]
    ):
        return False
    if not connected:
        return True
    return any(state.edge_owner[e] == player for e in topology.vertex_edges[vertex])


def place_settlement(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> None:
    if not can_place_settlement(state, player, vertex, connected=connected):
        raise ValueError(f"player {player} cannot settle vertex {vertex}")
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = Building.SETTLEMENT


def can_upgrade_to_city(state: GameState, player: int, vertex: int) -> bool:
    return (
        state.vertex_owner[vertex] == player
        and state.vertex_building[vertex] == Building.SETTLEMENT
        and city_count(state, player) < MAX_CITIES
    )


def upgrade_to_city(state: GameState, player: int, vertex: int) -> None:
    if not can_upgrade_to_city(state, player, vertex):
        raise ValueError(f"player {player} has no settlement on vertex {vertex}")
    state.vertex_building[vertex] = Building.CITY


def can_place_road(state: GameState, player: int, edge: int) -> bool:
    if state.edge_owner[edge] != NO_OWNER:
        return False
    if road_count(state, player) >= MAX_ROADS:
        return False
    topology = state.board.topology
    for v in topology.edges[edge]:
        owner = state.vertex_owner[v]
        if owner == player:
            return True
        # An opponent's building blocks a road network from continuing through it.
        if owner == NO_OWNER and any(
            state.edge_owner[e] == player for e in topology.vertex_edges[v]
        ):
            return True
    return False


def place_road(state: GameState, player: int, edge: int) -> None:
    if not can_place_road(state, player, edge):
        raise ValueError(f"player {player} cannot build road on edge {edge}")
    state.edge_owner[edge] = player


def production(state: GameState, roll: int) -> list[list[int]]:
    """Gross resource yield per player for `roll`, before any bank limit.

    Bank exhaustion is a later concern: the official rule depends on stock and
    on how many players are owed, so it cannot be resolved per hex.
    """
    gains = [[0] * NUM_RESOURCES for _ in range(state.num_players)]
    board = state.board
    topology = board.topology

    for h in board.hexes_by_roll[roll]:
        if h == state.robber:
            continue
        resource = TERRAIN_RESOURCE[board.terrain[h]]
        if resource is None:
            continue
        for v in topology.hex_vertices[h]:
            owner = state.vertex_owner[v]
            if owner != NO_OWNER:
                gains[owner][resource] += state.vertex_building[v]

    return gains


def gold_claims(state: GameState, roll: int) -> list[int]:
    """How many resources of their choice each player may claim from gold hexes."""
    claims = [0] * state.num_players
    board = state.board
    topology = board.topology

    for h in board.hexes_by_roll[roll]:
        if h == state.robber or board.terrain[h] is not Terrain.GOLD:
            continue
        for v in topology.hex_vertices[h]:
            owner = state.vertex_owner[v]
            if owner != NO_OWNER:
                claims[owner] += state.vertex_building[v]

    return claims
