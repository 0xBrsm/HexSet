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

__all__ = [
    "NO_OWNER",
    "OFF_BOARD",
    "BANK_PER_RESOURCE",
    "MAX_ROADS",
    "MAX_SETTLEMENTS",
    "MAX_CITIES",
    "Building",
    "HiddenRead",
    "Hidden",
    "HiddenHand",
    "HiddenCards",
    "HiddenDeck",
    "is_hidden",
    "pile_size",
    "GameState",
    "new_game",
    "copy_state",
    "observed_by",
    "settlement_count",
    "city_count",
    "road_count",
    "can_place_settlement",
    "settlement_placeable",
    "check_settlement",
    "place_settlement",
    "can_upgrade_to_city",
    "city_upgradeable",
    "check_city",
    "upgrade_to_city",
    "can_place_road",
    "road_placeable",
    "check_road",
    "place_road",
    "production",
    "gold_claims",
]


NO_OWNER = -1
#: `GameState.robber` before its first move on a board with no desert: the
#: robber stands beside the board until the first seven or knight places it.
OFF_BOARD = -1
BANK_PER_RESOURCE = 19

# The standard piece supply, per player, enforced in the placement legality so
# that `legal_actions` stops offering a piece that is not in the box.
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


class Building(IntEnum):
    """What stands on a vertex."""

    NONE = 0
    SETTLEMENT = 1
    CITY = 2


class HiddenRead(Exception):
    """A card *identity* was read off a pile that carries only a count.
    Deliberately not a `LookupError` or `TypeError`: nothing catches it."""


class Hidden:
    """`count` cards whose types the state's author cannot see.

    A referee's state holds every hand, holding and the deck as counts by type;
    a *seat*'s holds a `HiddenHand`, `HiddenCards` or `HiddenDeck` in the slots
    it cannot see. `len(pile)`, `pile[:]`, equality and `repr` work; indexing a
    type, iterating (so `sum`, `zip`, `tuple`, `max`, `random.shuffle`) and
    assignment raise `HiddenRead`. Read a size through `pile_size` or `len`."""

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
        # Truthy like the `[0] * 5` it replaces, so `if state.hands[seat]:`
        # does not change meaning on an observed state.
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

    # Mutable, like the concrete piles: not usable as dictionary keys.
    __hash__ = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.count})"

    def copy(self) -> "Hidden":
        return type(self)(self.count)

    def add(self, n: int) -> None:
        """`n` more cards -- fewer for a negative `n` -- whose types stay
        unseen: a steal between two other seats, another seat's draw from the
        deck. The count is all that moves."""
        count = self.count + integer_index(n)
        if count < 0:
            raise ValueError(f"{self.count} hidden {self.kind} cannot lose {-n}")
        self.count = count


class HiddenHand(Hidden):
    """Resource cards held by a seat this state's author cannot see.

    What it *can* see is every public change -- a production, a payment, a
    bank or player trade, a discard -- and `move` takes those: the count
    moves, and the identity is kept in `flow`, one running net total per
    resource, for `ledger.PublicLedger.apply_hand_diff` to read off a
    before/after pair exactly as it reads a concrete hand's. `flow` is
    bookkeeping for that diff and nothing else: it is not a composition, it
    is not part of what the pile *is* (equality is by count), and reading a
    type off the pile still raises."""

    __slots__ = ("flow",)
    kind = "resource cards"

    def __init__(self, count: int, flow: Sequence[int] | None = None) -> None:
        super().__init__(count)
        self.flow = [0] * NUM_RESOURCES if flow is None else list(flow)

    def __getitem__(self, index: object) -> "HiddenHand":
        if isinstance(index, slice) and index == slice(None):
            return HiddenHand(self.count, self.flow)
        return super().__getitem__(index)

    def copy(self) -> "HiddenHand":
        return HiddenHand(self.count, self.flow)

    def move(self, resource: int, n: int) -> None:
        """A public change of `n` cards of `resource` -- in, or out for a
        negative `n`. Out of a hand whose composition is unseen, only the
        count can be checked; that the seat held the type is the table's
        word, as a referee's state would have checked it."""
        self.add(n)
        self.flow[resource] += n


class HiddenCards(Hidden):
    """Development cards held by a seat this state's author cannot see."""

    kind = "development cards"


class HiddenDeck(Hidden):
    """A development deck known only by its length."""

    kind = "deck cards"

    def __bool__(self) -> bool:
        # A deck is variable-length, and an empty one must forbid card buys.
        return self.count > 0


def is_hidden(pile: object) -> bool:
    """Whether `pile` carries a count instead of a composition."""
    return isinstance(pile, Hidden)


def pile_size(pile: "Hidden | Sequence[int]") -> int:
    """How many cards `pile` holds, whether or not its types are visible -- the
    one primitive every size-only reader goes through."""
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
    # A seat the state's author cannot see holds a `Hidden` pile instead.
    hands: list[list[int] | HiddenHand] = field(default_factory=list)
    bank: list[int] = field(default_factory=list)
    deck: list[int] | HiddenDeck = field(default_factory=list)
    dev_cards: list[list[int] | HiddenCards] = field(default_factory=list)
    new_dev_cards: list[list[int] | HiddenCards] = field(default_factory=list)
    knights_played: list[int] = field(default_factory=list)
    # Road Building, Year of Plenty and Monopoly plays once resolved, indexed by
    # `DevCard`; the `KNIGHT` and `VICTORY_POINT` slots stay zero.
    dev_cards_played: list[int] = field(default_factory=list)
    # For each seat, how many of its own turns each development card it still
    # holds has been held through, oldest first -- public, as a purchase and
    # a play are, though the card is not. A play takes off the youngest card
    # it could have played (a seat plays its newest usable card and keeps the
    # one it cannot play). `None` for a seat whose history the state was
    # built without; one entry per seat either way (`__post_init__`).
    dev_ages: list[list[int] | None] = field(default_factory=list)
    # Each seat's longest-road length, cached so `victory.update_longest_road`
    # need only recompute the seats a placement could have changed. A path that
    # builds a `GameState` outside `copy_state` must compute it fresh; the
    # incremental update refuses a stale cache (`victory.StaleRoadLengths`).
    road_lengths: list[int] = field(default_factory=list)
    longest_road_holder: int = NO_OWNER
    largest_army_holder: int = NO_OWNER
    # The game type this position is played under, read wherever one of its
    # rules applies: the win check, the discard limit, the robber shield,
    # Road Building. Immutable and shared by reference.
    rules: Rules = STANDARD

    def __post_init__(self) -> None:
        # A state built without the card ages holds none, one entry a seat.
        if not self.dev_ages:
            self.dev_ages = [None] * self.num_players


def new_game(
    board: Board,
    num_players: int,
    rng: random.Random | None = None,
    *,
    rules: Rules = STANDARD,
) -> GameState:
    """A `GameState` at the start of setup on `board` for `num_players`: no
    pieces, empty hands, a full bank, the robber on the first desert (beside
    the board, `OFF_BOARD`, where there is none) and a deck shuffled by `rng`
    when one is given.
    Raises `ValueError` unless `num_players` is 2 to 6."""
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
        robber=deserts[0] if deserts else OFF_BOARD,
        hands=[[0] * NUM_RESOURCES for _ in range(num_players)],
        bank=[BANK_PER_RESOURCE] * NUM_RESOURCES,
        deck=make_deck(rng),
        dev_cards=[[0] * NUM_DEV_CARDS for _ in range(num_players)],
        new_dev_cards=[[0] * NUM_DEV_CARDS for _ in range(num_players)],
        knights_played=[0] * num_players,
        dev_cards_played=[0] * NUM_DEV_CARDS,
        dev_ages=[[] for _ in range(num_players)],
        road_lengths=[0] * num_players,
        rules=rules,
    )


def copy_state(state: GameState) -> GameState:
    """A state that can be mutated without touching the original. The board is
    shared rather than copied: it is frozen for the whole game and by far the
    largest object here. Hidden piles copy through the same `[:]` idiom."""
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
        dev_cards_played=state.dev_cards_played[:],
        dev_ages=[None if ages is None else ages[:] for ages in state.dev_ages],
        road_lengths=state.road_lengths[:],
        longest_road_holder=state.longest_road_holder,
        largest_army_holder=state.largest_army_holder,
        rules=state.rules,
    )


def observed_by(state: GameState, seat: int) -> GameState:
    """A copy of `state` holding only what `seat` can see: its own hand and
    development cards exact, every other seat's a `HiddenHand`/`HiddenCards` of
    the same size and the deck a `HiddenDeck`. Also the honesty probe -- a path
    that reads nothing it should not decides the same on this as on `state`."""
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
    """How many settlements `player` has on the board, cities not counted."""
    return sum(
        1
        for v, owner in enumerate(state.vertex_owner)
        if owner == player and state.vertex_building[v] == Building.SETTLEMENT
    )


def city_count(state: GameState, player: int) -> int:
    """How many cities `player` has on the board."""
    return sum(
        1
        for v, owner in enumerate(state.vertex_owner)
        if owner == player and state.vertex_building[v] == Building.CITY
    )


def road_count(state: GameState, player: int) -> int:
    """How many roads `player` has on the board."""
    return sum(1 for owner in state.edge_owner if owner == player)


def can_place_settlement(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> bool:
    """`connected` is False during initial placement, when roads are not required."""
    if settlement_count(state, player) >= MAX_SETTLEMENTS:
        return False
    return settlement_placeable(state, player, vertex, connected=connected)


def settlement_placeable(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> bool:
    """`can_place_settlement` without its piece-limit check: everything about
    *this vertex*, nothing about how many settlements the player has left. A
    caller with a single vertex in hand wants `can_place_settlement`."""
    if state.vertex_building[vertex] != Building.NONE:
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


def _check_index(index: int, count: int, what: str) -> None:
    """Refuse an index off the board, before a negative one wraps round to
    the far end of a list."""
    if not 0 <= index < count:
        raise ValueError(f"no such {what}: {index}")


def check_settlement(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> None:
    """Raise `ValueError` unless `place_settlement` would succeed. Writes
    nothing, so a caller can check every part of an action before applying
    any of it."""
    _check_index(vertex, state.board.topology.num_vertices, "vertex")
    if not can_place_settlement(state, player, vertex, connected=connected):
        raise ValueError(f"player {player} cannot settle vertex {vertex}")


def place_settlement(
    state: GameState, player: int, vertex: int, *, connected: bool = True
) -> None:
    """Put a settlement of `player`'s on `vertex`, charging nothing; `connected`
    as `can_place_settlement` takes it. Raises `ValueError` as
    `check_settlement` does."""
    check_settlement(state, player, vertex, connected=connected)
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = Building.SETTLEMENT


def can_upgrade_to_city(state: GameState, player: int, vertex: int) -> bool:
    """Whether `player` has a settlement on `vertex` and a city left in its
    supply."""
    return city_count(state, player) < MAX_CITIES and city_upgradeable(
        state, player, vertex
    )


def city_upgradeable(state: GameState, player: int, vertex: int) -> bool:
    """`can_upgrade_to_city` without its piece-limit check."""
    return (
        state.vertex_owner[vertex] == player
        and state.vertex_building[vertex] == Building.SETTLEMENT
    )


def check_city(state: GameState, player: int, vertex: int) -> None:
    """Raise `ValueError` unless `upgrade_to_city` would succeed; writes nothing."""
    _check_index(vertex, state.board.topology.num_vertices, "vertex")
    if not can_upgrade_to_city(state, player, vertex):
        raise ValueError(f"player {player} has no settlement on vertex {vertex}")


def upgrade_to_city(state: GameState, player: int, vertex: int) -> None:
    """Turn `player`'s settlement on `vertex` into a city, charging nothing.
    Raises `ValueError` as `check_city` does."""
    check_city(state, player, vertex)
    state.vertex_building[vertex] = Building.CITY


def can_place_road(state: GameState, player: int, edge: int) -> bool:
    """Whether `player` has a road left in its supply and `edge` is free and
    joins its buildings or roads; an opponent's building stops a road network
    continuing through its vertex."""
    if road_count(state, player) >= MAX_ROADS:
        return False
    return road_placeable(state, player, edge)


def road_placeable(state: GameState, player: int, edge: int) -> bool:
    """`can_place_road` without its piece-limit check: `road_count` scans every
    edge, so asking it per candidate edge is quadratic in the board."""
    if state.edge_owner[edge] != NO_OWNER:
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


def check_road(state: GameState, player: int, edge: int) -> None:
    """Raise `ValueError` unless `place_road` would succeed; writes nothing."""
    _check_index(edge, state.board.topology.num_edges, "edge")
    if not can_place_road(state, player, edge):
        raise ValueError(f"player {player} cannot build road on edge {edge}")


def place_road(state: GameState, player: int, edge: int) -> None:
    """Put a road of `player`'s on `edge`, charging nothing. Raises `ValueError`
    as `check_road` does."""
    check_road(state, player, edge)
    state.edge_owner[edge] = player


def production(state: GameState, roll: int) -> list[list[int]]:
    """Gross resource yield per player for `roll`, before any bank limit: bank
    exhaustion depends on how many players are owed, so it is not per-hex."""
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
    """How many resources of their choice each seat may claim from gold
    hexes on `roll`: one per adjacent settlement, two per city, none from
    the hex the robber stands on."""
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

