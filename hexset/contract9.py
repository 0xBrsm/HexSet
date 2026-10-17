# SPDX-License-Identifier: GPL-3.0-only
"""The contract-9 observation: one encoding for every seat.

Contract 8 (`hexset.encoding`) gives each seat's facts their own columns, so a
network reads "this hand affords a road" with separate weights for its own
hand and for each opponent's. Contract 9 encodes every seat the same way, as a
**seat token**, and the board as hexes and vertices only: a road is a link
between two vertices, and a building's or road's owner a pointer to its seat
token, never a column of its own. A network reads all four seat tokens with
the same weights; the seat's identity is a feature on the token.

Rows are seat-relative: the perspective seat is row 0 and the others follow in
turn order, as `encoding.to_frame` orders them, so a value head's output reads
"me" first. Every pointer (`vertex_owner`, `road_owner`) names a row, or `-1`.

Every field is read off the perspective's `hexset.view.View` or the public
state: an opponent's hand as the bounds the ledger certifies (`hand.least`,
`hand.most`), its development cards as what the perspective has not seen
could allow (`dev.most`, `dev.playable`), its own exact. Nothing of trading or
robber deals is read.

**Scales, one per unit** (a card, a victory point, a piece ÷ 10; a pip ÷ 10; a
turn ÷ 100 uncapped; an age `(age + 1) / 20`, 0 for none; a distance in roads
or knights `1 / (1 + n)`, 0 out of reach), so one unit reads the same on every
token: per resource, `bank + pool + Σ hand.least` is 19 cards on every view
whose ledger is consistent (`View.pool` pads where it is not).

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .board.board import Board, pips
from .board.terrain import NUM_RESOURCES, Terrain
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DevCard
from .devcards import dev_count
from .encoding import PLAYABLE_KINDS, knights_needed_all, to_frame
from .game import Game, Phase
from .roads import reach_all, roads_to_take_all
from .state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, NO_OWNER, Building, GameState, pile_size
from .victory import LARGEST_ARMY_VP, LONGEST_ROAD_VP
from .view import View

__all__ = [
    "CONTRACT",
    "SEAT_FEATURES",
    "TABLE_FEATURES",
    "HEX_FEATURES",
    "VERTEX_FEATURES",
    "AGE_SLOTS",
    "SEAT_COLUMNS",
    "TABLE_COLUMNS",
    "Observation9",
    "Layout9",
    "layout",
    "encode9",
    "flatten",
]

CONTRACT = "9"

# One unit, one scale (module docstring).
CARD_SCALE = 10.0
POINT_SCALE = 10.0
PIECE_SCALE = 10.0
PIP_SCALE = 10.0
TURN_SCALE = 100.0
AGE_SCALE = 20.0
# How many of a seat's longest-held development cards are read, oldest first:
# the deck's five victory point cards, all of them where one seat holds them.
AGE_SLOTS = 5
# A held card's age where the state was built without its history (an
# observed position rebuilt without it): unknown, not absent. Contract 8's
# typical ages for the first three slots (`encoding.AGE_FILL`); the fourth and
# fifth repeat the third.
AGE_FILL = (5.2, 3.1, 2.2, 2.2, 2.2)

# Columns of a seat token, in order, named and sized.
_SEAT_BLOCKS: tuple[tuple[str, int], ...] = (
    ("seat", 4),            # place in play order, counted from `game.first`
    ("on_turn", 1),         # the seat whose turn it is
    ("me", 1),              # the perspective
    ("hand_least", NUM_RESOURCES),
    ("hand_most", NUM_RESOURCES),
    ("hand_held", 1),
    ("dev_most", NUM_DEV_CARDS),
    ("dev_held", 1),
    ("dev_oldest", AGE_SLOTS),
    ("dev_playable", len(PLAYABLE_KINDS)),
    ("points", 1),          # buildings and awards: what every seat sees
    ("to_longest_road", 1),
    ("to_largest_army", 1),
    ("must_discard", 1),
    ("pieces", 3),          # settlements, cities, roads left to build
)
_TABLE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("bank", NUM_RESOURCES),
    ("pool", NUM_RESOURCES),
    ("deck", 1),
    ("dev_pool", NUM_DEV_CARDS),
    ("turn", 1),
    ("setup", 1),
    ("rolled", 1),
    ("robber", 1),
)


def _columns(blocks: Sequence[tuple[str, int]]) -> dict[str, slice]:
    out, at = {}, 0
    for name, width in blocks:
        out[name] = slice(at, at + width)
        at += width
    return out


SEAT_COLUMNS = _columns(_SEAT_BLOCKS)
TABLE_COLUMNS = _columns(_TABLE_BLOCKS)
SEAT_FEATURES = sum(width for _, width in _SEAT_BLOCKS)
TABLE_FEATURES = sum(width for _, width in _TABLE_BLOCKS)
# A hex: the five producing terrains one-hot (the desert none), its pips, the robber.
HEX_FEATURES = NUM_RESOURCES + 2
# A vertex: its port one-hot (generic, then one per resource), its building's
# points (1 a settlement, 2 a city).
VERTEX_FEATURES = 1 + NUM_RESOURCES + 1

assert SEAT_FEATURES == 39 and TABLE_FEATURES == 20


@dataclass(frozen=True)
class Observation9:
    """One position from one seat (`encode9`). Float arrays are float32;
    `vertex_owner` and `road_owner` are int64 row pointers, `-1` for none."""

    seats: np.ndarray          # (players, SEAT_FEATURES), the perspective first
    table: np.ndarray          # (TABLE_FEATURES,)
    hexes: np.ndarray          # (hexes, HEX_FEATURES)
    vertices: np.ndarray       # (vertices, VERTEX_FEATURES)
    vertex_owner: np.ndarray   # (vertices,) the building's owner's row
    road_owner: np.ndarray     # (edges,) the road's owner's row
    reach: np.ndarray          # (players, vertices) each row's reach closeness


@dataclass(frozen=True)
class Layout9:
    """Where each field of an `Observation9` sits in its `flatten`ed row."""

    players: int
    num_hexes: int
    num_vertices: int
    num_edges: int
    slices: dict[str, slice]
    shapes: dict[str, tuple[int, ...]]
    width: int


@lru_cache(maxsize=8)
def layout(players: int, num_hexes: int, num_vertices: int, num_edges: int) -> Layout9:
    """The flat layout for a board of this size seating `players`."""
    shapes = {
        "seats": (players, SEAT_FEATURES),
        "table": (TABLE_FEATURES,),
        "hexes": (num_hexes, HEX_FEATURES),
        "vertices": (num_vertices, VERTEX_FEATURES),
        "vertex_owner": (num_vertices,),
        "road_owner": (num_edges,),
        "reach": (players, num_vertices),
    }
    slices, at = {}, 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        slices[name] = slice(at, at + size)
        at += size
    return Layout9(players, num_hexes, num_vertices, num_edges, slices, shapes, at)


def flatten(observation: Observation9) -> np.ndarray:
    """One float32 row: the fields in `Layout9` order, pointers as floats."""
    return np.concatenate([
        observation.seats.ravel(), observation.table, observation.hexes.ravel(),
        observation.vertices.ravel(), observation.vertex_owner.astype(np.float32),
        observation.road_owner.astype(np.float32), observation.reach.ravel(),
    ]).astype(np.float32, copy=False)


# --- the board's fixed part -----------------------------------------------------


@dataclass(frozen=True)
class _Template:
    hexes: np.ndarray
    vertices: np.ndarray


_TEMPLATES: dict[int, tuple[Board, _Template]] = {}


def _template(board: Board) -> _Template:
    """Terrain, pips and ports, worked out once per board (by identity)."""
    entry = _TEMPLATES.get(id(board))
    if entry is not None and entry[0] is board:
        return entry[1]
    hexes = np.zeros((board.num_hexes, HEX_FEATURES), dtype=np.float32)
    for h in range(board.num_hexes):
        terrain = int(board.terrain[h])
        if terrain < NUM_RESOURCES:
            hexes[h, terrain] = 1.0
        hexes[h, NUM_RESOURCES] = pips(board.tokens[h]) / PIP_SCALE
    vertices = np.zeros((board.topology.num_vertices, VERTEX_FEATURES), dtype=np.float32)
    for port in board.ports:
        column = 0 if port.resource is None else 1 + int(port.resource)
        for v in port.vertices:
            vertices[v, column] = 1.0
    hexes.flags.writeable = False
    vertices.flags.writeable = False
    template = _Template(hexes, vertices)
    if len(_TEMPLATES) >= 4096:
        _TEMPLATES.pop(next(iter(_TEMPLATES)))
    _TEMPLATES[id(board)] = (board, template)
    return template


assert Terrain.FOREST == 0 and NUM_RESOURCES == 5  # producing terrains first, in Resource order


# --- reads ------------------------------------------------------------------------


def _closeness(n: int) -> float:
    """`1 / (1 + n)`, and 0 for `-1`, out of reach."""
    return 0.0 if n < 0 else 1.0 / (1.0 + n)


def _ages(state: GameState, seat: int) -> list[float]:
    """The seat's `AGE_SLOTS` longest-held cards' ages, oldest first, scaled;
    0 past its cards."""
    out = [0.0] * AGE_SLOTS
    ages = state.dev_ages[seat]
    if ages is None:
        for k in range(min(dev_count(state, seat), AGE_SLOTS)):
            out[k] = (AGE_FILL[k] + 1.0) / AGE_SCALE
        return out
    for k, age in enumerate(sorted(ages, reverse=True)[:AGE_SLOTS]):
        out[k] = (age + 1.0) / AGE_SCALE
    return out


def rolled(game: Game) -> bool:
    """Whether the dice have been thrown this turn: not before the roll, nor
    in a robber phase a knight opened before it, nor in setup."""
    phase = game.phase
    if phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.ROLL):
        return False
    if phase is Phase.ROBBER:
        return game.resume_phase is not Phase.ROLL
    return True


def _span(*names: str) -> np.ndarray:
    return np.concatenate([np.arange(SEAT_COLUMNS[n].start, SEAT_COLUMNS[n].stop) for n in names])


# The seat columns that are card counts over `CARD_SCALE`, and the rest, in
# the order `encode9` fills them.
_COUNT_COLUMNS = _span("hand_least", "hand_most", "dev_most", "dev_playable")
_DIRECT_COLUMNS = _span("seat", "on_turn", "me", "hand_held", "dev_held", "dev_oldest", "points", "to_longest_road",
                        "to_largest_army", "must_discard", "pieces")
assert sorted([*_COUNT_COLUMNS, *_DIRECT_COLUMNS]) == list(range(SEAT_FEATURES))

_PLACES = SEAT_COLUMNS["seat"].stop - SEAT_COLUMNS["seat"].start
_SETTLEMENT, _CITY = int(Building.SETTLEMENT), int(Building.CITY)

_REACH: dict[int, tuple[tuple, np.ndarray]] = {}


def _reach_closeness(reach: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """`roads.reach_all`'s distances as closeness, board-seat rows, read-only:
    worked out once per position's reach (which that function caches, so an
    unchanged position hands back the same tuple)."""
    held = _REACH.get(id(reach))
    if held is not None and held[0] is reach:
        return held[1]
    out = np.asarray(reach, dtype=np.float32)
    out = np.where(out < 0, np.float32(0.0), 1.0 / (1.0 + np.maximum(out, 0.0))).astype(np.float32)
    out.flags.writeable = False
    if len(_REACH) >= 4096:
        _REACH.pop(next(iter(_REACH)))
    _REACH[id(reach)] = (reach, out)
    return out


def encode9(game: Game, perspective: int) -> Observation9:
    """`game` as `perspective` may know it, in contract 9."""
    state = game._state
    players = state.num_players
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")
    board = state.board
    topology = board.topology
    rows = to_frame(range(players), perspective)          # row i is board seat rows[i]
    row_of = {seat: i for i, seat in enumerate(rows)}
    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()
    roads = roads_to_take_all(state)
    knights = knights_needed_all(state, perspective, unseen[DevCard.KNIGHT])
    mover = game.current_player

    # Board counts in one pass over the vertices: settlements, cities, and
    # building points (`vertex_building` is 1 a settlement, 2 a city: its points).
    settlements, cities, building = [0] * players, [0] * players, [0] * players
    for owner, kind in zip(state.vertex_owner, state.vertex_building):
        if owner >= 0:
            building[owner] += kind
            if kind == _SETTLEMENT:
                settlements[owner] += 1
            elif kind == _CITY:
                cities[owner] += 1

    # Each token in two parts: the card counts, divided by `CARD_SCALE` in
    # float32 together, and the rest worked out per seat as Python floats and
    # rounded to float32 once, the arithmetic each column always had.
    counts = np.empty((players, len(_COUNT_COLUMNS)), dtype=np.float32)
    direct = np.empty((players, len(_DIRECT_COLUMNS)), dtype=np.float64)
    awards = (state.longest_road_holder, state.largest_army_holder)
    quotas = game.discard_quota
    played_now = game.dev_card_played
    for i, s in enumerate(rows):
        place = [0.0] * _PLACES
        place[(s - game.first) % players] = 1.0
        least = view.known[s]
        most = [k + r for k, r in zip(least, view.room[s])] if s != perspective else least
        cards = dev_count(state, s)
        played = played_now and s == mover
        if s == perspective:
            dev_most = [state.dev_cards[s][k] + state.new_dev_cards[s][k] for k in range(NUM_DEV_CARDS)]
            playable = [0 if played else state.dev_cards[s][kind] for kind in PLAYABLE_KINDS]
        else:
            matured = pile_size(state.dev_cards[s])
            dev_most = [min(cards, unseen[k]) for k in range(NUM_DEV_CARDS)]
            playable = [0 if played else min(matured, unseen[kind]) for kind in PLAYABLE_KINDS]
        counts[i] = (*least, *most, *dev_most, *playable)
        points = building[s] + (LONGEST_ROAD_VP if awards[0] == s else 0) + (LARGEST_ARMY_VP if awards[1] == s else 0)
        quota = quotas[s] if quotas else 0
        direct[i] = (
            *place, float(s == mover), float(s == perspective), view.sizes[s] / CARD_SCALE, cards / CARD_SCALE,
            *_ages(state, s), points / POINT_SCALE, _closeness(roads[s]), _closeness(knights[s]),
            quota / CARD_SCALE,
            (MAX_SETTLEMENTS - settlements[s]) / PIECE_SCALE,
            (MAX_CITIES - cities[s]) / PIECE_SCALE,
            (MAX_ROADS - state.edge_owner.count(s)) / PIECE_SCALE,
        )
    seats = np.zeros((players, SEAT_FEATURES), dtype=np.float32)
    seats[:, _COUNT_COLUMNS] = counts / CARD_SCALE
    seats[:, _DIRECT_COLUMNS] = direct

    table = np.zeros(TABLE_FEATURES, dtype=np.float32)
    t = TABLE_COLUMNS
    table[t["bank"]] = np.asarray(state.bank, dtype=np.float32) / CARD_SCALE
    table[t["pool"]] = np.asarray(view.pool, dtype=np.float32) / CARD_SCALE
    table[t["deck"]] = pile_size(state.deck) / CARD_SCALE
    table[t["dev_pool"]] = np.asarray(unseen, dtype=np.float32) / CARD_SCALE
    table[t["turn"]] = game.turns / TURN_SCALE
    table[t["setup"]] = float(game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD))
    table[t["rolled"]] = float(rolled(game))
    table[t["robber"]] = float(game.phase in (Phase.DISCARD, Phase.ROBBER))

    template = _template(board)
    hexes = template.hexes.copy()
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        hexes[state.robber, NUM_RESOURCES + 1] = 1.0
    vertices = template.vertices.copy()
    vertices[:, -1] = np.asarray(state.vertex_building, dtype=np.float32) / POINT_SCALE
    relabel = np.full(players + 1, -1, dtype=np.int64)    # NO_OWNER (-1) reads the last entry
    for seat, row in row_of.items():
        relabel[seat] = row
    vertex_owner = relabel[np.asarray(state.vertex_owner, dtype=np.int64)]
    road_owner = relabel[np.asarray(state.edge_owner, dtype=np.int64)]
    reach = _reach_closeness(reach_all(state))[list(rows)]
    assert NO_OWNER == -1
    return Observation9(seats, table, hexes, vertices, vertex_owner, road_owner, reach)


def incidences(topology: Topology) -> tuple[tuple[int, int], ...]:
    """Every (vertex, edge) pair with the edge touching the vertex, vertex by
    vertex in `topology.vertex_edges` order: a setup turn's settlement with its
    road (`hexset.decisions`). 144 on the standard board."""
    # By identity first: each board dealt brings a `Topology` of its own,
    # equal to the others, and the cache below finds an equal key only by
    # comparing every index tuple, which each `hexset.decisions.tensors` paid.
    held = _INCIDENCES.get(id(topology))
    if held is not None and held[0] is topology:
        return held[1]
    out = _incidences(topology)
    if len(_INCIDENCES) >= 4096:
        _INCIDENCES.pop(next(iter(_INCIDENCES)))
    _INCIDENCES[id(topology)] = (topology, out)
    return out


_INCIDENCES: dict[int, tuple[Topology, tuple[tuple[int, int], ...]]] = {}


@lru_cache(maxsize=8)
def _incidences(topology: Topology) -> tuple[tuple[int, int], ...]:
    return tuple((v, e) for v, edges in enumerate(topology.vertex_edges) for e in edges)
