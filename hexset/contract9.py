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

**Contract numbers.** A minor version only appends columns to an existing
contract (9 to 9.1, 9d to 9d.1), so a checkpoint of the earlier one loads
with zero weights on the new columns and plays as before. Removing or
changing a column takes a new major number (11d).

**Scales, one per unit** (a card, a victory point, a piece ÷ 10; a pip ÷ 10; a
turn ÷ 100 uncapped; an age `(age + 1) / 20`, 0 for none; a distance in roads
or knights `1 / (1 + n)`, 0 out of reach), so one unit reads the same on every
token: per resource, `bank + pool + Σ hand.least` is 19 cards on every view
whose ledger is consistent (`View.pool` pads where it is not).

**Contract 9d is the two-seat observation**, defined for a table dealt to two
seats and refused at any other size. Across a table of two every card flow
is public and named except a steal, and a steal shows its card to the one
opponent, so each seat's hand is exact: the hand is one block, with no
bounds and no residual pool. The one hidden thing is the development deck.
A seat token holds only what both seats see, the same for each: the hand, the
development cards held, bought this turn and their ages, points, the award
distances (1 for the holder), a discard owed, pieces left, and whether the
seat opened the snake (the setup place, one bit). What only the perspective knows is the table's: its development cards
by kind and which it can play, beside the cards it has not seen by kind and
whether the mover has played one this turn. Nothing the other columns already
fix is read: not the bank (19 of each less the two exact hands), the deck
(the unseen less the opponent's held), nor the opponent's bounds by kind (its
held, bought and the unseen fix them). Points and pieces left, which the board
could be counted for, stay as the seat's summaries, and the turn count with
them. The table's rules are the table's constants and are not read. The next roll's odds sit
where they act: each hex carries the chance its number is the next roll
(`hex_features`: terrain, the number, that chance, the robber), under
independent dice its pips over 36 and under balanced dice the dice deck's
odds for the seat rolling next, read off the public rolls
(`hexset.chance.DiceDeck`); the table carries the seven's chance, the one
roll that is no hex. The number stays so hexes sharing one read as
producing together. Seat tokens are 22 wide, the table 20.

**Contract 9d.1** is 9d with each seat's economy appended to its token, both
public: its production by resource (`seat_production`, the cards a roll
yields it at the dice's long-run odds, a city twice, nothing from the
robber's hex) and its bank rate by resource (`hexset.economy.trade_ratios`,
over 4). 9d's columns stay put; seat tokens are 32 wide.

**Contract 11d** is the two-seat observation that puts each fact where it
belongs: private facts on my seat token, facts about a place on that place's
token, and public facts about the whole game on the board block (at four
seats that would be the bank, the development deck's size, the clock, and
the dice deck where dice are balanced). Two seat tokens of different widths,
the perspective's first, carry the same 21 public columns (the hand by
resource, points over the winning points, the two award distances, pieces
left, production and bank rates, as 9d.1 reads them). The opponent's token
adds its development cards held and their ages (27 wide); mine adds what I
alone know, my development cards by kind and those I may play and the cards
I have not seen by kind (35 wide). The board block holds the turn count, the
next roll's chance of a seven and whether the dice have been thrown this
turn (`BOARD_COLUMNS_11D`), each as 9d reads it. A hex reads its number one-hot
over the ten producing numbers (none for the desert) in place of the number
over 12, then the next roll's odds and the robber (17 wide,
`HEX_COLUMNS_11D`). `seats` is the two tokens end to end and `table` is the
board block (`MY_COLUMNS_11D`, `OTHER_COLUMNS_11D`). 11d removes and changes
columns of 9d.1, so it takes a new major number.

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
from .board.ports import BASE_TRADE_RATIO
from .economy import trade_ratios
from .encoding import PLAYABLE_KINDS, knights_needed_all, seat_frame, to_frame
from .game import Game, Phase, ROLL_ODDS, roll_odds
from .roads import reach_all, roads_to_take_all
from .state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, NO_OWNER, Building, GameState, pile_size
from .victory import LARGEST_ARMY_VP, LONGEST_ROAD_VP
from .view import View

__all__ = [
    "observation_row",
    "CONTRACT",
    "CONTRACTS",
    "features",
    "SEAT_COLUMNS_91",
    "TABLE_COLUMNS_91",
    "SEAT_COLUMNS_9D",
    "SEAT_COLUMNS_9D1",
    "MY_COLUMNS_11D",
    "OTHER_COLUMNS_11D",
    "HEX_COLUMNS_11D",
    "BOARD_COLUMNS_11D",
    "NUMBERS_11D",
    "DUEL_CONTRACTS",
    "seat_production",
    "TABLE_COLUMNS_9D",
    "HEX_COLUMNS_9D",
    "hex_features",
    "next_roll_odds",
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
CONTRACTS = ("9", "9.1", "9d", "9d.1", "11d")
# The two-seat contracts (`seat_frame`: a table of two, or more with all but two retired).
DUEL_CONTRACTS = ("9d", "9d.1", "11d")

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
# 9.1 appends public rule settings and the next roll's conditional odds.
SEAT_COLUMNS_91 = _columns((*_SEAT_BLOCKS, ("live", 1)))
TABLE_COLUMNS_91 = _columns((*_TABLE_BLOCKS, ("winning_points", 1), ("discard_limit", 1),
                             ("friendly_robber", 1), ("balanced_dice", 1), ("roll_odds", 11)))
# 9d, the two-seat contract (module docstring): a seat token is what both
# seats see, the perspective's own development cards are the table's, and
# nothing another column fixes is read.
_SEAT_BLOCKS_9D: tuple[tuple[str, int], ...] = (
    ("opened", 1),          # opened the setup snake: `game.first`
    ("on_turn", 1),
    ("me", 1),
    ("hand", NUM_RESOURCES),
    ("dev_held", 1),
    ("dev_new", 1),         # of those, bought this turn: not playable yet
    ("dev_oldest", AGE_SLOTS),
    ("points", 1),          # buildings and awards, over the game's winning points
    ("to_longest_road", 1),  # 1 for the holder
    ("to_largest_army", 1),  # 1 for the holder
    ("must_discard", 1),
    ("pieces", 3),          # settlements, cities, roads left to build
)
_TABLE_BLOCKS_9D: tuple[tuple[str, int], ...] = (
    ("dev_mine", NUM_DEV_CARDS),           # the perspective's cards by kind, bought this turn included
    ("dev_playable", len(PLAYABLE_KINDS)),  # of those, the ones it may play now
    ("dev_pool", NUM_DEV_CARDS),           # the cards it has not seen, by kind: the deck and the opponent's
    ("dev_played", 1),      # the mover has played a development card this turn
    ("turn", 1),
    ("setup", 1),
    ("rolled", 1),
    ("robber", 1),
    ("seven", 1),           # the next roll's chance of a seven
)
SEAT_COLUMNS_9D = _columns(_SEAT_BLOCKS_9D)
TABLE_COLUMNS_9D = _columns(_TABLE_BLOCKS_9D)
SEAT_FEATURES_9D = sum(width for _, width in _SEAT_BLOCKS_9D)
TABLE_FEATURES_9D = sum(width for _, width in _TABLE_BLOCKS_9D)
# 9d.1 appends each seat's economy to its token, both public: its production by
# resource (`seat_production`: expected cards a roll at the dice's long-run
# odds, a city twice, nothing from the robber's hex) and its bank rate by
# resource (`hexset.economy.trade_ratios`, over 4). 9d's columns stay put.
_SEAT_BLOCKS_9D1: tuple[tuple[str, int], ...] = (*_SEAT_BLOCKS_9D, ("production", NUM_RESOURCES),
                                                 ("rates", NUM_RESOURCES))
SEAT_COLUMNS_9D1 = _columns(_SEAT_BLOCKS_9D1)
SEAT_FEATURES_9D1 = sum(width for _, width in _SEAT_BLOCKS_9D1)
# 11d (module docstring): no table token, and two seat tokens of their own
# widths, the perspective's first. Each starts with the public columns both
# seats see alike, the same definitions and scales as 9d.1's.
_PUBLIC_BLOCKS_11D: tuple[tuple[str, int], ...] = (
    ("hand", NUM_RESOURCES),
    ("points", 1),          # buildings and awards, over the game's winning points
    ("to_longest_road", 1),  # 1 for the holder
    ("to_largest_army", 1),  # 1 for the holder
    ("pieces", 3),          # settlements, cities, roads left to build
    ("production", NUM_RESOURCES),  # `seat_production`
    ("rates", NUM_RESOURCES),       # bank rate by resource, over 4
)
_MY_BLOCKS_11D: tuple[tuple[str, int], ...] = (
    *_PUBLIC_BLOCKS_11D,
    ("dev_mine", NUM_DEV_CARDS),           # my cards by kind, bought this turn included
    ("dev_playable", len(PLAYABLE_KINDS)),  # of those, the ones I may play now
    ("dev_pool", NUM_DEV_CARDS),           # the cards I have not seen, by kind
)
_OTHER_BLOCKS_11D: tuple[tuple[str, int], ...] = (
    *_PUBLIC_BLOCKS_11D,
    ("dev_held", 1),
    ("dev_oldest", AGE_SLOTS),
)
MY_COLUMNS_11D = _columns(_MY_BLOCKS_11D)
OTHER_COLUMNS_11D = _columns(_OTHER_BLOCKS_11D)
MY_FEATURES_11D = sum(width for _, width in _MY_BLOCKS_11D)
OTHER_FEATURES_11D = sum(width for _, width in _OTHER_BLOCKS_11D)
# 11d's board block: public facts about the whole game.
_BOARD_BLOCKS_11D: tuple[tuple[str, int], ...] = (
    ("turn", 1),
    ("seven", 1),           # the next roll's chance of a seven
    ("rolled", 1),          # the dice have been thrown this turn
)
BOARD_COLUMNS_11D = _columns(_BOARD_BLOCKS_11D)
BOARD_FEATURES_11D = sum(width for _, width in _BOARD_BLOCKS_11D)


def features(contract: str = CONTRACT) -> tuple[int, int]:
    """Seat and table widths for an observation contract; old columns stay put.
    11d's two seat tokens differ in width: its seat width is both end to
    end, mine then the opponent's, and its table is the board block."""
    if contract not in CONTRACTS:
        raise ValueError(f"unknown observation contract {contract!r}: {CONTRACTS}")
    if contract == "9":
        return SEAT_FEATURES, TABLE_FEATURES
    if contract == "9.1":
        return SEAT_FEATURES + 1, TABLE_FEATURES + 15
    if contract == "9d.1":
        return SEAT_FEATURES_9D1, TABLE_FEATURES_9D
    if contract == "11d":
        return MY_FEATURES_11D + OTHER_FEATURES_11D, BOARD_FEATURES_11D
    return SEAT_FEATURES_9D, TABLE_FEATURES_9D


# A hex: the five producing terrains one-hot (the desert none), its pips, the robber.
HEX_FEATURES = NUM_RESOURCES + 2
# 9d's hex: the terrains, its number over 12, the chance its number is the
# next roll, the robber.
HEX_COLUMNS_9D = {"terrain": slice(0, NUM_RESOURCES), "number": NUM_RESOURCES, "odds": NUM_RESOURCES + 1,
                  "robber": NUM_RESOURCES + 2}
# 11d's hex: the terrains, its number one-hot over `NUMBERS_11D` (none for
# the desert), the chance its number is the next roll, the robber.
NUMBERS_11D = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
HEX_COLUMNS_11D = {"terrain": slice(0, NUM_RESOURCES), "number": slice(NUM_RESOURCES, NUM_RESOURCES + len(NUMBERS_11D)),
                   "odds": NUM_RESOURCES + len(NUMBERS_11D), "robber": NUM_RESOURCES + len(NUMBERS_11D) + 1}
# A vertex: its port one-hot (generic, then one per resource), its building's
# points (1 a settlement, 2 a city).
VERTEX_FEATURES = 1 + NUM_RESOURCES + 1

assert SEAT_FEATURES == 39 and TABLE_FEATURES == 20
assert SEAT_FEATURES_9D == 22 and TABLE_FEATURES_9D == 20
assert MY_FEATURES_11D == 35 and OTHER_FEATURES_11D == 27 and BOARD_FEATURES_11D == 3


def hex_features(contract: str = CONTRACT) -> int:
    """A hex token's width: `HEX_FEATURES` (terrain, pips, the robber); 9.1
    appends the number; 9d reads terrain, the number, the next roll's odds
    and the robber (`HEX_COLUMNS_9D`), and 11d its number one-hot in place of
    the number over 12 (`HEX_COLUMNS_11D`)."""
    if contract not in CONTRACTS:
        raise ValueError(f"unknown observation contract {contract!r}: {CONTRACTS}")
    if contract == "11d":
        return NUM_RESOURCES + len(NUMBERS_11D) + 2
    return HEX_FEATURES if contract == "9" else HEX_FEATURES + 1


@dataclass(frozen=True)
class Observation9:
    """One position from one seat (`encode9`). Float arrays are float32;
    `vertex_owner` and `road_owner` are int64 row pointers, `-1` for none."""

    seats: np.ndarray          # (players, SEAT_FEATURES), the perspective first; 11d's two tokens end to end
    table: np.ndarray          # (TABLE_FEATURES,); 11d's the board block
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
    contract: str = CONTRACT


@lru_cache(maxsize=8)
def layout(players: int, num_hexes: int, num_vertices: int, num_edges: int, contract: str = CONTRACT) -> Layout9:
    """The flat layout for a board of this size seating `players`."""
    if contract in DUEL_CONTRACTS and players != 2:
        raise ValueError(f"contract {contract} is the two-seat observation; this table seats {players}")
    seat_width, table_width = features(contract)
    shapes = {
        "seats": (seat_width,) if contract == "11d" else (players, seat_width),
        "table": (table_width,),
        "hexes": (num_hexes, hex_features(contract)),
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
    return Layout9(players, num_hexes, num_vertices, num_edges, slices, shapes, at, contract)


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
    numbers: np.ndarray     # each hex's number one-hot over `NUMBERS_11D` (11d)


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
    numbers = np.asarray([[float(token == n) for n in NUMBERS_11D] for token in board.tokens], dtype=np.float32)
    hexes.flags.writeable = False
    vertices.flags.writeable = False
    numbers.flags.writeable = False
    template = _Template(hexes, vertices, numbers)
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


def next_roll_odds(game: Game) -> dict[int, float]:
    """Each sum's chance on the next roll to come, by whoever throws it: the
    seat on turn until it has rolled (`rolled`), then the seat after it.
    `game.roll_odds` for that roller: `ROLL_ODDS` under independent dice, the
    dice deck's odds under balanced dice. A sum with no card left is absent."""
    state = game._state
    roller = game.current_player
    if rolled(game):
        n = state.num_players
        roller = next((s for s in ((roller + i) % n for i in range(1, n + 1)) if s not in game.locked), roller)
    dice = state.dice
    return dict(ROLL_ODDS if dice is None else dice.odds(roller))


def _pointers(state: GameState, rows: Sequence[int], players: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every pointer seat-relative (`vertex_owner`, `road_owner`), and each
    row's reach closeness: the board's part, the same under every contract."""
    relabel = np.full(players + 1, -1, dtype=np.int64)    # NO_OWNER (-1) reads the last entry
    for row, seat in enumerate(rows):
        relabel[seat] = row
    vertex_owner = relabel[np.asarray(state.vertex_owner, dtype=np.int64)]
    road_owner = relabel[np.asarray(state.edge_owner, dtype=np.int64)]
    reach = _reach_closeness(reach_all(state))[list(rows)]
    assert NO_OWNER == -1
    return vertex_owner, road_owner, reach


def seat_production(state: GameState, seat: int) -> list[float]:
    """`seat`'s expected cards a roll by resource at the dice's long-run odds
    (pips over 36): a settlement takes each adjacent producing hex once, a city
    twice, and the robber's hex gives nothing. It changes only when the seat
    builds or the robber moves."""
    board = state.board
    topology = board.topology
    out = [0.0] * NUM_RESOURCES
    for v, (owner, kind) in enumerate(zip(state.vertex_owner, state.vertex_building)):
        if owner != seat:
            continue
        for h in topology.vertex_hexes[v]:
            terrain = int(board.terrain[h])
            if terrain < NUM_RESOURCES and h != state.robber:
                out[terrain] += kind * pips(board.tokens[h]) / 36.0
    return out


def _encode9d(game: Game, perspective: int, contract: str = "9d") -> Observation9:
    """`game` as `perspective` may know it, in contract 9d (9d.1 appends each
    seat's economy; 11d is `_encode11d`): a table of two. The table may be
    dealt two seats, or more with all but two retired: its live seats are the
    two (`encoding.seat_frame`), and a retired seat is not read."""
    state = game._state
    if not 0 <= perspective < state.num_players:
        raise ValueError(f"no such player: {perspective}")
    try:
        frame = seat_frame(game, perspective, 2)
    except ValueError as e:
        raise ValueError(f"contract {contract} is the two-seat observation: {e}") from None
    players = 2
    board = state.board
    rows = frame.seats
    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()
    roads = roads_to_take_all(state)
    knights = knights_needed_all(state, perspective, unseen[DevCard.KNIGHT])
    mover = game.current_player
    quotas = game.discard_quota
    awards = (state.longest_road_holder, state.largest_army_holder)
    played_now = game.dev_card_played
    opponent = rows[1]
    # The opponent's hand: what the view certifies plus the residual pool,
    # which at two seats is its uncertified cards by resource (every card is
    # in the bank, in my hand, or in its).
    hands = {perspective: list(view.known[perspective]),
             opponent: [k + p for k, p in zip(view.known[opponent], view.pool)]}

    dealt = state.num_players
    settlements, cities, building = [0] * dealt, [0] * dealt, [0] * dealt
    for owner, kind in zip(state.vertex_owner, state.vertex_building):
        if owner >= 0:
            building[owner] += kind
            if kind == _SETTLEMENT:
                settlements[owner] += 1
            elif kind == _CITY:
                cities[owner] += 1

    if contract == "11d":
        return _encode11d(game, perspective, rows, hands, settlements, cities, building, roads, knights, unseen)
    economy = contract == "9d.1"
    seats = np.zeros((players, SEAT_FEATURES_9D1 if economy else SEAT_FEATURES_9D), dtype=np.float32)
    c = SEAT_COLUMNS_9D1 if economy else SEAT_COLUMNS_9D
    for i, s in enumerate(rows):
        cards = dev_count(state, s)
        points = building[s] + (LONGEST_ROAD_VP if awards[0] == s else 0) + (LARGEST_ARMY_VP if awards[1] == s else 0)
        seats[i, c["opened"]] = float(s == game.first)
        seats[i, c["on_turn"]] = float(s == mover)
        seats[i, c["me"]] = float(s == perspective)
        seats[i, c["hand"]] = np.asarray(hands[s], dtype=np.float32) / CARD_SCALE
        seats[i, c["dev_held"]] = cards / CARD_SCALE
        seats[i, c["dev_new"]] = (cards - pile_size(state.dev_cards[s])) / CARD_SCALE
        seats[i, c["dev_oldest"]] = _ages(state, s)
        seats[i, c["points"]] = points / state.rules.winning_points
        seats[i, c["to_longest_road"]] = _closeness(roads[s])
        seats[i, c["to_largest_army"]] = _closeness(knights[s])
        seats[i, c["must_discard"]] = (quotas[s] if quotas else 0) / CARD_SCALE
        seats[i, c["pieces"]] = [(MAX_SETTLEMENTS - settlements[s]) / PIECE_SCALE,
                                 (MAX_CITIES - cities[s]) / PIECE_SCALE,
                                 (MAX_ROADS - state.edge_owner.count(s)) / PIECE_SCALE]
        if economy:
            seats[i, c["production"]] = seat_production(state, s)
            seats[i, c["rates"]] = [r / BASE_TRADE_RATIO for r in trade_ratios(state, s)]

    mine = [state.dev_cards[perspective][k] + state.new_dev_cards[perspective][k] for k in range(NUM_DEV_CARDS)]
    played = played_now and perspective == mover
    playable = [0 if played else state.dev_cards[perspective][kind] for kind in PLAYABLE_KINDS]
    odds = next_roll_odds(game)
    table = np.zeros(TABLE_FEATURES_9D, dtype=np.float32)
    t = TABLE_COLUMNS_9D
    table[t["dev_mine"]] = np.asarray(mine, dtype=np.float32) / CARD_SCALE
    table[t["dev_playable"]] = np.asarray(playable, dtype=np.float32) / CARD_SCALE
    table[t["dev_pool"]] = np.asarray(unseen, dtype=np.float32) / CARD_SCALE
    table[t["dev_played"]] = float(bool(played_now))
    table[t["turn"]] = game.turns / TURN_SCALE
    table[t["setup"]] = float(game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD))
    table[t["rolled"]] = float(rolled(game))
    table[t["robber"]] = float(game.phase in (Phase.DISCARD, Phase.ROBBER))
    table[t["seven"]] = odds.get(7, 0.0)

    template = _template(board)
    h = HEX_COLUMNS_9D
    hexes = np.zeros((board.num_hexes, hex_features("9d")), dtype=np.float32)
    hexes[:, h["terrain"]] = template.hexes[:, :NUM_RESOURCES]
    hexes[:, h["number"]] = np.asarray(board.tokens, dtype=np.float32) / 12.0
    hexes[:, h["odds"]] = [odds.get(int(token), 0.0) for token in board.tokens]
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        hexes[state.robber, h["robber"]] = 1.0
    vertices = template.vertices.copy()
    vertices[:, -1] = np.asarray(state.vertex_building, dtype=np.float32) / POINT_SCALE
    vertex_owner, road_owner, reach = _pointers(state, rows, state.num_players)
    return Observation9(seats, table, hexes, vertices, vertex_owner, road_owner, reach)


def _encode11d(game: Game, perspective: int, rows: Sequence[int], hands: dict[int, list[int]],
               settlements: list[int], cities: list[int], building: list[int], roads, knights,
               unseen: Sequence[int]) -> Observation9:
    """Contract 11d from `_encode9d`'s reads of the position: my token, then
    the opponent's, end to end, and the board block. Each column is computed as 9d
    and 9d.1 compute it, so a column both contracts read holds the same bytes."""
    state = game._state
    board = state.board
    awards = (state.longest_road_holder, state.largest_army_holder)
    mover = game.current_player
    odds = next_roll_odds(game)
    tokens = []
    for i, (s, c) in enumerate(zip(rows, (MY_COLUMNS_11D, OTHER_COLUMNS_11D))):
        token = np.zeros(MY_FEATURES_11D if i == 0 else OTHER_FEATURES_11D, dtype=np.float32)
        points = building[s] + (LONGEST_ROAD_VP if awards[0] == s else 0) + (LARGEST_ARMY_VP if awards[1] == s else 0)
        token[c["hand"]] = np.asarray(hands[s], dtype=np.float32) / CARD_SCALE
        token[c["points"]] = points / state.rules.winning_points
        token[c["to_longest_road"]] = _closeness(roads[s])
        token[c["to_largest_army"]] = _closeness(knights[s])
        token[c["pieces"]] = [(MAX_SETTLEMENTS - settlements[s]) / PIECE_SCALE,
                              (MAX_CITIES - cities[s]) / PIECE_SCALE,
                              (MAX_ROADS - state.edge_owner.count(s)) / PIECE_SCALE]
        token[c["production"]] = seat_production(state, s)
        token[c["rates"]] = [r / BASE_TRADE_RATIO for r in trade_ratios(state, s)]
        if i == 0:
            mine = [state.dev_cards[s][k] + state.new_dev_cards[s][k] for k in range(NUM_DEV_CARDS)]
            played = game.dev_card_played and s == mover
            playable = [0 if played else state.dev_cards[s][kind] for kind in PLAYABLE_KINDS]
            token[c["dev_mine"]] = np.asarray(mine, dtype=np.float32) / CARD_SCALE
            token[c["dev_playable"]] = np.asarray(playable, dtype=np.float32) / CARD_SCALE
            token[c["dev_pool"]] = np.asarray(unseen, dtype=np.float32) / CARD_SCALE
        else:
            token[c["dev_held"]] = dev_count(state, s) / CARD_SCALE
            token[c["dev_oldest"]] = _ages(state, s)
        tokens.append(token)

    template = _template(board)
    h = HEX_COLUMNS_11D
    hexes = np.zeros((board.num_hexes, hex_features("11d")), dtype=np.float32)
    hexes[:, h["terrain"]] = template.hexes[:, :NUM_RESOURCES]
    hexes[:, h["number"]] = template.numbers
    hexes[:, h["odds"]] = [odds.get(int(token), 0.0) for token in board.tokens]
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        hexes[state.robber, h["robber"]] = 1.0
    vertices = template.vertices.copy()
    vertices[:, -1] = np.asarray(state.vertex_building, dtype=np.float32) / POINT_SCALE
    vertex_owner, road_owner, reach = _pointers(state, rows, state.num_players)
    table = np.zeros(BOARD_FEATURES_11D, dtype=np.float32)
    table[BOARD_COLUMNS_11D["turn"]] = game.turns / TURN_SCALE
    table[BOARD_COLUMNS_11D["seven"]] = odds.get(7, 0.0)
    table[BOARD_COLUMNS_11D["rolled"]] = float(rolled(game))
    return Observation9(np.concatenate(tokens), table, hexes, vertices, vertex_owner, road_owner, reach)


def observation_row(game: Game, perspective: int, contract: str = CONTRACT) -> np.ndarray:
    """`flatten(encode9(game, perspective, contract))`: the row a contract-9
    net reads. A native playout copy (`hexset.native.playout.NativeCopy`)
    encodes itself, to the same bytes."""
    if not isinstance(game, Game):
        return game.observation(perspective, contract)
    return flatten(encode9(game, perspective, contract))


def encode9(game: Game, perspective: int, contract: str = CONTRACT) -> Observation9:
    """`game` as `perspective` may know it, in contract 9 (9.1, or 9d, 9d.1 and 11d at two live seats)."""
    if contract in DUEL_CONTRACTS:
        return _encode9d(game, perspective, contract)
    seat_width, table_width = features(contract)
    state = game._state
    players = state.num_players
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")
    board = state.board
    topology = board.topology
    rows = to_frame(range(players), perspective)          # row i is board seat rows[i]
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
            *_ages(state, s), points / (state.rules.winning_points if contract == "9.1" else POINT_SCALE),
            _closeness(roads[s]), _closeness(knights[s]),
            quota / CARD_SCALE,
            (MAX_SETTLEMENTS - settlements[s]) / PIECE_SCALE,
            (MAX_CITIES - cities[s]) / PIECE_SCALE,
            (MAX_ROADS - state.edge_owner.count(s)) / PIECE_SCALE,
        )
    seats = np.zeros((players, seat_width), dtype=np.float32)
    seats[:, _COUNT_COLUMNS] = counts / CARD_SCALE
    seats[:, _DIRECT_COLUMNS] = direct

    table = np.zeros(table_width, dtype=np.float32)
    t = TABLE_COLUMNS
    table[t["bank"]] = np.asarray(state.bank, dtype=np.float32) / CARD_SCALE
    table[t["pool"]] = np.asarray(view.pool, dtype=np.float32) / CARD_SCALE
    table[t["deck"]] = len(state.deck) / CARD_SCALE
    table[t["dev_pool"]] = np.asarray(unseen, dtype=np.float32) / CARD_SCALE
    table[t["turn"]] = game.turns / TURN_SCALE
    table[t["setup"]] = float(game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD))
    table[t["rolled"]] = float(rolled(game))
    table[t["robber"]] = float(game.phase in (Phase.DISCARD, Phase.ROBBER))
    if contract == "9.1":
        seats[:, SEAT_COLUMNS_91["live"]] = [[float(s not in game.locked)] for s in rows]
        t91 = TABLE_COLUMNS_91
        table[t91["winning_points"]] = state.rules.winning_points / POINT_SCALE
        table[t91["discard_limit"]] = state.rules.discard_limit / CARD_SCALE
        table[t91["friendly_robber"]] = float(state.rules.friendly_robber)
        table[t91["balanced_dice"]] = float(state.rules.balanced_dice)
        odds = dict(roll_odds(game))
        table[t91["roll_odds"]] = [odds.get(total, 0.0) for total in range(2, 13)]

    template = _template(board)
    hexes = template.hexes.copy()
    if contract == "9.1":
        # Pips alone identify neither the roll sum nor its conditional odds.
        hexes = np.column_stack((hexes, np.asarray(board.tokens, dtype=np.float32) / 12.0))
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        hexes[state.robber, NUM_RESOURCES + 1] = 1.0
    vertices = template.vertices.copy()
    vertices[:, -1] = np.asarray(state.vertex_building, dtype=np.float32) / POINT_SCALE
    vertex_owner, road_owner, reach = _pointers(state, rows, players)
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
