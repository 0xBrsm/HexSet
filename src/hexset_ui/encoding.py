"""Turn a game state into the heterogeneous graph the model reads.

Two properties matter here and are enforced by construction rather than by
convention:

*Seat-relative.* Seats are rotated so the player to move is always seat 0. The
network therefore learns one policy rather than one per seat, and a position is
encoded identically however the table is numbered.

*Information-set correct.* Only what the perspective player may legally know is
encoded. Own hand and own development cards are exact; opponents contribute
counts alone. Nothing downstream can accidentally read a hidden card, which is
the gap the published agents for this game leave open.

**This module is frozen at its current (`contract=1`) feature layout, by
design, and is not kept in sync with dev-hexset's own `encoding.py`.**
dev-hexset has since widened its global feature block with a live-offer
section and a per-seat public-ledger section (see `hexset_ui.record` and
`docs/bot-api.md`'s `contract=2`, which does carry both). Every `contract=1`
checkpoint under `models/` was trained against *this* narrower layout —
widening it here would silently break every one of them, which is exactly
what the `contract` metadata prop (`onnxbot.py`) exists to prevent by
letting a v1 file keep its v1 features forever. New work, ledger included,
goes through `record.py`'s `contract=2` path instead of here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .board.board import Board, pips
from .board.terrain import NUM_RESOURCES, Terrain
from .board.topology import Topology
from .cards import DECK_SIZE
from .game import Game, Phase
from .state import NO_OWNER, Building, GameState

NUM_TERRAIN = len(Terrain)
NUM_BUILDINGS = len(Building)
NUM_PHASES = len(Phase)

MAX_TOKEN_PIPS = 5
BANK_SCALE = 19.0
HAND_SCALE = 10.0
TURN_SCALE = 200.0


@dataclass(frozen=True)
class StaticGraph:
    """Adjacency that depends only on the board, so it is built once per board.

    Kept apart from features because during search the structure never changes
    while the features change at every node.
    """

    num_hexes: int
    num_vertices: int
    num_edges: int
    hex_vertex: np.ndarray
    vertex_edge: np.ndarray
    hex_hex: np.ndarray
    vertex_vertex: np.ndarray


def _pairs(groups) -> np.ndarray:
    out = [(i, j) for i, members in enumerate(groups) for j in members]
    return np.array(out, dtype=np.int64).reshape(-1, 2).T


@lru_cache(maxsize=8)
def static_graph(topology: Topology) -> StaticGraph:
    return StaticGraph(
        num_hexes=topology.num_hexes,
        num_vertices=topology.num_vertices,
        num_edges=topology.num_edges,
        hex_vertex=_pairs(topology.hex_vertices),
        vertex_edge=_pairs(topology.vertex_edges),
        hex_hex=_pairs(topology.hex_neighbors),
        vertex_vertex=_pairs(topology.vertex_neighbors),
    )


@dataclass(frozen=True)
class Observation:
    hexes: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    graph: StaticGraph

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "hexes": self.hexes.shape,
            "vertices": self.vertices.shape,
            "edges": self.edges.shape,
            "globals": self.globals.shape,
        }


HEX_FEATURES = NUM_TERRAIN + 3


def vertex_features(players: int) -> int:
    return NUM_BUILDINGS + (players + 1) + 1 + NUM_RESOURCES


def edge_features(players: int) -> int:
    return players + 1


def _seat(seat: int, perspective: int, players: int) -> int:
    """Rotate so the perspective player is seat 0."""
    return (seat - perspective) % players


@dataclass(frozen=True)
class _Template:
    """The part of an observation that no move can change.

    A search evaluates thousands of positions on one board, and terrain,
    tokens, pips and ports are identical in every one of them. Building these
    per position was about a quarter of `encode`.
    """

    hexes: np.ndarray
    vertices: np.ndarray


# One entry per board, and a vectorised collector holds one board per lane. At
# maxsize 8 a sixteen-lane rollout missed on every single call and rebuilt the
# template it exists to avoid: 7.5k actions/sec against 8.5k, 1.13x, reproduced
# over three alternating runs. A template is about 3 KB, so the headroom is free.
#
# 64 was the same mistake one size up. PPO wants the largest batch the dispatch
# toll can be amortised over, which measured out at 512 lanes, and 512 boards
# miss a 64-entry cache every time. A/B'd in one process, alternating, on the
# torch-free rollout: 84.2 -> 67.6 us per position at 128 lanes (1.25x), 118.5
# -> 101.1 at 512 (1.17x), 126.9 -> 115.8 at 1024 (1.10x). Note the remainder
# is real -- the cost still climbs with lanes once the cache stops missing, so
# most of that rise is working set rather than this, and sizing the cache is not
# the whole answer to it.
@lru_cache(maxsize=4096)
def _template_by_value(board: Board, players: int) -> _Template:
    hexes = np.zeros((board.num_hexes, HEX_FEATURES), dtype=np.float32)
    for h in range(board.num_hexes):
        hexes[h, int(board.terrain[h])] = 1.0
        token = board.tokens[h]
        hexes[h, NUM_TERRAIN] = 1.0 if token else 0.0
        hexes[h, NUM_TERRAIN + 1] = pips(token) / MAX_TOKEN_PIPS

    port_base = NUM_BUILDINGS + players + 1
    vertices = np.zeros(
        (board.topology.num_vertices, vertex_features(players)), dtype=np.float32
    )
    for port in board.ports:
        column = port_base if port.resource is None else port_base + 1 + int(port.resource)
        for v in port.vertices:
            vertices[v, column] = 1.0

    # Handed out by reference to every caller, so a stray write would corrupt
    # every later encode on this board rather than just one observation.
    hexes.flags.writeable = False
    vertices.flags.writeable = False
    return _Template(hexes=hexes, vertices=vertices)


_TEMPLATE_IDENTITIES: dict[tuple[int, int], tuple[Board, _Template]] = {}


def _template(board: Board, players: int) -> _Template:
    """Board template with a fast identity hit before structural hashing.

    Frozen dataclasses make `Board` a safe value-cache key, but hashing one
    recursively walks the topology. A live collector asks about the exact same
    board thousands of times, so key that hot path by identity and retain the
    value cache for distinct-but-equal boards and other callers.
    """
    key = (id(board), players)
    cached = _TEMPLATE_IDENTITIES.get(key)
    if cached is not None and cached[0] is board:
        return cached[1]

    template = _template_by_value(board, players)
    if len(_TEMPLATE_IDENTITIES) >= 4096:
        _TEMPLATE_IDENTITIES.pop(next(iter(_TEMPLATE_IDENTITIES)))
    _TEMPLATE_IDENTITIES[key] = (board, template)
    return template


def _slots(players: int, perspective: int) -> list[int]:
    """Seat column per raw owner value, with NO_OWNER last.

    A table indexed by the owner as stored means the rotation and the
    unowned case both become a lookup, and `NO_OWNER` being -1 lands on the
    last row under numpy's negative indexing without a branch.
    """
    return [_seat(owner, perspective, players) for owner in range(players)] + [players]


def _vertex_key(building: int, owner: int, players: int) -> int:
    """One index standing for a vertex's building and owner together.

    Two lookups become one, which matters because crossing a Python list into
    numpy costs more per element than the loop this replaced: one array of 54
    keys is cheaper than two arrays of 54 fields.
    """
    return building * (players + 1) + owner + 1


@lru_cache(maxsize=32)
def _vertex_rows(players: int, perspective: int) -> np.ndarray:
    """`rows[key]` is the building and owner block of a vertex."""
    width = NUM_BUILDINGS + players + 1
    slots = _slots(players, perspective)
    rows = np.zeros((NUM_BUILDINGS * (players + 1), width), dtype=np.float32)
    for building in range(NUM_BUILDINGS):
        for owner in range(NO_OWNER, players):
            key = _vertex_key(building, owner, players)
            rows[key, building] = 1.0
            rows[key, NUM_BUILDINGS + slots[owner]] = 1.0
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=32)
def _edge_rows(players: int, perspective: int) -> np.ndarray:
    """`rows[owner]` is the whole feature row of an edge."""
    rows = np.zeros((players + 1, edge_features(players)), dtype=np.float32)
    for owner, slot in enumerate(_slots(players, perspective)):
        rows[owner, slot] = 1.0
    rows.flags.writeable = False
    return rows


_BUILDING_VALUE = np.arange(NUM_BUILDINGS, dtype=np.float32)


def _building_points(vertices: np.ndarray, players: int) -> np.ndarray:
    """Every seat's building victory points, in seat-relative order.

    The encoded vertices already carry what a building is worth — the values
    1 and 2 are both the `Building` enum and the points it scores — and who
    owns it, so the score is a contraction of that block rather than another
    walk over the board. Pinned to `victory.building_points` by test.
    """
    unowned = NUM_BUILDINGS + players
    return (vertices[:, :NUM_BUILDINGS] @ _BUILDING_VALUE) @ vertices[
        :, NUM_BUILDINGS:unowned
    ]


def _encode_hexes(state: GameState, template: _Template) -> np.ndarray:
    out = template.hexes.copy()
    out[state.robber, NUM_TERRAIN + 2] = 1.0
    return out


def _encode_vertices(
    state: GameState, perspective: int, template: _Template
) -> np.ndarray:
    players = state.num_players
    span = players + 1
    keys = np.asarray(
        [
            building * span + owner + 1
            for building, owner in zip(state.vertex_building, state.vertex_owner)
        ],
        dtype=np.intp,
    )
    out = template.vertices.copy()
    out[:, : NUM_BUILDINGS + span] = _vertex_rows(players, perspective)[keys]
    return out


def _encode_edges(state: GameState, perspective: int) -> np.ndarray:
    owners = np.asarray(state.edge_owner, dtype=np.intp)
    return _edge_rows(state.num_players, perspective)[owners]


def _encode_globals(
    game: Game, perspective: int, building_points: np.ndarray
) -> np.ndarray:
    from .victory import award_points

    state = game.state
    players = state.num_players
    seats = [(perspective + i) % players for i in range(players)]

    parts: list[float] = []
    parts.extend(n / HAND_SCALE for n in state.hands[perspective])
    parts.extend(sum(state.hands[s]) / HAND_SCALE for s in seats[1:])
    parts.extend(n / BANK_SCALE for n in state.bank)

    own_cards = [
        held + fresh
        for held, fresh in zip(
            state.dev_cards[perspective], state.new_dev_cards[perspective]
        )
    ]
    parts.extend(n / 5.0 for n in own_cards)
    parts.extend(
        (sum(state.dev_cards[s]) + sum(state.new_dev_cards[s])) / 5.0 for s in seats[1:]
    )

    parts.extend(state.knights_played[s] / 5.0 for s in seats)
    parts.extend(
        (building_points[i] + award_points(state, s)) / 10.0
        for i, s in enumerate(seats)
    )

    for holder in (state.longest_road_holder, state.largest_army_holder):
        slot = players if holder == NO_OWNER else _seat(holder, perspective, players)
        one_hot = [0.0] * (players + 1)
        one_hot[slot] = 1.0
        parts.extend(one_hot)

    phase = [0.0] * NUM_PHASES
    phase[int(game.phase)] = 1.0
    parts.extend(phase)

    parts.append(game.free_roads / 2.0)
    parts.append(len(state.deck) / DECK_SIZE)
    parts.append(min(game.turns / TURN_SCALE, 1.0))

    return np.array(parts, dtype=np.float32)


def encode(game: Game, perspective: int | None = None) -> Observation:
    """Encode the position as seen by `perspective`, defaulting to the mover."""
    state = game.state
    if perspective is None:
        perspective = game.current_player
    if not 0 <= perspective < state.num_players:
        raise ValueError(f"no such player: {perspective}")

    template = _template(state.board, state.num_players)
    vertices = _encode_vertices(state, perspective, template)

    return Observation(
        hexes=_encode_hexes(state, template),
        vertices=vertices,
        edges=_encode_edges(state, perspective),
        globals=_encode_globals(
            game, perspective, _building_points(vertices, state.num_players)
        ),
        graph=static_graph(state.board.topology),
    )
