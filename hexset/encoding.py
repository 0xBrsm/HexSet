# SPDX-License-Identifier: GPL-3.0-only
"""Turn a game state into the heterogeneous graph the model reads.

Two properties matter here and are enforced by construction rather than by
convention:

*Seat-relative.* Seats are rotated so the player to move is always seat 0. The
network therefore learns one policy rather than one per seat, and a position is
encoded identically however the table is numbered.

*Information-set correct.* Only what the perspective player may legally know is
encoded. Own hand and own development cards are exact; opponents contribute
counts alone, plus (`_ledger_parts`) the public-knowledge reconstruction of
their composition that `hexset.ledger` tracks from public events — never
anything only the perspective seat could not have derived from the log.
Nothing downstream can accidentally read a hidden card, which is the gap the
published Catan agents leave open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from .board.board import Board, pips
from .board.terrain import NUM_RESOURCES, Terrain
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DECK_SIZE
from .game import Game, Phase
from .state import NO_OWNER, Building, GameState
from .trading import responders as offer_responders

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
    _packed: np.ndarray | None = field(default=None, repr=False, compare=False)
    _row: int = field(default=-1, repr=False, compare=False)

    def __reduce__(self):
        """Serialize only this position, never the other rows from its tick."""
        return (
            Observation,
            (self.hexes, self.vertices, self.edges, self.globals, self.graph),
        )

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


def global_features(players: int) -> int:
    return (
        NUM_RESOURCES  # own hand
        + (players - 1)  # opponent hand sizes
        + NUM_RESOURCES  # bank
        + NUM_DEV_CARDS  # own development cards
        + (players - 1)  # opponent development card counts
        + players  # knights played
        + players  # public victory points
        + 2 * (players + 1)  # longest road and largest army holders
        + NUM_PHASES
        + 3  # free roads, deck size, turn
        + 2 * NUM_RESOURCES  # live trade offer: give, want
        + 2 * players  # live trade offer: proposer seat, who has answered
        + (players - 1) * (NUM_RESOURCES + 1)  # ledger: known[5] + unknown per opponent
    )


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


@lru_cache(maxsize=8)
def _vertex_rows_all(players: int) -> np.ndarray:
    """The vertex lookup tables stacked for batched perspective indexing."""
    rows = np.stack([_vertex_rows(players, seat) for seat in range(players)])
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=8)
def _edge_rows_all(players: int) -> np.ndarray:
    """The edge lookup tables stacked for batched perspective indexing."""
    rows = np.stack([_edge_rows(players, seat) for seat in range(players)])
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


def _offer_parts(game: Game, perspective: int) -> list[float]:
    """The live trade offer, as the perspective seat may legally see it.

    While an offer stands (`Phase.TRADE_RESPOND`), its give/want bundles and
    its proposer are public — everyone at the table heard it. Who has already
    answered is deliberately *not* public here: the trading design (part 3)
    moves to every-player-responds-then-the-proposer-chooses, approximating
    simultaneous responses, so a responder must not condition on earlier
    declines. Only the proposer's own perspective carries the answered block
    (in today's engine an accept ends the offer at once, so "answered" means
    "declined so far"). With no offer standing the whole block is zero, which
    is also what every checkpoint migrated by `hexnet.migrate` reads it as
    until it trains further.

    One list, four parts, in this order: give (scaled like hands), want,
    proposer seat one-hot (seat-relative, no "none" slot — all zero when no
    offer stands), answered-by-seat (seat-relative).
    """
    players = game.state.num_players
    give = [0.0] * NUM_RESOURCES
    want = [0.0] * NUM_RESOURCES
    proposer = [0.0] * players
    answered = [0.0] * players
    offer = game.offer
    if offer is not None:
        give = [n / HAND_SCALE for n in offer.give]
        want = [n / HAND_SCALE for n in offer.want]
        proposer[_seat(offer.proposer, perspective, players)] = 1.0
        if perspective == offer.proposer:
            declined = set(offer_responders(game.state, offer)) - set(
                game.pending_responders
            )
            for s in declined:
                answered[_seat(s, perspective, players)] = 1.0
    return give + want + proposer + answered


def _ledger_parts(game: Game, perspective: int) -> list[float]:
    """Each opponent's reconstructed hand composition (`hexset.ledger`), in
    seat-relative order, own seat excluded — own hand is already exact
    (`_encode_globals`'s own-hand block above) and never needs a ledger
    entry. Per opponent: `known[5]` (certified per-resource counts, scaled
    like a hand) then `unknown` (cards the ledger cannot type), so this is
    `(players - 1) * (NUM_RESOURCES + 1)` floats — `global_features`'s
    ledger term.
    """
    players = game.state.num_players
    parts: list[float] = []
    for i in range(1, players):
        seat = (perspective + i) % players
        seat_ledger = game.ledger.seats[seat]
        parts.extend(k / HAND_SCALE for k in seat_ledger.known)
        parts.append(seat_ledger.unknown / HAND_SCALE)
    return parts


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

    parts.extend(_offer_parts(game, perspective))
    parts.extend(_ledger_parts(game, perspective))

    return np.array(parts, dtype=np.float32)


def _encode_globals_batch(
    games: Sequence[Game], perspectives: np.ndarray, building_points: np.ndarray
) -> np.ndarray:
    """Write the small global blocks once per batch instead of once per game."""
    batch = len(games)
    players = games[0].state.num_players
    rows = np.arange(batch)[:, None]
    seats = (perspectives[:, None] + np.arange(players)) % players

    hands = np.asarray([game.state.hands for game in games], dtype=np.int16)
    banks = np.asarray([game.state.bank for game in games], dtype=np.int16)
    cards = np.asarray([game.state.dev_cards for game in games], dtype=np.int16)
    fresh = np.asarray(
        [game.state.new_dev_cards for game in games], dtype=np.int16
    )
    knights = np.asarray(
        [game.state.knights_played for game in games], dtype=np.int16
    )

    out = np.zeros((batch, global_features(players)), dtype=np.float32)
    cursor = 0

    def append(values: np.ndarray, scale: float) -> None:
        nonlocal cursor
        block = np.asarray(values, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        width = block.shape[1]
        out[:, cursor : cursor + width] = block / scale
        cursor += width

    append(hands[np.arange(batch), perspectives], HAND_SCALE)
    hand_sizes = hands.sum(axis=2)
    append(hand_sizes[rows, seats[:, 1:]], HAND_SCALE)
    append(banks, BANK_SCALE)

    all_cards = cards + fresh
    append(all_cards[np.arange(batch), perspectives], 5.0)
    card_counts = all_cards.sum(axis=2)
    append(card_counts[rows, seats[:, 1:]], 5.0)
    append(knights[rows, seats], 5.0)

    longest = np.asarray(
        [game.state.longest_road_holder for game in games], dtype=np.intp
    )
    army = np.asarray(
        [game.state.largest_army_holder for game in games], dtype=np.intp
    )
    awards = 2 * (longest[:, None] == seats) + 2 * (army[:, None] == seats)
    append(building_points.astype(np.float64) + awards, 10.0)

    batch_rows = np.arange(batch)
    for holders in (longest, army):
        slots = np.where(
            holders == NO_OWNER,
            players,
            (holders - perspectives) % players,
        )
        out[batch_rows, cursor + slots] = 1.0
        cursor += players + 1

    phases = np.asarray([int(game.phase) for game in games], dtype=np.intp)
    out[batch_rows, cursor + phases] = 1.0
    cursor += NUM_PHASES

    append(np.asarray([game.free_roads for game in games]), 2.0)
    append(np.asarray([len(game.state.deck) for game in games]), DECK_SIZE)
    turns = np.minimum(
        np.asarray([game.turns for game in games], dtype=np.float64) / TURN_SCALE,
        1.0,
    )
    append(turns, 1.0)

    # The offer block is 18 floats a row and offers are one phase of many, so
    # this stays on the canonical per-game path — `_offer_parts` is the single
    # source of the block's semantics, and reusing it keeps the fast path
    # byte-identical to the oracle by construction.
    offer_block = np.asarray(
        [_offer_parts(game, int(p)) for game, p in zip(games, perspectives)],
        dtype=np.float64,
    )
    append(offer_block, 1.0)

    # Same reasoning as the offer block just above: `_ledger_parts` is the
    # single source of the ledger block's semantics, reused per game rather
    # than re-derived, so this fast path stays byte-identical to the oracle
    # by construction.
    ledger_block = np.asarray(
        [_ledger_parts(game, int(p)) for game, p in zip(games, perspectives)],
        dtype=np.float64,
    )
    append(ledger_block, 1.0)

    if cursor != out.shape[1]:
        raise AssertionError(f"wrote {cursor} global features into {out.shape[1]}")
    return out


def encode_batch(
    games: Sequence[Game], perspectives: Sequence[int]
) -> list[Observation]:
    """Encode a collector tick with one set of vector operations.

    The canonical single-position path stays deliberately plain and is the
    oracle for this fast path. Collection asks about a couple dozen independent
    games at once; crossing all of their Python lists into NumPy together avoids
    repeating dozens of tiny allocations and dispatches per position.
    """
    if len(games) != len(perspectives):
        raise ValueError("one perspective is required per game")
    if not games:
        return []

    states = [game.state for game in games]
    players = states[0].num_players
    if any(state.num_players != players for state in states):
        raise ValueError("one batch cannot mix player counts")

    perspective = np.asarray(perspectives, dtype=np.intp)
    if np.any((perspective < 0) | (perspective >= players)):
        raise ValueError("a perspective does not name a player")

    batch = len(games)
    rows = np.arange(batch)
    templates = [_template(state.board, players) for state in states]

    shapes = (
        (states[0].board.num_hexes, HEX_FEATURES),
        (states[0].board.topology.num_vertices, vertex_features(players)),
        (states[0].board.topology.num_edges, edge_features(players)),
        (global_features(players),),
    )
    packed = np.empty(
        (batch, sum(int(np.prod(shape)) for shape in shapes)), dtype=np.float32
    )
    blocks = []
    start = 0
    for shape in shapes:
        stop = start + int(np.prod(shape))
        block = packed[:, start:stop].reshape(batch, *shape)
        if block.base is None:
            raise AssertionError("packed observation slice did not stay a view")
        blocks.append(block)
        start = stop
    hexes, vertices, edges, globals_ = blocks

    np.stack([template.hexes for template in templates], out=hexes)
    robbers = np.asarray([state.robber for state in states], dtype=np.intp)
    hexes[rows, robbers, NUM_TERRAIN + 2] = 1.0

    buildings = np.asarray(
        [state.vertex_building for state in states], dtype=np.intp
    )
    owners = np.asarray([state.vertex_owner for state in states], dtype=np.intp)
    keys = buildings * (players + 1) + owners + 1
    np.stack([template.vertices for template in templates], out=vertices)
    dynamic = NUM_BUILDINGS + players + 1
    vertices[:, :, :dynamic] = _vertex_rows_all(players)[
        perspective[:, None], keys
    ]

    edge_owners = np.asarray([state.edge_owner for state in states], dtype=np.intp)
    edges[:] = _edge_rows_all(players)[perspective[:, None], edge_owners]

    building_value = vertices[:, :, :NUM_BUILDINGS] @ _BUILDING_VALUE
    building_points = np.sum(
        building_value[:, :, None]
        * vertices[:, :, NUM_BUILDINGS : NUM_BUILDINGS + players],
        axis=1,
    )
    globals_[:] = _encode_globals_batch(games, perspective, building_points)

    graph = static_graph(states[0].board.topology)
    return [
        Observation(hexes[i], vertices[i], edges[i], globals_[i], graph, packed, i)
        for i in range(batch)
    ]


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
