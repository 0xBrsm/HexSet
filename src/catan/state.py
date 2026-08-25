from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum

from .board.board import Board
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Terrain
from .cards import NUM_DEV_CARDS, make_deck

NO_OWNER = -1
BANK_PER_RESOURCE = 19

# Standard Catan piece supply, per player.
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


class Building(IntEnum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2


@dataclass
class GameState:
    """Mutable occupancy of a board, plus what each player is holding."""

    board: Board
    num_players: int
    vertex_owner: list[int]
    vertex_building: list[int]
    edge_owner: list[int]
    robber: int
    hands: list[list[int]] = field(default_factory=list)
    bank: list[int] = field(default_factory=list)
    deck: list[int] = field(default_factory=list)
    dev_cards: list[list[int]] = field(default_factory=list)
    new_dev_cards: list[list[int]] = field(default_factory=list)
    knights_played: list[int] = field(default_factory=list)
    longest_road_holder: int = NO_OWNER
    largest_army_holder: int = NO_OWNER


def new_game(
    board: Board, num_players: int, rng: random.Random | None = None
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
    )


def copy_state(state: GameState) -> GameState:
    """A state that can be mutated without touching the original.

    The board is shared rather than copied: it is frozen and never changes
    during a game, and it is by far the largest object here.
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
    )


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
