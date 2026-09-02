# SPDX-License-Identifier: GPL-3.0-only
"""Translates a catanatron `CatanMap` into a dev-catan `Board`.

Both engines use the same cube coordinate system for hexes (confirmed by
comparing `catanatron.models.coordinate_system.UNIT_VECTORS` against
`hexset.board.coords.DIRECTIONS`: identical (q, r, s) unit vectors under the
same direction names). A hex's coordinate is therefore already the shared key
-- no translation needed there.

A vertex is not so simply keyed on either side: dev-catan indexes vertices by
insertion order out of `topology.build`, catanatron by an autoincrementing
`node_id` assigned in map-generation order. But both derive a vertex's
*identity* the same way -- the frozenset of (up to) three hex coordinates
that meet at it -- so that frozenset is the thing to match on, computed
independently on each side rather than trusted from one engine's internal
bookkeeping.

dev-catan computes it directly: `hexset.board.topology._corner(h, i)` is
`frozenset((h, neighbor(h, i), neighbor(h, i + 1)))` for corner index `i`
against `hexset.board.coords.DIRECTIONS = (EAST, NORTHEAST, NORTHWEST, WEST,
SOUTHWEST, SOUTHEAST)`.

catanatron never states the same formula, so it is reconstructed here from
`catanatron.models.map.get_nodes_and_edges`'s corner-sharing rules and
verified by hand against two of them (the SOUTHEAST-neighbour rule, worked
through in the design notes for this module): a catanatron `NodeRef` corner
of hex `h` is the same physical vertex as dev-catan's numbered corner of the
same `h`, per this fixed table. It does not depend on which neighbouring
tiles catanatron happened to register (land, water or port all carry `.nodes`
and agree), so it is exact all the way to the map's outer edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from hexset.board.board import Board, make_board
from hexset.board.coords import Hex, neighbor
from hexset.board.ports import GENERIC_RATIO, SPECIFIC_RATIO, Port as CatanPort
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology

from catanatron.models.enums import NodeRef
from catanatron.models.map import CatanMap, PORT_DIRECTION_TO_NODEREFS
from catanatron.models.tiles import Port as CatanatronPort

# dev-catan's corner index for each catanatron NodeRef, derived from
# `get_nodes_and_edges`'s sharing rules (see module docstring).
NODE_REF_TO_CORNER: dict[NodeRef, int] = {
    NodeRef.NORTHEAST: 0,
    NodeRef.NORTH: 1,
    NodeRef.NORTHWEST: 2,
    NodeRef.SOUTHWEST: 3,
    NodeRef.SOUTH: 4,
    NodeRef.SOUTHEAST: 5,
}

_RESOURCE_TO_TERRAIN: dict[str | None, Terrain] = {
    "WOOD": Terrain.FOREST,
    "BRICK": Terrain.HILLS,
    "SHEEP": Terrain.PASTURE,
    "WHEAT": Terrain.FIELDS,
    "ORE": Terrain.MOUNTAINS,
    None: Terrain.DESERT,
}

_RESOURCE_NAME_TO_RESOURCE: dict[str, Resource] = {
    "WOOD": Resource.WOOD,
    "BRICK": Resource.BRICK,
    "SHEEP": Resource.SHEEP,
    "WHEAT": Resource.WHEAT,
    "ORE": Resource.ORE,
}


def _hex_corner_key(h: Hex, corner: int) -> frozenset[Hex]:
    return frozenset((h, neighbor(h, corner), neighbor(h, corner + 1)))


@dataclass(frozen=True)
class BoardMapping:
    """The board plus the lookup tables translating catanatron ids into it.

    The `_of` tables translate catanatron ids into dev-catan indices, for
    reading a catanatron `Game`'s state. `coord_of`/`node_of`/`catanatron_edge_of`
    go the other way, for turning a dev-catan `Action` back into one of
    catanatron's `playable_actions`.
    """

    board: Board
    hex_of: dict[tuple[int, int, int], int]
    vertex_of: dict[int, int]
    edge_of: dict[tuple[int, int], int]
    coord_of: dict[int, tuple[int, int, int]]
    node_of: dict[int, int]
    catanatron_edge_of: dict[int, tuple[int, int]]


def translate_board(catan_map: CatanMap) -> BoardMapping:
    hexes = tuple(sorted(Hex(*coord) for coord in catan_map.land_tiles))
    topology = build_topology(hexes)
    hex_of = {(h.q, h.r, h.s): i for i, h in enumerate(topology.hexes)}

    vertex_key_index = {key: i for i, key in enumerate(topology.vertices)}
    vertex_of: dict[int, int] = {}
    for coord, tile in catan_map.tiles.items():
        h = Hex(*coord)
        for node_ref, node_id in tile.nodes.items():
            key = _hex_corner_key(h, NODE_REF_TO_CORNER[node_ref])
            index = vertex_key_index.get(key)
            if index is None:
                continue  # off dev-catan's board entirely (deep water); never buildable
            if node_id in vertex_of and vertex_of[node_id] != index:
                raise ValueError(
                    f"node {node_id} maps to two different vertices: "
                    f"{vertex_of[node_id]} and {index}"
                )
            vertex_of[node_id] = index

    edge_of: dict[tuple[int, int], int] = {}
    node_id_of_vertex = {v: k for k, v in vertex_of.items()}
    for our_edge_index, (va, vb) in enumerate(topology.edges):
        a, b = node_id_of_vertex.get(va), node_id_of_vertex.get(vb)
        if a is None or b is None:
            continue
        edge_of[(min(a, b), max(a, b))] = our_edge_index

    terrain = tuple(
        _RESOURCE_TO_TERRAIN[catan_map.land_tiles[coord].resource] for coord in
        sorted(catan_map.land_tiles, key=lambda c: hex_of[c])
    )
    tokens = tuple(
        catan_map.land_tiles[coord].number or 0 for coord in
        sorted(catan_map.land_tiles, key=lambda c: hex_of[c])
    )

    ports: list[CatanPort] = []
    for tile in catan_map.tiles.values():
        if not isinstance(tile, CatanatronPort):
            continue
        ref_a, ref_b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        node_a, node_b = tile.nodes[ref_a], tile.nodes[ref_b]
        va, vb = vertex_of.get(node_a), vertex_of.get(node_b)
        if va is None or vb is None:
            continue
        edge_index = topology.edge_index.get((min(va, vb), max(va, vb)))
        if edge_index is None:
            continue
        ratio = SPECIFIC_RATIO if tile.resource is not None else GENERIC_RATIO
        resource = (
            None
            if tile.resource is None
            else _RESOURCE_NAME_TO_RESOURCE[tile.resource]
        )
        ports.append(
            CatanPort(edge=edge_index, vertices=(va, vb), resource=resource, ratio=ratio)
        )

    board = make_board(topology, terrain, tokens, tuple(ports))
    coord_of = {i: coord for coord, i in hex_of.items()}
    node_of = {v: n for n, v in vertex_of.items()}
    catanatron_edge_of = {i: pair for pair, i in edge_of.items()}
    return BoardMapping(
        board=board,
        hex_of=hex_of,
        vertex_of=vertex_of,
        edge_of=edge_of,
        coord_of=coord_of,
        node_of=node_of,
        catanatron_edge_of=catanatron_edge_of,
    )