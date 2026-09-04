# SPDX-License-Identifier: GPL-3.0-only
"""Translates a catanatron `CatanMap` into a dev-catan `Board`, and back.

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

`catanatron_map` runs the same translation backwards, for a catanatron bot
sitting at a hexset table (`bot.py`): it builds the `CatanMap` that
`translate_board` would turn back into the given `Board`. Terrain and tokens go
through `initialize_tiles`'s own parameters; ports need one extra step, because
hexset spaces them evenly around the coast (`hexset.board.ports.place_ports`)
rather than at the official positions, so the template's nine port coordinates
are not where a hexset board's ports are. Every third-layer tile therefore
starts as water and the ports are re-seated on the coastal edges the board
actually has, which `PORT_DIRECTION_TO_NODEREFS`, read backwards, names a
direction for.
"""

from __future__ import annotations

from dataclasses import dataclass

from hexset.board.board import Board, make_board
from hexset.board.coords import Hex, neighbor
from hexset.board.ports import GENERIC_RATIO, SPECIFIC_RATIO, Port as CatanPort
from hexset.board.terrain import TERRAIN_RESOURCE, Resource, Terrain
from hexset.board.topology import build as build_topology

from catanatron.models.enums import NodeRef
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    CatanMap,
    PORT_DIRECTION_TO_NODEREFS,
    initialize_tiles,
)
from catanatron.models.tiles import LandTile, Port as CatanatronPort, Water

from .names import RESOURCE_INDEX, RESOURCE_NAMES

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

# What a hex pays, named each engine's way: catanatron names a land tile by its
# resource (`None` for the desert), dev-catan by its terrain. Derived from
# dev-catan's own `TERRAIN_RESOURCE` rather than spelled out again, so a new
# terrain cannot drift out of step with it -- the two water-only terrains (SEA,
# GOLD) pay no fixed resource and have no catanatron tile.
RESOURCE_OF_TERRAIN: dict[Terrain, str | None] = {
    Terrain.DESERT: None,
    **{
        terrain: RESOURCE_NAMES[resource]
        for terrain, resource in TERRAIN_RESOURCE.items()
        if resource is not None
    },
}

TERRAIN_OF_RESOURCE: dict[str | None, Terrain] = {
    resource: terrain for terrain, resource in RESOURCE_OF_TERRAIN.items()
}

# Which two corners of a coastal water tile a port on that side is reachable
# from, read backwards: a coastal edge's two node refs name the direction the
# port faces, which is what a catanatron `Port` tile is built with.
DIRECTION_OF_NODE_REFS = {
    frozenset(refs): direction
    for direction, refs in PORT_DIRECTION_TO_NODEREFS.items()
}


def _hex_corner_key(h: Hex, corner: int) -> frozenset[Hex]:
    return frozenset((h, neighbor(h, corner), neighbor(h, corner + 1)))


@dataclass(frozen=True)
class BoardMapping:
    """The board plus the lookup tables translating catanatron ids into it.

    The `_of` tables translate catanatron ids into dev-catan indices, for
    reading a catanatron `Game`'s state. `coord_of`/`node_of`/`catanatron_edge_of`
    go the other way, for turning a dev-catan `Action` back into one of
    catanatron's `playable_actions` and for mirroring a dev-catan `GameState`
    into a catanatron `State` (`state.to_catanatron`).
    """

    catan_map: CatanMap
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
        TERRAIN_OF_RESOURCE[catan_map.land_tiles[coord].resource] for coord in
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
            None if tile.resource is None else Resource(RESOURCE_INDEX[tile.resource])
        )
        ports.append(
            CatanPort(edge=edge_index, vertices=(va, vb), resource=resource, ratio=ratio)
        )

    board = make_board(topology, terrain, tokens, tuple(ports))
    coord_of = {i: coord for coord, i in hex_of.items()}
    node_of = {v: n for n, v in vertex_of.items()}
    catanatron_edge_of = {i: pair for pair, i in edge_of.items()}
    return BoardMapping(
        catan_map=catan_map,
        board=board,
        hex_of=hex_of,
        vertex_of=vertex_of,
        edge_of=edge_of,
        coord_of=coord_of,
        node_of=node_of,
        catanatron_edge_of=catanatron_edge_of,
    )


def catanatron_map(board: Board) -> CatanMap:
    """The `CatanMap` `translate_board` turns back into `board` -- its inverse.

    Exact for any board on the base 19-hex layout, ports included; a layout
    catanatron has no template for is out of scope and says so.
    """
    topology = BASE_MAP_TEMPLATE.topology
    land = [coord for coord, kind in topology.items() if kind is LandTile]
    if len(land) != board.topology.num_hexes:
        raise ValueError(f"no catanatron template for a {board.num_hexes}-hex board")
    hex_of = {(h.q, h.r, h.s): i for i, h in enumerate(sorted(Hex(*c) for c in land))}
    terrain = [board.terrain[hex_of[coord]] for coord in land]

    # `initialize_tiles` pops each list from the end as it walks the template's
    # topology in order, and asks for a number only where there is a resource,
    # so both lists are per-land-tile in topology order and reversed.
    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=[
            board.tokens[hex_of[coord]]
            for coord, hex_terrain in zip(land, terrain)
            if hex_terrain is not Terrain.DESERT
        ][::-1],
        shuffled_port_resources_param=[None] * len(BASE_MAP_TEMPLATE.port_resources),
        shuffled_tile_resources_param=[RESOURCE_OF_TERRAIN[t] for t in terrain][::-1],
        number_placement="random",
    )

    sea = {
        coord: Water(tile.nodes, tile.edges)
        for coord, tile in tiles.items()
        if not isinstance(tile, LandTile)
    }
    tiles.update(sea)
    node_of = translate_board(CatanMap.from_tiles(tiles)).node_of
    for port_id, port in enumerate(board.ports):
        nodes = {node_of[v] for v in port.vertices}
        for coord, tile in sea.items():
            refs = [ref for ref, node in tile.nodes.items() if node in nodes]
            if len(refs) == 2:
                break
        else:
            raise ValueError(f"port {port} is not on a coastal edge of the base map")
        if not isinstance(tiles[coord], Water):
            raise ValueError(f"two ports on the water tile at {coord}")
        tiles[coord] = CatanatronPort(
            port_id,
            None if port.resource is None else RESOURCE_NAMES[port.resource],
            DIRECTION_OF_NODE_REFS[frozenset(refs)],
            tile.nodes,
            tile.edges,
        )
    return CatanMap.from_tiles(tiles)
