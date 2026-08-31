# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Iterable

from .coords import Hex, neighbor

VertexKey = frozenset[Hex]


def _corner(h: Hex, i: int) -> VertexKey:
    """Corner `i` of hex `h`, shared with the neighbours in directions i and i+1.

    Identifying a vertex by the three hex positions touching it makes the key
    canonical regardless of which hex we reached it from. Positions off the map
    are kept in the key so that border vertices stay canonical too.
    """
    return frozenset((h, neighbor(h, i), neighbor(h, i + 1)))


def _vertex_sort_key(key: VertexKey) -> tuple[Hex, ...]:
    return tuple(sorted(key))


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, eq=True)
class Topology:
    """Board connectivity, derived from a bare set of hex coordinates.

    Terrain-agnostic on purpose: base Catan, Seafarers scenarios and arbitrary
    multi-island layouts all produce a Topology the same way, so map variety
    costs no new code here and no new parameters in a graph model.

    Index arrays are stable across builds for a given input, so they can be
    turned into tensors and cached.
    """

    hexes: tuple[Hex, ...]
    vertices: tuple[VertexKey, ...]
    edges: tuple[tuple[int, int], ...]

    hex_vertices: tuple[tuple[int, ...], ...]
    hex_edges: tuple[tuple[int, ...], ...]
    hex_neighbors: tuple[tuple[int, ...], ...]
    vertex_hexes: tuple[tuple[int, ...], ...]
    vertex_edges: tuple[tuple[int, ...], ...]
    vertex_neighbors: tuple[tuple[int, ...], ...]
    edge_hexes: tuple[tuple[int, ...], ...]

    # Lookup tables are derived from the tuples above, so they are left out of
    # equality and hashing — which also keeps Topology usable as a cache key.
    hex_index: dict[Hex, int] = dataclass_field(compare=False)
    vertex_index: dict[VertexKey, int] = dataclass_field(compare=False)
    edge_index: dict[tuple[int, int], int] = dataclass_field(compare=False)

    @property
    def num_hexes(self) -> int:
        return len(self.hexes)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def build(hex_coords: Iterable[Hex]) -> Topology:
    hexes = tuple(sorted(set(hex_coords)))
    if not hexes:
        raise ValueError("cannot build a topology from zero hexes")
    hex_index = {h: i for i, h in enumerate(hexes)}

    corners = {h: tuple(_corner(h, i) for i in range(6)) for h in hexes}

    vertices = tuple(
        sorted({c for cs in corners.values() for c in cs}, key=_vertex_sort_key)
    )
    vertex_index = {k: i for i, k in enumerate(vertices)}

    hex_sides = {
        h: tuple(
            _ordered(vertex_index[corners[h][i - 1]], vertex_index[corners[h][i]])
            for i in range(6)
        )
        for h in hexes
    }
    edges = tuple(sorted({side for sides in hex_sides.values() for side in sides}))
    edge_index = {e: i for i, e in enumerate(edges)}

    hex_vertices = tuple(
        tuple(vertex_index[c] for c in corners[h]) for h in hexes
    )
    hex_edges = tuple(tuple(edge_index[s] for s in hex_sides[h]) for h in hexes)

    touching: list[list[int]] = [[] for _ in edges]
    for h, sides in enumerate(hex_edges):
        for e in sides:
            touching[e].append(h)
    hex_neighbors = tuple(
        tuple(
            hex_index[n]
            for n in (neighbor(h, i) for i in range(6))
            if n in hex_index
        )
        for h in hexes
    )
    vertex_hexes = tuple(
        tuple(sorted(hex_index[h] for h in key if h in hex_index))
        for key in vertices
    )

    incident: list[list[int]] = [[] for _ in vertices]
    adjacent: list[list[int]] = [[] for _ in vertices]
    for e, (a, b) in enumerate(edges):
        incident[a].append(e)
        incident[b].append(e)
        adjacent[a].append(b)
        adjacent[b].append(a)

    return Topology(
        hexes=hexes,
        vertices=vertices,
        edges=edges,
        hex_vertices=hex_vertices,
        hex_edges=hex_edges,
        hex_neighbors=hex_neighbors,
        vertex_hexes=vertex_hexes,
        vertex_edges=tuple(tuple(sorted(xs)) for xs in incident),
        vertex_neighbors=tuple(tuple(sorted(xs)) for xs in adjacent),
        edge_hexes=tuple(tuple(sorted(hs)) for hs in touching),
        hex_index=hex_index,
        vertex_index=vertex_index,
        edge_index=edge_index,
    )


def coastal_edges(topology: Topology) -> tuple[int, ...]:
    """Edges with land on only one side."""
    return tuple(
        e for e in range(topology.num_edges) if len(topology.edge_hexes[e]) == 1
    )


def coastal_rings(topology: Topology) -> tuple[tuple[int, ...], ...]:
    """Coastal edges walked in order around each landmass."""
    remaining = set(coastal_edges(topology))
    by_vertex: dict[int, list[int]] = {}
    for e in remaining:
        for v in topology.edges[e]:
            by_vertex.setdefault(v, []).append(e)

    rings: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.discard(start)
        ring = [start]
        _, cursor = topology.edges[start]
        while True:
            step = next((e for e in by_vertex[cursor] if e in remaining), None)
            if step is None:
                break
            remaining.discard(step)
            ring.append(step)
            a, b = topology.edges[step]
            cursor = b if a == cursor else a
        rings.append(tuple(ring))

    return tuple(rings)
