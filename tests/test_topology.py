from __future__ import annotations

import pytest

from catan.board import BASE_LAYOUT, MINI_LAYOUT, Hex, hexagon, islands
from catan.board import topology as topo


@pytest.mark.parametrize(
    ("layout", "hexes", "vertices", "edges"),
    [
        (BASE_LAYOUT, 19, 54, 72),
        (MINI_LAYOUT, 7, 24, 30),
        (tuple(hexagon(0)), 1, 6, 6),
    ],
)
def test_known_layout_sizes(layout, hexes, vertices, edges):
    t = topo.build(layout)
    assert (t.num_hexes, t.num_vertices, t.num_edges) == (hexes, vertices, edges)


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
def test_euler_characteristic(radius):
    t = topo.build(hexagon(radius))
    faces = t.num_hexes + 1
    assert t.num_vertices - t.num_edges + faces == 2


def test_hex_has_six_distinct_corners():
    t = topo.build(BASE_LAYOUT)
    for corners in t.hex_vertices:
        assert len(corners) == 6
        assert len(set(corners)) == 6


def test_vertex_degrees_are_two_or_three():
    t = topo.build(BASE_LAYOUT)
    for edges in t.vertex_edges:
        assert len(edges) in (2, 3)
    assert sum(len(e) for e in t.vertex_edges) == 2 * t.num_edges


def test_vertex_touches_one_to_three_on_map_hexes():
    t = topo.build(BASE_LAYOUT)
    for hexes in t.vertex_hexes:
        assert 1 <= len(hexes) <= 3
    interior = [v for v, hs in enumerate(t.vertex_hexes) if len(hs) == 3]
    assert len(interior) == 24


def test_adjacency_is_symmetric():
    t = topo.build(BASE_LAYOUT)
    for v, others in enumerate(t.vertex_neighbors):
        for u in others:
            assert v in t.vertex_neighbors[u]
    for h, others in enumerate(t.hex_neighbors):
        for g in others:
            assert h in t.hex_neighbors[g]


def test_edges_agree_with_vertex_edges():
    t = topo.build(BASE_LAYOUT)
    for e, (a, b) in enumerate(t.edges):
        assert a < b
        assert e in t.vertex_edges[a]
        assert e in t.vertex_edges[b]
        assert b in t.vertex_neighbors[a]


def test_shared_corners_are_deduplicated():
    t = topo.build(MINI_LAYOUT)
    centre = t.hex_index[Hex(0, 0, 0)]
    for ring in t.hex_neighbors[centre]:
        shared = set(t.hex_vertices[centre]) & set(t.hex_vertices[ring])
        assert len(shared) == 2


def test_build_is_deterministic():
    a = topo.build(BASE_LAYOUT)
    b = topo.build(reversed(BASE_LAYOUT))
    assert a.hexes == b.hexes
    assert a.vertices == b.vertices
    assert a.edges == b.edges
    assert a.hex_vertices == b.hex_vertices


def test_disconnected_islands_are_supported():
    layout = islands(Hex(0, 0, 0), Hex(9, -9, 0), radius=1)
    t = topo.build(layout)
    single = topo.build(hexagon(1))
    assert t.num_hexes == 2 * single.num_hexes
    assert t.num_vertices == 2 * single.num_vertices
    assert t.num_edges == 2 * single.num_edges


def test_touching_islands_share_geometry():
    layout = islands(Hex(0, 0, 0), Hex(3, -3, 0), radius=1)
    t = topo.build(layout)
    separate = topo.build(hexagon(1))
    assert t.num_hexes == 2 * separate.num_hexes
    assert t.num_vertices < 2 * separate.num_vertices


def test_empty_layout_rejected():
    with pytest.raises(ValueError):
        topo.build([])
