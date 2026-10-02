# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

from hexset.board import BASE_LAYOUT, Hex, hexagon, islands
from hexset.board import topology as topo


@pytest.mark.parametrize(
    ("layout", "hexes", "vertices", "edges"),
    [
        (BASE_LAYOUT, 19, 54, 72),
    ],
)
def test_known_layout_sizes(layout, hexes, vertices, edges):
    t = topo.build(layout)
    assert (t.num_hexes, t.num_vertices, t.num_edges) == (hexes, vertices, edges)


@pytest.mark.parametrize("radius", [2])
def test_euler_characteristic(radius):
    t = topo.build(hexagon(radius))
    faces = t.num_hexes + 1
    assert t.num_vertices - t.num_edges + faces == 2


def test_vertex_degrees_are_two_or_three():
    t = topo.build(BASE_LAYOUT)
    for edges in t.vertex_edges:
        assert len(edges) in (2, 3)
    assert sum(len(e) for e in t.vertex_edges) == 2 * t.num_edges


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


def test_empty_layout_rejected():
    with pytest.raises(ValueError):
        topo.build([])
