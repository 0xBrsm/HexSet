# SPDX-License-Identifier: GPL-3.0-only
"""The board translation must be exact, not merely plausible: a wrong
`NODE_REF_TO_CORNER` scrambles adjacency rather than crashing. Structural:
every catanatron node/edge lands on exactly one hexset vertex/edge, all
54/72 of them.
"""

import pytest

# A submodule, not bare "catanatron": this directory is named `catanatron`
# too, so with `tests/` on sys.path a bare import can resolve to it as an
# empty namespace package and skip nothing. `catanatron.game` exists only in
# the real distribution.
pytest.importorskip("catanatron.game")

from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE

from hexset.catanatron._board import translate_board


def test_translation_is_a_bijection_on_the_base_map():
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    mapping = translate_board(catan_map)

    assert mapping.board.topology.num_hexes == 19
    assert len(mapping.hex_of) == 19

    land_node_ids = {
        node_id for tile in catan_map.land_tiles.values() for node_id in tile.nodes.values()
    }
    assert len(land_node_ids) == 54
    mapped = {mapping.vertex_of[n] for n in land_node_ids if n in mapping.vertex_of}
    assert len(mapped) == 54, "not a bijection: two node_ids collided on one vertex"
    assert mapped == set(range(54))

    from catanatron.models.board import get_edges

    land_edges = get_edges(frozenset(land_node_ids))
    assert len(land_edges) == 72
    mapped_edges = {
        mapping.edge_of[(min(a, b), max(a, b))]
        for a, b in land_edges
        if (min(a, b), max(a, b)) in mapping.edge_of
    }
    assert len(mapped_edges) == 72
    assert mapped_edges == set(range(72))
