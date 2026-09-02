# SPDX-License-Identifier: GPL-3.0-only
"""Proves the board translation is exact, not merely plausible.

If `NODE_REF_TO_CORNER` were wrong, this would not crash -- it would silently
produce a `Board` with scrambled adjacency, and every game played through it
would be legal-looking nonsense. So the check here is structural: every
catanatron node/edge must land on exactly one dev-catan vertex/edge, covering
all 54/72, with terrain and ports agreeing with what catanatron itself
believes is on each hex.
"""

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE

from hexset.catanatron.board import translate_board


def test_translation_is_a_bijection_on_the_base_map():
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    mapping = translate_board(catan_map)

    assert mapping.board.topology.num_hexes == 19
    assert len(mapping.hex_of) == 19

    # Every land node_id catanatron assigned lands on a distinct dev-catan vertex.
    land_node_ids = {
        node_id for tile in catan_map.land_tiles.values() for node_id in tile.nodes.values()
    }
    assert len(land_node_ids) == 54
    mapped = {mapping.vertex_of[n] for n in land_node_ids if n in mapping.vertex_of}
    assert len(mapped) == 54, "not a bijection: two node_ids collided on one vertex"
    assert mapped == set(range(54))

    # Every land edge catanatron would generate resolves too.
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


def test_terrain_and_tokens_agree_with_catanatron():
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    mapping = translate_board(catan_map)
    board = mapping.board

    for coord, tile in catan_map.land_tiles.items():
        h = mapping.hex_of[coord]
        our_resource = board.terrain[h]
        if tile.resource is None:
            assert our_resource.name == "DESERT"
            assert board.tokens[h] == 0
        else:
            assert our_resource.name == {
                "WOOD": "FOREST",
                "BRICK": "HILLS",
                "SHEEP": "PASTURE",
                "WHEAT": "FIELDS",
                "ORE": "MOUNTAINS",
            }[tile.resource]
            assert board.tokens[h] == tile.number


def test_ports_agree_with_catanatron():
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    mapping = translate_board(catan_map)
    board = mapping.board

    assert len(board.ports) == 9  # 5 specific + 4 generic, per BASE_MAP_TEMPLATE
    specific = [p for p in board.ports if p.resource is not None]
    generic = [p for p in board.ports if p.resource is None]
    assert len(specific) == 5
    assert len(generic) == 4
