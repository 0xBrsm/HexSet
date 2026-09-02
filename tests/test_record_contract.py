"""`hexset_ui.record` against dev-HexNet's own definition of the record.

`record.py` is the one engine-adjacent module this package still carries its
own copy of, and it is a copy only because the canonical version
(`hexset.onnx_record.RECORD_FIELDS` / `record_from_game`, with the per-field
shapes in `hexset.export_onnx._shapes` and the number in `_CONTRACT_VERSION`)
imports torch at module scope, and this package ships an onnxruntime-only
image. `docs/engine-divergence-2026-09-02.md` files that as change request R1;
this module is the guard until it lands.

Everything here skips without torch, which is the normal case on a serving
box and in CI here. That is a real limitation and not a hidden one: on a
machine that *does* have torch — a developer's, or dev-HexNet's own — these
are the tests that catch the two repos' contracts drifting apart, which is the
failure PR #2 shipped.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import build_space
from hexset.board.board import random_base_board
from hexset.game import start

from hexset_ui.record import build_record
from hexset_ui.rules import options_for

from conftest import step_randomly

torch = pytest.importorskip(
    "torch", reason="dev-HexNet's canonical record definition imports torch"
)
onnx_record = pytest.importorskip("hexset.onnx_record")
export_onnx = pytest.importorskip("hexset.export_onnx")


def _position(seed: int = 4, steps: int = 40):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    for _ in range(steps):
        if game.won_by is not None:
            break
        step_randomly(game, rng)
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, 4
    )
    return game, space


def test_the_field_set_and_its_order_match_dev_hexsets():
    game, space = _position()
    record = build_record(game, 0, tuple(options_for(game)), space)
    assert tuple(record) == onnx_record.RECORD_FIELDS


def test_every_field_has_the_shape_dev_exports_it_with():
    """`export_onnx._shapes` is the one table the sample inputs, the read-back
    check and dev's own tests all read. A field that agrees in name and
    disagrees in shape is exactly the failure that does not surface until a
    served checkpoint's first move."""
    game, space = _position()
    record = build_record(game, 0, tuple(options_for(game)), space)
    graph = export_onnx.static_graph(export_onnx._base_topology())
    shapes = export_onnx._shapes(graph, 4, space)
    for name, value in record.items():
        assert np.asarray(value).shape == shapes[name], name


def test_every_field_has_the_dtype_dev_exports_it_with():
    game, space = _position()
    record = build_record(game, 0, tuple(options_for(game)), space)
    for name, value in record.items():
        expected = np.bool_ if name in export_onnx._BOOL_INPUTS else np.int64
        assert np.asarray(value).dtype == expected, name


def test_the_contract_number_this_repo_serves_includes_the_one_dev_exports():
    """dev assigns the number; this repo reads it. If dev bumps to `"5"` and
    nothing here changes, a fresh export stops loading — loudly, by
    `onnxbot._load_cached`, and this test says so first."""
    from hexset_ui.constants import RECORD_CONTRACTS

    assert export_onnx._CONTRACT_VERSION in RECORD_CONTRACTS


def test_the_record_agrees_with_dev_field_for_field_on_the_same_position():
    """The whole reason the copy is tolerable: on the omniscient option list
    dev's own `record_from_game` uses, the two builders must agree exactly.

    `options` is passed here only to make the two comparable — every seat at a
    real HexSet table gets `rules.fair_legal_actions` instead, which is the
    one deliberate difference and the reason `options` is a parameter at all
    (change request R1b asks dev for the same hook).
    """
    from hexset.actions import legal_actions

    game, space = _position()
    seat = game.current_player
    mine = build_record(game, seat, tuple(legal_actions(game)), space)
    theirs = onnx_record.record_from_game(game, seat, space)

    assert mine.keys() == theirs.keys()
    for name in mine:
        assert np.array_equal(np.asarray(mine[name]), np.asarray(theirs[name])), name
