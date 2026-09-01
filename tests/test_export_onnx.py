# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
onnx = pytest.importorskip("onnx", reason="only needed to run hexset.export_onnx")
ort = pytest.importorskip(
    "onnxruntime", reason="only needed to run hexset.export_onnx"
)

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import static_graph  # noqa: E402
from hexset.export_onnx import _INPUT_NAMES, _OUTPUT_NAMES, _sample_inputs, export  # noqa: E402
from hexset.game import start  # noqa: E402
from hexset.model import HexNet, ModelConfig  # noqa: E402
from hexset.onnx_record import RECORD_FIELDS  # noqa: E402
from hexset.policy import NUM_PAIRS  # noqa: E402


def a_checkpoint(path, *, players: int = 4, max_offers: int | None = 3, seed: int = 0):
    """A checkpoint in the shape `hexset.train.save` writes, tiny enough to
    export quickly. Mirrors `test_netbot.py::a_checkpoint` — kept as its own
    copy rather than a cross-file import, same as that file does for itself.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = HexNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1))
    torch.save(
        {
            "iteration": 7,
            "net": net.state_dict(),
            "args": {
                "players": players,
                "width": 16,
                "rounds": 1,
                "max_offers": max_offers,
            },
        },
        path,
    )
    return board


def test_export_writes_a_parity_checked_onnx_file(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "latest.onnx"

    # export() runs its own numerical-parity check internally (see
    # hexset.export_onnx._verify_parity) and raises if the eager and
    # onnxruntime paths disagree — a clean return is itself the load-bearing
    # assertion here.
    result = export(str(checkpoint), out, topology=board.topology)

    assert result == out
    assert out.exists()


def test_the_graph_is_shaped_the_way_hexset_ui_feeds_and_reads_it_v2(tmp_path):
    """The v2 consumer is `hexset_ui.onnxbot`'s contract-2 path: it feeds the
    23-field information-set record and reads back two argmaxed indices and
    three distributions -- nothing about masking or rotation left for the
    caller to do."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "latest.onnx"
    export(str(checkpoint), out, topology=board.topology)

    model = onnx.load(str(out))
    assert [t.name for t in model.graph.input] == list(RECORD_FIELDS)
    assert [t.name for t in model.graph.output] == list(_OUTPUT_NAMES)

    bool_inputs = {"action_mask", "pair_mask"}
    for tensor in model.graph.input:
        expected = onnx.TensorProto.BOOL if tensor.name in bool_inputs else onnx.TensorProto.INT64
        assert tensor.type.tensor_type.elem_type == expected, tensor.name
        assert tensor.type.tensor_type.shape.dim[0].dim_param == "batch", tensor.name

    int_outputs = {"action_index", "pair_index"}
    for tensor in model.graph.output:
        expected = onnx.TensorProto.INT64 if tensor.name in int_outputs else onnx.TensorProto.FLOAT
        assert tensor.type.tensor_type.elem_type == expected, tensor.name
        assert tensor.type.tensor_type.shape.dim[0].dim_param == "batch", tensor.name

    game = start(board, 4, random.Random(1))
    space = space_for(game)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        inputs = _sample_inputs(space, 4, batch=batch, seed=batch)
        action_index, pair_index, prior, pair_prior, value = session.run(
            list(_OUTPUT_NAMES), inputs
        )
        assert action_index.shape == (batch,)
        assert pair_index.shape == (batch,)
        assert prior.shape == (batch, space.size)
        assert pair_prior.shape == (batch, NUM_PAIRS)
        assert value.shape == (batch, 4)
        assert action_index.dtype == np.int64
        assert pair_index.dtype == np.int64
        assert all(a.dtype == np.float32 for a in (prior, pair_prior, value))
        # Every row's prior is a probability distribution over the legal
        # actions the record's own `action_mask` names -- masking, softmax
        # and normalisation all happened inside the graph.
        for row in range(batch):
            legal = inputs["action_mask"][row]
            assert np.all(prior[row][~legal] == 0.0)
            assert prior[row].sum() == pytest.approx(1.0, abs=1e-4)


def test_the_exported_metadata_says_contract_4_and_matches_the_checkpoints_own_args(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, players=3, max_offers=5)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["contract"] == "4"
    assert meta["players"] == "3"
    assert meta["max_offers"] == "5"
    assert meta["iteration"] == "7"
    assert meta["num_hexes"] == str(board.topology.num_hexes)
    assert meta["num_vertices"] == str(board.topology.num_vertices)
    assert meta["num_edges"] == str(board.topology.num_edges)
    assert meta["source_checkpoint"] == str(checkpoint)
    # A plain export says nothing about search, so the UI plays one forward
    # pass — the cheap default every pre-`search` export already gets.
    assert "search" not in meta
    assert "simulations" not in meta
    assert "wave" not in meta


def test_a_checkpoint_that_omits_max_offers_exports_empty_metadata_not_none(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, max_offers=None)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map["max_offers"] == ""


def test_a_search_export_declares_itself_in_the_keys_hexset_ui_reads(tmp_path):
    """`hexset_ui.modelmeta.search_config`: `search == "mcts"` turns the
    search on, `simulations`/`wave` are its budget, and the UI clamps them."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "mcts256.onnx"

    export(
        str(checkpoint), out, topology=board.topology,
        search="mcts", simulations=256, wave=32,
    )

    meta = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    ).get_modelmeta().custom_metadata_map
    assert meta["contract"] == "4"
    assert meta["search"] == "mcts"
    assert meta["simulations"] == "256"
    assert meta["wave"] == "32"


def test_a_search_budget_without_a_search_is_refused_not_written(tmp_path):
    """The UI ignores `simulations` unless `search=mcts`, so a file carrying
    one without the other would silently be a plain policy."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="search='mcts'"):
        export(str(checkpoint), tmp_path / "x.onnx", topology=board.topology, simulations=64)
    with pytest.raises(ValueError, match="search must be one of"):
        export(str(checkpoint), tmp_path / "x.onnx", topology=board.topology, search="puct")


def test_input_names_are_exactly_the_record_fields():
    """Pins the exporter's input order to `onnx_record.RECORD_FIELDS` --
    hexset-ui's phase-3 record builder has to produce a dict with exactly
    these keys, in a contract both repos can only agree on by name."""
    assert _INPUT_NAMES == RECORD_FIELDS
