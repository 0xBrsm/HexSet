from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
onnx = pytest.importorskip("onnx", reason="only needed to run catan.export_onnx")
ort = pytest.importorskip(
    "onnxruntime", reason="only needed to run catan.export_onnx"
)

from catan.board.board import random_base_board  # noqa: E402
from catan.board.terrain import NUM_RESOURCES  # noqa: E402
from catan.encoding import encode, static_graph  # noqa: E402
from catan.export_onnx import export  # noqa: E402
from catan.model import CatanNet, ModelConfig  # noqa: E402
from catan.actions import space_for  # noqa: E402
from catan.game import start  # noqa: E402


def a_checkpoint(path, *, players: int = 4, max_offers: int | None = 3, seed: int = 0):
    """A checkpoint in the shape `catan.train.save` writes, tiny enough to
    export quickly. Mirrors `test_netbot.py::a_checkpoint` — kept as its own
    copy rather than a cross-file import, same as that file does for itself.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = CatanNet(space_for(game), graph, players, ModelConfig(width=16, rounds=1))
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
    # catan.export_onnx._verify_parity) and raises if torch and onnxruntime
    # disagree — a clean return is itself the load-bearing assertion here.
    result = export(str(checkpoint), out, topology=board.topology)

    assert result == out
    assert out.exists()


def test_the_graph_is_shaped_the_way_hexset_ui_feeds_and_reads_it(tmp_path):
    """The consumer is `hexset_ui.onnxbot.OnnxPolicy._forward`: it `np.stack`s
    `encode()`'s four arrays under these exact input names, asks for these
    exact four outputs by name, and applies the legal mask itself. So the
    graph must take raw observations of any batch size and hand back raw
    per-slot logits, the two offer heads and a per-seat value — nothing
    masked, nothing argmaxed, nothing rotated."""
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint)
    out = tmp_path / "latest.onnx"
    export(str(checkpoint), out, topology=board.topology)

    model = onnx.load(str(out))
    assert [t.name for t in model.graph.input] == ["hexes", "vertices", "edges", "globals"]
    assert [t.name for t in model.graph.output] == ["logits", "give", "want", "value"]
    for tensor in (*model.graph.input, *model.graph.output):
        assert tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, tensor.name
        assert tensor.type.tensor_type.shape.dim[0].dim_param == "batch", tensor.name

    # Real observations, stacked exactly as the UI stacks them — one for a
    # single decision, several for a wave of MCTS leaves — through one session.
    game = start(board, 4, random.Random(1))
    space = space_for(game)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        observations = [encode(game, seat % 4) for seat in range(batch)]
        inputs = {
            name: np.stack([getattr(o, name) for o in observations]).astype(np.float32)
            for name in ("hexes", "vertices", "edges", "globals")
        }
        logits, give, want, value = session.run(["logits", "give", "want", "value"], inputs)
        assert logits.shape == (batch, space.size)
        assert give.shape == (batch, NUM_RESOURCES)
        assert want.shape == (batch, NUM_RESOURCES)
        assert value.shape == (batch, 4)
        assert all(a.dtype == np.float32 for a in (logits, give, want, value))


def test_the_exported_metadata_matches_the_checkpoints_own_args(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, players=3, max_offers=5)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
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
