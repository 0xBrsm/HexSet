from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
onnx = pytest.importorskip("onnx", reason="only needed to run catan.export_onnx")
ort = pytest.importorskip(
    "onnxruntime", reason="only needed to run catan.export_onnx"
)

from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import static_graph  # noqa: E402
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


def test_a_checkpoint_that_omits_max_offers_exports_empty_metadata_not_none(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    board = a_checkpoint(checkpoint, max_offers=None)
    out = tmp_path / "latest.onnx"

    export(str(checkpoint), out, topology=board.topology)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map["max_offers"] == ""
