"""`spawn_bot` on an ONNX spec — the path that regressed to a
`ModuleNotFoundError` when `hexset.server.onnxbot` moved to
`hexset.clients.onnxbot` (the one-distribution restructure) and `api.py`'s
deferred import was left pointing at the old, now-nonexistent module.
`spec in PRESETS` (`search2`, `heximax`) never touches this import at all, so
`tests/server/test_api.py`'s all-`search2` suite could not have caught it —
this is the only server-side test that walks the `.onnx` branch of
`spawn_bot` rather than stubbing it out.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="the .onnx spawn_bot branch needs onnxruntime installed")

from hexset.actions import legal_actions  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.server.api import Config, spawn_bot  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract5.onnx"


def test_spawn_bot_resolves_the_onnx_spec_without_a_module_not_found_error():
    """`spawn_bot` deferred-imports `hexset.clients.onnxbot` (see its
    docstring: the module boundary keeps onnxruntime out of every caller that
    never seats a checkpoint). Before the fix this raised `ModuleNotFoundError:
    No module named 'hexset.server.onnxbot'` for any `.onnx` spec -- any
    model picked in the web UI. `stub-contract5.onnx` is the only contract the
    server still serves (`hexset.server.constants.RECORD_CONTRACTS`), built by
    `tests/clients/fixtures/build_stub.py`: uniform-over-legal prior, zero
    value, no learned weights.
    """
    board = random_base_board(random.Random(0))
    bot = spawn_bot(str(FIXTURE), board, random.Random(1), Config())

    from hexset.game import start

    game = start(board, 4, random.Random(2))
    action = bot.choose(game)
    assert action in legal_actions(game)
