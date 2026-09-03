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
from hexset.game import Phase  # noqa: E402
from hexset.server.api import Config, spawn_bot  # noqa: E402
from conftest import new_tables  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract5.onnx"
VALUED_FIXTURE = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract5-valued.onnx"


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


def test_a_network_seats_first_action_publishes_a_nonzero_valuation(monkeypatch, tmp_path):
    """The one-event trade mechanic (`hexset.trading`, `agents/reference/
    trading-design.md` §8) has both halves wired end to end: `Tables.act`
    already calls `publish_valuation` for any traded seat once it moves
    (PR #18); this pins that an *ONNX* seat's `NetworkBot.valuation` is the
    thing that answers, not the zero default a bot with no `valuation`
    method gets (`hexset.trading.NO_VALUATION`).

    `stub-contract5-valued.onnx` reads `own_hand` through fixed, distinct,
    non-zero per-resource weights (`fixtures/build_stub.py --valued`), so
    every one-card-more delta is that resource's own weight regardless of
    the seat's actual hand -- unlike `stub-contract5.onnx`, whose value head
    really is all zero and could never tell "published nothing" apart from
    "published five honest zeros".

    The table is dealt with the network bot seated the normal way
    (`bots=["valued"]`, via a `MODELS_DIR` this test points at a tmp
    directory), which starts a polling `BotRunner` thread same as any other
    bot seat -- stopped immediately below, because this repo's tests never
    wait on one (a background thread's timing has no place in an assertion,
    see `conftest.stop_bot_runners`). Its one decision is then driven by
    hand through `LocalSearchBrain`/`Tables.act`, the identical two calls
    the runner's own loop would have made, with none of the raciness.
    """
    import shutil

    from hexset.clients.botclient import LocalSearchBrain, LocalTransport
    from hexset.server import api

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    shutil.copy(VALUED_FIXTURE, model_dir / "valued.onnx")
    monkeypatch.setattr(api, "MODELS_DIR", model_dir)

    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": ["valued"]}, None)
    table = registry.get(data["code"])
    seat = next(i for i, s in enumerate(table.seats) if s.name == "valued")

    for i, (runner, thread) in enumerate(table.runners):
        if runner.seat == seat:
            runner.stop.set()
            thread.join(timeout=2.0)
            del table.runners[i]
            break

    game = table.session.game
    bot = table.session.traders[seat]
    # Park the game in MAIN with the network seat to move -- past setup, so
    # `legal_actions` offers more than one option and the stub's
    # uniform-over-legal policy has an ordinary turn to choose from.
    game.phase = Phase.MAIN
    game.current_player = seat

    assert all(v == 0.0 for v in table.session.game.valuations[seat])

    brain = LocalSearchBrain(bot=bot, game=game)
    token = table.seats[seat].token
    wire = brain.decide(LocalTransport(registry), token, seat)
    result = registry.handle("POST", "/api/action", {"action": wire}, token)
    assert "error" not in result

    assert any(v != 0.0 for v in table.session.game.valuations[seat])
