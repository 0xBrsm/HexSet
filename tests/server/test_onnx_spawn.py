"""`spawn_bot` on an ONNX spec: the only server-side test that walks the
`.onnx` branch rather than stubbing it out. A preset spec never reaches the
deferred `hexset.clients.onnxbot` import at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="the .onnx spawn_bot branch needs onnxruntime installed")

from hexset.game import Phase  # noqa: E402
from conftest import new_tables  # noqa: E402

VALUED_FIXTURE = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract6-valued.onnx"


def test_a_network_seat_plays_through_local_search_brain(monkeypatch, tmp_path):
    """`LocalSearchBrain.decide` -- the call an embedded bot runner makes --
    drives an ONNX seat's one decision without a live runner thread. The table
    is dealt with the bot seated the normal way, which starts a polling
    `BotRunner`; it is stopped below, because no assertion here should depend
    on a background thread's timing.
    """
    from hexset.clients.onnxmeta import stamp
    from hexset.server._runner import LocalSearchBrain, LocalTransport
    from hexset.server import api

    model_dir = tmp_path / "models"
    stamp(VALUED_FIXTURE, model_dir / "stub.onnx", name="valued", modes=["standard"], version="1", commit="0")
    monkeypatch.setattr(api, "MODELS_DIR", model_dir)

    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": ["valued"]}, None)
    table = registry.get(data["code"])
    seat = next(i for i, s in enumerate(table.seats) if s.name == "valued")

    # The runner may be parked on the table's long poll: `stop_runners` wakes
    # it with a version bump and joins it, where a bare `stop.set()` does not.
    table.stop_runners()

    game = table.session.game
    bot = table.session.traders[seat]
    # Past setup, so `legal_actions` offers more than one option and the
    # stub's uniform-over-legal policy has an ordinary turn to choose from.
    game.phase = Phase.MAIN
    game.current_player = seat

    brain = LocalSearchBrain(bot=bot, game=game)
    token = table.seats[seat].token
    wire = brain.decide(LocalTransport(registry), token, seat)
    result = registry.handle("POST", "/api/action", {"action": wire}, token)
    assert "error" not in result


def test_every_seat_of_a_rollout_file_shares_the_servers_worker_pool(monkeypatch, tmp_path):
    """With `rollout_workers`, a rollout file seated at two tables plays its
    worlds out in one pool, which `Tables.close` stops."""
    from hexset.clients.onnxmeta import stamp, with_metadata
    from hexset.server import api

    stub9 = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract9.onnx"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    rollouts = tmp_path / "rollouts.onnx"
    rollouts.write_bytes(with_metadata(stub9.read_bytes(), {"search": "rollout", "worlds": "3", "top": "2",
                                                            "horizon": "2"}))
    stamp(rollouts, model_dir / "r.onnx", name="roller", modes=["standard"], version="1", commit="0")
    monkeypatch.setattr(api, "MODELS_DIR", model_dir)

    registry = new_tables(rollout_workers=2)
    tables = [registry.get(registry.handle("POST", "/api/games", {"bots": ["roller"]}, None)["code"])
              for _ in range(2)]
    for table in tables:
        table.stop_runners()
    pools = {id(t.session.traders[s].pool) for t in tables for s, seat in enumerate(t.seats) if seat.name == "roller"}
    assert len(pools) == 1
    [pool] = [t.session.traders[s].pool for t in tables[:1] for s, seat in enumerate(t.seats) if seat.name == "roller"]
    assert not pool.broken
    registry.close()
    assert pool.broken
