"""`botclient.py` driven end to end — the module the PI's review found had no
test importing it at all.

Everything here runs over `LocalTransport`, which is `Tables.handle` behind the
same `get`/`post` pair `HttpTransport` presents, so a bot plays exactly the
route an external `python -m hexset.clients.botclient` process plays: join, poll
`/api/state`, fetch `/api/record` when it is on move, post one action. No
privileged access to the session, and no HTTP server to start.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="botclient needs onnxruntime installed")

from hexset.clients.botclient import (  # noqa: E402
    BotRunner,
    LocalTransport,
    RecordBrain,
    _within_offer_budget,
)

from conftest import new_tables  # noqa: E402

STUB4 = Path(__file__).parent / "fixtures" / "stub-contract4.onnx"
DEV_CONTRACT2 = Path(__file__).parent / "fixtures" / "dev-contract2.onnx"


def _table_with_one_open_seat(registry):
    data = registry.handle("POST", "/api/games", {"bots": [], "name": "Ada"}, None)
    return data["code"], data["token"]


def test_a_record_brain_joins_and_plays_its_own_seat():
    """The whole external path, in process: join over `POST /api/join`, then
    let the runner drive the seat off `/api/state` and `/api/record`. Every
    action it plays is one the server offered it — `GameSession.submit`
    rejects anything else, so a single completed ply is the assertion."""
    registry = new_tables()
    code, creator_token = _table_with_one_open_seat(registry)
    transport = LocalTransport(registry)

    joined = transport.post("/api/join", "", {"code": code, "name": "Bot"})
    token, seat = joined["token"], joined["seat"]

    brain = RecordBrain.load(str(STUB4))
    runner = BotRunner(seat=seat, token=token, transport=transport, brain=brain)

    played = 0
    for _ in range(60):
        view = transport.get("/api/state", token)
        if view.get("game_over"):
            break
        if view.get("to_move") == seat:
            before = view["round"], view["phase"]
            assert runner.run_once() is True
            after = transport.get("/api/state", token)
            assert (after["round"], after["phase"]) != before or after["to_move"] != seat
            played += 1
            continue
        # Not this bot's turn: the creator's seat moves the game along, the
        # way another client would.
        creator = transport.get("/api/state", creator_token)
        if creator["to_move"] != creator["seat"]:
            break  # a locked or absent seat; nothing here can advance it
        transport.post(
            "/api/action", creator_token, {"action": creator["legal_actions"][0]}
        )
    assert played > 0, "the bot never got a turn"


def test_a_record_brain_is_refused_the_seat_of_a_game_it_is_not_on():
    """`RecordBrain` has no session access at all: everything it knows comes
    back through the transport, and the transport is token-gated."""
    registry = new_tables()
    _table_with_one_open_seat(registry)
    transport = LocalTransport(registry)
    assert "error" in transport.get("/api/state", "not-a-token")
    assert "error" in transport.get("/api/record", "not-a-token")


def test_a_search_flagged_checkpoint_is_refused_as_an_external_bot():
    """A search needs the true game state to simulate forward, which no
    external client may have. `dev-contract2.onnx` really does ask for one
    (`search=mcts` in its metadata), so this is the real refusal, not a
    contrived one — and it must be refused for *that*, not for its contract."""
    with pytest.raises(ValueError) as caught:
        RecordBrain.load(str(DEV_CONTRACT2))
    message = str(caught.value)
    assert "asks to be searched" in message
    assert "declares contract=" not in message  # not the contract check


def test_the_runner_stops_when_the_game_is_over():
    """`run_once` returning `False` is the only thing that ends a runner's
    loop short of `stop` being set; a runner that kept polling a finished
    game is a thread that never joins."""
    registry = new_tables()
    code, token = _table_with_one_open_seat(registry)
    table = registry.get(code)
    table.session.game.phase = type(table.session.game.phase).GAME_OVER

    runner = BotRunner(seat=0, token=token, transport=LocalTransport(registry), brain=None)
    assert runner.run_once() is False


def test_the_offer_budget_is_read_off_the_wire_not_off_a_live_game():
    """`_within_offer_budget` is `actions.within_offer_budget` without a
    `Game` — the external bot has none. Same rule: trim proposals once the
    budget is spent, unless that would leave nothing legal at all."""
    from hexset.actions import Action, ActionType

    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 1, 0, 0, 0))
    end = Action(ActionType.END_TURN)

    assert _within_offer_budget([offer, end], 0, 1) == [offer, end]
    assert _within_offer_budget([offer, end], 1, 1) == [end]
    assert _within_offer_budget([offer, end], 5, None) == [offer, end]
    # Never leaves a seat with nothing to play.
    assert _within_offer_budget([offer], 1, 1) == [offer]
