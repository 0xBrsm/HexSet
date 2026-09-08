# SPDX-License-Identifier: GPL-3.0-only
"""The trade round over the API (`hexset.trading`, "The trade round";
`docs/bot-api.md` §3): `POST .../trade/round`, `.../trade/round/answer`,
`.../trade/round/choose`, and the hold a bot's turn takes while a person
has an offer to answer.

Positions are set directly on the engine (phase, current player, hands),
as `test_api.py` does; no bot seats are dealt, so no runner thread races
the test -- gates are installed with `set_trader` instead.
"""

from __future__ import annotations

import pytest

from hexset.board.terrain import Resource
from hexset.game import Phase
from hexset.server.api import ApiError
from hexset.server.seating import locked_of

from conftest import new_tables

WOOD_FOR_ORE = [-1, 0, 0, 0, 1]  # signed towards the actor: gives one wood, gets one ore


class _Wants:
    def __init__(self, resource: int):
        self.resource = resource

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]


class _NeverWants:
    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def _table(*, actor_is_human: bool, other_gate):
    """A four-seat table: the human at `human`, `other` gated by
    `other_gate`, the rest never trading. The current player holds one
    wood and the other side one ore; everyone else has nothing."""
    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    table = registry.get(code)
    game = table.session.game
    human = table.seat_of(token)
    other = next(s for s in range(game.num_players) if s != human)
    table.session.set_trader(other, other_gate)
    for s in range(game.num_players):
        if s not in (human, other):
            table.session.set_trader(s, _NeverWants())
    actor, responder = (human, other) if actor_is_human else (other, human)
    game.phase = Phase.MAIN
    game.current_player = actor
    state = game.state(0, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[actor][Resource.WOOD] = 1
    state.hands[responder][Resource.ORE] = 1
    return registry, table, code, token, human, other


def test_human_offer_collects_bot_answers_and_a_pick_executes_one():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    state = table.session.game._state

    data = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert data["trade_round"]["offer"] == {"actor": human, "bundle": WOOD_FOR_ORE}
    assert data["trade_round"]["responses"] == [{"seat": bot, "kind": "accept", "bundle": WOOD_FOR_ORE}]
    assert data["trade_round"]["awaiting"] == []
    assert table.session.game.trades == [], "a person's pick is explicit"

    data = registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                           {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert (data["trades"][0]["a"], data["trades"][0]["b"]) == (human, bot)
    assert state.hands[human][Resource.ORE] == 1 and state.hands[bot][Resource.WOOD] == 1
    assert data["trade_round"] is None
    assert any("traded" in line for line in data["log"])


def test_the_view_keeps_the_lap_number_and_the_open_round_apart():
    """`round` is the lap number every log line is tagged with; the open
    trade round is `trade_round`. They shared the one key once, so the lap
    number never reached a client and the page's current-round log filter
    matched nothing."""
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))

    view = table.view(human)
    assert isinstance(view["round"], int)
    assert view["trade_round"] is None
    assert all(line.split("\t")[0].isdigit() for line in view["log"])

    data = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert isinstance(data["round"], int)
    assert data["trade_round"]["offer"]["actor"] == human


def test_bot_offer_holds_the_bots_turn_until_the_human_answers():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    state = table.session.game._state

    table.session.begin_round()  # the session's MAIN-entry hook for a bot actor

    view = table.view(human)
    assert view["pending"] == [{"actor": bot, "bundle": WOOD_FOR_ORE}]
    assert view["trade_wait"] == [human] and view["to_move"] is None
    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/action", {"action": {"type": "END_TURN"}}, token)
    assert excinfo.value.status == 409

    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "accept"}, token)
    assert (data["trades"][0]["a"], data["trades"][0]["b"]) == (bot, human)
    assert state.hands[bot][Resource.ORE] == 1 and state.hands[human][Resource.WOOD] == 1
    assert data["pending"] == [] and data["trade_wait"] == []
    assert table.view(human)["to_move"] == bot


def test_human_pass_releases_the_bot_and_a_stale_answer_is_refused():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    table.session.begin_round()

    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "pass"}, token)
    assert data["pending"] == [] and data["trade_wait"] == [] and data["trades"] == []
    assert table.session.open_round is None

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": bot, "received": WOOD_FOR_ORE, "kind": "accept"}, token)
    assert excinfo.value.status == 409


def test_human_counter_to_a_bot_offer_is_picked_by_the_bots_gate():
    """The bot offered wood for ore; the human counters asking for the same
    wood plus nothing else but gives sheep instead of ore -- the bot's gate
    prices only ore, so it declines the counter and the round closes with
    nothing moved. Then the mirror case: a counter the bot does want."""
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    state = table.session.game._state
    state.hands[human][Resource.SHEEP] = 1
    table.session.begin_round()

    # counter, signed towards the actor: bot gives wood, gets one sheep
    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "counter",
                            "bundle": [-1, 0, 1, 0, 0]}, token)
    assert data["trades"] == [] and data["trade_wait"] == []
    assert table.session.open_round is None

    table.session.begin_round()
    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "counter",
                            "bundle": [-1, 0, 0, 0, 1]}, token)
    assert len(data["trades"]) == 1
    assert state.hands[bot][Resource.ORE] == 1 and state.hands[human][Resource.WOOD] == 1


def test_a_round_closes_with_the_turn_and_a_late_choose_is_refused():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    registry.handle("POST", f"/api/games/{code}/trade/round/choose", {"decline": True}, token)
    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                        {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert excinfo.value.status == 409
    assert table.session.game._state.hands[human][Resource.WOOD] == 1


def test_the_old_one_to_one_routes_are_gone():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    for method, path in [("POST", "trade"), ("GET", "trade/acceptable"), ("POST", "trade/confirm")]:
        with pytest.raises(ApiError) as excinfo:
            registry.handle(method, f"/api/games/{code}/{path}", {}, token)
        assert excinfo.value.status == 404


def test_leave_refuses_the_round_s_own_actor():
    """`leave_seat` must not strand a round nobody else can close: the
    actor is the only seat `choose_trade` will act for."""
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)
    assert "open trade round" in excinfo.value.args[0]
    assert human not in locked_of(table.session.game)


def test_leave_refuses_a_round_s_still_awaiting_manual_responder():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    table.session.begin_round()
    assert human in table.session.open_round.awaiting

    with pytest.raises(ApiError) as excinfo:
        registry.handle("POST", "/api/leave", {}, token)
    assert "open trade round" in excinfo.value.args[0]
    assert human not in locked_of(table.session.game)
