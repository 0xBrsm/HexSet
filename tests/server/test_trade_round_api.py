# SPDX-License-Identifier: GPL-3.0-only
"""The trade round over the API: `POST .../trade/round`, `.../round/answer`,
`.../round/choose`, and the hold a bot's turn takes while a person has an
offer to answer. No bot seats are dealt, so no runner thread races the test.
"""

from __future__ import annotations

import pytest

from hexset.board.terrain import Resource
from hexset.game import Phase
from hexset.server.api import ApiError

from conftest import new_tables

WOOD_FOR_ORE = [-1, 0, 0, 0, 1]  # signed towards the actor: gives one wood, gets one ore


class _Wants:
    trade_floor = 0.0

    def __init__(self, resource: int):
        self.resource = resource

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]


class _NeverWants:
    trade_floor = 0.0

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def _table(*, actor_is_human: bool, other_gate):
    """The human at `human`, `other` gated by `other_gate`, the rest never
    trading. The current player holds one wood and the other side one ore.
    """
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
        for r, n in enumerate(hand):  # every card back in the bank, as a table keeps it
            state.bank[r] += n
        hand[:] = [0, 0, 0, 0, 0]
    for seat, r in ((actor, Resource.WOOD), (responder, Resource.ORE)):
        state.bank[r] -= 1
        state.hands[seat][r] = 1
    _certify(game)
    return registry, table, code, token, human, other


def _certify(game) -> None:
    """Make the ledger say what the hands hold, as a roll's distribution
    would have: a bot actor offers only for cards the record certifies."""
    from hexset.ledger import SeatLedger

    state = game.state(0, hidden=False)
    for seat, hand in enumerate(state.hands):
        game.ledger.seats[seat] = SeatLedger()
        for resource, count in enumerate(hand):
            if count:
                game.ledger.receive(seat, resource, count)


def test_human_offer_collects_bot_answers_and_a_pick_executes_one():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    state = table.session.game._state

    data = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert data["trade_round"]["offer"] == {"actor": human, "bundle": WOOD_FOR_ORE}
    # Every seat's answer is in the round, a pass included (bundle null).
    answers = {r["seat"]: r for r in data["trade_round"]["responses"]}
    assert answers[bot] == {"seat": bot, "kind": "accept", "bundle": WOOD_FOR_ORE}
    assert [answers[s]["kind"] for s in answers if s != bot] == ["pass", "pass"]
    assert all(answers[s]["bundle"] is None for s in answers if s != bot)
    assert data["trade_round"]["awaiting"] == []
    assert table.session.game.trades == [], "a person's pick is explicit"

    data = registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                           {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert (data["trades"][0]["a"], data["trades"][0]["b"]) == (human, bot)
    assert state.hands[human][Resource.ORE] == 1 and state.hands[bot][Resource.WOOD] == 1
    assert data["trade_round"] is None
    assert any("Traded with" in line for line in data["log"])


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


def test_a_bot_broadcasts_once_a_turn_however_often_main_is_entered():
    """A knight played in MAIN re-enters MAIN after its robber move and `apply_action`
    calls `begin_round` on every entry, so the second call must be a no-op.
    """
    _registry, table, _code, _token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    session = table.session

    session.begin_round()
    first = session.open_round
    assert first is not None and first.awaiting == {human}
    session.begin_round()

    assert session.open_round is first
    assert [e.note.kind for e in session.events if e.note is not None].count("offer") == 1
    assert table.view(human)["pending"] == [{"actor": bot, "bundle": WOOD_FOR_ORE}]


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


class _ChangesItsMind:
    """Wants wood when it answers, and nothing at all afterwards -- the
    position a bot's accept is in when a person takes their time to pick."""

    trade_floor = 0.0

    def __init__(self):
        self.answered = False

    def respond(self, view, offer):
        from hexset.trading import RESPONSE_ACCEPT, Response
        self.answered = True
        return Response(view.perspective, RESPONSE_ACCEPT, offer.received)

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def test_a_bots_accept_binds_it_when_the_person_picks_it_later():
    """The bot answered the offer; the person's pick executes it without the
    bot being asked again -- its answer was its consent."""
    bot_gate = _ChangesItsMind()
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=bot_gate)
    state = table.session.game._state

    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert bot_gate.answered
    data = registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                           {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert (data["trades"][0]["a"], data["trades"][0]["b"]) == (human, bot)
    assert state.hands[human][Resource.ORE] == 1 and state.hands[bot][Resource.WOOD] == 1


def test_a_bot_prices_a_persons_acceptance_by_who_they_are():
    """The bot's offer went out with another seat in mind; the person took it.
    The bot's gate prices dealing with the person below its floor, so nothing
    executes and no cards move."""

    class NotWithThePerson(_Wants):
        person = None

        def gains_many(self, view, received, counterparties):
            return [-1.0 if c == self.person else g
                    for g, c in zip(super().gains_many(view, received, counterparties), counterparties)]

    bot_gate = NotWithThePerson(Resource.ORE)
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=bot_gate)
    bot_gate.person = human
    state = table.session.game._state
    table.session.begin_round()

    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "accept"}, token)
    assert data["trades"] == []
    assert state.hands[bot][Resource.WOOD] == 1 and state.hands[human][Resource.ORE] == 1


def test_human_counter_to_a_bot_offer_is_picked_by_the_bots_gate():
    """A counter is priced by the actor's own gate: the first offers sheep, which
    this gate does not price; the second the ore it wants.
    """
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    state = table.session.game._state
    state.hands[human][Resource.SHEEP] = 1
    table.session.game.ledger.receive(human, int(Resource.SHEEP), 1)
    table.session.begin_round()

    # counter, signed towards the actor: bot gives wood, gets one sheep
    data = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                           {"actor": bot, "received": WOOD_FOR_ORE, "kind": "counter",
                            "bundle": [-1, 0, 1, 0, 0]}, token)
    assert data["trades"] == [] and data["trade_wait"] == []
    assert table.session.open_round is None

    table.session.game.turns += 1  # a new turn: a bot broadcasts once a turn
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


def test_a_gate_declaring_no_offer_limit_still_offers():
    """`trade_offer_budget` is `-1` for a gate that declares no limit of its
    own -- every checkpoint that says nothing about `max_offers`. Read as a
    spent budget it would silence the seat, which is the one bug a bot that
    never opens a round looks exactly like.
    """
    _registry, table, _code, _token, human, _bot = _table(
        actor_is_human=False, other_gate=_Wants(Resource.ORE)
    )
    session = table.session
    gate = session.traders[session.game.current_player]
    gate.trade_offer_budget = -1

    session.begin_round()

    assert session.open_round is not None
    assert table.view(human)["pending"] == [{"actor": session.game.current_player, "bundle": WOOD_FOR_ORE}]


def test_a_person_offers_any_card_for_ore_and_the_bot_names_the_card():
    """Any card for your ore: the bot counters naming the card it wants, and
    the person picks that counter like any other."""
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    state = table.session.game._state
    data = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [0, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1], "give_any": 1}, token)
    assert data["trade_round"]["offer"] == {"actor": human, "bundle": [0, 0, 0, 0, 1], "any": -1}
    answers = {r["seat"]: r for r in data["trade_round"]["responses"]}
    assert answers[bot] == {"seat": bot, "kind": "counter", "bundle": WOOD_FOR_ORE}
    assert any("any 1 card for 1 Ore" in line for line in data["log"])
    registry.handle("POST", f"/api/games/{code}/trade/round/choose", {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert state.hands[human][Resource.ORE] == 1 and state.hands[bot][Resource.WOOD] == 1


def test_an_open_offer_is_refused_as_an_accept_and_taken_as_a_counter():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_NeverWants())
    session = table.session
    session.open_round_for(bot, (0, 0, 0, 0, 1), -1)        # the bot: any card for your ore
    view = registry.handle("GET", "/api/state", None, token)
    assert view["pending"] == [{"actor": bot, "bundle": [0, 0, 0, 0, 1], "any": -1}]
    with pytest.raises(ApiError, match="counter naming them"):
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": bot, "received": [0, 0, 0, 0, 1], "kind": "accept"}, token)
    # The counter names the card: the person's ore for the bot's wood.
    answered = registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                               {"actor": bot, "received": [0, 0, 0, 0, 1], "kind": "counter",
                                "bundle": WOOD_FOR_ORE}, token)
    assert answered["pending"] == []


def test_an_invitation_needs_only_the_named_cards_its_actor_gives():
    """The actor may invite a counter for more any cards than it holds --
    it can refuse whatever comes back -- but not name cards it lacks."""
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_NeverWants())
    data = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [0, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1], "give_any": 3}, token)
    assert data["trade_round"]["offer"]["any"] == -3
    with pytest.raises(ApiError, match="cover"):
        registry.handle("POST", f"/api/games/{code}/trade/round",
                        {"give": [0, 1, 0, 0, 0], "want": [0, 0, 0, 0, 1], "give_any": 1}, token)
    with pytest.raises(ApiError, match="one side"):
        registry.handle("POST", f"/api/games/{code}/trade/round",
                        {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 0], "give_any": 1, "want_any": 1}, token)


def test_a_persons_offer_and_the_answers_go_on_the_ledger():
    """What a person offers at a table here is public: the ledger keeps it as
    what they want and will give up, and each bot answer the same way, for
    every bot's later offers to read (`PublicLedger.show`)."""
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    ledger = table.session.game.ledger

    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert ledger.seats[human].want[Resource.ORE] == 1
    assert ledger.seats[human].waste[Resource.WOOD] == 1
    # The bot accepted: it wants the wood and will give up the ore.
    assert ledger.seats[bot].want[Resource.WOOD] == 1
    assert ledger.seats[bot].waste[Resource.ORE] == 1

    registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                    {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    # The cards moved: both wants met, both wastes used up.
    assert ledger.seats[human].want == [0] * 5 and ledger.seats[human].waste == [0] * 5
    assert ledger.seats[bot].want == [0] * 5 and ledger.seats[bot].waste == [0] * 5


# --- who may answer, what an answer must cover, and when a round may run ---------


class _Budgeted(_Wants):
    """`_Wants`, declaring its offer budget the way a gate does: through
    `trade_params`, with no loose `trade_offer_budget` attribute."""

    def __init__(self, resource: int, max_offers: int):
        super().__init__(resource)
        from hexset.trading import TradeParams

        self.trade_params = TradeParams(max_offers=max_offers, trade_floor=0.0)


class _WaitsForItsMoment(_Wants):
    """A gate that goes to the table only once it has built something."""

    def __init__(self, resource: int):
        super().__init__(resource)
        self.ready = False

    def trade_now(self, game) -> bool:
        return self.ready


def test_an_accept_the_seat_cannot_cover_is_refused_like_a_counter():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    table.session.begin_round()
    state = table.session.game._state
    state.hands[human][Resource.ORE] = 0  # the ore it would give went elsewhere
    state.bank[Resource.ORE] += 1

    with pytest.raises(ApiError) as accepted:
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": bot, "received": WOOD_FOR_ORE, "kind": "accept"}, token)
    with pytest.raises(ApiError) as countered:
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": bot, "received": WOOD_FOR_ORE, "kind": "counter",
                         "bundle": [-2, 0, 0, 0, 1]}, token)

    assert accepted.value.status == countered.value.status == 409
    assert str(accepted.value) == str(countered.value)
    assert table.session.open_round.awaiting == {human}, "still waiting on a real answer"


def test_the_actor_cannot_answer_its_own_round():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_NeverWants())
    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)

    with pytest.raises(ApiError, match="not waiting on your answer") as refused:
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": human, "received": WOOD_FOR_ORE, "kind": "accept"}, token)
    assert refused.value.status == 409
    assert all(r.seat != human for r in table.session.open_round.responses)


def test_a_seat_answers_a_round_once():
    registry, table, code, token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    session = table.session
    # Two people at the table, so the first answer leaves the round open.
    second = next(s for s in range(4) if s not in (human, bot))
    session.confirm_mode(second)
    session.game._state.hands[second][Resource.WOOD] = 1
    session.begin_round()
    registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                    {"actor": bot, "received": WOOD_FOR_ORE, "kind": "pass"}, token)

    with pytest.raises(ApiError, match="not waiting on your answer"):
        registry.handle("POST", f"/api/games/{code}/trade/round/answer",
                        {"actor": bot, "received": WOOD_FOR_ORE, "kind": "accept"}, token)


def test_undo_is_refused_while_the_seats_own_round_is_open():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    state = table.session.game._state
    state.hands[human][Resource.BRICK] = 4
    state.bank[Resource.BRICK] -= 4
    bank_trade = next(a for a in registry.handle("GET", "/api/state", {}, token)["legal_actions"]
                      if a["type"] == "BANK_TRADE" and a["a"] == Resource.BRICK)
    registry.handle("POST", "/api/action", {"action": bank_trade}, token)
    assert registry.handle("GET", "/api/state", {}, token)["can_undo"] is True

    view = registry.handle("POST", f"/api/games/{code}/trade/round",
                           {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)

    assert view["can_undo"] is False
    with pytest.raises(ApiError, match="open trade round"):
        registry.handle("POST", "/api/undo", {}, token)
    registry.handle("POST", f"/api/games/{code}/trade/round/choose", {"decline": True}, token)
    assert registry.handle("GET", "/api/state", {}, token)["can_undo"] is True


def _owe_road_building(table) -> None:
    """Leave the actor two Road Building roads to place, with somewhere to
    place them."""
    game = table.session.game
    game._state.edge_owner[0] = game.current_player
    game.free_roads = 2


def test_no_round_opens_while_road_building_roads_are_owed():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    _owe_road_building(table)

    with pytest.raises(ApiError, match="Road Building") as refused:
        registry.handle("POST", f"/api/games/{code}/trade/round",
                        {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    assert refused.value.status == 409
    assert table.session.open_round is None


def test_no_answer_is_picked_while_road_building_roads_are_owed():
    registry, table, code, token, human, bot = _table(actor_is_human=True, other_gate=_Wants(Resource.WOOD))
    registry.handle("POST", f"/api/games/{code}/trade/round",
                    {"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 1]}, token)
    _owe_road_building(table)

    with pytest.raises(ApiError, match="Road Building"):
        registry.handle("POST", f"/api/games/{code}/trade/round/choose",
                        {"seat": bot, "bundle": WOOD_FOR_ORE}, token)
    assert table.session.game.trades == []


def test_a_bot_does_not_open_a_round_while_road_building_roads_are_owed():
    _registry, table, _code, _token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    _owe_road_building(table)

    table.session.begin_round()

    assert table.session.open_round is None


def test_a_bot_offers_no_more_than_its_declared_budget():
    _registry, table, _code, _token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    session = table.session
    session.set_trader(bot, _Budgeted(Resource.ORE, max_offers=0))

    session.begin_round()

    assert session.open_round is None, "a budget of no offers makes none"


def test_a_bot_with_trade_now_opens_its_round_when_it_says_so():
    _registry, table, _code, _token, human, bot = _table(actor_is_human=False, other_gate=_Wants(Resource.ORE))
    session = table.session
    gate = _WaitsForItsMoment(Resource.ORE)
    session.set_trader(bot, gate)

    session.begin_round()
    assert session.open_round is None, "not yet: the window stays open"
    gate.ready = True
    session.begin_round(entering=False)  # after one of its main-phase actions
    assert session.open_round is not None
    assert table.view(human)["pending"] == [{"actor": bot, "bundle": WOOD_FOR_ORE}]
