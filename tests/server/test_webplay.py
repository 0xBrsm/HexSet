from __future__ import annotations

import json

import random

from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions

from hexset.board.board import random_base_board

from hexset.cards import DevCard

from conftest import RandomBot

from hexset.game import Phase, is_over, may_act, start, to_move

from hexset.server._journal import open_journal, replayable

from hexset.trading import RESPONSE_ACCEPT, RESPONSE_PASS, Trade

from hexset.server._webplay import GameSession
from hexset.server.wire import (
    RESOURCE_NAMES,
    action_to_wire,
    round_bundle_from_wire,
    wire_to_action,
)


class _Wants:
    """Prices a candidate positively iff it hands this seat more of `resource`."""

    trade_floor = 0.0

    def __init__(self, resource: int):
        self.resource = resource

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng, first=0)


def a_session(game, claimed, **kwargs) -> GameSession:
    return GameSession(game=game, claimed_seats=set(claimed), **kwargs)


def test_wire_round_trips_across_a_played_out_game():
    game = a_game(seed=11)
    rng = random.Random(99)
    steps = 0
    while not is_over(game) and steps < 500:
        options = legal_actions(game)
        for action in options:
            assert wire_to_action(action_to_wire(action)) == action
        apply(game, rng.choice(options))
        steps += 1
    assert steps > 50


def test_wire_to_action_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        wire_to_action({"type": "TELEPORT", "a": 0, "b": 0})


def test_session_rejects_an_action_not_currently_legal():
    game = a_game(seed=2)
    seat = to_move(game)
    session = a_session(game, {seat})

    forged = action_to_wire(Action(ActionType.ROLL))
    with pytest.raises(ValueError):
        session.submit(seat, forged)
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert all(owner == -1 for owner in game._state.vertex_owner)


def test_session_rejects_an_action_from_a_seat_that_has_not_claimed_it():
    game = a_game(seed=4)
    mover = to_move(game)
    other = (mover + 1) % game._state.num_players
    session = a_session(game, {other})

    legal_for_mover = action_to_wire(legal_actions(game)[0])
    with pytest.raises(ValueError):
        session.submit(other, legal_for_mover)


def test_legal_wire_actions_never_depend_on_an_opponents_hand():
    game = a_game(seed=19)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game._state
    session = a_session(game, {0})
    before = session.legal_wire_actions(0)

    for seat in range(1, state.num_players):
        for r in range(len(state.hands[seat])):
            state.hands[seat][r] = 0

    assert session.legal_wire_actions(0) == before


def test_playing_a_knight_is_undoable_until_the_robber_actually_moves():
    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    session = a_session(game, {0})

    knight = next(a for a in legal_actions(game) if a.type is ActionType.PLAY_KNIGHT)
    session.submit(0, action_to_wire(knight))
    assert game.phase is Phase.ROBBER
    assert session.state_view(0)["can_undo"] is True

    session.undo_last_build(0)

    assert game.phase is Phase.MAIN
    assert game._state.dev_cards[0][DevCard.KNIGHT] == 1
    # `dev_card_played` lives on `Game`, not `GameState`, so the `set_state`
    # swap cannot reach it: checked directly rather than inferred.
    assert game.dev_card_played is False
    assert session.state_view(0)["can_undo"] is False

    knight_again = next(a for a in legal_actions(game) if a.type is ActionType.PLAY_KNIGHT)
    session.submit(0, action_to_wire(knight_again))
    assert game.phase is Phase.ROBBER

    move = next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER)
    session.submit(0, action_to_wire(move))
    assert game.phase is Phase.MAIN
    assert session.state_view(0)["can_undo"] is False


def _discard_all(session: GameSession, seat: int) -> None:
    while (
        session.game.phase is Phase.DISCARD
        and to_move(session.game) == seat
    ):
        action = next(a for a in legal_actions(session.game) if a.type is ActionType.DISCARD)
        session.apply_action(seat, action)


def _owing_game(seed: int, seat: int, hand: list[int]) -> GameSession:
    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = seat
    game._state.hands[seat] = list(hand)
    game.discard_quota = [0] * game._state.num_players
    game.discard_quota[seat] = sum(hand) // 2
    return game


def test_a_spectators_log_redacts_nothing_a_seats_log_would():
    game = _owing_game(seed=22, seat=1, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0, 1})

    _discard_all(session, 1)

    theirs = session.log_for(1)[0]
    across = session.log_for(0)[0]
    watching = session.log_for(None, omniscient=True)[0]

    assert any(r in theirs for r in RESOURCE_NAMES)
    assert "discarded 4 cards" in across
    assert not any(r in across for r in RESOURCE_NAMES)
    assert watching == theirs


def test_state_view_reveals_the_log_once_the_game_is_over():
    game = _owing_game(seed=22, seat=1, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0, 1})

    _discard_all(session, 1)
    game.won_by = 0
    game.phase = Phase.GAME_OVER

    across = session.state_view(0)["log"][0]
    assert any(r in across for r in RESOURCE_NAMES)


def test_state_view_carries_the_public_ledger_for_every_seat():
    """Resource *counting* is public knowledge -- only a steal's identity and
    dev-card types are hidden -- so every seat carries `known`/`unknown`.
    """
    game = a_game(seed=8)
    seat = to_move(game)
    other = (seat + 1) % game._state.num_players
    session = a_session(game, {seat})

    game.ledger.receive(other, 0, 2)

    view = session.state_view(seat)
    players = {p["seat"]: p for p in view["players"]}
    assert players[other]["known"]["Wood"] == 2
    assert players[other]["unknown"] == 0
    assert "hand" not in players[other]


SEED = 42


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    directory = tmp_path_factory.mktemp("games")
    # Two independent `random.Random(SEED)` instances, as `api.build_session`
    # does: the board spends one stream and `start` gets a fresh one.
    board = random_base_board(random.Random(SEED))
    game = start(board, 4, random.Random(SEED), first=0)
    session = GameSession(
        game=game,
        claimed_seats={0, 1, 2, 3},
        seed=SEED,
        journal=open_journal(SEED, str(directory)),
    )

    driver = RandomBot(rng=random.Random(2))
    steps = 0
    while not is_over(session.game) and steps < 4000:
        seat = to_move(session.game)
        action = driver.choose(session.game)
        session.submit(seat, action_to_wire(action))
        steps += 1
    assert is_over(session.game)
    return session, directory


def journal_events(directory) -> list[dict]:
    """The one per-game journal in `directory`, parsed."""
    files = list(Path(directory).glob("*.jsonl"))
    assert len(files) == 1, f"expected one game journal, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_a_journalled_game_replays_clean(played):
    """Through `replayable` and `restore`, the two calls `api.reopen_session`
    makes, rather than a replay written for the test.
    """
    session, directory = played
    events = journal_events(directory)
    header = events[0]
    assert header["seed"] == SEED
    assert header["first"] == 0

    board = random_base_board(random.Random(SEED))
    resumed = GameSession(
        game=start(board, header["num_players"], random.Random(SEED), first=header["first"]),
        claimed_seats=set(header["human_seats"]),
        seed=SEED,
    )
    resumed.restore(replayable(events))

    assert resumed.game.won_by == session.game.won_by
    assert resumed.game.turns == session.game.turns


def _setup_complete_game(seed: int, players: int = 4):
    board = random_base_board(random.Random(seed))
    game = start(board, players, random.Random(seed), first=0)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, next(iter(legal_actions(game))))
    return game


def test_a_pre_split_journalled_knight_play_still_replays():
    """An older journal records a played Knight as one step, with the target hex
    and victim as its `a`/`b`, where this engine resolves it in two.
    """
    scout = _setup_complete_game(seed=3)
    scout.phase = Phase.MAIN
    scout.current_player = 0
    scout._state.dev_cards[0][DevCard.KNIGHT] = 1
    apply(scout, Action(ActionType.PLAY_KNIGHT), seat=0)
    move = next(
        a for a in legal_actions(scout) if a.type is ActionType.MOVE_ROBBER and a.b < 4
    )
    target, victim = move.a, move.b

    game = _setup_complete_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    game._state.hands[victim] = [1, 1, 1, 1, 1]  # something for the knight to take
    session = GameSession(game=game, claimed_seats={0, victim})

    events = [{"kind": "action", "actor": 0, "type": "PLAY_KNIGHT", "a": target, "b": victim}]
    session.restore(replayable(events))

    assert game._state.robber == target
    assert game.phase is Phase.MAIN
    assert sum(game._state.hands[victim]) == 4
    # The file recorded one step, not the two this engine needed, so
    # `_apply_knight` must fold the second `apply_action`'s count back out or every
    # later step replays against the wrong `undo.back_to`/`note.step`.
    assert session.steps == 1


def test_an_undone_placement_is_written_down_not_erased(tmp_path):
    game = a_game(seed=5)
    seat = to_move(game)
    session = a_session(game, {seat}, journal=open_journal(5, str(tmp_path)))
    settlement = next(
        a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT
    )
    session.submit(seat, action_to_wire(settlement))
    session.undo_last_build(seat)

    events = journal_events(tmp_path)
    assert [e["kind"] for e in events] == ["game", "action", "undo"]
    assert events[1]["type"] == "SETUP_SETTLEMENT"
    assert events[2]["back_to"] == 0


class _Planner:
    """A gate carrying a multi-offer plan -- the served seam's view of the
    fragmented policy: `propose` is honoured, `trade_round_finished` is told
    how each round ended, and `trade_offer_budget` bounds the turn."""

    trade_floor = 0.0

    def __init__(self, plan, budget=2):
        self.plan = list(plan)
        self.trade_offer_budget = budget
        self.attempt = 0
        self.finished = []

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        if turn is None or self.attempt >= len(self.plan):
            return []
        return [(counterparties[0], self.plan[self.attempt])]

    def offer(self, view, candidates):
        return 0 if candidates else None

    def trade_round_finished(self, view, offer, responses, trade, *, turn):
        self.finished.append((tuple(offer.received), trade))
        self.attempt += 1

    def allow_repeated_offer(self, view, bundle, *, turn):
        return False


def _planning_table(plan, budget=2):
    """Seat 0 (a `_Planner`) holds one wood and one brick; seat 1 holds two
    ore and wants wood. The session drives trading (`trade_mode="external"`),
    as the served table does."""
    from hexset.board.terrain import Resource
    from hexset.game import roll_dice

    game = a_game(seed=13)
    game.phase = Phase.ROLL
    game.current_player = 0
    game.trade_mode = "external"
    for hand in game._state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[0][Resource.BRICK] = 1
    game._state.hands[1][Resource.ORE] = 2
    session = a_session(game, set())
    planner = _Planner(plan, budget)
    session.set_trader(0, planner)
    session.set_trader(1, _Wants(Resource.WOOD))
    roll_dice(game, 8)
    assert game.phase is Phase.MAIN and not game.trades
    return game, session, planner


def test_a_served_bot_actor_offers_its_whole_plan_and_is_told_how_each_round_ended():
    from hexset.board.terrain import Resource

    wood_for_ore = [0] * 5
    wood_for_ore[Resource.WOOD], wood_for_ore[Resource.ORE] = -1, 1
    brick_for_ore = [0] * 5
    brick_for_ore[Resource.BRICK], brick_for_ore[Resource.ORE] = -1, 1
    game, session, planner = _planning_table([tuple(wood_for_ore), tuple(brick_for_ore)])

    session.begin_round()

    offers = [tuple(e.note.bundle) for e in session.events if e.note is not None and e.note.kind == "offer"]
    assert offers == [tuple(wood_for_ore), tuple(brick_for_ore)]
    # seat 1 wanted the wood and not the brick: one exchange, and the planner
    # learned the outcome of both rounds in order
    assert [t.received for t in game.trades] == [tuple(wood_for_ore)]
    assert [received for received, _ in planner.finished] == offers
    assert planner.finished[0][1] is not None and planner.finished[1][1] is None
    assert session.open_round is None


def test_round_bundle_from_wire_refuses_a_shared_resource():
    with pytest.raises(ValueError):
        round_bundle_from_wire([1, 0, 0, 0, 0], [1, 0, 0, 0, 0])


def test_round_bundle_from_wire_caps_nothing_but_needs_a_card_a_side():
    """Nothing on the wire caps a trade's size: what a seat will move is that
    seat's own, and a manual seat's submission is its own consent."""
    wide = round_bundle_from_wire([1, 1, 1, 1, 0], [0, 0, 0, 0, 5])
    assert sum(max(-n, 0) for n in wide) == 4 and sum(max(n, 0) for n in wide) == 5
    with pytest.raises(ValueError, match="give must be at least one card"):
        round_bundle_from_wire([0, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="want must be at least one card"):
        round_bundle_from_wire([1, 0, 0, 0, 0], [0, 0, 0, 0, 0])


def test_the_trade_round_is_in_the_log_and_a_pass_is_an_answer():
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    game._state.hands[2][Resource.SHEEP] = 1
    session = a_session(game, {0}, player_names={0: "Ada"})
    session.confirm_mode(0)
    session.set_trader(1, _Wants(Resource.WOOD))  # accepts: it gets wood
    session.set_trader(2, _Wants(Resource.BRICK))
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1
    received = tuple(received)

    session.open_round_for(0, received)

    view = session.state_view(0)["trade_round"]
    assert view["offer"] == {"actor": 0, "bundle": list(received)}
    assert {r["seat"]: r["kind"] for r in view["responses"]} == {1: RESPONSE_ACCEPT, 2: RESPONSE_PASS}
    assert next(r for r in view["responses"] if r["seat"] == 2)["bundle"] is None
    line = session.log_for(None)[-1]
    assert line.endswith("offers 1 Wood for 1 Ore. Player 2 (bot) accepts.")
    assert "passes" not in line and len(session.events) == 3

    session.decline_round(0)
    assert session.log_for(None)[-1].endswith("offers 1 Wood for 1 Ore. Player 2 (bot) accepts. Player 1 (Ada) declines.")
    assert session.open_round is None

    session.open_round_for(0, received)
    session.execute_round_choice(0, 1, received)
    line = session.log_for(None)[-1]
    assert line.endswith("offers 1 Wood for 1 Ore. Player 2 (bot) accepts. Traded with Player 2 (bot).")
    assert game._state.hands[0][Resource.ORE] == 1 and game._state.hands[1][Resource.WOOD] == 1


def test_a_manual_seat_with_no_cards_passes_at_once():
    from hexset.board.terrain import NUM_RESOURCES, Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    for seat in range(4):
        game._state.hands[seat] = [0] * NUM_RESOURCES
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    session = a_session(game, {0, 1, 2, 3}, player_names={0: "Ada"})
    for seat in range(4):
        session.confirm_mode(seat)
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    session.open_round_for(0, tuple(received))

    round_ = session.open_round
    assert round_.awaiting == {1}, "only the seat that holds cards is asked"
    assert {r.seat for r in round_.responses if r.kind == RESPONSE_PASS} == {2, 3}
    assert [t.b for t in game.pending] == [1]
    assert "Everyone declines" not in session.log_for(None)[-1]


def test_state_view_pending_is_filtered_per_viewer():
    game = a_game(seed=3)
    game.pending.append(Trade(0, 1, (1, 0, 0, 0, -1)))
    session = a_session(game, {0, 1})
    assert session.state_view(1)["pending"] == [{"actor": 0, "bundle": [1, 0, 0, 0, -1]}]
    assert session.state_view(0)["pending"] == []
    assert session.state_view(None)["pending"] == []


def test_a_manually_executed_trade_appears_in_the_log():
    from hexset.board.terrain import Resource
    from hexset.game import Phase

    game = a_game(seed=3)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    session = a_session(game, {0, 1})
    session.confirm_mode(0)
    session.set_trader(1, _Wants(Resource.WOOD))
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    trade = session._execute_round_trade(0, 1, tuple(received))

    assert trade.received == tuple(received)
    assert game._state.hands[0][Resource.ORE] == 1
    assert game._state.hands[1][Resource.WOOD] == 1
    line = next(line for line in session.log_for(None) if " to Player " in line)
    assert "traded" in line and "Wood" in line and "Ore" in line
    assert session.state_view(0)["can_undo"] is False


def test_a_manually_executed_trade_survives_journal_and_resume(tmp_path):
    """Without its own action-less journal step a restart would rebuild hands
    from recorded actions alone and forget the cards this moved.
    """
    from hexset.board.terrain import Resource
    from hexset.game import Phase
    from hexset.server._journal import replayable

    game = a_game(seed=3)
    session = a_session(game, {0, 1}, journal=open_journal(3, str(tmp_path)))
    session.confirm_mode(0)
    session.set_trader(1, _Wants(Resource.WOOD))
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0][Resource.WOOD] = 1
    game._state.hands[1][Resource.ORE] = 1
    received = [0, 0, 0, 0, 0]
    received[Resource.ORE] = 1
    received[Resource.WOOD] = -1

    session._execute_round_trade(0, 1, tuple(received))

    events = journal_events(tmp_path)
    assert any(e.get("kind") == "trade" for e in events)

    resumed_game = a_game(seed=3)
    resumed = a_session(resumed_game, {0, 1})
    resumed_game.phase = Phase.MAIN
    resumed_game.current_player = 0
    resumed_game._state.hands[0][Resource.WOOD] = 1
    resumed_game._state.hands[1][Resource.ORE] = 1

    resumed.restore(replayable(events))

    assert resumed_game._state.hands[0][Resource.ORE] == 1
    assert resumed_game._state.hands[0][Resource.WOOD] == 0
    assert resumed_game._state.hands[1][Resource.WOOD] == 1
    assert resumed_game._state.hands[1][Resource.ORE] == 0
    assert resumed.steps == session.steps == 1
    line = next(line for line in resumed.log_for(None) if " to Player " in line)
    assert "traded" in line


# --- a seven's discards are simultaneous, and the log says so ------------------


def _two_owing(seed: int = 11) -> GameSession:
    from hexset.board.terrain import NUM_RESOURCES, Resource

    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = 1
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[0][Resource.WOOD] = 4
    game._state.hands[3][Resource.ORE] = 4
    game.discard_quota = [2, 0, 0, 2]
    return a_session(game, {0, 1, 2, 3})


def _discards(session: GameSession, seat: int | None, **kwargs) -> list[str]:
    return [line for line in session.log_for(seat, **kwargs) if "discard" in line]


def _discard_wire(resource) -> dict:
    return {"type": "DISCARD", "a": int(resource)}


def test_no_discard_is_logged_until_the_whole_round_has_resolved():
    from hexset.board.terrain import Resource

    session = _two_owing()

    session.submit(3, _discard_wire(Resource.ORE))
    session.submit(3, _discard_wire(Resource.ORE))
    assert session.game.discard_quota == [2, 0, 0, 0]
    assert _discards(session, 3) == []
    assert _discards(session, 0) == []
    assert _discards(session, None) == []
    assert _discards(session, None, omniscient=True) == []

    session.submit(0, _discard_wire(Resource.WOOD))
    assert _discards(session, 0) == []

    session.submit(0, _discard_wire(Resource.WOOD))

    lines = _discards(session, None, omniscient=True)
    assert len(lines) == 2
    assert "Player 1 " in lines[0] and "Player 4 " in lines[1]


def test_the_revealed_round_is_still_redacted_per_reader():
    from hexset.board.terrain import Resource

    session = _two_owing()
    for seat, resource in ((3, Resource.ORE), (0, Resource.WOOD)) * 2:
        session.submit(seat, _discard_wire(resource))

    mine = next(line for line in _discards(session, 3) if "Player 4" in line)
    across = next(line for line in _discards(session, 0) if "Player 4" in line)

    assert "Ore" in mine
    assert "discarded 2 cards" in across and not any(r in across for r in RESOURCE_NAMES)


def test_a_round_closed_by_a_locked_seat_still_reveals_the_rest():
    from hexset.board.terrain import Resource
    from hexset.game import lock_seat

    session = _two_owing()
    session.submit(3, _discard_wire(Resource.ORE))
    session.submit(3, _discard_wire(Resource.ORE))
    assert _discards(session, None, omniscient=True) == []

    lock_seat(session.game, 0)

    lines = _discards(session, None, omniscient=True)
    assert len(lines) == 1 and "2 Ore" in lines[0]


def _finish_setup(game) -> None:
    """Reaches `Phase.MAIN`, where `GameSession.round` counts laps."""
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        action = next(
            a
            for a in legal_actions(game)
            if a.type in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD)
        )
        apply(game, action)


def _take_turn(game) -> None:
    from hexset.game import end_turn, roll_dice

    roll_dice(game, roll=8)
    end_turn(game)


def test_a_lap_is_two_turns_when_two_seats_are_retired():
    from hexset.game import lock_seat

    game = a_game(seed=3)
    lock_seat(game, 2)
    lock_seat(game, 3)
    _finish_setup(game)
    session = a_session(game, {0})

    assert session.round == 1
    _take_turn(game)
    assert session.round == 1
    _take_turn(game)
    assert session.round == 2  # four dealt seats, only two playing

    for _ in range(8):
        _take_turn(game)
    assert session.round == 6
    assert session.game.turns == 10


# --- The setup turn's own end (GameSession.awaiting_confirm) ----------------


def _manual_session(seed=5):
    """`clients` carries the `kind` `api.Tables` records at seat-up, and the
    setup hold is scoped to `"web"` alone.
    """
    session = a_session(a_game(seed=seed), {0, 1, 2, 3})
    for seat in range(4):
        session.confirm_mode(seat)
        session.clients[seat] = {"id": None, "kind": "web"}
    return session


def _setup_turn(session, seat):
    for kind in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD):
        session.apply_action(seat, next(a for a in legal_actions(session.game) if a.type is kind))


def test_the_next_seat_cannot_move_until_the_setup_turn_is_ended():
    session = _manual_session()
    _setup_turn(session, 0)
    nxt = to_move(session.game)
    assert nxt != 0
    settlement = next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    with pytest.raises(ValueError, match="has not finished its setup turn"):
        session.submit(nxt, action_to_wire(settlement))

    session.submit(0, action_to_wire(Action(ActionType.END_TURN)))
    assert session.awaiting_confirm is None
    session.submit(nxt, action_to_wire(settlement))


def test_a_setup_placement_is_still_undoable_while_the_turn_is_held():
    session = _manual_session()
    _setup_turn(session, 0)
    assert session.state_view(0)["can_undo"] is True
    session.undo_last_build(0)
    assert session.awaiting_confirm is None
    assert session.state_view(0)["awaiting_confirm"] is None
    assert to_move(session.game) == 0


def test_the_held_seat_is_offered_an_end_turn_it_would_not_otherwise_have():
    session = _manual_session()
    _setup_turn(session, 0)
    assert not may_act(session.game, 0)
    assert session.state_view(0)["legal_actions"] == [action_to_wire(Action(ActionType.END_TURN))]
    with pytest.raises(ValueError, match="end your setup turn first"):
        session.submit(0, action_to_wire(Action(ActionType.SETUP_SETTLEMENT, 0, 0)))


def test_an_llm_seat_is_never_held_through_setup():
    session = a_session(a_game(seed=5), {0, 1, 2, 3})
    for seat in range(4):
        session.confirm_mode(seat)
        session.clients[seat] = {"id": None, "kind": "mcp"}
    _setup_turn(session, 0)
    assert session.awaiting_confirm is None


def _round_table(seed: int = 13, *, certify_ore: bool = True):
    """A bot-only table one wood-for-ore exchange away, the record certifying
    both cards (or seat 1's ore held untyped), and the same game copied for
    the engine's loop."""
    from hexset.board.terrain import Resource
    from hexset.game import imagine
    from hexset.ledger import SeatLedger

    game = a_game(seed=seed)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game._state
    for seat, hand in enumerate(state.hands):
        for r, n in enumerate(hand):  # every card back in the bank, as a table keeps it
            state.bank[r] += n
        hand[:] = [0, 0, 0, 0, 0]
        game.ledger.seats[seat] = SeatLedger()
    for seat, r in ((0, Resource.WOOD), (1, Resource.ORE)):
        state.bank[r] -= 1
        state.hands[seat][r] = 1
    game.ledger.receive(0, int(Resource.WOOD), 1)
    if certify_ore:
        game.ledger.receive(1, int(Resource.ORE), 1)
    else:
        game.ledger.gain_unknown(1, 1)
    gates = (_Wants(Resource.ORE), _Wants(Resource.WOOD), _Wants(Resource.SHEEP), _Wants(Resource.SHEEP))
    return game, imagine(game, random.Random(0), randomize_deck=False), gates


def test_the_served_round_is_the_engines_round():
    """A bot-only table's `begin_round` opens, answers and resolves through
    the same stages `hexset.trading.trade_round` runs, so the two tables
    reach the same offer and the same exchange from the same position."""
    from hexset.trading import trade_round

    game, mirror, gates = _round_table()
    session = a_session(game, set())
    for seat, gate in enumerate(gates):
        session.set_trader(seat, gate)

    session.begin_round()
    engine = trade_round(mirror, gates)

    assert len(game.trades) == 1 and game.trades == engine
    assert game._state.hands == mirror._state.hands
    assert session.open_round is None


def test_the_served_round_publishes_the_trade_event_to_observers():
    """What the engine's loop tells `observe_trade` after its event, the
    served table tells after its round closes: same fields, same facts."""
    from hexset.game import run_trade_event

    class Watching(_Wants):
        def __init__(self, resource):
            super().__init__(resource)
            self.seen = []

        def observe_trade(self, **event):
            self.seen.append(event)

    game, mirror, _ = _round_table()
    served = (Watching(0 + 4), Watching(0), Watching(2), Watching(2))  # 4: ore
    engine = (Watching(0 + 4), Watching(0), Watching(2), Watching(2))
    session = a_session(game, set())
    for seat, gate in enumerate(served):
        session.set_trader(seat, gate)
    session.begin_round()

    mirror.gates = engine
    mirror.trade_mode = "round"
    mirror.trade_event_turn = -1
    run_trade_event(mirror)

    assert served[2].seen and served[2].seen == engine[2].seen
    event = served[2].seen[0]
    assert event["actor"] == 0 and event["trade_participants"] == ((0, 1),)
    assert event["hand_sizes"] == (1, 1, 0, 0)


def test_a_trade_filed_after_a_bank_trade_or_a_build_is_in_the_log():
    """A journalled record files an exchange after whichever action it
    followed. The bank and build lines are rewritten in place as a run grows,
    so the trade has to be written after the line and end the run."""
    from hexset.actions import Action, ActionType
    from hexset.server._webplay import _Event, _Snapshot, render_log
    from hexset.trading import Trade

    def snap(hand):
        return _Snapshot(hands=[hand, [0, 0, 0, 0, 0]], held=[[0] * 5, [0] * 5])

    trade = Trade(0, 1, (1, 0, 0, -1, 0))
    bank = Action(ActionType.BANK_TRADE, 2, 4)
    events = [
        _Event(3, 0, bank, snap([0, 0, 8, 1, 0]), snap([0, 0, 4, 1, 1]), None, (trade,)),
        _Event(3, 0, bank, snap([0, 0, 4, 1, 1]), snap([0, 0, 0, 1, 2]), None),
        _Event(3, 0, Action(ActionType.BUILD_ROAD, 0), snap([1, 1, 0, 0, 2]),
               snap([0, 0, 0, 0, 2]), None, (trade,)),
    ]
    lines = [line.split("\t", 1)[1] for line in render_log(events, None, {0: "A", 1: "B"}, None)]

    assert len([line for line in lines if "to Player 2 (B)" in line]) == 2
    assert [line for line in lines if "with the bank" in line] == [
        "Player 1 (A) traded 4 Sheep for 1 Ore with the bank.",
        "Player 1 (A) traded 4 Sheep for 1 Ore with the bank.",
    ]
