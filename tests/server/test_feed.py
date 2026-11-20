"""A view's `feed`, each action as its own entry hidden as the log is, with
what it changed on each seat's row (`after`).
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.game import Phase, is_over, start, to_move
from hexset.server._runner import BotRunner
from hexset.server._webplay import GameSession, _Event, _feed_entry, _Snapshot, render_feed
from hexset.server.api import ApiError
from hexset.server.wire import RESOURCE_NAMES, action_to_wire
from hexset.trading import Trade


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng, first=0)


def a_session(game, claimed) -> GameSession:
    return GameSession(game=game, claimed_seats=set(claimed))


def _snapshot(hands, points=None):
    seats = len(hands)
    return _Snapshot(
        hands=[h[:] for h in hands], held=[[0] * 5 for _ in hands],
        points=list(points or [0] * seats), public_points=list(points or [0] * seats),
        knights=[0] * seats, roads=[0] * seats, longest_road=-1, largest_army=-1)


# --- the feed ---------------------------------------------------------------


def test_the_feed_has_one_entry_per_action_numbered_by_its_place_in_the_game():
    game = a_game(seed=3)
    session = a_session(game, range(4))
    rng = random.Random(3)
    for _ in range(300):
        if is_over(game):
            break
        seat = to_move(game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(game))))
    feed = session.state_view(0)["feed"]
    numbers = [entry["n"] for entry in feed["events"]]
    assert feed["epoch"] == 0
    assert numbers == list(range(len(session.events) - len(numbers), len(session.events)))
    assert all("kind" in entry and "actor" in entry for entry in feed["events"])
    kinds = {entry["kind"] for entry in feed["events"]}
    assert {"roll", "build", "end"} <= kinds


def test_a_bought_card_is_named_only_to_its_buyer_and_an_omniscient_reader():
    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0] = [0, 0, 1, 1, 1]
    session = a_session(game, {0})
    buy = next(a for a in legal_actions(game) if a.type is ActionType.BUY_DEV_CARD)
    session.submit(0, action_to_wire(buy))

    mine = session.state_view(0)["feed"]["events"][-1]
    theirs = session.state_view(1)["feed"]["events"][-1]
    watching = session.state_view(None, omniscient=True)["feed"]["events"][-1]
    assert mine["kind"] == theirs["kind"] == "buy"
    assert mine["card"] is not None and mine["card"] == watching["card"]
    assert theirs["card"] is None


def test_a_stolen_card_is_named_only_to_the_thief_and_the_victim():
    before = [[0] * 5, [0, 0, 0, 2, 0], [0] * 5, [0] * 5]
    after = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0] * 5, [0] * 5]
    event = _Event(round_num=3, actor=0, action=Action(ActionType.MOVE_ROBBER, 5, 1),
                   before=_snapshot(before), after=_snapshot(after), last_roll=7)
    seen = {viewer: _feed_entry(event, 9, viewer, False) for viewer in (0, 1, 2, None)}
    assert seen[0]["card"] == seen[1]["card"] == RESOURCE_NAMES[3]
    assert seen[2]["card"] is None and seen[None]["card"] is None
    assert all(e["hex"] == 5 and e["victim"] == 1 and e["stole"] for e in seen.values())
    assert _feed_entry(event, 9, None, True)["card"] == RESOURCE_NAMES[3]


def test_a_roll_carries_every_seats_production_to_every_reader():
    before = [[0] * 5, [1, 0, 0, 0, 0], [0] * 5, [0] * 5]
    after = [[0, 2, 0, 0, 0], [1, 0, 0, 0, 1], [0] * 5, [0] * 5]
    event = _Event(round_num=1, actor=2, action=Action(ActionType.ROLL),
                   before=_snapshot(before), after=_snapshot(after), last_roll=6)
    entry = _feed_entry(event, 0, 3, False)
    assert entry["kind"] == "roll" and entry["value"] == 6
    assert entry["gains"] == {"0": [0, 2, 0, 0, 0], "1": [0, 0, 0, 0, 1]}


def _hand(*counts):
    return dict(zip(RESOURCE_NAMES, counts))


def test_an_entry_says_what_it_changed_on_each_row_and_hides_cards_as_the_row_does():
    before = [[0] * 5, [0, 0, 0, 2, 0], [0] * 5, [0] * 5]
    after = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0] * 5, [0] * 5]
    event = _Event(round_num=3, actor=0, action=Action(ActionType.MOVE_ROBBER, 5, 1),
                   before=_snapshot(before), after=_snapshot(after), last_roll=7)
    seen = {viewer: _feed_entry(event, 9, viewer, False)["after"] for viewer in (0, 1, 2, None)}
    # Every reader is told both hands' sizes; a seat is told its own cards too.
    assert seen[2] == seen[None] == {"0": {"hand_size": 1}, "1": {"hand_size": 1}}
    assert seen[0] == {"0": {"hand_size": 1, "hand": _hand(0, 0, 0, 1, 0)}, "1": {"hand_size": 1}}
    assert seen[1] == {"0": {"hand_size": 1}, "1": {"hand_size": 1, "hand": _hand(0, 0, 0, 1, 0)}}
    watching = _feed_entry(event, 9, None, True)["after"]
    assert watching["0"]["hand"] == watching["1"]["hand"] == _hand(0, 0, 0, 1, 0)


def test_an_entry_that_changed_no_row_carries_no_after():
    hands = [[1, 0, 0, 0, 0], [0] * 5]
    event = _Event(round_num=2, actor=0, action=Action(ActionType.END_TURN),
                   before=_snapshot(hands), after=_snapshot(hands), last_roll=5)
    assert "after" not in _feed_entry(event, 4, 0, False)
    assert "bank" not in _feed_entry(event, 4, 0, False)


def test_a_build_changes_its_seats_points_as_each_reader_sees_them():
    before = [[1, 1, 1, 1, 0], [0] * 5]
    after = [[0] * 5, [0] * 5]
    event = _Event(round_num=4, actor=0, action=Action(ActionType.BUILD_SETTLEMENT, 7),
                   before=_snapshot(before, points=[2, 2]), after=_snapshot(after, points=[3, 2]), last_roll=5)
    assert _feed_entry(event, 0, 1, False)["after"] == {"0": {"victory_points": 3, "hand_size": 0}}
    assert _feed_entry(event, 0, 0, False)["after"] == {
        "0": {"victory_points": 3, "hand_size": 0, "hand": _hand(0, 0, 0, 0, 0)}}


def test_each_exchange_cleared_with_an_entry_says_what_it_changed_in_turn():
    # Seat 0 rolls and collects a Wood, then trades it to seat 1 for a Brick,
    # then that Brick to seat 2 for a Sheep.
    before = [[0] * 5, [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    after = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    trades = (Trade(a=0, b=1, received=(-1, 1, 0, 0, 0)), Trade(a=0, b=2, received=(0, -1, 1, 0, 0)))
    event = _Event(round_num=5, actor=0, action=Action(ActionType.ROLL),
                   before=_snapshot(before), after=_snapshot(after), last_roll=6, trades=trades)
    entry = _feed_entry(event, 3, 0, False)
    # The roll alone, then each exchange: the sizes move only where they changed.
    assert entry["after"] == {"0": {"hand_size": 1, "hand": _hand(1, 0, 0, 0, 0)}}
    first, second = entry["trades"]
    assert first["after"] == {"0": {"hand": _hand(0, 1, 0, 0, 0)}}
    assert second["after"] == {"0": {"hand": _hand(0, 0, 1, 0, 0)}}
    assert "after" not in _feed_entry(event, 3, None, False)["trades"][0]


_ROW_KEYS = ("victory_points", "knights_played", "road_length", "longest_road", "largest_army",
             "hand_size", "dev_card_count", "hand", "dev_cards")


def _rows_of(view):
    return {str(p["seat"]): {key: p[key] for key in _ROW_KEYS if key in p} for p in view["players"]}


@pytest.mark.parametrize("viewer, omniscient", [(0, False), (2, False), (None, False), (None, True)])
def test_applying_each_entrys_after_to_the_rows_before_it_gives_the_rows_after_it(viewer, omniscient):
    game = a_game(seed=5)
    session = a_session(game, range(4))
    rng = random.Random(5)
    first = session.state_view(viewer, omniscient=omniscient)
    rows, bank = _rows_of(first), first["bank"]
    taken, steps, banked = -1, 0, 0
    for _ in range(1200):
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
        else:
            session.submit(to_move(game), action_to_wire(rng.choice(legal_actions(game))))
        if is_over(game):
            break
        view = session.state_view(viewer, omniscient=omniscient)
        for entry in view["feed"]["events"]:
            if entry["n"] <= taken:
                continue
            taken = entry["n"]
            if "bank" in entry:
                bank = entry["bank"]
                banked += 1
            for step in [entry, *entry.get("trades", [])]:
                for seat, changed in step.get("after", {}).items():
                    rows[seat].update(changed)
                    steps += 1
        # A discard round still open holds its entries back while the view
        # already shows the cards gone.
        if game.phase is not Phase.DISCARD:
            assert rows == _rows_of(view), (taken, view["phase"])
            assert bank == view["bank"], (taken, view["phase"])
    assert steps > 200 and banked > 50


def test_an_open_discard_round_is_held_back_until_every_seat_has_discarded():
    game = a_game(seed=22)
    game.phase = Phase.DISCARD
    game.current_player = 1
    game._state.hands[1] = [4, 4, 0, 0, 0]
    game.discard_quota = [0, 4, 0, 0]
    session = a_session(game, {0, 1})

    discard = next(a for a in legal_actions(game) if a.type is ActionType.DISCARD)
    session.apply_action(1, discard)
    assert not [e for e in session.state_view(1)["feed"]["events"] if e["kind"] == "discard"]

    while game.phase is Phase.DISCARD:
        session.apply_action(1, next(a for a in legal_actions(game) if a.type is ActionType.DISCARD))
    theirs = [e for e in session.state_view(1)["feed"]["events"] if e["kind"] == "discard"]
    across = [e for e in session.state_view(0)["feed"]["events"] if e["kind"] == "discard"]
    assert len(theirs) == len(across) == 4
    assert all(e["card"] in RESOURCE_NAMES for e in theirs)
    assert all(e["card"] is None for e in across)


def test_an_undo_starts_a_new_epoch_and_drops_the_entry_it_took_back():
    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    session = a_session(game, {0})
    knight = next(a for a in legal_actions(game) if a.type is ActionType.PLAY_KNIGHT)
    session.submit(0, action_to_wire(knight))
    assert session.state_view(0)["feed"]["events"][-1]["kind"] == "play"

    session.undo_last_build(0)

    feed = session.state_view(0)["feed"]
    assert feed["epoch"] == 1
    assert not [e for e in feed["events"] if e["kind"] == "play"]


def test_the_feed_carries_only_the_newest_window_of_entries():
    game = a_game(seed=3)
    session = a_session(game, range(4))
    rng = random.Random(3)
    for _ in range(40):
        session.submit(to_move(game), action_to_wire(rng.choice(legal_actions(game))))
    entries = render_feed(session.events, 0, window=5)
    assert [e["n"] for e in entries] == list(range(len(session.events) - 5, len(session.events)))


# --- the runner --------------------------------------------------------


class _OneMove:
    def __init__(self):
        self.views = iter([{"version": 1, "your_move": "act"}, {"version": 2, "your_move": "game_over"}])
        self.posted = []

    def get(self, path, token):
        return next(self.views)

    def post(self, path, token, body):
        self.posted.append(body)
        return {}


class _Brain:
    def decide(self, transport, token, seat):
        return {"type": "ROLL", "a": 0, "b": 0}


def test_the_runner_sends_the_move_it_decides_and_stops_at_the_end():
    transport = _OneMove()
    runner = BotRunner(seat=0, token="t", transport=transport, brain=_Brain())
    assert runner.run_once() is False
    assert transport.posted == [{"action": {"type": "ROLL", "a": 0, "b": 0}}]


# --- warming the offered bots -------------------------------------------------


def test_warming_spawns_every_offered_bot_once_per_game_type_and_skips_one_that_fails(monkeypatch):
    from hexset.server import api

    spawned = []

    def spawn(name, spec, board, config, players, game_type=None):
        spawned.append((game_type, name))
        if name == "broken":
            raise ApiError("cannot seat broken")
        return object()

    catalogue = {"standard": {"a": None, "broken": None}, "duel-variant": {"a": None}}
    monkeypatch.setattr(api, "model_catalogue", lambda: catalogue)
    monkeypatch.setattr(api, "_seated_spec", lambda name, game_type: name)
    monkeypatch.setattr(api, "seat_spawn", spawn)
    warmed = api.warm_models(api.Config(games_dir=""))
    assert sorted(spawned) == [("duel-variant", "a"), ("standard", "a"), ("standard", "broken")]
    assert sorted(warmed) == ["duel-variant/a", "standard/a"]
