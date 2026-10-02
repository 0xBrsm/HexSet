"""The server's own bot seats: `LocalSearchBrain` deciding on the table's
`Game`, and `BotRunner` meeting the moves and discards `your_move` says the
seat owes.
"""

from __future__ import annotations

import random
import threading

import pytest

from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.bots import RandomBot
from hexset.game import Phase, start, to_move
from hexset.server._runner import BotRunner, LocalSearchBrain
from hexset.server.wire import action_to_wire


def _discard_round():
    """Seat 1 rolled a seven; seats 0 and 2 owe two cards each out of
    disjoint hands, so a discard chosen for the wrong seat names a card this
    seat does not hold."""
    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    game.phase = Phase.DISCARD
    game.current_player = 1
    for seat in range(4):
        game._state.hands[seat] = [0] * NUM_RESOURCES
    game._state.hands[0] = [4, 0, 0, 0, 0]  # WOOD
    game._state.hands[2] = [0, 0, 0, 0, 4]  # ORE
    game.discard_quota = [2, 0, 2, 0]
    return game


def test_an_embedded_seat_discards_for_itself_while_a_lower_seat_owes_too():
    game = _discard_round()
    assert to_move(game) == 0
    brain = LocalSearchBrain(RandomBot(random.Random(0)), game)
    wire = brain.decide(transport=None, token="", seat=2)
    assert wire == action_to_wire(Action(ActionType.DISCARD, Resource.ORE))
    assert game.discard_quota == [2, 0, 2, 0], "deciding is not acting"


def test_an_embedded_seat_decides_holding_the_table_s_lock():
    game = _discard_round()
    lock = threading.Lock()
    held = []

    class Watching(RandomBot):
        def choose(self, game):
            held.append(lock.locked())
            return super().choose(game)

    brain = LocalSearchBrain(Watching(random.Random(0)), game, lock=lock)
    brain.decide(transport=None, token="", seat=0)
    assert held == [True] and not lock.locked()


def test_the_runner_acts_on_a_discard_it_owes_while_another_seat_is_to_move():
    views = iter([
        {"version": 1, "to_move": 0, "your_move": "discard"},
        {"version": 2, "to_move": 0, "your_move": "wait"},
        {"version": 3, "to_move": 0, "your_move": "wait"},
    ])
    posted = []

    class Transport:
        def get(self, path, token):
            return next(views)

        def post(self, path, token, body):
            posted.append(body)
            return {}

    class Brain:
        def decide(self, transport, token, seat):
            return {"type": "DISCARD", "seat": seat}

    runner = BotRunner(seat=2, token="t", transport=Transport(), brain=Brain(), poll_interval=0.0)
    assert runner.run_once() is True
    assert posted == [{"action": {"type": "DISCARD", "seat": 2}}]


def test_a_checkpoint_for_another_player_count_is_refused_when_seated():
    from hexset.clients.netbot import NetworkBot

    three = start(random_base_board(random.Random(0)), 3, random.Random(0))
    with pytest.raises(ValueError, match="4"):
        LocalSearchBrain(NetworkBot(policy=None, players=4), three)


def test_a_runner_whose_brain_refuses_the_position_stops_instead_of_retrying():
    """A `ValueError` from `decide` is the position refused, which asking
    again cannot change: the runner stops on it rather than spinning."""
    asked = []

    class Transport:
        def get(self, path, token):
            return {"version": 1, "to_move": 0, "your_move": "act"}

        def post(self, path, token, body):
            raise AssertionError("nothing was decided, so nothing is posted")

    class Refuses:
        def decide(self, transport, token, seat):
            asked.append(seat)
            raise ValueError("trained for 4 players, not this table's 3")

    runner = BotRunner(seat=0, token="t", transport=Transport(), brain=Refuses(), poll_interval=0.0)
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    thread.join(timeout=10)
    alive = thread.is_alive()
    runner.stop.set()
    assert not alive, "the runner kept retrying a refused position"
    assert asked == [0]
