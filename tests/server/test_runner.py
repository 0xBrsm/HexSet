"""The server's own bot seats: `LocalSearchBrain` deciding on the table's
`Game`, and `BotRunner` meeting the moves and discards `your_move` says the
seat owes.
"""

from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.game import start
from hexset.server._runner import BotRunner, LocalSearchBrain


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
