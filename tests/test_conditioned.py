# SPDX-License-Identifier: GPL-3.0-only
"""A rollout's other seats played by a model of them (`hexset.clients.conditioned`)."""
from __future__ import annotations

import random

import numpy as np

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.bots.base import Closes, TradesBy, watch, watches
from hexset.clients import conditioned as C
from hexset.game import Phase, start, to_move
from hexset.rollout import paired_copy


def after_setup(seed: int = 0):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


class Uniform:
    """A stand-in for the other seats' policy: the last option of each row."""

    width = 3

    def __init__(self) -> None:
        self.conditions: dict[int, np.ndarray] = {}
        self.rows: list = []

    def act_rows(self, rows):
        self.rows += rows
        return [options[-1] for _, _, options in rows]


class Own:
    def __init__(self) -> None:
        self.rows: list = []

    def act_rows(self, rows):
        self.rows += rows
        return [options[0] for _, _, options in rows]

    def value_rows(self, rows):
        return [(0.25,) * 4 for _ in rows]

    def score_rows(self, rows):
        return [(np.full(len(o), 1 / len(o)), (0.25,) * 4) for _, _, o in rows]


def test_a_playout_routes_this_seat_to_its_own_policy_and_the_rest_to_the_model():
    game = after_setup()
    rows = []
    for seat in range(4):
        g = after_setup()
        g.current_player = seat
        g.phase = Phase.ROLL
        rows.append((g, seat, tuple(legal_actions(g))))
    own, others = Own(), Uniform()
    routed = C.SeatRouted(own, others, seat=2)
    out = routed.act_rows(rows)
    assert [s for _, s, _ in own.rows] == [2]
    assert [s for _, s, _ in others.rows] == [0, 1, 3]
    assert out[2] == rows[2][2][0] and out[0] == rows[0][2][-1]
    assert routed.value_rows([(game, 0)]) == [(0.25,) * 4]


def test_the_context_carries_the_seat_and_every_condition_to_a_worker():
    sender = C.SeatRouted(Own(), Uniform(), seat=1)
    sender.others.conditions = {0: np.ones(3, np.float32), 3: np.zeros(3, np.float32)}
    worker = C.SeatRouted(Own(), Uniform(), seat=None)
    worker.set_context(sender.context())
    assert worker.seat == 1
    assert set(worker.others.conditions) == {0, 3}
    sender.others.conditions[0][:] = 5               # a copy, not the sender's own arrays
    assert worker.others.conditions[0][0] == 1


def test_copies_of_one_world_carry_its_seed_for_an_opponent_model_to_key_draws_by():
    world = after_setup()
    a, b, c = paired_copy(world, 7, 0), paired_copy(world, 7, 0), paired_copy(world, 8, 0)
    assert a.paired_seed == b.paired_seed == 7 and c.paired_seed == 8


class Watching:
    def __init__(self) -> None:
        self.seen_moves = []

    def seen(self, game, actor, action, viewer):
        self.seen_moves.append((actor, action.type, viewer))

    def choose(self, game):
        return legal_actions(game)[0]


def test_watch_reaches_the_mover_through_its_wrappers():
    game = after_setup()
    inner = Watching()
    seat = TradesBy(Closes(inner, object()), object())
    action = legal_actions(game)[0]
    watch(seat, game, 1, action, 2)
    assert inner.seen_moves == [(1, action.type, 2)]
    assert watches(seat) and not watches(TradesBy(Closes(Own(), object()), object()))


def test_a_seat_watches_each_move_before_it_is_applied():
    from hexset.game import observe
    from hexset.seat import Seat

    before = []

    class Phases(Watching):
        def seen(self, g, actor, action, viewer):
            before.append((g.phase, actor, viewer))

    game = after_setup()
    seat = Seat(observe(game, 0), 0, Phases())
    mover = to_move(game)
    action = next(a for a in legal_actions(game) if a.type is ActionType.ROLL)
    seat.play(mover, action, roll=8)
    assert before == [(Phase.ROLL, mover, 0)]
