# SPDX-License-Identifier: GPL-3.0-only
"""A rollout's other seats played by a conditioned model (`hexset.clients.conditioned`)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.bots.base import Closes, TradesBy, watch, watches
from hexset.clients import conditioned as C
from hexset.clients._modelmeta import search_config
from hexset.game import Phase, start, to_move
from hexset.rollout import paired_copy


def after_setup(seed: int = 0):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


class Uniform:
    """A stand-in for `ConditionedPolicy`: a flat prior over each row's options."""

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


def test_the_pair_is_named_beside_the_file_both_or_neither():
    meta = {"search": "rollout", "opponents": "a.onnx", "opponent_encoder": "b.onnx"}
    assert search_config(meta).rollout.opponents == ("a.onnx", "b.onnx")
    assert search_config({"search": "rollout"}).rollout.opponents is None
    for bad in ({"opponents": "a.onnx"}, {"opponent_encoder": "b.onnx"},
                {"opponents": "../a.onnx", "opponent_encoder": "b.onnx"}):
        with pytest.raises(ValueError):
            search_config({"search": "rollout", **bad})


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


def _policy_without_a_session(width=3, seed=0):
    p = C.ConditionedPolicy.__new__(C.ConditionedPolicy)
    p.width, p.conditions, p.rng = width, {}, random.Random(seed)
    return p


def test_copies_of_one_world_draw_alike_and_other_worlds_do_not():
    world = after_setup()
    a, b, c = paired_copy(world, 7, 0), paired_copy(world, 7, 0), paired_copy(world, 8, 0)
    p = _policy_without_a_session()
    da = [p._uniform(a, 1) for _ in range(5)]
    db = [p._uniform(b, 1) for _ in range(5)]
    dc = [p._uniform(c, 1) for _ in range(5)]
    assert da == db and da != dc
    assert p._uniform(a, 2) != da[0]                # each seat its own stream
    assert p.condition(3).tolist() == [0.0, 0.0, 0.0]


def test_a_move_is_sampled_from_the_prior_by_its_draw():
    p = _policy_without_a_session()
    game = after_setup()
    options = ("first", "second", "third")             # act_rows only indexes them
    p.prior_rows = lambda rows: [np.array([0.2, 0.5, 0.3])]
    picks = {}
    for u, want in ((0.1, 0), (0.3, 1), (0.69, 1), (0.71, 2)):
        p._uniform = lambda g, s, u=u: u
        picks[u] = p.act_rows([(game, 0, options)])[0]
        assert picks[u] == options[want]


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


class Encoder:
    def __init__(self) -> None:
        self.steps = []

    def step(self, game, seat, action):
        self.steps.append((seat, action))
        return np.full(3, len(self.steps), np.float32)


def _opponents():
    o = C.Opponents.__new__(C.Opponents)
    o.rng, o.policy, o.encoder, o.moves = random.Random(0), _policy_without_a_session(), Encoder(), 0
    return o


def test_the_model_reads_other_seats_moves_with_a_choice_never_its_own():
    game = after_setup()
    game.phase, game.current_player = Phase.ROLL, 1
    o = _opponents()
    roll = next(a for a in legal_actions(game) if a.type is ActionType.ROLL)
    o.seen(game, 0, roll, 0)                         # our own move: not read
    assert o.encoder.steps == []
    # a seat whose only legal move is the roll has no choice to read
    o.seen(game, 1, roll, 0)
    assert o.encoder.steps == [] or len(legal_actions(game, 1)) > 1
    game.phase = Phase.MAIN
    end = next(a for a in legal_actions(game) if a.type is ActionType.END_TURN)
    if len(legal_actions(game, 1)) > 1:
        o.seen(game, 1, end, 0)
        assert o.encoder.steps[-1] == (1, end)
        assert o.policy.conditions[1].tolist() == [len(o.encoder.steps)] * 3


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
