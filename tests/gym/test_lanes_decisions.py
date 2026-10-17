# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes` with a random discard order and with seats that act in
whole decisions (`hexset.decisions`): the same game a seat acting a move at a
time plays, and records that replay."""

from __future__ import annotations

import random

import pytest

from hexset import decisions as D
from hexset.actions import Action, ActionType, options_for
from hexset.game import UNSTRUCTURED_TURN_CAP, Phase, players_owing_discards, to_move
from hexset.gym.lanes import LaneEnv as _LaneEnv
from hexset.record import advance, moves, open_record, replay
from hexset.victory import victory_points

SEED = 3


def LaneEnv(*args, **kwargs):
    kwargs.setdefault("turn_cap", UNSTRUCTURED_TURN_CAP)
    return _LaneEnv(*args, **kwargs)


class First:
    """Deterministic, so one object can serve every lane and seat."""

    trade_floor = 0.0

    def choose(self, game):
        return options_for(game)[-1]


class Drawn:
    """Uniform over the legal actions, from a stream of its own per lane
    position, so the play does not depend on the order lanes are asked in."""

    trade_floor = 0.0

    def choose(self, game):
        state = game.state(0, hidden=False)
        rng = random.Random(f"{game.turns}:{sum(state.hands[0])}:{len(game.discard_quota)}:{game.phase}")
        return rng.choice(options_for(game))


def chooser(request) -> D.Option:
    """A whole decision for a request that has one, drawn from the decision's
    own options by its game and step."""
    rng = random.Random(f"{request.index}:{request.step}")
    return rng.choice(D.options(request.game, request.seat, double_turn=False))


def answers(requests):
    return [chooser(r) if r.decision is not None else None for r in requests]


def watched(env, answer=lambda requests: [None] * len(requests)):
    """Play `env` out, noting every discard request: (index, step, seat, the
    seats owing then, the seat's quota then)."""
    asked, episodes = [], []
    while env.running:
        requests = env.requests()
        for r in requests:
            if r.game.phase is Phase.DISCARD:
                asked.append((r.index, r.step, r.seat, tuple(players_owing_discards(r.game)),
                              r.game.discard_quota[r.seat]))
        episodes.extend(env.step(answer(requests)))
    return asked, episodes


def test_a_random_discard_order_draws_an_owing_seat_and_lets_it_finish():
    env = LaneEnv(4, SEED, 3, deal=6, action_cap=1500, bots={0: lambda board: First()},
                  records=True, discard_order="random")
    asked, episodes = watched(env)

    assert asked, "no discard round in these games"
    out_of_order = 0
    for (index, step, seat, owing, quota), following in zip(asked, asked[1:] + [None]):
        assert seat in owing
        if following is not None and following[0] == index and following[1] == step + 1 and quota > 1:
            assert following[2] == seat, "a seat was left part-way through its discard"
    rounds = {}
    for index, step, seat, owing, _ in asked:
        # The first ask of each round: the step before it was not a discard.
        if (index, step - 1) not in rounds:
            out_of_order += seat != owing[0]
        rounds[(index, step)] = seat
    assert out_of_order, "every round went lowest seat first"
    for episode in episodes:
        assert any(episode.record.actors) or not any(
            d.action.type is ActionType.DISCARD for d in episode.stream()
        )
        replayed = replay(episode.record).state(0, hidden=False)
        assert tuple(victory_points(replayed, s) for s in range(4)) == episode.outcome.points


def test_the_discard_order_is_each_games_own_whatever_the_lanes():
    played = {}
    for lanes in (1, 4):
        env = LaneEnv(4, SEED, lanes, deal=4, action_cap=900, bots={0: lambda board: First()},
                      records=True, discard_order="random")
        played[lanes] = {e.index: e.record for e in env.drain()}
    assert played[1] == played[4]


def test_seat_order_names_no_actor():
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=900, bots={0: lambda board: First()}, records=True)
    assert all(e.record.actors == () for e in env.drain())


def test_an_unknown_discard_order_is_refused():
    with pytest.raises(ValueError, match="discard order"):
        LaneEnv(4, SEED, 1, deal=1, discard_order="lowest")


@pytest.mark.parametrize("order", ["seat", "random"])
def test_a_decision_lane_plays_the_game_its_moves_play_one_at_a_time(order):
    env = LaneEnv(4, SEED, 2, deal=3, action_cap=600, records=True, decisions={0}, discard_order=order)
    kinds: set = set()

    def answer(requests):
        out = []
        for r in requests:
            assert r.decision is not None and r.options == ()
            option = chooser(r)
            if isinstance(option, D.Compound):
                kinds.add(option.kind)
            out.append(option)
        return out

    _, episodes = watched(env, answer)
    # The snake seat's double turn is asked as two setup turns.
    assert D.CompoundKind.SETUP in kinds and D.CompoundKind.SETUP_DOUBLE not in kinds
    for episode in episodes:
        record = episode.record
        replayed = replay(record).state(0, hidden=False)
        assert tuple(victory_points(replayed, s) for s in range(4)) == episode.outcome.points
        assert len(record.actions) == episode.outcome.actions == len(episode)
        # The same moves answered one at a time by an atomic lane: the same record.
        atomic = LaneEnv(4, SEED, 1, deal=1, first_game=episode.index, action_cap=600, records=True,
                         discard_order=order)
        stream = [Action(ActionType(t), a, b) for t, a, b in record.actions]
        redone = []
        while atomic.running:
            redone.extend(atomic.step([stream[r.step] for r in atomic.requests()]))
        assert [e.record for e in redone] == [record]


def test_bot_seats_act_a_move_at_a_time_beside_decision_seats():
    env = LaneEnv(4, SEED, 2, deal=3, action_cap=800, records=True, caster=lambda index: (0, 1, 0, 1),
                  bots={1: lambda board: Drawn()}, decisions={0}, discard_order="random")
    asked, episodes = [], []
    while env.running:
        requests = env.requests()
        for r in requests:
            assert (r.decision is not None) == (r.policy == 0)
            assert bool(r.options) == (r.policy == 1)
        asked.extend(r.policy for r in requests)
        episodes.extend(env.step(answers(requests)))
    assert {0, 1} <= set(asked)
    for episode in episodes:
        replay(episode.record)
        for seat, decisions in enumerate(episode.decisions):
            assert all(d.seat == seat and d.policy == (0, 1, 0, 1)[seat] for d in decisions)


def test_an_answer_outside_the_decision_advances_no_lane():
    env = LaneEnv(4, SEED, 2, deal=2, records=True, decisions={0})
    requests = env.requests()
    good = answers(requests)
    bad = list(good)
    first = good[0]
    bad[1] = D.Compound(D.CompoundKind.ROAD_BUILDING, (Action(ActionType.PLAY_ROAD_BUILDING),))
    with pytest.raises(ValueError, match="illegal action"):
        env.step(bad)
    assert env.pending() == (0, 0)
    env.step(good)
    assert env.pending() == (len(first.parts),) * 2


def test_a_bot_cannot_be_asked_for_decisions():
    with pytest.raises(ValueError, match="cannot be asked for decisions"):
        LaneEnv(4, SEED, 1, deal=1, bots={0: lambda board: First()}, decisions={0})


def regrouped(record) -> list[int]:
    """Each move of `record` with the run of options it falls in, worked out
    afresh off its replay by `hexset.decisions.closes` a move at a time."""
    game = open_record(record)
    out, running, runs = [], {}, 0
    for actor, action, trades in moves(record):
        seat = actor if actor is not None else (
            players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game))
        if seat not in running:
            running[seat], runs = runs, runs + 1
        out.append(running[seat])
        advance(game, action, trades, actor)
        if D.closes(game, action, seat):
            del running[seat]
    return out


@pytest.mark.parametrize("order", ["seat", "random"])
def test_every_move_is_filed_under_its_run_and_the_record_regroups_the_same(order):
    env = LaneEnv(4, SEED, 2, deal=3, action_cap=700, records=True, caster=lambda index: (0, 1, 0, 0),
                  bots={1: lambda board: Drawn()}, decisions={0}, discard_order=order)
    asked, episodes = [], []
    while env.running:
        requests = env.requests()
        asked.extend((r.index, r.seat, r.decision_id, r.part, r.step) for r in requests)
        episodes.extend(env.step(answers(requests)))
    for episode in episodes:
        stream = episode.stream()
        assert [d.decision_id for d in stream] == regrouped(episode.record)
        # A run is one seat's, its parts asked in order from 0, its ids counted in the order begun.
        mine = sorted((a for a in asked if a[0] == episode.index), key=lambda a: a[4])
        parts: dict = {}
        for _, seat, run, part, _ in mine:
            held = parts.setdefault(run, (seat, -1))
            assert held == (seat, part - 1)
            parts[run] = (seat, part)
        assert sorted(parts) == list(range(len(parts)))
        firsts = [run for _, _, run, part, _ in mine if part == 0]
        assert firsts == sorted(firsts)
        # The snake seat's two setup turns share one run; the other first turns are one each.
        setup = [d for d in stream if d.action.type is ActionType.SETUP_SETTLEMENT]
        assert len(setup) == 8 and len({d.decision_id for d in setup}) == 7
        assert setup[3].decision_id == setup[4].decision_id
        # Some run of a learner seat has several parts past the setup (a build run).
        long = [run for run, (seat, part) in parts.items() if part >= 1 and run not in {d.decision_id for d in setup}]
        assert long or len(stream) < 100
        replayed = replay(episode.record).state(0, hidden=False)
        assert tuple(victory_points(replayed, s) for s in range(4)) == episode.outcome.points
