# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.driver`: one loop over Python and native lanes.

A game is the same function of `(seed, index)` on either lane kind: the same
record, the same decisions put to the same seats, the same observation bytes.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from hexset import contract9, decisions as D
from hexset.actions import space_for
from hexset.arena import deal_game
from hexset.gym.driver import Driver, draw_key, observations, seat_points
from hexset.gym.lanes import LaneEnv
from hexset.gym.native import available
from hexset.record import to_json

SEED = 11
KINDS = (D.DecisionKind.FLAT, D.DecisionKind.SETUP, D.DecisionKind.ROAD_BUILDING, D.DecisionKind.DISCARD)


class Answer:
    def __init__(self, action, decision):
        self.action, self.decision = action, decision


class Uniform:
    """Any legal option of a whole decision, drawn from the request's own key."""

    def __init__(self, players, contract, log=None):
        game = deal_game(SEED, 0, players)
        self.space = space_for(game)
        self.pairs = contract9.incidences(game.state(0, hidden=False).board.topology)
        self.players, self.native_contract, self.log = players, contract, log

    def choose_many(self, requests):
        out = []
        for request, observation in zip(requests, observations(requests, self.native_contract)):
            decision = request.decision
            blocks = [(kind, list(values)) for kind, values in zip(KINDS, (
                np.flatnonzero(decision.flat), np.flatnonzero(decision.setup),
                range(len(decision.road_building)), range(len(decision.discard)))) if len(values)]
            rng = random.Random(request.draw)
            kind, values = rng.choice(blocks)
            at = int(rng.choice(values))
            frame = None if request.game is None else D.frame_for(request.game, request.seat, self.space)
            option = D.option_of(decision, kind, at, self.pairs, request.seat, self.space, self.players, frame)
            if self.log is not None:
                self.log.append((request.index, request.step, request.seat, request.decision_id, request.part,
                                 int(kind), at, np.asarray(observation, np.float32).tobytes()))
            out.append(Answer(option, (int(kind), at)))
        return out


def play(players, lanes, *, kind, games=5, discard_order="random", cap=600):
    env = LaneEnv(players, SEED, lanes, deal=games, action_cap=cap, discard_order=discard_order,
                  decisions={0}, first_game=3, stride=2)
    log, decided, kept = [], [], []
    contract = "9d" if players == 2 else "9"
    seat = Uniform(players, contract, log)
    driver = Driver(env, {0: seat}, keep=lambda episode: kept.append(episode),
                    on_decision=lambda request, answer: decided.append(
                        (request.index, request.step, answer.decision, driver.targets().points[request.row])),
                    lanes=kind)
    returned = driver.play()
    assert returned == kept and not driver.running
    return driver, kept, sorted(log), sorted(decided)


@pytest.mark.skipif(not available(), reason="optional native extension")
@pytest.mark.parametrize("players", [2, 4])
@pytest.mark.parametrize("order", ["seat", "random"])
def test_native_and_python_lanes_play_the_same_games(players, order):
    python, a, log_a, decided_a = play(players, 2, kind="python", discard_order=order)
    native, b, log_b, decided_b = play(players, 2, kind="native", discard_order=order)
    assert python.lanes == "python" and native.lanes == "native"
    assert [e.index for e in a] and sorted(e.index for e in a) == sorted(e.index for e in b) == [3, 5, 7, 9, 11]
    by_index = {e.index: e for e in b}
    for episode in a:
        other = by_index[episode.index]
        assert to_json(episode.record) == to_json(other.record)
        assert episode.outcome == other.outcome
    assert log_a == log_b
    assert decided_a == decided_b
    assert native.games_started() == python.games_started()


@pytest.mark.skipif(not available(), reason="optional native extension")
def test_auto_takes_native_lanes_only_where_every_seat_reads_them():
    env = LaneEnv(4, SEED, 2, deal=2, decisions={0}, action_cap=50)
    assert Driver(env, {0: Uniform(4, "9")}, keep=lambda e: None, lanes="auto").lanes == "native"
    env = LaneEnv(4, SEED, 2, deal=2, decisions={0}, action_cap=50)
    reader = Uniform(4, "9")
    reader.native_contract = None
    assert Driver(env, {0: reader}, keep=lambda e: None, lanes="auto").lanes == "python"
    env = LaneEnv(4, SEED, 2, deal=2, decisions={0}, action_cap=50, caster=lambda i: (0, 1, 0, 1),
                  bots={1: lambda board: _First()})
    driver = Driver(env, {0: Uniform(4, "9")}, keep=lambda e: None, lanes="auto")
    assert driver.lanes == "python"
    with pytest.raises(ValueError, match="native lanes cannot play"):
        Driver(LaneEnv(4, SEED, 2, deal=2, decisions={0}, caster=lambda i: (0, 1, 0, 1),
                       bots={1: lambda board: _First()}), {0: Uniform(4, "9")}, keep=lambda e: None,
               lanes="native")


class _First:
    def choose(self, game):
        from hexset.actions import options_for
        return options_for(game)[0]


def test_bot_seats_are_played_by_the_environment_and_targets_are_the_true_points():
    env = LaneEnv(4, SEED, 3, deal=3, decisions={0}, action_cap=300, caster=lambda i: (0, 1, 1, 0),
                  bots={1: lambda board: _First()})
    seat = Uniform(4, "9")
    seen = []

    def decided(request, answer):
        assert request.policy == 0
        seen.append(request)

    driver = Driver(env, {0: seat}, keep=lambda e: None, on_decision=decided, lanes="python")
    while driver.running:
        for request, points in zip(driver.requests(), driver.targets().points):
            assert points == seat_points(request.game)
            assert request.draw == draw_key(SEED, request.index, request.seat, request.step)
        driver.tick()
    assert seen and {r.seat for r in seen} <= {0, 3}


def test_keep_is_required_and_every_policy_needs_a_seat():
    env = LaneEnv(4, SEED, 1, deal=1, decisions={0}, caster=lambda i: (0, 2, 0, 2))
    with pytest.raises(ValueError, match="keep"):
        Driver(env, {0: Uniform(4, "9")}, keep=None)
    driver = Driver(env, {0: Uniform(4, "9")}, keep=lambda e: None)
    with pytest.raises(ValueError, match="no seat answers"):
        while driver.running:
            driver.tick()


def test_a_listed_environment_deals_exactly_its_indices_as_the_strided_deal_does():
    bots = {0: lambda board: _First()}
    strided = LaneEnv(4, SEED, 1, deal=4, first_game=3, stride=2, bots=bots, action_cap=300)
    listed = LaneEnv(4, SEED, 1, indices=[9, 5], bots=bots, action_cap=300)
    a = {e.index: to_json(e.record) for e in strided.drain()}
    b = {e.index: to_json(e.record) for e in listed.drain()}
    assert set(b) == {5, 9} and all(b[i] == a[i] for i in b)
