# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.seats`: batched policy seats on the driver, and their labels.

The batched seat plays the same games and keeps the same labels as the same
seat asked one row at a time, its product is the committee's, and a draw
reads only the request's key.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import options_for
from hexset.clients.committee import Committee
from hexset.gym.driver import Driver
from hexset.gym.lanes import LaneEnv
from hexset.gym.seats import Labels, Play, PolicySeat, draw_index, product, tempered
from hexset.record import to_json

SEED = 11


class Fake:
    """A batch-invariant policy: each row's prior and values are a fixed
    function of the policy, the seat and the options, with some zeros."""

    def __init__(self, k: int, players: int) -> None:
        self.k, self.players, self.calls, self.space = k, players, 0, None

    def score_rows(self, rows):
        self.calls += 1
        out = []
        for _, seat, options in rows:
            rng = random.Random(f"{self.k}:{seat}:{len(options)}:{options[0]!r}:{options[-1]!r}")
            p = np.array([rng.random() for _ in options])
            if len(p) > 2 and self.k % 2:
                p[rng.randrange(len(p))] = 0.0
            values = tuple(rng.random() for _ in range(self.players))
            out.append((p / p.sum(), values))
        return out


class First:
    def choose(self, game):
        return options_for(game)[0]


def play(players, *, batch, lanes=3, games=4):
    policies = [Fake(k, players) for k in range(4)]
    own = Fake(9, players)
    plays = {0: Play(members=(0, 1, 2), temperature=0.5), 1: Play(members=(3,), temperature=0.5),
             2: Play(policy=own, temperature=0.5), 3: Play(members=(0, 1, 2))}
    seat = PolicySeat(policies, plays, record=True, batch=batch)
    cast = ((0, 1, 2, 4), (3, 4, 0, 0)) if players == 4 else ((0, 1), (2, 4), (3, 0))
    env = LaneEnv(players, SEED, lanes, deal=games, action_cap=400, caster=lambda i: cast[i % len(cast)],
                  bots={4: lambda board: First()}, no_trade={0, 1, 2, 3, 4})
    labels, kept = Labels(), {}

    def keep(episode):
        kept[episode.index] = (to_json(episode.record), episode.outcome, labels.take(episode.index))

    driver = Driver(env, {pid: seat for pid in plays}, keep=keep, on_decision=labels)
    driver.play()
    return kept, policies, driver, seat


@pytest.mark.parametrize("players", [2, 4])
def test_batched_seat_plays_the_same_games_and_labels_as_one_row_at_a_time(players):
    batched, policies, driver, seat = play(players, batch=True)
    single, *_ = play(players, batch=False)
    assert sorted(batched) == sorted(single) == [0, 1, 2, 3]
    rows = 0
    for index, (record, outcome, labels) in batched.items():
        other_record, other_outcome, other_labels = single[index]
        assert record == other_record and outcome == other_outcome
        assert len(labels) == len(other_labels) > 0
        for a, b in zip(labels, other_labels):
            assert (a.seat, a.chosen) == (b.seat, b.chosen)
            assert a.priors.dtype == np.float32 and a.priors.shape[0] == 4
            assert a.values.shape == (4, players)
            np.testing.assert_array_equal(a.priors, b.priors)
            np.testing.assert_array_equal(a.values, b.values)
        rows += len(labels)
    assert seat.rows == rows
    # One forward per read policy per tick that had a multi-option row.
    assert all(p.calls == policies[0].calls for p in policies) and policies[0].calls < rows


def test_product_and_top_are_the_committees():
    members = [Fake(k, 4) for k in range(3)]
    committee = Committee(members)
    for n in (2, 5, 9, 30):
        options = tuple(range(n))
        row = (None, 1, options)
        priors = np.array([m.score_rows([row])[0][0] for m in members])
        np.testing.assert_allclose(product(priors), committee.score_rows([row])[0][0], rtol=0, atol=1e-12)
        from hexset.gym.seats import _top
        assert options[_top(priors)] == committee.act_rows([row])[0]
    np.testing.assert_allclose(product(np.zeros((2, 3))), np.full(3, 1 / 3))


def test_draws_read_the_key_alone():
    p = tempered(np.array([0.5, 0.3, 0.2, 0.0]), 0.5)
    np.testing.assert_allclose(p, np.array([0.25, 0.09, 0.04, 0.0]) / 0.38)
    rng = random.Random(3)
    keys = [rng.getrandbits(64) for _ in range(20000)]
    picks = [draw_index(p, key) for key in keys]
    assert picks == [draw_index(p, key) for key in keys]
    assert 3 not in picks
    shares = np.bincount(picks, minlength=4) / len(picks)
    np.testing.assert_allclose(shares, p, atol=0.015)
    assert draw_index(np.array([0.0, 1.0, 0.0]), (1 << 64) - 1) == 1


def test_one_seat_object_is_asked_once_a_tick_for_all_its_ids():
    calls = []

    class Counting(PolicySeat):
        def choose_many(self, requests):
            calls.append(sorted({r.policy for r in requests}))
            return super().choose_many(requests)

    policies = [Fake(k, 4) for k in range(2)]
    seat = Counting(policies, {0: Play(members=(0,)), 1: Play(members=(1,))})
    env = LaneEnv(4, SEED, 4, deal=4, action_cap=60, caster=lambda i: (0, 1, 0, 1))
    Driver(env, {0: seat, 1: seat}, keep=lambda e: None).play()
    assert calls and any(c == [0, 1] for c in calls)
    assert all(p.calls <= len(calls) for p in policies)


def test_plays_are_checked_and_native_or_decision_requests_refused():
    with pytest.raises(ValueError):
        Play()
    with pytest.raises(ValueError):
        Play(members=(0,), policy=Fake(0, 4))
    with pytest.raises(ValueError):
        PolicySeat([Fake(0, 4)], {0: Play(members=(1,))})
    seat = PolicySeat([Fake(0, 4)], {0: Play(members=(0,))})
    env = LaneEnv(4, SEED, 1, deal=1, decisions={0})
    with pytest.raises(ValueError, match="atomic"):
        Driver(env, {0: seat}, keep=lambda e: None).tick()


def test_no_trade_ids_are_embargoed_with_every_other_seat():
    env = LaneEnv(4, SEED, 1, deal=1, caster=lambda i: (0, 1, 0, 1), no_trade={1})
    game = next(env.in_flight())
    assert game.embargoes == frozenset((s, o) for s in (1, 3) for o in range(4) if o != s)


@pytest.mark.parametrize("batch", [True, False])
def test_a_joint_scorer_reads_as_each_policy_does(batch):
    policies = [Fake(k, 4) for k in range(3)]
    calls = []

    def scorer(rows):
        calls.append(len(rows))
        return [p.score_rows(rows) for p in policies]

    plays = {0: Play(members=(0, 1, 2), temperature=0.5)}
    games = {}
    for joint in (False, True):
        seat = PolicySeat(policies, plays, record=True, batch=batch, scorer=scorer if joint else None)
        env = LaneEnv(4, SEED, 3, deal=3, action_cap=200, no_trade={0})
        labels, kept = Labels(), {}
        Driver(env, {0: seat}, on_decision=labels,
               keep=lambda e: kept.update({e.index: (to_json(e.record), labels.take(e.index))})).play()
        games[joint] = kept
    assert calls and (max(calls) > 1) == batch
    for index, (record, labels) in games[False].items():
        other_record, other_labels = games[True][index]
        assert record == other_record and len(labels) == len(other_labels)
        for a, b in zip(labels, other_labels):
            assert a.chosen == b.chosen and np.array_equal(a.priors, b.priors)
