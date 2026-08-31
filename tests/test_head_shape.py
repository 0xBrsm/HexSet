# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import pytest

from benchmarks.head_shape import load_corpus, rows, split


class Outcome:
    points = (10, 4, 4, 2)
    actions = 20


class Transition:
    def __init__(self, observation) -> None:
        self.observation = observation


class Episode:
    def __init__(self, lengths) -> None:
        self.outcome = Outcome()
        self.trajectories = [
            [Transition(f"{seat}.{step}") for step in range(length)]
            for seat, length in enumerate(lengths)
        ]


def test_the_holdout_is_taken_by_game_so_no_game_reaches_both_sides():
    corpus = list(range(128))
    fit, held = split(corpus, 0.2, random.Random(1))
    assert len(held) == 26
    assert len(fit) + len(held) == len(corpus)
    assert not set(fit) & set(held)


def test_two_games_still_leave_one_on_each_side():
    fit, held = split([0, 1], 0.2, random.Random(0))
    assert len(fit) == 1 and len(held) == 1


def test_one_game_is_a_corpus_that_cannot_be_held_out_of():
    with pytest.raises(ValueError):
        split([0], 0.2, random.Random(0))


def test_a_holdout_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        split(list(range(10)), 1.0, random.Random(0))


def test_a_corpus_that_is_not_episodes_is_refused(tmp_path):
    path = tmp_path / "not.corpus"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_corpus(str(path))


def test_every_decision_becomes_one_row():
    pytest.importorskip("torch", reason="`rows` rotates with `hexset.ppo`")
    observations, targets = rows([Episode((3, 2, 1, 0)), Episode((1, 1, 1, 1))])
    assert len(observations) == 6 + 4
    assert targets.shape == (10, 4)


def test_a_seat_sees_its_own_payoff_first_at_every_one_of_its_decisions():
    pytest.importorskip("torch", reason="`rows` rotates with `hexset.ppo`")
    from hexset.ppo import rotate
    from hexset.rewards import reward

    episode = Episode((2, 2, 2, 2))
    _, targets = rows([episode])
    payoff = reward(episode.outcome)
    for seat in range(4):
        wanted = rotate(payoff, seat)
        for step in range(2):
            assert tuple(targets[seat * 2 + step]) == pytest.approx(wanted)
    # Seat 0 won, so its own component leads and the row is still zero-sum.
    assert targets[0][0] == pytest.approx(payoff[0])
    # abs= because the targets are float32: the four components cancel to ~3e-8,
    # not to zero, and approx's default tolerance against 0.0 is 1e-12.
    assert float(targets[0].sum()) == pytest.approx(0.0, abs=1e-6)
