from __future__ import annotations

import pytest

from benchmarks.horizon import labels
from catan.rewards import reward
from catan.selfplay import Outcome


class Transition:
    def __init__(self, value=()):
        self.value = value


def an_episode(trajectory, seat=1, points=(10, 4, 6, 3)):
    class Episode:
        pass

    episode = Episode()
    episode.outcome = Outcome(
        winner=0, points=points, turns=60, actions=700, truncated=False
    )
    episode.trajectories = [[] for _ in points]
    episode.trajectories[seat] = trajectory
    return episode


def test_the_bootstrapped_label_is_the_estimate_that_many_decisions_later():
    trajectory = [Transition((0.1 * i, 0.0, 0.0, 0.0)) for i in range(10)]
    boot, terminal, reached = labels(an_episode(trajectory), seat=1, horizon=8)
    assert reached
    assert boot == pytest.approx(0.8)
    assert terminal == pytest.approx(reward(an_episode(trajectory).outcome)[1])


def test_the_label_is_read_in_the_movers_own_frame():
    # `NetworkPolicy` puts the mover first, so element 0 is the seat's own
    # estimate and no rotation applies. `SearchPolicy` does not, which is why
    # `distill._value_targets` rotates and this does not.
    trajectory = [Transition((0.5, -0.9, -0.9, -0.9)) for _ in range(4)]
    boot, _, _ = labels(an_episode(trajectory, seat=2), seat=2, horizon=2)
    assert boot == pytest.approx(0.5)


def test_a_trajectory_shorter_than_the_horizon_falls_back_to_the_terminal():
    trajectory = [Transition((0.4, 0.0, 0.0, 0.0)) for _ in range(5)]
    boot, terminal, reached = labels(an_episode(trajectory), seat=1, horizon=8)
    assert not reached
    assert boot == terminal


def test_a_forced_move_at_the_horizon_carries_no_estimate_and_falls_back():
    # A decision with one legal option is filed with an empty value, so the
    # horizon lands on nothing to bootstrap from. Counting it as reached would
    # quietly mix terminal labels into the bootstrapped variance.
    trajectory = [Transition((0.4, 0.0, 0.0, 0.0)) for _ in range(10)]
    trajectory[8] = Transition(())
    boot, terminal, reached = labels(an_episode(trajectory), seat=1, horizon=8)
    assert not reached
    assert boot == terminal


def test_a_horizon_of_zero_reads_the_position_itself():
    """The degenerate end, which the benchmark exists to make visible.

    At horizon 0 the label is the head's own opinion of `s`, identical across
    every rollout from `s`, so its variance is exactly zero and the ratio
    collapses. That is what a too-short horizon looks like in the report.
    """
    trajectory = [Transition((0.3, 0.0, 0.0, 0.0)) for _ in range(6)]
    boot, _, reached = labels(an_episode(trajectory), seat=1, horizon=0)
    assert reached
    assert boot == pytest.approx(0.3)
