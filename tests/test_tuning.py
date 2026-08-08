from __future__ import annotations

import random
from dataclasses import fields

import pytest

from catan.evaluate import Weights
from catan.tuning import ANCHOR, TUNABLE, as_source, climb, duel, perturb


def changed_fields(before: Weights, after: Weights) -> set[str]:
    return {
        f.name
        for f in fields(Weights)
        if getattr(before, f.name) != getattr(after, f.name)
    }


def test_the_anchor_is_never_tuned():
    assert ANCHOR not in TUNABLE
    assert set(TUNABLE) | {ANCHOR} == {f.name for f in fields(Weights)}


def test_perturbing_moves_only_the_requested_number_of_weights():
    start = Weights()
    for seed in range(20):
        moved = changed_fields(start, perturb(start, random.Random(seed), sigma=0.4, count=2))
        assert len(moved) <= 2
        assert ANCHOR not in moved


def test_perturbing_can_revive_a_weight_sitting_at_zero():
    start = Weights(reach=0.0)
    revived = any(
        perturb(start, random.Random(seed), sigma=0.4, count=len(TUNABLE)).reach != 0.0
        for seed in range(10)
    )
    assert revived


def test_perturbing_leaves_the_original_alone():
    start = Weights()
    perturb(start, random.Random(0), sigma=0.9, count=4)
    assert start == Weights()


def test_a_duel_returns_wins_out_of_the_games_that_finished():
    wins, decided = duel(
        Weights(), Weights(), 4, seed=0, depth=1, width=None
    )
    assert 0 <= wins <= decided <= 4


def test_a_zero_step_proposes_the_incumbent_unchanged():
    start = Weights()
    assert perturb(start, random.Random(0), sigma=0.0, count=4) == start


def test_a_stricter_bar_accepts_no_more_than_a_looser_one():
    """The whole point of `z`: it trades progress against drift monotonically.

    One round only. As soon as the two runs disagree about an acceptance their
    incumbents differ, and every later duel is a different experiment.
    """
    strict = climb(rounds=1, games=4, sigma=0.5, seed=3, z=1.96)[1][0]
    loose = climb(rounds=1, games=4, sigma=0.5, seed=3, z=0.0)[1][0]

    assert strict.wins == loose.wins
    assert strict.lower <= loose.lower
    assert strict.accepted <= loose.accepted


def test_a_climb_records_every_round_it_tried():
    _, history = climb(rounds=2, games=4, sigma=0.5, seed=1)
    assert [step.round for step in history] == [0, 1]
    assert all(step.decided <= 4 for step in history)


def test_the_climb_reports_each_step_as_it_goes():
    seen = []
    climb(rounds=2, games=4, sigma=0.5, seed=1, report=seen.append)
    assert len(seen) == 2


def test_weights_round_trip_through_their_source_form():
    original = Weights(production=1.234, surplus_card=-0.5)
    restored = eval(as_source(original), {"Weights": Weights})
    for f in fields(Weights):
        assert getattr(restored, f.name) == pytest.approx(getattr(original, f.name))
