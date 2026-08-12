from __future__ import annotations

import random

import numpy as np
import pytest

from benchmarks.sibling import Probing, Spread, rows
from catan.actions import ActionType
from catan.selfplay import Choice, Collector, Request

from test_mcts import a_game


class Ranked:
    """Values that step by one per leaf, so the spread is known in advance."""

    def __init__(self, players: int = 4) -> None:
        self.players = players
        self.calls = 0

    def evaluate(self, leaves):
        out = []
        for leaf in leaves:
            value = [0.0] * self.players
            value[leaf.seat] = float(self.calls)
            self.calls += 1
            out.append((np.full(max(len(leaf.options), 1), 0.5), tuple(value)))
        return out


class First:
    """A policy that always plays the first option, and records nothing."""

    def act(self, requests):
        return [Choice(action=request.options[0]) for request in requests]


def a_probing(evaluator=None, *, rate=1.0, seed=0):
    return Probing(
        First(),
        evaluator or Ranked(),
        max_offers=3,
        rate=rate,
        rng=random.Random(seed),
    )


def a_request(game, options) -> Request:
    return Request(
        lane=0,
        seat=0,
        observation=None,
        mask=np.zeros(0, dtype=bool),
        options=options,
        game=game,
    )


def test_a_probe_reports_the_spread_of_the_values_it_was_given():
    probing = a_probing()
    game = a_game()
    spread = probing._probe(game)
    assert spread is not None
    assert spread.options > 1
    # Values 0, 1, ... n-1, whose statistics are exact.
    row = np.arange(spread.options, dtype=np.float64)
    assert spread.spread == pytest.approx(float(row.std()))
    assert spread.span == pytest.approx(spread.options - 1)
    assert spread.best_gap == pytest.approx(1.0)


def test_a_probe_scores_every_legal_child_exactly_once():
    probing = a_probing()
    game = a_game()
    spread = probing._probe(game)
    assert probing.evaluator.calls == spread.options


def test_a_probe_leaves_the_position_it_was_handed_alone():
    probing = a_probing()
    game = a_game()
    before = probing._options(game)
    probing._probe(game)
    assert probing._options(game) == before


def test_a_roll_position_is_skipped_rather_than_scored():
    probing = a_probing()
    game = a_game()
    options = probing._options(game)
    while not any(a.type is ActionType.ROLL for a in options):
        from catan.actions import apply

        apply(game, options[0])
        options = probing._options(game)
    assert probing._probe(game) is None


def test_the_probe_rate_decides_how_many_positions_carry_a_measurement():
    probing = a_probing(rate=0.0)
    collector = Collector(probing, lanes=1, seed=7, players=4)
    collector.run(40)
    assert probing.probed == 0
    assert probing.skipped == 0


def test_a_probed_decision_carries_its_measurement_on_the_choice():
    probing = a_probing(rate=1.0)
    game = a_game()
    options = probing._options(game)
    (choice,) = probing.act([a_request(game, options)])
    assert isinstance(choice.aux, Spread)
    assert choice.action == options[0]
    assert probing.probed == 1


def test_rows_pairs_an_error_with_every_probe_and_ignores_the_rest():
    pytest.importorskip("torch", reason="`rows` rotates with `catan.ppo`")

    class Episode:
        pass

    class Outcome:
        points = (10, 4, 4, 2)
        actions = 20

    class Transition:
        def __init__(self, aux, value):
            self.aux = aux
            self.value = value

    episode = Episode()
    episode.outcome = Outcome()
    marked = Spread(seat=0, options=3, spread=0.1, span=0.3, best_gap=0.05)
    episode.trajectories = [
        [Transition(marked, (0.25,)), Transition(None, (0.9,))],
        [Transition(marked, ())],
    ]
    errors, spreads = rows([episode])
    assert len(spreads) == 1
    assert errors.shape == (1,)
    # Seat 0 finished 10 against a mean of 3.33 for the others, over 10 points.
    assert errors[0] == pytest.approx((10 - 10 / 3) / 10 - 0.25)
