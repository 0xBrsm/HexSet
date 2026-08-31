# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
import zlib

import numpy as np
import pytest

from benchmarks.sibling import Probing, Spread, rows
from hexset.actions import ActionType
from hexset.board.board import random_base_board
from hexset.bots import greedy
from hexset.evaluate import Evaluator
from hexset.mcts import draws_hidden
from hexset.selfplay import BotPolicy, Choice, Collector, Request

from test_mcts import a_game, a_purchase, a_steal


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
        from hexset.actions import apply

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
    pytest.importorskip("torch", reason="`rows` rotates with `hexset.ppo`")

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


# ---------------------------------------------------------------------------
# Averaging a chance child. The module already skips roll positions, because the
# spread across dice outcomes is chance and not a decision. Until 2026-08-28 the
# same argument was missed one level down: `imagine` then `apply` froze one
# stolen or bought card into each child, so the spread across siblings was
# partly a spread across decks.


class Hashing:
    """A value that is a deterministic function of the child's own holdings.

    A real head reads the same fields — `encoding.py` extends per-resource hand
    counts and per-type development-card counts for the perspective seat — so a
    stub that reads them is what makes a frozen draw visible. A stub returning a
    constant would hide the defect exactly as the harness did.
    """

    def __init__(self, players: int = 4) -> None:
        self.players = players
        self.calls = 0

    def evaluate(self, leaves):
        out = []
        for leaf in leaves:
            self.calls += 1
            state = leaf.game.state
            key = (
                tuple(tuple(hand) for hand in state.hands),
                tuple(tuple(cards) for cards in state.new_dev_cards),
                tuple(tuple(cards) for cards in state.dev_cards),
            )
            digest = zlib.crc32(repr(key).encode()) / 2**32
            value = [0.0] * self.players
            value[leaf.seat] = 0.03 * (digest - 0.5)
            out.append((np.full(max(len(leaf.options), 1), 0.5), tuple(value)))
        return out


def a_hashing_probing(*, draws, seed=0, chance_seed=99):
    return Probing(
        First(),
        Hashing(),
        max_offers=3,
        rate=1.0,
        rng=random.Random(seed),
        chance_draws=draws,
        chance_seed=chance_seed,
    )


def test_a_chance_free_probe_is_bit_identical_however_many_draws_are_asked():
    """Exact equality, not `approx`. The opening placement row resolves no
    hidden information, so asking for eight draws must produce the same three
    statistics and the same number of forward passes as asking for one."""
    one, many = a_hashing_probing(draws=1), a_hashing_probing(draws=8)
    game = a_game()
    assert not any(draws_hidden(game, a) for a in one._options(game))

    before, after = one._probe(a_game()), many._probe(a_game())

    assert before == after
    assert before.chance_children == 0
    assert one.evaluator.calls == many.evaluator.calls


def test_a_chance_probe_averages_its_draws_and_says_how_far_they_moved():
    game, _ = a_steal()
    one = a_hashing_probing(draws=1)._probe(game)
    many = a_hashing_probing(draws=8)._probe(game)

    assert one.chance_children > 0
    assert one.chance_children == many.chance_children
    # A single draw measures no spread because it takes no second draw, and
    # reporting 0.0 there would read as "no contamination".
    assert one.chance_spread == 0.0
    assert many.chance_spread > 0.0
    # Averaging shrinks the row's own spread toward the pre-draw one, because
    # the per-child chance shocks it was carrying are gone.
    assert one.spread != many.spread


def test_a_bought_card_is_averaged_beside_deterministic_siblings():
    game, _ = a_purchase()
    probe = a_hashing_probing(draws=8)
    spread = probe._probe(game)

    assert spread.chance_children == 1
    assert spread.options > spread.chance_children


class Sweeping:
    """Probes every decision of real games and keeps every row, with a flag for
    whether the position held a chance child."""

    def __init__(self, draws, seed, max_offers=3):
        self.probing = Probing(
            BotPolicy(
                lambda b: greedy(
                    Evaluator(b), random.Random(seed), max_offers=max_offers
                )
            ),
            Hashing(),
            max_offers=max_offers,
            rate=1.0,
            rng=random.Random(seed + 2),
            chance_draws=draws,
            chance_seed=seed + 4,
        )
        self.rows: list[tuple[bool, Spread]] = []

    def act(self, requests):
        for request in requests:
            hot = any(
                draws_hidden(request.game, a)
                for a in self.probing._options(request.game)
            )
            spread = self.probing._probe(request.game)
            if spread is not None:
                self.rows.append((hot, spread))
        return self.probing.policy.act(requests)


def sweep(draws, games=2, seed=0, max_offers=3):
    board = random_base_board(random.Random(seed))
    sweeping = Sweeping(draws, seed, max_offers)
    Collector(
        policy=sweeping,
        lanes=8,
        players=4,
        seed=seed + 1,
        max_offers=max_offers,
        deal=games,
        board=board,
    ).drain()
    # The post-sweep draw off the shared stream, the check `3e9d03a` used: a
    # stream shift that has not yet reached a row is invisible in the rows.
    return sweeping.rows, sweeping.probing.rng.random()


def test_the_fix_is_off_path_over_a_thousand_chance_free_rows():
    """The anchor, proven at scale rather than on one fixture.

    Two full games of `greedy` self-play, every decision probed, both arms at
    one seed. Every chance-free row must be bit-identical and the shared stream
    must end in the same state — so a chance row cannot move a chance-free row
    that comes after it. And the chance rows must actually move, or the fix
    would be doing nothing anywhere.
    """
    one, tail_one = sweep(1)
    many, tail_many = sweep(8)

    assert len(one) == len(many) > 1000
    assert [hot for hot, _ in one] == [hot for hot, _ in many]

    clean = [(a, b) for (hot, a), (_, b) in zip(one, many) if not hot]
    hot = [(a, b) for (hot, a), (_, b) in zip(one, many) if hot]

    assert len(clean) > 1000
    assert [a for a, _ in clean] == [b for _, b in clean]
    assert tail_one == tail_many

    assert len(hot) > 20
    assert sum(a != b for a, b in hot) > 0.9 * len(hot)
