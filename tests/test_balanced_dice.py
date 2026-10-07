# SPDX-License-Identifier: GPL-3.0-only
"""`chance.Balanced`: the 1v1 ladder variant's dice, as `Rules.balanced_dice`.

Specified algorithm: one deck of all 36 two-dice combinations, drawn and
discarded, reshuffled whole at 12 cards remaining. A draw picks a sum by
weight -- the cards of that sum left, less 34% for each time the sum came up
in the last five rolls -- then one of its cards. A seven's weight is scaled
for the roller: up while it holds less than its share of the sevens, down
while it holds more, and by 0.4 for each seven of the current run, down when
the run is the roller's own, the scale kept within 0 and 2.
"""
from __future__ import annotations

import collections
import random

import pytest

from hexset.chance import (
    BALANCED_SEVEN_ODDS,
    DICE_DECK_RESHUFFLE_AT,
    DICE_RECENT_DISCOUNT,
    SEVEN_ADJUST_MAX,
    Balanced,
    Live,
    dice_deck,
)
from hexset.rules import DUEL_VARIANT_GAME, DUEL_VARIANT, STANDARD, STANDARD_GAME

# Two dice, independent: sum over p^2 of the eleven outcomes.
IID_REPEAT_RATE = 0.1127
# The same deck with the recent-sum discount, two seats rolling in turn, measured off this implementation.
DISCOUNTED_REPEAT_RATE = 0.0675


def rolls(n: int, seed: int = 0) -> list[int]:
    source = Balanced(random.Random(seed))
    return [source.roll_by(i % 2) for i in range(n)]


def repeat_rate(seq: list[int]) -> float:
    return sum(a == b for a, b in zip(seq, seq[1:])) / (len(seq) - 1)


def test_the_deck_is_the_thirty_six_combinations():
    deck = dice_deck()
    assert len(deck) == 36
    assert len(set(deck)) == 36
    assert collections.Counter(a + b for a, b in deck) == {
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }


def test_the_deck_reshuffles_before_it_runs_out():
    """A deck is never played past the threshold, so the 36 combinations are
    not a hard quota over any 36-roll window."""
    source = Balanced(random.Random(0))
    seen = []
    for i in range(200):
        source.roll_by(i % 2)
        seen.append(source.cards)
    assert min(seen) == DICE_DECK_RESHUFFLE_AT
    assert max(seen) == 35


def test_a_sum_that_came_up_lately_is_discounted_per_time():
    source = Balanced(random.Random(0))
    source.recent = [8, 8, 3]
    w = source.weights()
    assert w[8] == pytest.approx(5 / 36 * (1 - 2 * DICE_RECENT_DISCOUNT))
    assert w[3] == pytest.approx(2 / 36 * (1 - DICE_RECENT_DISCOUNT))
    assert w[6] == pytest.approx(5 / 36)


def test_the_recent_discount_suppresses_repeats():
    """Independent dice repeat 11.3% of the time; the discounted deck about
    6.8%. 20,000 rolls put the standard error near 0.002."""
    discounted = repeat_rate(rolls(20_000))
    assert abs(discounted - DISCOUNTED_REPEAT_RATE) < 0.01
    assert discounted < IID_REPEAT_RATE - 0.03


def test_a_seven_is_steered_toward_the_seat_short_of_its_share():
    source = Balanced(random.Random(0))
    source.sevens = {0: 0, 1: 2}
    source.run = (1, 2)
    assert source.seven_scale(0) == SEVEN_ADJUST_MAX       # 2 for the share, +0.8 for the other's run, capped
    assert source.seven_scale(1) == 0.0                    # 0 for the share, -0.8 for its own run, floored
    source.sevens = {0: 1, 1: 1}
    source.run = (1, 1)
    assert source.seven_scale(0) == pytest.approx(1.4)
    assert source.seven_scale(1) == pytest.approx(0.6)


def test_no_steering_before_every_roller_could_have_one():
    source = Balanced(random.Random(0))
    source.sevens = {0: 0, 1: 1}
    source.run = (None, 0)
    assert source.seven_scale(0) == 1.0
    assert source.seven_scale(None) == 1.0


def test_rolled_sevens_follow_the_roller_share():
    """Two seats in turn over game-length runs: a seat holding fewer than half
    the sevens so far rolls a seven far more often than one holding more."""
    hit = collections.Counter()
    seen = collections.Counter()
    for game in range(600):
        source = Balanced(random.Random(game))
        sevens = collections.Counter()
        for i in range(70):
            seat = i % 2
            total = sum(sevens.values())
            band = None if total < 2 else ("low" if sevens[seat] * 2 < total else "high" if sevens[seat] * 2 > total else None)
            r = source.roll_by(seat)
            if band:
                seen[band] += 1
                hit[band] += r == 7
            if r == 7:
                sevens[seat] += 1
    assert hit["low"] / seen["low"] > 2 * hit["high"] / seen["high"]


def test_the_engine_names_the_roller():
    from hexset.arena import play
    from hexset.bots import RandomBot
    from hexset.board.board import random_base_board

    board = random_base_board(random.Random(0))
    game = play([RandomBot(random.Random(1)), RandomBot(random.Random(2))], board, random.Random(3),
                game_type=DUEL_VARIANT_GAME, turn_cap=10_000)
    assert set(game.chance.sevens) == {0, 1}


def test_the_rules_select_the_source():
    from hexset.board.board import random_base_board
    from hexset.game import start

    board = random_base_board(random.Random(0))
    assert DUEL_VARIANT.balanced_dice is True
    assert STANDARD.balanced_dice is False
    assert isinstance(start(board, 2, random.Random(0), game_type=DUEL_VARIANT_GAME).chance, Balanced)
    assert type(start(board, 2, random.Random(0), game_type=STANDARD_GAME).chance) is Live


def test_the_published_seven_rate_matches_the_deck_it_describes():
    """`BALANCED_SEVEN_ODDS` is measured off `Balanced` with two seats in
    turn, so it has to be re-measurable. 200,000 rolls put the standard error
    near 0.0008; 0.004 is five of those."""
    seq = rolls(200_000, seed=4242)
    assert abs(seq.count(7) / len(seq) - BALANCED_SEVEN_ODDS) < 0.004
    assert BALANCED_SEVEN_ODDS < 6 / 36


def test_a_seat_yet_to_roll_is_counted_with_no_sevens():
    source = Balanced(random.Random(0))
    source.sevens = {0: 1}
    source.run = (0, 1)
    assert source.seven_scale(1) == pytest.approx(1.4)   # no share term until each seat could hold one; +0.4 for the other's run
