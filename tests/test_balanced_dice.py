# SPDX-License-Identifier: GPL-3.0-only
"""`chance.Balanced`: the 1v1 ladder variant's dice deck, as
`Rules.balanced_dice`.

Specified algorithm: one deck of all 36 two-dice combinations, drawn and
discarded, reshuffled whole at 12 cards remaining, with a 30% discount on a
card that repeats the previous number. The discount is what these tests pin:
the deck alone repeats about as often as independent dice, the discount is
what pulls the repeat rate down.

Not modelled, and asserted here so the gap stays visible: the variant also
damps seven streaks and steers each player's share of the sevens.
"""
from __future__ import annotations

import collections
import random

from hexset.chance import (
    DICE_DECK_RESHUFFLE_AT,
    Balanced,
    Live,
    dice_deck,
)
from hexset.rules import DUEL_VARIANT_GAME, DUEL_VARIANT, STANDARD, STANDARD_GAME

# Two dice, independent: sum over p^2 of the eleven outcomes.
IID_REPEAT_RATE = 0.1127
# The same deck drawn with the 30% discount, measured off this implementation.
DISCOUNTED_REPEAT_RATE = 0.064


def rolls(n: int, seed: int = 0) -> list[int]:
    source = Balanced(random.Random(seed))
    return [source.roll() for _ in range(n)]


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
    not a hard quota over any 36-roll window, and disjoint windows need not
    each hold an exact deck."""
    source = Balanced(random.Random(0))
    seen = []
    for _ in range(200):
        source.roll()
        seen.append(len(source.deck))
    assert min(seen) == DICE_DECK_RESHUFFLE_AT
    assert max(seen) == 35


def test_the_repeat_discount_is_what_suppresses_repeats():
    """Independent dice repeat 11.3% of the time. The deck with the discount
    lands near 6.4%, well below that -- the discount is doing the work, not
    the deck. 20,000 rolls put the standard error near 0.002."""
    discounted = repeat_rate(rolls(20_000))
    assert abs(discounted - DISCOUNTED_REPEAT_RATE) < 0.01
    assert discounted < IID_REPEAT_RATE - 0.03


def test_the_seven_adjustment_is_not_modelled():
    """A table running the seven adjustment rolls fewer sevens than the
    textbook rate. This source rolls the textbook rate, because the
    adjustment is specified as behaviour with no numbers to implement."""
    seq = rolls(20_000)
    sevens = seq.count(7) / len(seq)
    assert abs(sevens - 6 / 36) < 0.01


def test_the_rules_select_the_source():
    from hexset.board.board import random_base_board
    from hexset.game import start

    board = random_base_board(random.Random(0))
    assert DUEL_VARIANT.balanced_dice is True
    assert STANDARD.balanced_dice is False
    assert isinstance(start(board, 2, random.Random(0), game_type=DUEL_VARIANT_GAME).chance, Balanced)
    assert type(start(board, 2, random.Random(0), game_type=STANDARD_GAME).chance) is Live


def test_the_published_seven_rate_matches_the_deck_it_describes():
    """`BALANCED_SEVEN_ODDS` is measured off `Balanced`, so it has to be
    re-measurable. 200,000 rolls put the standard error near 0.0008; 0.004 is
    five of those, tight enough to catch a changed deck and loose enough not to
    flake on the seed."""
    from hexset.chance import BALANCED_SEVEN_ODDS, Balanced

    dice = Balanced(random.Random(4242))
    rolls = 200_000
    sevens = sum(1 for _ in range(rolls) if dice.roll() == 7)
    assert abs(sevens / rolls - BALANCED_SEVEN_ODDS) < 0.004
    # The discount is what moves it, and it moves it downwards.
    assert BALANCED_SEVEN_ODDS < 6 / 36
