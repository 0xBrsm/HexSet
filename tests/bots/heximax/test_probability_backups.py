"""Chance must average utility, not apply nonlinear utility to mean scores."""
import random
from types import SimpleNamespace

import pytest

from hexset.board.board import random_base_board
from hexset.bots.heximax import heximax
from hexset.game import Phase


def bot(**kwargs):
    return heximax(random_base_board(random.Random(1)), **kwargs)


def leaf(values, winner=None):
    return SimpleNamespace(num_players=len(values), values=values, won_by=winner,
                           phase=Phase.MAIN if winner is None else Phase.GAME_OVER)


def test_expected_probability_prefers_better_odds_over_large_terminal_score():
    baseline=bot()
    candidate=bot(probability_backups=True)
    for b in (baseline,candidate):
        b.evaluator.evaluate_game=lambda game,knower: game.values
    outcomes=[leaf([100,0,0,0],winner=0),leaf([0,10,10,10])]
    secure=leaf([5,0,0,0])
    def utility(b):
        values=[b._leaf(g,0) for g in outcomes]
        mean=[sum(v[p] for v in values)/2 for p in range(4)]
        return b._search_rank(mean,0),b._search_rank(b._leaf(secure,0),0)
    gamble, certain=utility(baseline)
    assert gamble>certain
    gamble,certain=utility(candidate)
    assert gamble<certain
    assert .5<gamble<.51


def test_terminal_winner_is_one_hot_even_if_hidden_vp_is_not_estimated_as_ten():
    candidate=bot(probability_backups=True)
    candidate.evaluator.evaluate_game=lambda *args: pytest.fail('terminal utility must use the declared winner')
    assert candidate._leaf(leaf([0,0,0,0],winner=2),0)==[0,0,1,0]


def test_leaf_probabilities_sum_to_one_and_other_valuation_stays_unchanged():
    baseline=bot()
    candidate=bot(probability_backups=True)
    values=[4,7,6,3]
    candidate.evaluator.evaluate_game=lambda *args: values
    assert sum(candidate._leaf(leaf(values),0))==pytest.approx(1)
    assert candidate._rank(values,0)==baseline._rank(values,0)
    assert baseline.probability_backups is False
    with pytest.raises(ValueError,match='win stance'):
        bot(probability_backups=True,stance='own')
