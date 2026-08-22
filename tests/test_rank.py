"""The sibling-ranking scorer, against rows whose answer is known by construction.

The measurement decides between two different campaigns — trust the value head
for credit assignment, or replace its target — so the arithmetic that turns
rollouts into a verdict is worth pinning separately from the rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.rank import (
    assess,
    backed_up,
    correlation,
    ranks,
    pooled,
    resolution,
    standardised,
    summarise,
)
from catan.mcts import Node

TRUE = np.array([0.9, 0.5, 0.1])


def test_ties_get_the_average_rank_rather_than_an_invented_order():
    assert list(ranks(np.array([1.0, 2.0, 2.0, 3.0]))) == [0.0, 1.5, 1.5, 3.0]


def test_a_head_that_orders_the_row_correctly_scores_one_and_regrets_nothing():
    got = assess(np.array([0.3, 0.2, 0.1]), TRUE)

    assert got["spearman"] == pytest.approx(1.0)
    assert got["top1"] is True
    assert got["regret"] == pytest.approx(0.0)


def test_a_head_that_orders_it_backwards_scores_minus_one_and_pays_the_span():
    got = assess(np.array([0.1, 0.2, 0.3]), TRUE)

    assert got["spearman"] == pytest.approx(-1.0)
    assert got["top1"] is False
    assert got["regret"] == pytest.approx(0.8)


def test_a_head_with_no_opinion_correlates_with_nothing():
    """A constant row must read zero, not nan — `corrcoef` would give nan."""
    got = assess(np.array([0.2, 0.2, 0.2]), TRUE)

    assert got["spearman"] == pytest.approx(0.0)
    assert got["pearson"] == pytest.approx(0.0)


def test_chance_top_one_is_reported_so_the_hit_rate_can_be_read():
    assert assess(np.array([0.3, 0.2, 0.1]), TRUE)["chance_top1"] == pytest.approx(1 / 3)


def test_pairing_shrinks_the_standard_error_when_the_noise_is_shared():
    """The whole reason siblings are rolled out from one seed."""
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 0.19, 512)
    a = 0.02 + shared + rng.normal(0, 0.02, 512)
    b = shared + rng.normal(0, 0.02, 512)

    got = resolution([a, b], np.array([a.mean(), b.mean()]))

    assert got["paired_se"] < got["unpaired_se"] / 5
    assert got["pairing_gain"] > 5
    assert got["resolved"] is True


def test_a_gap_inside_the_noise_is_reported_unresolved():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.19, 32)
    b = rng.normal(0, 0.19, 32)

    got = resolution([a, b], np.array([a.mean(), b.mean()]))

    # Two draws from one distribution: whatever gap appears is inside the noise.
    assert got["resolved"] is False


def test_the_summary_reads_top_one_against_its_own_chance_rate():
    rows = [
        dict(assess(np.array([0.3, 0.2, 0.1]), TRUE), paired_se=0.001,
             unpaired_se=0.01, pairing_gain=10.0, resolved=True),
        dict(assess(np.array([0.1, 0.2, 0.3]), TRUE), paired_se=0.001,
             unpaired_se=0.01, pairing_gain=10.0, resolved=True),
    ]

    got = summarise(rows)

    assert got["positions"] == 2
    assert got["top1_rate"] == pytest.approx(0.5)
    assert got["top1_chance"] == pytest.approx(1 / 3)
    assert got["spearman_mean"] == pytest.approx(0.0)
    assert got["regret_mean_victory_points"] == pytest.approx(4.0)


def test_a_constant_column_does_not_produce_a_nan_correlation():
    assert correlation(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0.0


def test_pooled_recovers_a_correlation_that_noise_attenuated():
    """The whole point of the correction: noisy truth, honest answer."""
    from benchmarks.rank import pooled

    rng = np.random.default_rng(7)
    heads, trues, ses = [], [], []
    noise = 0.03
    for _ in range(400):
        head = rng.normal(0, 0.012, 6)
        truth = head + rng.normal(0, 0.006, 6)          # true r is high, not 1
        trues.append(truth + rng.normal(0, noise, 6))   # observed with error
        heads.append(head)
        ses.append(np.full(6, noise))

    got = pooled(heads, trues, ses)

    assert got["pearson_observed"] < 0.4          # attenuated hard by the noise
    assert got["pearson_corrected"] > 0.75        # recovered
    assert 0.0 < got["reliability"] < 0.5


def test_pooled_does_not_manufacture_a_correlation_from_noise():
    """Real signal in the truth, none of it related to the head: still zero.

    The truth carries genuine spread here, so reliability is well above zero and
    the correction actually runs — which is the case worth guarding. A truth
    that is *pure* noise has nothing to correct toward and is covered below.
    """
    from benchmarks.rank import pooled

    rng = np.random.default_rng(11)
    heads = [rng.normal(0, 0.012, 6) for _ in range(400)]
    trues = [rng.normal(0, 0.013, 6) + rng.normal(0, 0.03, 6) for _ in range(400)]
    ses = [np.full(6, 0.03) for _ in range(400)]

    got = pooled(heads, trues, ses)

    assert got["reliability"] > 0.05
    assert abs(got["pearson_corrected"]) < 0.25


def test_pooled_reports_zero_reliability_rather_than_dividing_by_it():
    from benchmarks.rank import pooled

    rng = np.random.default_rng(3)
    heads = [rng.normal(0, 0.01, 4) for _ in range(50)]
    trues = [rng.normal(0, 0.01, 4) for _ in range(50)]
    ses = [np.full(4, 1.0) for _ in range(50)]        # noise swamps everything

    got = pooled(heads, trues, ses)

    assert got["reliability"] == 0.0
    assert got["pearson_corrected"] != got["pearson_corrected"]   # nan, not inf


def searched(visits, totals) -> Node:
    """A root as `Search.run_many` leaves it, with the game it played omitted."""
    node = Node(game=None, mover=0, options=("a", "b"))
    node.visits = np.asarray(visits, dtype=np.float64)
    node.totals = np.asarray(totals, dtype=np.float64)
    return node


def test_the_backed_up_value_is_the_visit_weighted_mean_of_what_came_back():
    # Edge a returned 0.3 for seat 0 on each of three visits, edge b -0.4 once.
    root = searched([3.0, 1.0], [[0.9, -0.3], [-0.4, 0.4]])
    assert backed_up(root, 0) == pytest.approx((3 * 0.3 + 1 * -0.4) / 4)
    assert backed_up(root, 1) == pytest.approx(0.1 / 4)


def test_a_finished_or_forced_child_has_no_backed_up_value_to_give():
    assert backed_up(Node(game=None, mover=0, options=()), 0) is None
    assert backed_up(Node(game=None, mover=0, options=("a",)), 0) is None


def test_a_tree_that_never_ran_reports_nothing_rather_than_a_confident_zero():
    # Zero is an ordinary value on this scale, so the empty case has to be
    # distinguishable from it or it pools into the correlation unnoticed.
    assert backed_up(searched([0.0, 0.0], np.zeros((2, 4))), 0) is None


def widest_row_disagrees():
    """Nine tight rows the head orders correctly, and one wide row it inverts.

    The wide row is one position of ten and should read as one position. Under
    `pooled` it is not: Pearson weights by variance, so a row whose spread is
    ten times larger carries a hundred times the leverage.
    """
    heads = [[0.0, 0.01, 0.02]] * 9 + [[0.2, 0.1, 0.0]]
    trues = [[0.0, 0.01, 0.02]] * 9 + [[0.0, 0.1, 0.2]]
    ses = [[1e-6] * 3] * 10
    return heads, trues, ses


def test_one_wide_row_can_outvote_nine_tight_ones_when_rows_are_not_scaled():
    heads, trues, ses = widest_row_disagrees()
    assert pooled(heads, trues, ses)["pearson_observed"] < 0.0


def test_scaling_each_row_first_lets_the_nine_outvote_the_one():
    heads, trues, ses = widest_row_disagrees()
    assert standardised(heads, trues, ses)["pearson_observed"] > 0.5


def test_a_row_with_one_child_carries_no_ordering_and_is_dropped():
    out = standardised([[0.1], [0.0, 0.1]], [[0.2], [0.0, 0.1]], [[0.0], [0.0, 0.0]])
    assert out["positions"] == 1
