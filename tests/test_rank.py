"""The sibling-ranking scorer, against rows whose answer is known by construction.

The measurement decides between two different campaigns — trust the value head
for credit assignment, or replace its target — so the arithmetic that turns
rollouts into a verdict is worth pinning separately from the rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest

import random
import zlib

from benchmarks.rank import (
    assess,
    backed_up,
    chance,
    correlation,
    head_row,
    lane_plan,
    probeable,
    ranks,
    pooled,
    resolution,
    share,
    standardised,
    summarise,
    teacher_row,
)
from catan.actions import legal_actions
from catan.mcts import Node, draws_hidden

from test_mcts import a_game, a_purchase, a_steal

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


TRUTH = np.array([0.30, 0.10, 0.00])  # child 0 is genuinely best


def test_a_search_that_finds_the_best_child_the_prior_missed_counts_as_improved():
    cell = teacher_row(
        prior=np.array([0.2, 0.7, 0.1]),
        visits=np.array([60.0, 30.0, 10.0]),
        true=TRUTH,
    )
    assert cell["moved"] and cell["improved"] and not cell["damaged"]
    assert cell["visits_top1"] and not cell["prior_top1"]


def test_a_search_that_walks_away_from_a_correct_prior_counts_as_damaged():
    cell = teacher_row(
        prior=np.array([0.7, 0.2, 0.1]),
        visits=np.array([20.0, 70.0, 10.0]),
        true=TRUTH,
    )
    assert cell["moved"] and cell["damaged"] and not cell["improved"]


def test_a_search_that_agrees_with_the_prior_is_neither():
    # The case expert iteration cannot learn from: whatever the tree spent, the
    # target it hands back is the argmax the policy already had.
    cell = teacher_row(
        prior=np.array([0.7, 0.2, 0.1]),
        visits=np.array([70.0, 20.0, 10.0]),
        true=TRUTH,
    )
    assert not cell["moved"] and not cell["improved"] and not cell["damaged"]


def test_regret_is_zero_for_whichever_side_picked_the_best_child():
    cell = teacher_row(
        prior=np.array([0.1, 0.8, 0.1]),
        visits=np.array([80.0, 10.0, 10.0]),
        true=TRUTH,
    )
    assert cell["visits_regret"] == pytest.approx(0.0)
    assert cell["prior_regret"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Averaging a chance child. Until 2026-08-28 a `BUY_DEV_CARD`, `PLAY_KNIGHT` or
# `Phase.ROBBER` child was built with `imagine` then `apply`, which froze one
# sampled outcome into the row, so head-versus-truth across siblings compared
# draws as well as decisions. These are the tests that would have caught it.


class Hashing:
    """A value that is a deterministic function of the child's own holdings.

    Torch-free, and the right stand-in for the question at hand: `encoding.py`
    extends per-resource hand counts and per-type development-card counts for
    the perspective seat, so a stolen card or a bought card *is* visible to a
    real head. A stub that reads the same fields is what makes a frozen draw
    show up at all; a stub returning a constant would hide the defect exactly
    the way the harness did.
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
            # Stable across processes — `hash` of a tuple of ints is, but saying
            # so in the arithmetic is cheaper than relying on it. Scaled to the
            # 0.0155 the real head's sibling spread was measured at.
            digest = zlib.crc32(repr(key).encode()) / 2**32
            value = [0.0] * self.players
            value[leaf.seat] = 0.03 * (digest - 0.5)
            out.append((np.full(max(len(leaf.options), 1), 0.5), tuple(value)))
        return out


MAX_OFFERS = 3


def scored(game, *, draws, seed=0, chance_seed=99, evaluator=None):
    """One position's row, the way `main` builds it."""
    children = probeable(game, MAX_OFFERS)
    assert len(children) >= 2
    return children, head_row(
        game,
        children,
        game.current_player,
        evaluator=evaluator or Hashing(),
        max_offers=MAX_OFFERS,
        draws=draws,
        rng=random.Random(seed),
        extra=lambda: random.Random(chance_seed),
    )


def test_a_budget_is_split_exactly_however_it_divides():
    assert share(384, 8) == [48] * 8
    assert sum(share(384, 7)) == 384
    assert share(10, 4) == [3, 3, 2, 2]
    # Fewer rollouts than draws is a smoke setting, and it still sums.
    assert sum(share(3, 8)) == 3


def test_one_draw_plans_exactly_the_single_collector_the_harness_had_before():
    """The byte-identity claim for the truth column, stated as arithmetic: one
    draw must ask for every lane at offset zero, which is the one
    `PairedBranching` the row built before the chance fix."""
    assert lane_plan(384, 1) == [(384, 0)]


def test_every_child_covers_the_same_lane_streams_however_many_draws_it_has():
    """`resolution` pairs the top two children lane by lane, so a chance child
    spread over eight draws and a deterministic sibling on one must walk the
    same stream offsets in the same order or the pairing silently dissolves."""
    for draws in (1, 2, 3, 8, 16):
        plan = lane_plan(384, draws)
        assert sum(lanes for lanes, _ in plan) == 384
        walked = [
            offset + lane
            for lanes, offset in plan
            for lane in range(lanes)
        ]
        assert walked == list(range(384))


def test_a_chance_free_row_is_bit_identical_however_many_draws_are_asked():
    """The off-path anchor. Exact equality, not `approx`: the opening placement
    row resolves no hidden information, so asking for eight draws must produce
    the same floats and the same number of forward passes as asking for one.
    """
    one_eval, many_eval = Hashing(), Hashing()
    _, one = scored(a_game(), draws=1, evaluator=one_eval)
    _, many = scored(a_game(), draws=8, evaluator=many_eval)

    assert not any(one.hot)
    assert one.head == many.head
    assert one.drawn == many.drawn
    assert [len(g) for g in many.games] == [1] * len(many.games)
    assert one_eval.calls == many_eval.calls


def test_a_chance_free_row_never_touches_the_chance_stream():
    """A stream this row does not draw from cannot shift the rows after it —
    the same post-run-draw check `3e9d03a` used to prove its own off-path."""
    stream = random.Random(99)
    children = probeable(a_game(), MAX_OFFERS)
    head_row(
        a_game(),
        children,
        0,
        evaluator=Hashing(),
        max_offers=MAX_OFFERS,
        draws=8,
        rng=random.Random(0),
        extra=lambda: stream,
    )
    assert stream.random() == random.Random(99).random()


def test_a_chance_row_moves_when_the_frozen_draw_stops_being_frozen():
    """The on-path half. A `Phase.ROBBER` row is all chance children, so the
    fix has to change it — a fix that changed nothing anywhere would be the
    other failure mode."""
    game, _ = a_steal()
    _, one = scored(game, draws=1)
    _, many = scored(game, draws=8)

    assert any(one.hot)
    assert one.head != many.head


def test_a_chance_child_is_scored_as_the_mean_of_its_draws():
    game, _ = a_steal()
    children, row = scored(game, draws=8)

    for index, action in enumerate(children):
        assert len(row.drawn[index]) == (8 if draws_hidden(game, action) else 1)
        assert row.head[index] == pytest.approx(float(np.mean(row.drawn[index])))


def test_a_bought_card_is_averaged_the_same_way_a_steal_is():
    """The other draw. `BUY_DEV_CARD` resolves inside `apply` off the deck, and
    it sits in an ordinary `Phase.MAIN` row beside deterministic siblings — so
    this also pins that a mixed row averages only the child that drew."""
    game, index = a_purchase()
    children, row = scored(game, draws=8)
    purchase = legal_actions(game)[index]
    at = children.index(purchase)

    assert row.hot[at]
    assert len(row.drawn[at]) == 8
    assert sum(row.hot) < len(children)
    for other, action in enumerate(children):
        if not draws_hidden(game, action):
            assert len(row.drawn[other]) == 1


def test_averaging_shrinks_what_the_draw_can_do_to_a_chance_child():
    """The measurement the power calculation rests on.

    Re-running the probe at a different seed re-draws the hidden card. Under one
    draw the child's score moves by the full spread of the head across outcomes;
    under N it moves by that over root N, because the score is now a mean of N
    of them. This is the noise the metric was carrying and the reason the
    recommended draw count is what it is.

    Measured as the standard deviation of one chance child's score across 96
    seeds. Theory says the ratio is sqrt(8) = 2.83; the assertion is loose
    enough to survive a two-outcome victim's discreteness.
    """
    game, _ = a_steal()
    children = probeable(game, MAX_OFFERS)
    at = next(i for i, a in enumerate(children) if draws_hidden(game, a))
    cold = next(i for i, a in enumerate(children) if not draws_hidden(game, a))

    def spread(draws):
        values, others = [], []
        for trial in range(96):
            _, row = scored(game, draws=draws, seed=trial, chance_seed=5000 + trial)
            values.append(row.head[at])
            others.append(row.head[cold])
        # A child that draws nothing does not move at all, whatever the seeds do
        # — the control that says the shrinkage below is about the draw.
        assert len(set(others)) == 1
        return float(np.std(values))

    one, many = spread(1), spread(8)
    assert one > 0.0
    assert one / many > 2.0


def test_the_chance_block_reports_no_spread_rather_than_a_reassuring_zero():
    """At one draw there is no second draw to measure a spread against, and
    zero is an ordinary value on this scale that would read as "no
    contamination" — the exact misreading the old path invited."""
    rows = [{"chance_children": 2, "chance_spread": []}]
    got = chance(rows, 1)

    assert got["spread"] is None and got["residual"] is None
    assert got["rows"] == 1 and got["children"] == 2 and got["row_share"] == 1.0


def test_the_chance_block_reports_the_residual_that_averaging_leaves():
    rows = [
        {"chance_children": 1, "chance_spread": [0.02]},
        {"chance_children": 0, "chance_spread": []},
    ]
    got = chance(rows, 16)

    assert got["spread"] == pytest.approx(0.02)
    assert got["residual"] == pytest.approx(0.02 / 4.0)
    assert got["row_share"] == pytest.approx(0.5)
