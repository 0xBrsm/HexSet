# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

from helpers import clear_hand, give

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, imagine, start, to_move
from hexset.rollout import PairedRollouts, paired_copy


def after_setup(seed: int = 0):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def a_road_or_not(seed: int = 0):
    """Seat 0 to move in its main phase holding exactly one road's cards."""
    game = after_setup(seed)
    game.phase = Phase.MAIN
    game.current_player = 0
    clear_hand(game._state, 0)
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 0, Resource.BRICK, 1)
    return game


def ends_turns(options):
    """Ends the turn where it can, else the first option."""
    for option in options:
        if option.type is ActionType.END_TURN:
            return option
    return options[0]


class Stub:
    """Prior `prior(options)`; plays `ends_turns`; values each seat's hand
    size over 10."""

    def __init__(self, prior) -> None:
        self.prior = prior
        self.acted = 0
        self.valued = 0

    def score_rows(self, rows):
        return [(self.prior(options), self.values(game)) for game, _, options in rows]

    def act_rows(self, rows):
        self.acted += len(rows)
        return [ends_turns(options) for _, _, options in rows]

    def value_rows(self, rows):
        self.valued += len(rows)
        return [self.values(game) for game, _ in rows]

    @staticmethod
    def values(game):
        hands = game.state(0, hidden=False).hands
        return tuple(sum(hand) / 10 for hand in hands)


def favouring(kind: ActionType, sure: float):
    """`sure` on the first option of `kind`, most of the rest on END_TURN,
    and what is left shared by the other options."""

    def prior(options):
        out = np.full(len(options), 0.0)
        first = next(i for i, o in enumerate(options) if o.type is kind)
        end = next(i for i, o in enumerate(options) if o.type is ActionType.END_TURN)
        rest = [i for i in range(len(options)) if i not in (first, end)]
        out[first] = sure
        out[end] = (1 - sure) * (0.9 if rest else 1.0)
        out[rest] = (1 - sure) * 0.1 / max(1, len(rest))
        return out

    return prior


def rolls(game, turns: int):
    out = []
    for _ in range(turns):
        while game.phase is not Phase.ROLL:
            apply(game, ends_turns(legal_actions(game)))
        apply(game, next(o for o in legal_actions(game) if o.type is ActionType.ROLL))
        out.append(game.last_roll)
    return out


def test_two_paired_copies_roll_the_same_dice_whatever_was_played_between():
    game = a_road_or_not()
    world = imagine(game, random.Random(1), randomize_deck=False)
    built, waited = paired_copy(world, 99, 0), paired_copy(world, 99, 0)
    road = next(o for o in legal_actions(built) if o.type is ActionType.BUILD_ROAD)
    apply(built, road)
    assert rolls(built, 12) == rolls(waited, 12)
    assert rolls(paired_copy(world, 99, 0), 12) != rolls(paired_copy(world, 100, 0), 12)


def test_a_decisive_root_plays_the_policys_own_move_without_a_playout():
    """The prior favours the road; the policy, played alone, ends its turn:
    the root unsearched plays what the policy plays, not the prior's top."""
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.95))
    search = PairedRollouts(stub, worlds=8, decisive=0.9, rng=random.Random(1))
    decision = search.decide(a_road_or_not())
    assert decision.worlds == 0 and decision.scores == ()
    assert decision.options[decision.chosen].type is ActionType.END_TURN
    assert decision.move.type is ActionType.END_TURN and decision.candidates == (decision.chosen,)
    assert stub.acted == 1


def sure_of_the_first(options):
    """0.99 on the first option, the rest shared."""
    if len(options) == 1:
        return np.ones(1)
    return np.array([0.99] + [0.01 / (len(options) - 1)] * (len(options) - 1))


def test_full_setup_searches_a_confident_opening_root_and_no_later_one():
    """`decisive` skips a root the policy is sure of, unless `full_setup`
    and the root is the opening's; after the opening it skips as before."""
    opening = start(random_base_board(random.Random(4)), 4, random.Random(4))
    assert opening.phase is Phase.SETUP_SETTLEMENT

    def search(full_setup):
        return PairedRollouts(Stub(sure_of_the_first), worlds=4, top=2, horizon=1, marks=(),
                              decisive=0.97, full_setup=full_setup, rng=random.Random(5))

    assert search(False).decide(opening).worlds == 0
    searched = search(True).decide(opening)
    assert searched.worlds == 4 and len(searched.scores) == 2
    assert search(True).decide(a_road_or_not()).worlds == 0


def test_the_move_that_plays_out_better_is_chosen_over_the_favourite():
    """Holding on to the road's cards keeps a bigger hand at the mover's next
    turn in every world, which is what the stub's values count. The policy's
    own move (ending the turn) is the first candidate, the prior's top next."""
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=16, top=2, horizon=1, marks=(), rng=random.Random(3))
    decision = search.decide(a_road_or_not())
    played = [decision.options[i].type for i in decision.candidates]
    assert played == [ActionType.END_TURN, ActionType.BUILD_ROAD]
    assert decision.worlds == 16 and len(decision.scores) == 2
    assert decision.scores[0] > decision.scores[1]
    assert decision.options[decision.chosen].type is ActionType.END_TURN
    assert stub.acted > 0 and stub.valued == 32


def test_a_horizon_stops_each_playout_at_the_movers_next_turn():
    stopped = []

    class Watching(Stub):
        def value_rows(self, rows):
            stopped.extend((game.turns, to_move(game), game.phase) for game, _ in rows)
            return super().value_rows(rows)

    game = a_road_or_not()
    search = PairedRollouts(Watching(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=4, horizon=1,
                            marks=(), rng=random.Random(2))
    search.decide(game)
    assert stopped and all(turns >= game.turns + 1 and seat == 0 and phase is Phase.ROLL
                           for turns, seat, phase in stopped)


def test_the_same_seed_decides_the_same_way():
    def decide(seed):
        stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
        return PairedRollouts(stub, worlds=8, horizon=2, rng=random.Random(seed)).decide(a_road_or_not())

    assert decide(5).scores == decide(5).scores


def test_a_playout_that_never_ends_is_scored_where_it_stands(monkeypatch):
    import hexset.rollout as rollout

    monkeypatch.setattr(rollout, "STEP_LIMIT", 5)
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=4, marks=(), rng=random.Random(4))
    decision = search.decide(a_road_or_not())
    assert decision.worlds == 4 and stub.valued == 4 * len(decision.candidates)
    assert search.last_truncated_rollouts == search.truncated_rollouts == stub.valued


def test_a_playout_cut_short_is_read_from_the_movers_own_frame(monkeypatch):
    """Five steps on, another seat is to move; the read is still the root
    mover's, so its own cards in that world stay known to the value head."""
    import hexset.rollout as rollout

    monkeypatch.setattr(rollout, "STEP_LIMIT", 5)
    frames, cut_at = [], []

    class Watching(Stub):
        def value_rows(self, rows):
            frames.extend(seat for _, seat in rows)
            cut_at.extend(to_move(game) for game, _ in rows)
            return super().value_rows(rows)

    PairedRollouts(Watching(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=4,
                   rng=random.Random(4)).decide(a_road_or_not())
    assert frames and set(frames) == {0}
    assert any(seat != 0 for seat in cut_at)


def test_a_decisive_root_cuts_no_playout_short():
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=2, top=2, horizon=1, rng=random.Random(3))
    search.decide(a_road_or_not())
    assert search.last_truncated_rollouts == 4
    search.decisive = 0.5
    search.decide(a_road_or_not())
    assert search.last_truncated_rollouts == 0 and search.truncated_rollouts == 4


@pytest.mark.parametrize("kwargs, match", [
    ({"worlds": 0}, "world"), ({"top": 1}, "two"), ({"decisive": 1.5}, "decisive"),
    ({"horizon": 0}, "horizon"),
])
def test_the_parameters_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PairedRollouts(Stub(lambda options: np.ones(len(options)) / len(options)), **kwargs)


def test_every_decision_is_logged_with_what_it_was_made_from(monkeypatch):
    import json
    import hexset.rollout as rollout

    monkeypatch.setattr(rollout, "STEP_LIMIT", 40)
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=4, top=2, marks=(2, 4), rng=random.Random(4))
    game = a_road_or_not()
    decision = search.decide(game)
    assert search.decision_log == [decision]
    assert (decision.turn, decision.phase, decision.seat) == (game.turns, "MAIN", 0)
    assert len(decision.points) == 4
    assert len(decision.outcomes) == 4 and all(len(row) == 2 for row in decision.outcomes)
    assert decision.scores == tuple(np.mean(decision.outcomes, axis=0))
    assert decision.marks == (2, 4) and len(decision.mark_values) == 2
    assert all(len(table) == 4 and all(len(row) == 2 for row in table)
               for table in decision.mark_values)
    assert not np.isnan(np.asarray(decision.mark_values)).any()
    # Right after the candidate, before any playout, the stub reads hand sizes:
    # ending the turn (the policy's own move, first) keeps two cards the road spends.
    after = np.asarray(decision.after_values)
    assert after.shape == (4, 2) and (after[:, 1] < after[:, 0]).all()
    assert decision.steps > 0 and decision.cpu_seconds >= 0
    assert 0 < decision.mark_row_steps[0] <= decision.mark_row_steps[1] <= decision.row_steps
    line = json.loads(json.dumps(decision.to_json()))
    assert line["worlds"] == 4 and line["candidates"] == list(decision.candidates)
    assert [a[0] for a in line["actions"]] == ["END_TURN", "BUILD_ROAD"]
    assert len(line["mark_values"]) == 2 and len(line["outcomes"]) == 4
    assert len(line["after_values"]) == 4


def test_marks_change_no_decision(monkeypatch):
    import hexset.rollout as rollout

    monkeypatch.setattr(rollout, "STEP_LIMIT", 40)

    def decide(marks):
        rng = random.Random(9)
        search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=4,
                                marks=marks, rng=rng)
        decision = search.decide(a_road_or_not())
        return decision.chosen, decision.scores, decision.outcomes, rng.getstate()

    assert decide(()) == decide((2, 4, 8))
    search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=2, marks=(),
                            rng=random.Random(9))
    assert search.decide(a_road_or_not()).after_values == ()


def test_a_decisive_root_is_logged_without_worlds():
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.95))
    search = PairedRollouts(stub, worlds=8, decisive=0.9, rng=random.Random(1))
    search.decide(a_road_or_not())
    (logged,) = search.decision_log
    line = logged.to_json()
    assert line["worlds"] == 0 and line["top_prior"] == 0.95 and "outcomes" not in line


def test_marks_are_checked():
    for marks in ((0,), (4, 2), (2, 2)):
        with pytest.raises(ValueError, match="marks"):
            PairedRollouts(Stub(lambda options: np.ones(len(options)) / len(options)), marks=marks)


def favours_the_road(options):
    return favouring(ActionType.BUILD_ROAD, 0.6)(options)


def a_stub():
    """The pool tests' policy: a worker process loads its own."""
    return Stub(favours_the_road)


def never_loads():
    raise OSError("no such model")


def decided(pool, seed: int = 7):
    search = PairedRollouts(a_stub(), worlds=7, top=2, horizon=3, marks=(1, 2),
                            rng=random.Random(seed), pool=pool)
    d = search.decide(a_road_or_not())
    return (d.chosen, d.scores, d.outcomes, d.mark_values, d.after_values, d.steps, d.row_steps,
            d.mark_row_steps, d.truncated, search.truncated_rollouts, search.rng.getstate())


def test_a_pool_decides_as_one_process_does():
    from hexset.rollout import RolloutPool

    alone = decided(None)
    with RolloutPool(a_stub, workers=2) as pool:
        assert decided(pool) == alone
        assert not pool.broken


def test_shared_pools_keep_one_pool_a_file_and_retire_it_when_the_file_changes():
    from hexset.rollout import SharedPools

    alone = decided(None)
    pools = SharedPools()
    try:
        first = pools.get("/m/a.onnx", ("v1",), a_stub, 2)
        assert pools.get("/m/a.onnx", ("v1",), a_stub, 2) is first, "every seat of one file shares it"
        assert pools.get("/m/b.onnx", ("v1",), a_stub, 2) is not first
        replaced = pools.get("/m/a.onnx", ("v2",), a_stub, 2)
        assert replaced is not first and first.broken and not replaced.broken
        # A seat still holding the retired pool decides as it would have, here.
        assert decided(first) == decided(replaced) == alone
    finally:
        pools.close()
    assert replaced.broken


def persistent_batch(policy=None, marks=(2, 4, 8)):
    from hexset.rollout import RolloutBatch

    game = a_road_or_not(3)
    moves = (next(o for o in legal_actions(game) if o.type is ActionType.BUILD_ROAD),
             next(o for o in legal_actions(game) if o.type is ActionType.END_TURN))
    return RolloutBatch.from_tasks(policy or a_stub(), [(game, 19), (game, 29)],
                                   moves, 0, game.turns, marks)


def assert_same_lanes(a, b):
    np.testing.assert_array_equal(a.outcomes, b.outcomes)
    np.testing.assert_array_equal(a.after, b.after)
    np.testing.assert_array_equal(a.reads, b.reads)
    assert (a.steps, a.row_steps, a.mark_row_steps, a.truncated) == (
        b.steps, b.row_steps, b.mark_row_steps, b.truncated)


def test_resumed_copies_keep_their_prefix_marks_and_chance():
    batch = persistent_batch()
    first = batch.advance(2)
    frozen = first.reads.copy()
    prefix_work = first.row_steps
    early_counts = (batch.policy.acted, batch.policy.valued)
    batch.advance(4)
    assert (batch.policy.acted, batch.policy.valued) == early_counts
    final = batch.advance(8)
    assert_same_lanes(final, persistent_batch().advance(8))
    np.testing.assert_array_equal(first.reads, frozen)
    assert 0 < prefix_work < final.row_steps
    assert batch.policy.acted == final.row_steps
    acted, valued = batch.policy.acted, batch.policy.valued
    assert_same_lanes(batch.advance(8), final)
    assert batch.policy.acted == acted
    assert batch.policy.valued == valued


def test_selective_extension_keeps_paused_copies_untouched():
    batch = persistent_batch()
    batch.advance(2)
    paused = [(batch.games[i].turns, batch.counts[i]) for i in (0, 2)]
    part = batch.advance(8, [3, 1])
    assert paused == [(batch.games[i].turns, batch.counts[i]) for i in (0, 2)]
    reference = persistent_batch().advance(8, [3, 1])
    assert_same_lanes(part, reference)
    assert_same_lanes(batch.advance(8), persistent_batch().advance(8))


def test_resume_keeps_a_cumulative_step_limit_and_the_root_value_frame(monkeypatch):
    import hexset.rollout as rollout

    frames = []

    class Watching(Stub):
        def value_rows(self, rows):
            frames.extend(seat for _, seat in rows)
            return super().value_rows(rows)

    batch = persistent_batch(Watching(favours_the_road))
    early = batch.advance(2)
    monkeypatch.setattr(rollout, 'STEP_LIMIT', early.steps + 3)
    end = batch.advance(None)
    assert max(batch.counts) == early.steps + 3
    assert batch.policy.acted == end.row_steps
    assert frames and set(frames) == {0}
    assert_same_lanes(batch.advance(None), end)


def test_persistent_batch_checks_monotonic_horizons_and_distinct_copies():
    from hexset.rollout import RolloutBatch

    batch = persistent_batch()
    batch.advance(4, [0])
    with pytest.raises(ValueError, match='backwards'):
        batch.advance(2, [0])
    batch.advance(2, [1])
    for indices in ([0, 0], [-1], [4]):
        with pytest.raises(ValueError, match='indices'):
            batch.advance(8, indices)
    with pytest.raises(ValueError, match='positive'):
        batch.advance(0)
    with pytest.raises(ValueError, match='distinct game'):
        RolloutBatch(a_stub(), [batch.games[0]] * 2, 0, 0)
    empty = batch.advance(8, [])
    assert empty.row_steps == empty.steps == empty.truncated == 0


def test_new_worlds_can_be_added_without_replaying_existing_copies():
    batch = persistent_batch()
    batch.advance(2)
    prefix_counts = batch.counts.copy()
    game = a_road_or_not(3)
    moves = (next(o for o in legal_actions(game) if o.type is ActionType.BUILD_ROAD),
             next(o for o in legal_actions(game) if o.type is ActionType.END_TURN))
    added = batch.add_tasks([(game, 39)], moves)
    assert added == (4, 5)
    np.testing.assert_array_equal(batch.counts[:4], prefix_counts)
    assert not batch.counts[4:].any()
    full = batch.advance(8)
    from hexset.rollout import RolloutBatch
    direct = RolloutBatch.from_tasks(a_stub(), [(game, 19), (game, 29), (game, 39)],
                                     moves, 0, game.turns, (2, 4, 8)).advance(8)
    assert_same_lanes(full, direct)
    with pytest.raises(ValueError, match='distinct game'):
        batch.add([batch.games[0]])


def test_finished_copies_remain_exact_scores_without_policy_work():
    from hexset.rollout import RolloutBatch

    game = a_road_or_not()
    game.won_by = 0
    game.phase = Phase.GAME_OVER
    stub = a_stub()
    batch = RolloutBatch(stub, [game], 0, game.turns, (2, 4), after=True)
    first = batch.advance(2)
    assert_same_lanes(first, batch.advance(None))
    np.testing.assert_array_equal(first.outcomes, [1.])
    np.testing.assert_array_equal(first.reads, [[1.], [1.]])
    assert stub.acted == stub.valued == first.row_steps == first.truncated == 0


def test_a_dead_worker_costs_speed_not_the_decision():
    import os
    from concurrent.futures.process import BrokenProcessPool
    from hexset.rollout import RolloutPool

    alone = decided(None)
    with RolloutPool(a_stub, workers=2) as pool:
        with pytest.raises(BrokenProcessPool):
            pool._executor.submit(os._exit, 1).result()
        assert decided(pool) == alone
        assert pool.broken
        assert decided(pool) == alone


def test_a_pool_that_cannot_load_its_policy_fails_at_once():
    from hexset.rollout import RolloutPool

    with pytest.raises(RuntimeError, match="could not load"):
        RolloutPool(never_loads, workers=1)
    with pytest.raises(ValueError, match="at least one worker"):
        RolloutPool(a_stub, workers=0)


# --- the mixed strategy (`mix`) ----------------------------------------------------------------------------

def test_the_mixed_strategy_is_the_prior_regularised_towards_the_scores():
    from hexset.rollout import mixed_strategy

    scores, prior = np.array([0.50, 0.52, 0.40]), np.array([0.7, 0.2, 0.1])
    pi = mixed_strategy(scores, prior, 0.02)
    expected = prior * np.exp(scores / 0.02)
    assert np.allclose(pi, expected / expected.sum()) and np.isclose(pi.sum(), 1.0)
    # Cold, it plays the best score; hot, the prior.
    assert np.argmax(mixed_strategy(scores, prior, 1e-4)) == 1
    assert mixed_strategy(scores, prior, 1e-4)[1] > 1 - 1e-12
    assert np.allclose(mixed_strategy(scores, prior, 1e6), prior)
    # Its expected shortfall from the best score is bounded by mix * ln(1 / prior(best)).
    rng = np.random.default_rng(20260909)
    for _ in range(200):
        s, p, t = rng.random(6), rng.dirichlet(np.ones(6)), float(rng.choice([0.005, 0.02, 0.1]))
        shortfall = float(mixed_strategy(s, p, t) @ (s.max() - s))
        assert shortfall <= t * np.log(1 / p[np.argmax(s)]) + 1e-12
    # A candidate the prior gives nothing is never played; a prior of nothing counts them alike.
    assert mixed_strategy(scores, np.array([0.5, 0.0, 0.5]), 0.02)[1] == 0.0
    assert np.allclose(mixed_strategy(np.zeros(3), np.zeros(3), 0.02), 1 / 3)
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="mix"):
            mixed_strategy(scores, prior, bad)


def test_the_anchor_flattens_the_prior_the_mix_is_drawn_towards():
    from hexset.rollout import mixed_strategy

    scores, prior = np.array([0.50, 0.505, 0.40]), np.array([0.98, 0.01, 0.01])
    assert np.allclose(mixed_strategy(scores, prior, 0.02, 1.0), mixed_strategy(scores, prior, 0.02))
    half = prior ** 0.5 * np.exp(scores / 0.02)
    assert np.allclose(mixed_strategy(scores, prior, 0.02, 0.5), half / half.sum())
    flat = np.exp(scores / 0.02)
    assert np.allclose(mixed_strategy(scores, prior, 0.02, 0.0), flat / flat.sum())
    # Flattening the anchor mixes more among the close pair.
    assert mixed_strategy(scores, prior, 0.02, 0.0)[1] > mixed_strategy(scores, prior, 0.02, 0.5)[1] \
        > mixed_strategy(scores, prior, 0.02)[1]
    # The bound holds against the anchor's own distribution.
    rng = np.random.default_rng(20260910)
    for _ in range(200):
        s, p, t, b = rng.random(6), rng.dirichlet(np.ones(6)), 0.02, float(rng.choice([0.0, 0.5, 1.0]))
        q = p ** b / (p ** b).sum()
        assert float(mixed_strategy(s, p, t, b) @ (s.max() - s)) <= t * np.log(1 / q[np.argmax(s)]) + 1e-12
    for bad in (-0.5, float("nan")):
        with pytest.raises(ValueError, match="anchor"):
            mixed_strategy(scores, prior, 0.02, bad)
        with pytest.raises(ValueError, match="anchor"):
            PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), mix=0.02, anchor=bad)


def logged(decision):
    """A decision as its log line reads it, less the time it took."""
    line = decision.to_json()
    line.pop("cpu_seconds")
    return line


def mixed_decision(mix, seed: int = 9, game=None, **kwargs):
    rng = random.Random(seed)
    search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=6, top=3, horizon=2,
                            marks=(2,), rng=rng, mix=mix, **kwargs)
    return search, search.decide(a_road_or_not() if game is None else game)


def test_mix_zero_is_the_greedy_search():
    _, plain = mixed_decision(0.0)
    rng = random.Random(9)
    search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=6, top=3, horizon=2,
                            marks=(2,), rng=rng)
    greedy = search.decide(a_road_or_not())
    assert logged(plain) == logged(greedy) and plain.chosen == greedy.chosen
    assert plain.mix == () and plain.greedy is None and "mix" not in plain.to_json()
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mix"):
            PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), mix=bad)


def test_a_mixed_search_plays_out_the_greedy_worlds_and_logs_its_draw():
    """The draw comes off a stream of its own: the worlds, scores and the
    rollouts' stream are the greedy search's, and only `chosen` may differ."""
    greedy_search, greedy = mixed_decision(0.0)
    import json
    for mix in (0.01, 0.5, 100.0):
        search, decision = mixed_decision(mix)
        assert (decision.scores, decision.outcomes, decision.mark_values, decision.candidates) == (
            greedy.scores, greedy.outcomes, greedy.mark_values, greedy.candidates)
        assert search.rng.getstate() == greedy_search.rng.getstate()
        assert decision.greedy == greedy.chosen and decision.chosen in decision.candidates
        assert np.allclose(decision.mix, mixed_strategy_of(decision, mix)) and np.isclose(sum(decision.mix), 1)
        flat = mixed_decision(mix, anchor=0.5)[1]
        assert flat.scores == decision.scores and np.allclose(flat.mix, mixed_strategy_of(flat, mix, 0.5))
        line = json.loads(json.dumps(decision.to_json()))
        assert line["greedy"] == greedy.chosen and len(line["mix"]) == len(line["candidates"])
        assert line["chosen"] == decision.chosen
        # Reproducible from the seed.
        assert logged(mixed_decision(mix)[1]) == logged(decision)


def mixed_strategy_of(decision, mix, anchor=1.0):
    from hexset.rollout import mixed_strategy

    return mixed_strategy(decision.scores, decision.prior[list(decision.candidates)], mix, anchor)


def test_mixed_draws_follow_their_probabilities_and_never_touch_another_searchers():
    """Hot, the draws follow the logged probabilities; a searcher's draws
    are its own, whatever another seat decides between them (one at a time
    or interleaved, as a batched table runs them)."""
    def run(seed, positions, other=None):
        search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=2, top=3, horizon=1,
                                marks=(), rng=random.Random(seed), mix=100.0)
        out = []
        for p in positions:
            out.append(search.decide(a_road_or_not(p)))
            if other is not None:
                other.decide(a_road_or_not(p + 1000))
        return out

    positions = list(range(40))
    alone = run(3, positions)
    other = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=2, top=3, horizon=1,
                           marks=(), rng=random.Random(4), mix=100.0)
    assert [logged(d) for d in run(3, positions, other)] == [logged(d) for d in alone]
    drawn = [d.candidates.index(d.chosen) for d in alone]
    assert len(set(drawn)) == 3, "every candidate is drawn at a temperature that plays the prior"
    assert any(d.chosen != d.greedy for d in alone)
    # Over many decisions the first candidate's share sits near its mean probability.
    share = np.mean([j == 0 for j in drawn])
    assert abs(share - np.mean([d.mix[0] for d in alone])) < 0.25


def test_a_pool_draws_as_one_process_does():
    from hexset.rollout import RolloutPool

    def drawn(pool):
        search = PairedRollouts(a_stub(), worlds=7, top=3, horizon=3, marks=(1, 2),
                                rng=random.Random(11), pool=pool, mix=0.05)
        return [search.decide(a_road_or_not(seed)) for seed in range(3)]

    alone = drawn(None)
    with RolloutPool(a_stub, workers=2) as pool:
        assert [(d.chosen, d.mix, d.scores) for d in drawn(pool)] == [(d.chosen, d.mix, d.scores) for d in alone]


def test_a_root_decided_without_playouts_draws_nothing():
    search = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.95)), worlds=8, decisive=0.9,
                            rng=random.Random(1), mix=1.0)
    state = search._draws.getstate()
    decision = search.decide(a_road_or_not())
    assert decision.worlds == 0 and decision.mix == () and search._draws.getstate() == state


# --- the deterministic search (`det`) -----------------------------------------------------------------------

def det_search(seed: int, **kwargs):
    kwargs = {"worlds": 6, "top": 3, "horizon": 2, "marks": (2,), "det": True, **kwargs}
    return PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), rng=random.Random(seed), **kwargs)


def searched(decision):
    return decision.chosen, decision.scores, decision.outcomes, decision.mark_values, decision.det_key


def test_a_deterministic_search_searches_a_visible_position_alike_on_any_searcher():
    """Fresh searchers on different seeds search one position on the same
    worlds and seeds and play the same move; the searcher's own stream is
    left untouched, so it is no part of the decision."""
    first, other = det_search(1), det_search(2)
    before = first.rng.getstate()
    a, b = first.decide(a_road_or_not()), other.decide(a_road_or_not())
    assert a.worlds == 6 and a.det_key and a.salt == 0
    assert searched(a) == searched(b) and logged(a) == logged(b)
    assert first.rng.getstate() == before
    assert first.decision_key(a_road_or_not(), 0).hex() == a.det_key
    # Again on the same searcher, after other decisions: the same.
    first.decide(a_road_or_not(5))
    assert searched(first.decide(a_road_or_not())) == searched(a)
    line = logged(a)
    assert line["det_key"] == a.det_key and line["salt"] == 0


def test_the_salt_reseeds_every_decision():
    a, b = det_search(1).decide(a_road_or_not()), det_search(1, salt=7).decide(a_road_or_not())
    assert a.det_key != b.det_key and a.outcomes != b.outcomes and b.salt == 7
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="salt"):
            det_search(1, salt=bad)


def redealt(game, seat, seed):
    """`game` with every other seat's hand, development cards and the deck
    redealt among themselves (sizes kept): what `seat` cannot see moves."""
    state = game._state
    others = [s for s in range(state.num_players) if s != seat]
    rng = random.Random(seed)
    cards = [r for s in others for r, n in enumerate(state.hands[s]) for _ in range(n)]
    rng.shuffle(cards)
    sizes = [sum(state.hands[s]) for s in others]
    for s, size in zip(others, sizes):
        hand = [0] * len(state.hands[s])
        for r in cards[:size]:
            hand[r] += 1
        state.hands[s], cards = hand, cards[size:]
    rng.shuffle(state.deck)
    return game


def test_hidden_information_moves_neither_the_key_nor_the_search():
    from hexset.game import observed_digest

    plain = a_road_or_not()
    hidden = redealt(a_road_or_not(), 0, 3)
    assert [plain._state.hands[s] for s in (1, 2, 3)] != [hidden._state.hands[s] for s in (1, 2, 3)] \
        or plain._state.deck != hidden._state.deck, "the redeal moved nothing"
    assert plain._state.hands[0] == hidden._state.hands[0]
    assert observed_digest(plain, 0) == observed_digest(hidden, 0)
    assert searched(det_search(1).decide(plain)) == searched(det_search(2).decide(hidden))


def test_what_the_seat_can_see_moves_the_key():
    from hexset.game import observed_digest

    game = a_road_or_not()
    built = a_road_or_not()
    apply(built, next(o for o in legal_actions(built) if o.type is ActionType.BUILD_ROAD))
    other_seed = a_road_or_not(1)
    own_hand = a_road_or_not()
    give(own_hand._state, 0, Resource.ORE, 1)
    digests = {observed_digest(g, 0) for g in (game, built, other_seed, own_hand)}
    assert len(digests) == 4
    assert observed_digest(game, 0) != observed_digest(game, 1)


def test_a_deterministic_mix_draws_alike_on_any_searcher():
    runs = [[det_search(seed, mix=0.5).decide(a_road_or_not(p)) for p in range(12)] for seed in (1, 2)]
    assert [logged(d) for d in runs[0]] == [logged(d) for d in runs[1]]
    assert all(d.mix and d.det_key for d in runs[0])
    assert len({d.candidates.index(d.chosen) for d in runs[0]}) > 1, "every draw fell on one candidate"


def test_a_deterministic_search_draws_alike_in_another_process_and_in_a_pool():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from hexset.rollout import RolloutPool

    alone = det_search(1, mix=0.5).decide(a_road_or_not())
    with RolloutPool(a_stub, workers=2) as pool:
        pooled = PairedRollouts(a_stub(), worlds=6, top=3, horizon=2, marks=(2,), det=True, mix=0.5,
                                rng=random.Random(9), pool=pool).decide(a_road_or_not())
        assert not pool.broken
    assert searched(pooled) == searched(alone) and pooled.mix == alone.mix
    here = Path(__file__).resolve().parent
    script = (
        "import json, random, sys; sys.path[:0] = [sys.argv[1]]\n"
        "from test_rollout import det_search, a_road_or_not, logged\n"
        "print(json.dumps(logged(det_search(3, mix=0.5).decide(a_road_or_not()))))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", script, str(here)], env=env, capture_output=True, text=True,
                         check=True, cwd=here)
    assert json.loads(out.stdout) == json.loads(json.dumps(logged(alone)))


def test_det_off_draws_from_the_searcher_as_before():
    rng = random.Random(9)
    plain = PairedRollouts(Stub(favouring(ActionType.BUILD_ROAD, 0.6)), worlds=6, top=3, horizon=2,
                           marks=(2,), rng=rng).decide(a_road_or_not())
    off = det_search(9, det=False).decide(a_road_or_not())
    assert searched(off) == searched(plain) and off.det_key == "" and off.salt is None
    assert "det_key" not in logged(off)


def test_a_position_rebuilt_from_its_record_keeps_its_digest(tmp_path):
    """Every decision of a recorded game, walked again from the record, sees
    the digest it was played with: what `det` seeds by survives a rebuild,
    whatever cap the table played under."""
    from hexset.bots import RandomBot
    from hexset.game import observed_digest
    from hexset.record import advance, moves, open_record, record_game, shown_by_step
    from hexset.rules import GAME_TYPES

    played = []

    class Watching(RandomBot):
        def choose(self, game):
            played.append(observed_digest(game, to_move(game)))
            return super().choose(game)

    board = random_base_board(random.Random(6))
    record = record_game([Watching(random.Random(i)) for i in range(2)], board, 98000, turn_cap=60,
                         game_type=GAME_TYPES["duel-variant"], journal=tmp_path / "games.jsonl")
    game, shown, again = open_record(record), shown_by_step(record), []
    for step, (actor, action, trades) in enumerate(moves(record)):
        again.append(observed_digest(game, to_move(game) if actor is None else actor))
        advance(game, action, trades, actor, shown.get(step, ()))
    assert len(played) == len(again) > 100 and played == again
