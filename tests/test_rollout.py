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


def test_a_decisive_root_plays_its_favourite_without_a_playout():
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.95))
    search = PairedRollouts(stub, worlds=8, decisive=0.9, rng=random.Random(1))
    decision = search.decide(a_road_or_not())
    assert decision.worlds == 0 and decision.scores == ()
    assert decision.options[decision.chosen].type is ActionType.BUILD_ROAD
    assert stub.acted == 0


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
    turn in every world, which is what the stub's values count."""
    stub = Stub(favouring(ActionType.BUILD_ROAD, 0.6))
    search = PairedRollouts(stub, worlds=16, top=2, horizon=1, marks=(), rng=random.Random(3))
    decision = search.decide(a_road_or_not())
    played = [decision.options[i].type for i in decision.candidates]
    assert played == [ActionType.BUILD_ROAD, ActionType.END_TURN]
    assert decision.worlds == 16 and len(decision.scores) == 2
    assert decision.scores[1] > decision.scores[0]
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
    # building the road spends two cards, ending the turn keeps them.
    after = np.asarray(decision.after_values)
    assert after.shape == (4, 2) and (after[:, 0] < after[:, 1]).all()
    assert decision.steps > 0 and decision.cpu_seconds >= 0
    assert 0 < decision.mark_row_steps[0] <= decision.mark_row_steps[1] <= decision.row_steps
    line = json.loads(json.dumps(decision.to_json()))
    assert line["worlds"] == 4 and line["candidates"] == list(decision.candidates)
    assert [a[0] for a in line["actions"]] == ["BUILD_ROAD", "END_TURN"]
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
