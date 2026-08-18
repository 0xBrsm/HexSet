from __future__ import annotations

import random
from typing import Iterator, Sequence

import numpy as np
import pytest

from catan.actions import apply, legal_mask, space_for
from catan.encoding import HAND_SCALE, encode
from catan.board.terrain import NUM_RESOURCES
from catan.game import Game, to_move
from catan.selfplay import (
    Choice,
    Collector,
    Episode,
    RandomPolicy,
    Request,
    Transition,
    new_game,
)


def replay(episode: Episode) -> Iterator[tuple[Game, Transition]]:
    """Rebuild the game from its seed and walk it alongside the recorded stream.

    The whole point of the collector is that a lane's interleaved actions come
    back split by seat, so the check that matters is whether the engine agrees:
    at every recorded step, is the seat the transition is filed under actually
    the seat the engine was asking?
    """
    game = new_game(episode.seed, episode.index, episode.players)
    for transition in episode.stream():
        yield game, transition
        apply(game, transition.action)


class Counting:
    """Wraps a policy and records the shape of every batch it was given."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.batches: list[int] = []

    def act(self, requests: Sequence[Request]) -> Sequence[Choice]:
        self.batches.append(len(requests))
        return self.inner.act(requests)


class First:
    """Takes the first legal action, and stamps the extras PPO will want kept."""

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        return [
            Choice(
                action=request.options[0],
                log_prob=-float(request.seat),
                value=tuple(float(request.seat + i) for i in range(4)),
            )
            for request in requests
        ]


def test_one_batch_per_tick_covering_every_lane():
    """The reason this module exists: a forward costs 1.5 ms plus 25 µs a
    position, so a tick must be one call carrying every lane."""
    policy = Counting(RandomPolicy(random.Random(0)))
    collector = Collector(policy, lanes=5, seed=1, action_cap=60)
    collector.run(30)

    assert len(policy.batches) == collector.ticks == 30
    assert set(policy.batches) == {5}


def test_seats_are_demultiplexed_against_a_replay():
    collector = Collector(RandomPolicy(random.Random(3)), lanes=1, seed=11)
    episode = collector.collect(1)[0]
    stream = episode.stream()

    off_turn = 0
    for step, (game, transition) in enumerate(replay(episode)):
        assert transition.step == step
        assert to_move(game) == transition.seat
        if game.current_player != transition.seat:
            off_turn += 1
        expected = encode(game, transition.seat)
        assert np.array_equal(expected.hexes, transition.observation.hexes)
        assert np.array_equal(expected.vertices, transition.observation.vertices)
        assert np.array_equal(expected.edges, transition.observation.edges)
        assert np.array_equal(expected.globals, transition.observation.globals)

    # None of the above means anything unless the stream really was interleaved
    # and every seat really was demultiplexed out of it.
    assert len(stream) == len(episode) == episode.outcome.actions
    assert all(len(seat) > 1 for seat in episode.trajectories)
    changes = sum(1 for a, b in zip(stream, stream[1:]) if a.seat != b.seat)
    assert changes > episode.players
    # And the hard case actually happened: decisions taken by a seat whose turn
    # it is not, where filing by `current_player` would have looked fine.
    assert off_turn > 0
    for seat in episode.trajectories:
        assert max(b.step - a.step for a, b in zip(seat, seat[1:])) > 1


def test_a_seat_asked_off_turn_is_encoded_from_its_own_perspective():
    """Discarding on a seven and answering an offer belong to somebody other
    than the player whose turn it is, and `encode` defaults to the turn holder.
    """
    collector = Collector(RandomPolicy(random.Random(6)), lanes=1, seed=2)
    episode = collector.collect(1)[0]

    checked = 0
    for game, transition in replay(episode):
        if game.current_player == transition.seat:
            continue
        own_hand = transition.observation.globals[:NUM_RESOURCES] * HAND_SCALE
        assert own_hand == pytest.approx(game.state.hands[transition.seat], abs=1e-4)
        checked += 1
    assert checked > 0


def test_the_mask_marks_exactly_the_legal_actions():
    collector = Collector(RandomPolicy(random.Random(8)), lanes=1, seed=4, action_cap=150)
    episode = collector.collect(1)[0]
    space = space_for(new_game(episode.seed, episode.index, episode.players))

    wide = 0
    for game, transition in replay(episode):
        assert np.array_equal(transition.mask, np.array(legal_mask(game, space)))
        assert transition.mask[transition.index]
        wide += transition.mask.sum() > 1
    # A mask with one bit set is trivially right, so insist most were choices.
    assert wide > len(episode) // 2


def test_a_finished_lane_is_replaced_without_stalling_the_others():
    ticks = 3000
    collector = Collector(RandomPolicy(random.Random(2)), lanes=3, seed=13)
    episodes = collector.run(ticks)

    assert collector.games == len(episodes) > 0
    # Every tick moved every lane, whatever any other lane was doing.
    recorded = sum(len(e) for e in episodes) + sum(collector.pending())
    assert recorded == collector.steps == 3 * ticks
    # Lanes really were at different points, so the replacement was not a
    # lockstep restart of the whole batch.
    assert len(set(collector.pending())) > 1
    assert len({e.index for e in episodes}) == len(episodes)


def test_the_action_cap_truncates_rather_than_running_forever():
    collector = Collector(RandomPolicy(random.Random(4)), lanes=2, seed=9, action_cap=25)
    episodes = collector.run(75)

    assert len(episodes) == 6
    for episode in episodes:
        assert episode.outcome.truncated
        assert episode.outcome.winner is None
        assert episode.outcome.actions == len(episode) == 25


def test_an_outcome_carries_both_candidate_rewards():
    """Reward design is still open, so a collector reports the terminal facts
    and leaves the scalarisation to whoever consumes it."""
    collector = Collector(RandomPolicy(random.Random(5)), lanes=2, seed=3)
    episodes = collector.collect(2)

    for episode in episodes:
        outcome = episode.outcome
        assert not outcome.truncated
        assert outcome.winner is not None
        assert len(outcome.points) == episode.players
        assert outcome.points[outcome.winner] >= 10
        assert outcome.points[outcome.winner] == max(outcome.points)
        assert outcome.turns > 0


def test_the_policys_extras_are_recorded_against_the_seat_that_acted():
    collector = Collector(First(), lanes=2, seed=17, action_cap=200)
    episodes = collector.run(200)

    assert episodes
    seen = set()
    for episode in episodes:
        for seat, trajectory in enumerate(episode.trajectories):
            for transition in trajectory:
                assert transition.seat == seat
                assert transition.log_prob == -float(seat)
                assert transition.value == tuple(float(seat + i) for i in range(4))
                seen.add(seat)
    assert len(seen) > 1


def test_the_same_seed_collects_the_same_games():
    def once() -> list[tuple]:
        collector = Collector(RandomPolicy(random.Random(21)), lanes=2, seed=19)
        return [
            (e.index, e.outcome, tuple(t.index for t in e.stream()))
            for e in collector.collect(2)
        ]

    assert once() == once()


def test_a_bounded_collector_deals_exactly_the_cohort_and_stops():
    """The point is the count, not the filter.

    An evaluation wants a fixed set of game indices; the naive way to get one is
    to let the collector refill freed lanes and discard the replacements, which
    plays every one of them in full first. `deal` stops them being started.
    """
    collector = Collector(RandomPolicy(random.Random(3)), lanes=4, seed=5, deal=6)
    episodes = collector.drain()

    assert sorted(e.index for e in episodes) == [0, 1, 2, 3, 4, 5]
    assert collector.games_started() == 6
    assert not collector.running
    assert collector.tick() == []


def test_asking_a_bounded_collector_for_more_than_it_has_fails_loudly():
    """Otherwise `collect` spins on empty ticks instead of blocking on a game."""
    collector = Collector(RandomPolicy(random.Random(3)), lanes=2, seed=5, deal=3)
    with pytest.raises(ValueError, match="4 games wanted, 3 left"):
        collector.collect(4)


def test_an_unbounded_collector_refuses_to_drain():
    collector = Collector(RandomPolicy(random.Random(3)), lanes=2, seed=5)
    with pytest.raises(ValueError, match="never drains"):
        collector.drain()


def test_a_cohort_smaller_than_the_lane_count_leaves_lanes_empty():
    collector = Collector(RandomPolicy(random.Random(3)), lanes=8, seed=5, deal=2)
    assert len(list(collector.in_flight())) == 2
    assert len(collector.drain()) == 2


def test_a_bound_does_not_change_the_games_themselves():
    """Bounding the collector must not perturb the cohort it does play.

    Under a *stateless* policy a game is a pure function of the seed and its
    index, so the bounded and unbounded runs have to agree game for game. They
    do not agree under `RandomPolicy`, which draws every lane from one shared
    stream — the discarded replacement games consume draws and shift everything
    after them. That is a property of the policy, not of the bound, but it is
    the reason an eval must fix its cohort by index rather than by arrival.
    """

    class First:
        def act(self, requests):
            return [Choice(action=r.options[0]) for r in requests]

    def cohort(deal: int | None) -> list[tuple]:
        collector = Collector(First(), lanes=2, seed=19, deal=deal, action_cap=300)
        played = collector.drain() if deal else collector.collect(4)
        return sorted(
            (e.index, e.outcome, tuple(t.index for t in e.stream())) for e in played
        )

    bounded = cohort(4)
    assert [i for i, _, _ in bounded] == [0, 1, 2, 3]
    assert bounded == [e for e in cohort(None) if e[0] < 4]


def test_a_policy_that_answers_the_wrong_number_of_requests_is_rejected():
    class Short:
        def act(self, requests):
            return [Choice(action=requests[0].options[0])]

    collector = Collector(Short(), lanes=2, seed=0)
    with pytest.raises(ValueError, match="answered 1 of 2"):
        collector.tick()


def test_the_offer_budget_clears_proposals_from_the_mask():
    """A budget the policy cannot even see the slots for.

    Filtering has to happen before the mask is built. If a forbidden action
    stayed set, the policy could be trained to want something the engine will
    then refuse, and the shorter horizon the budget was adopted for would not
    materialise either.
    """
    budget = 2
    collector = Collector(
        RandomPolicy(random.Random(12)), lanes=2, seed=6, max_offers=budget
    )
    space = collector.space
    proposals = [
        index
        for action, index in ((a, space.index(a)) for a in _every_proposal(collector))
    ]
    assert proposals, "expected the space to contain proposal slots"

    spent = 0
    for episode in collector.collect(2):
        for game, transition in replay(episode):
            assert game.offers_made <= budget
            if game.offers_made >= budget:
                spent += 1
                assert not transition.mask[proposals].any()
    assert spent > 0, "no turn ever reached the budget, so nothing was tested"


def _every_proposal(collector: Collector) -> list:
    from catan.actions import Action, ActionType
    from catan.board.terrain import NUM_RESOURCES

    out = []
    for given in range(NUM_RESOURCES):
        for wanted in range(NUM_RESOURCES):
            if given == wanted:
                continue
            out.append(
                Action(
                    ActionType.PROPOSE_TRADE,
                    give=tuple(1 if r == given else 0 for r in range(NUM_RESOURCES)),
                    want=tuple(1 if r == wanted else 0 for r in range(NUM_RESOURCES)),
                )
            )
    return out


def test_a_cast_lane_routes_each_seat_and_records_only_the_learner():
    learner = Counting(RandomPolicy(random.Random(0)))
    opponent = Counting(RandomPolicy(random.Random(1)))
    collector = Collector(
        learner,
        lanes=2,
        seed=5,
        action_cap=900,
        opponents=[opponent],
        caster=lambda index: (0, 1, 0, 1),
    )
    episode = collector.collect(1)[0]

    assert episode.cast == (0, 1, 0, 1)
    # The learner's seats have trajectories; the opponent's are empty, which is
    # what keeps its decisions out of `catan.ppo.assemble` without a filter.
    assert all(episode.trajectories[seat] for seat in (0, 2))
    assert all(not episode.trajectories[seat] for seat in (1, 3))
    # The game itself still ran through every seat.
    assert len(episode) < episode.outcome.actions
    # Each policy was asked at most once per tick — the batching survived.
    assert len(learner.batches) <= collector.ticks
    assert len(opponent.batches) <= collector.ticks
    assert opponent.batches, "the opponent was never consulted"


def test_the_cast_is_taken_from_the_game_index_not_the_lane():
    def swaps(index: int) -> tuple[int, ...]:
        return (0, 1, 0, 1) if index % 2 == 0 else (1, 0, 1, 0)

    collector = Collector(
        RandomPolicy(random.Random(2)),
        lanes=3,
        seed=9,
        action_cap=700,
        opponents=[RandomPolicy(random.Random(3))],
        caster=swaps,
    )
    episodes = collector.collect(4)
    assert {e.index % 2 for e in episodes} == {0, 1}, "both parities must appear"
    for episode in episodes:
        assert episode.cast == swaps(episode.index)
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_a_caster_without_opponents_is_rejected():
    with pytest.raises(ValueError):
        Collector(RandomPolicy(), lanes=1, caster=lambda index: (0, 0, 0, 0))


def test_a_cast_reaching_past_the_opponents_fails_loudly():
    with pytest.raises(ValueError):
        Collector(
            RandomPolicy(),
            lanes=1,
            opponents=[RandomPolicy()],
            caster=lambda index: (0, 2, 0, 2),
        )


def test_a_bot_policy_spawns_one_bot_per_board_and_reuses_it():
    from catan.bots import RandomBot
    from catan.selfplay import BotPolicy

    spawned = []

    def spawn(board):
        spawned.append(board)
        return RandomBot(random.Random(4))

    collector = Collector(
        RandomPolicy(random.Random(5)),
        lanes=2,
        seed=13,
        opponents=[BotPolicy(spawn)],
        caster=lambda index: (0, 1, 0, 1),
    )
    collector.run(40)

    # Two lanes, two boards, two bots — and no respawn on any later request.
    assert len(spawned) == 2
    assert len({id(board) for board in spawned}) == 2


def test_a_bot_policy_over_capacity_respawns_rather_than_growing():
    from catan.bots import RandomBot
    from catan.selfplay import BotPolicy

    spawned = []

    def spawn(board):
        spawned.append(board)
        return RandomBot(random.Random(6))

    policy = BotPolicy(spawn, capacity=1)
    collector = Collector(
        RandomPolicy(random.Random(7)),
        lanes=2,
        seed=17,
        opponents=[policy],
        caster=lambda index: (0, 1, 0, 1),
    )
    collector.run(60)

    # Room for one bot serving two live boards: the cache must not grow past
    # its capacity, and the cost shows up as respawns rather than wrong bots.
    assert len(policy._bots) == 1
    assert len({id(board) for board in spawned}) == 2
    assert len(spawned) > 2


def test_a_strided_collector_deals_every_kth_index():
    collector = Collector(
        RandomPolicy(random.Random(1)),
        lanes=2,
        seed=7,
        action_cap=500,
        first_game=3,
        stride=4,
        deal=4,
    )
    episodes = collector.drain()

    # Worker w of K takes first_game = base + w, stride = K: the indices are
    # base + w mod K, so two workers' sets are disjoint by construction.
    assert sorted(e.index for e in episodes) == [3, 7, 11, 15]
    assert collector.games_started() == 3 + 4 * 4


def test_a_cohort_returns_the_block_it_dealt_and_ends_with_empty_lanes():
    """The on-policy guarantee, which `collect` does not give.

    A PPO iteration wants every position in its batch played under one set of
    weights. That holds exactly when the collector starts empty, deals a known
    block, and leaves nothing behind for the next iteration to inherit.
    """
    collector = Collector(
        RandomPolicy(random.Random(1)), lanes=8, fill=False, seed=0, max_offers=3
    )

    first = collector.cohort(16)
    assert {episode.index for episode in first} == set(range(16))
    assert list(collector.in_flight()) == []

    second = collector.cohort(16)
    assert {episode.index for episode in second} == set(range(16, 32))
    assert list(collector.in_flight()) == []


def test_a_cohort_larger_than_the_lane_count_refills_until_it_is_dealt_out():
    """Lanes are the concurrency, not the cohort.

    Keeping them independent is what lets the inference batch be chosen for
    throughput without deciding how many positions an update trains on.
    """
    collector = Collector(
        RandomPolicy(random.Random(4)), lanes=4, fill=False, seed=0, max_offers=3
    )
    assert {episode.index for episode in collector.cohort(12)} == set(range(12))
    assert list(collector.in_flight()) == []


def test_streaming_collection_returns_replacements_while_older_games_run_on():
    """The defect `cohort` removes, pinned so `stream` cannot quietly become it.

    `collect` refills a lane the moment its game ends, so a replacement dealt
    late can finish and enter the batch while a longer game dealt earlier is
    still being played. The PPO batch inherited both consequences for five
    runs: it selects for short games, and the unfinished lanes carry across the
    learner's weight sync into the next iteration's data.
    """
    collector = Collector(RandomPolicy(random.Random(1)), lanes=8, seed=0, max_offers=3)

    returned = {episode.index for episode in collector.collect(16)}

    assert len(returned) == 16
    assert returned != set(range(16)), "no replacement outran an older game"
    assert len(list(collector.in_flight())) == 8


def test_a_cohort_collector_must_start_empty():
    collector = Collector(RandomPolicy(random.Random(1)), lanes=4, seed=0)
    with pytest.raises(RuntimeError, match="still hold games"):
        collector.cohort(4)


def test_a_collector_bounded_at_build_time_refuses_a_second_cohort():
    collector = Collector(RandomPolicy(random.Random(1)), lanes=2, seed=0, deal=2)
    with pytest.raises(ValueError, match="one cohort at build time"):
        collector.cohort(2)
