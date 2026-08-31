from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.collect import ParallelCollector, WorkerSpec  # noqa: E402


def spec(worker: int, workers: int, **overrides) -> WorkerSpec:
    base = dict(
        seed=5,
        players=4,
        lanes=2,
        action_cap=500,
        max_offers=3,
        first_game=worker,
        stride=workers,
        width=8,
        rounds=1,
        torch_seed=1000 + worker,
        mix=(),
        parent="",
    )
    base.update(overrides)
    return WorkerSpec(**base)


def test_workers_deal_disjoint_strided_indices_and_ship_valid_episodes():
    collector = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        episodes = collector.collect(4)

        assert len(episodes) == 4
        indices = sorted(e.index for e in episodes)
        assert len(set(indices)) == 4, f"an index was dealt twice: {indices}"
        # Both workers contributed: the two stride residues both appear.
        assert {i % 2 for i in indices} == {0, 1}
        for episode in episodes:
            assert len(episode) > 0
            assert episode.outcome.actions > 0
        assert collector.games == 4
    finally:
        collector.close()


def test_collection_can_be_started_then_finished_around_other_work():
    collector = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        collector.start_collect(4)
        with pytest.raises(RuntimeError, match="already in flight"):
            collector.start_collect(2)

        # Production uses this window for the preceding batch's GPU update.
        assert collector.games == 0
        episodes = collector.finish_collect()

        assert len(episodes) == 4
        assert collector.games == 4
        assert collector.last_collect_seconds > 0.0
        with pytest.raises(RuntimeError, match="nothing in flight"):
            collector.finish_collect()
    finally:
        collector.close()


def test_a_parallel_resume_never_redeals_a_seen_index():
    first = ParallelCollector([spec(0, 2), spec(1, 2)])
    try:
        seen = {e.index for e in first.collect(4)}
        base = first.games_started()
    finally:
        first.close()

    # The resume rule: base is the max over worker counters, so indices the
    # slow worker never reached are skipped — unused seeds, never replays.
    assert base > max(seen)
    second = ParallelCollector([spec(base + 0, 2), spec(base + 1, 2)])
    try:
        fresh = {e.index for e in second.collect(2)}
    finally:
        second.close()
    assert not (seen & fresh)
    assert min(fresh) >= base


def test_workers_cast_mix_opponents_and_record_only_the_learner():
    collector = ParallelCollector(
        [
            spec(0, 2, mix=(("greedy", 1.0),)),
            spec(1, 2, mix=(("greedy", 1.0),)),
        ]
    )
    try:
        episodes = collector.collect(2)
    finally:
        collector.close()

    for episode in episodes:
        assert any(pid == 1 for pid in episode.cast)
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)


def test_sync_ships_weights_the_workers_actually_load():
    from catan.actions import build_space
    from catan.board.board import random_base_board
    from catan.encoding import static_graph
    from catan.model import CatanNet, ModelConfig

    board = random_base_board(random.Random(5))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, 4
    )
    net = CatanNet(space, static_graph(topology), 4, ModelConfig(width=8, rounds=1))

    collector = ParallelCollector([spec(0, 1, lanes=2)])
    try:
        collector.sync(net)
        episodes = collector.collect(1)
        assert episodes and len(episodes[0]) > 0
    finally:
        collector.close()


def test_a_searched_worker_returns_episodes_whose_transitions_carry_targets():
    """The whole point of sharding the searched path: the corpus must still be
    distillable, which means every transition needs its `Target` to survive the
    pipe as well as the search."""
    from catan.collect import ParallelCollector, WorkerSpec
    from catan.expert import Target

    collector = ParallelCollector(
        [
            WorkerSpec(
                seed=5,
                players=4,
                lanes=2,
                action_cap=600,
                max_offers=3,
                first_game=0,
                stride=1,
                width=16,
                rounds=1,
                torch_seed=5,
                simulations=8,
                wave=4,
            )
        ]
    )
    try:
        episodes = collector.collect(1)
    finally:
        collector.close()
    assert episodes
    targets = [
        t.aux
        for e in episodes
        for traj in e.trajectories
        for t in traj
        if t.aux is not None
    ]
    assert targets
    assert all(isinstance(t, Target) for t in targets)
    # The prior rides along too, or `contested_only` has nothing to filter on.
    searched = [t for t in targets if len(t.options) > 1]
    assert searched and all(t.prior is not None for t in searched)


def test_a_worker_with_no_simulations_is_the_plain_policy_path():
    from catan.collect import ParallelCollector, WorkerSpec

    collector = ParallelCollector(
        [
            WorkerSpec(
                seed=6,
                players=4,
                lanes=2,
                action_cap=600,
                max_offers=3,
                first_game=0,
                stride=1,
                width=16,
                rounds=1,
                torch_seed=6,
            )
        ]
    )
    try:
        episodes = collector.collect(1)
    finally:
        collector.close()
    assert episodes
    # The invariant is that no search target appears -- `aux` is a general
    # pocket and the plain path is free to use it for other things.
    from catan.expert import Target

    auxes = [t.aux for e in episodes for traj in e.trajectories for t in traj]
    assert auxes
    assert not any(isinstance(a, Target) for a in auxes)


def test_the_flat_wire_format_rebuilds_byte_identical_episodes():
    """`Flattened` is a container change, not a data change.

    The same cohort, shipped flat and rebuilt, must assemble to a Batch equal
    tensor-for-tensor to the one the original objects assemble to — that is
    the acceptance test the perf review set. The rebuilt observations must
    also share one packed buffer, because restoring `pack`'s gather path is
    half the point.
    """
    import pickle

    from catan.actions import space_for
    from catan.board.board import random_base_board
    from catan.collect import Flattened
    from catan.encoding import static_graph
    from catan.game import start
    from catan.model import CatanNet, ModelConfig, packing
    from catan.policy import NetworkPolicy
    from catan.ppo import PPOConfig, assemble
    from catan.selfplay import Collector

    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(0)
    net = CatanNet(space_for(game), graph, 4, ModelConfig(width=16, rounds=1))
    policy = NetworkPolicy(net, space_for(game), packing(graph, 4))
    episodes = Collector(policy, lanes=4, seed=3, action_cap=3000).collect(3)

    flat = pickle.loads(pickle.dumps(Flattened(episodes, policy.layout)))
    rebuilt = flat.episodes()

    assert [e.index for e in rebuilt] == [e.index for e in episodes]
    assert [e.outcome for e in rebuilt] == [e.outcome for e in episodes]
    assert [len(e) for e in rebuilt] == [len(e) for e in episodes]

    config = PPOConfig()
    original = assemble(episodes, policy.layout, config)
    again = assemble(rebuilt, policy.layout, config)
    for field in (
        "buffer",
        "mask",
        "pair",
        "chosen",
        "offer",
        "log_prob",
        "advantage",
        "value_target",
    ):
        assert torch.equal(getattr(original, field), getattr(again, field)), field

    shared = {
        id(t.observation._packed)
        for e in rebuilt
        for seat in e.trajectories
        for t in seat
    }
    assert shared == {id(flat.buffer)}


def test_a_league_worker_records_every_seat_and_loads_a_weight_list():
    """Stage 2's worker contract: two learner nets share the table, every seat
    records under its caster's id, and sync ships one dict per learner."""
    collector = ParallelCollector([spec(w, 2, learners=2) for w in range(2)])
    try:
        from catan.collect import _build  # the same construction the worker runs

        policies, _ = _build(spec(0, 2, learners=2))
        assert len(policies) == 2
        collector.sync_many([policies[0].net, policies[1].net])

        episodes = collector.collect(4)
        assert len(episodes) == 4
        for episode in episodes:
            assert set(episode.cast) == {0, 1}
            for seat, trajectory in enumerate(episode.trajectories):
                assert trajectory, "both ids are learners; every seat records"
                assert all(t.seat == seat for t in trajectory)
    finally:
        collector.close()


def test_a_league_spec_refuses_a_mix():
    from catan.collect import _build

    with pytest.raises(ValueError):
        _build(spec(0, 1, learners=2, mix=(("greedy", 0.15),)))


def test_the_league_caster_balances_seats_and_fixes_adjacency():
    """The property that made a permutation option necessary.

    Rotation alone balances every learner over every board seat -- which is why
    board position was excluded as the cause of the noise heats' two tight pairs
    -- but it leaves the cyclic order round the table invariant, so learner 0's
    turn-order successor is learner 1 in every game ever played.
    """
    from catan.collect import league_caster

    caster = league_caster(4, 4)
    casts = [caster(i) for i in range(4)]
    for learner in range(4):
        seats = [cast.index(learner) for cast in casts]
        assert sorted(seats) == [0, 1, 2, 3], "each learner takes each seat once"
    for cast in casts:
        # Successor of learner k round the table is always k+1 (mod 4).
        for seat, learner in enumerate(cast):
            assert cast[(seat + 1) % 4] == (learner + 1) % 4


def test_a_permuted_learner_order_reseats_the_cycle_without_unbalancing_seats():
    from catan.collect import league_caster

    caster = league_caster(4, 4, order=(0, 2, 1, 3))
    casts = [caster(i) for i in range(4)]
    for learner in range(4):
        seats = [cast.index(learner) for cast in casts]
        assert sorted(seats) == [0, 1, 2, 3], "the share stays balanced"
    successor = {}
    for cast in casts:
        for seat, learner in enumerate(cast):
            successor.setdefault(learner, cast[(seat + 1) % 4])
            assert successor[learner] == cast[(seat + 1) % 4], "adjacency is fixed"
    # 0 now sits before 2 rather than before 1, which is the whole point.
    assert successor[0] == 2 and successor[2] == 1


def test_a_learner_order_that_is_not_a_permutation_is_refused():
    import pytest

    from catan.collect import league_caster

    with pytest.raises(ValueError, match="permutation"):
        league_caster(4, 4, order=(0, 1, 1, 3))


def test_a_paired_caster_casts_both_halves_of_a_pair_identically():
    from catan.collect import league_caster, paired_caster

    plain = league_caster(4, 4)
    caster = paired_caster(plain)
    for k in range(12):
        assert caster(2 * k) == caster(2 * k + 1) == plain(k)


def test_a_paired_league_balances_every_seat_over_a_doubled_window():
    from collections import Counter

    from catan.collect import league_caster, paired_caster

    learners, players = 4, 4
    caster = paired_caster(league_caster(learners, players))
    # The documented cost of pairing: the exact balance `league_caster` gives
    # over any `learners`-game window now takes `2 * learners` games — and it
    # holds at every offset, not just on pair boundaries.
    for offset in range(2 * learners):
        window = [caster(offset + i) for i in range(2 * learners)]
        for seat in range(players):
            share = Counter(cast[seat] for cast in window)
            assert share == {k: 2 for k in range(learners)}


def test_a_paired_worker_wraps_its_caster_and_deals_paired_boards():
    from catan.collect import _build, league_caster

    _, collector = _build(spec(0, 1, learners=2, pair_boards=True))
    assert collector.pair_boards
    plain = league_caster(2, 4)
    for k in range(6):
        assert collector.caster(2 * k) == collector.caster(2 * k + 1) == plain(k)


# --- `--mix` routed through `named_opponent` -------------------------------
#
# The compatibility claim these pin: `greedy` and `parent` must resolve to
# exactly what they resolved to before any entrant spec was accepted, because 34
# recorded runs were collected against them and their numbers cannot be allowed
# to change meaning without their configuration changing.


def _cast_cohort(opponents, *, seed=11, games=4, lanes=2):
    """A cohort with every game cast, played by a torch-free learner.

    `RandomPolicy` rather than a network on purpose: two runs of this with
    identically seeded policies play identical games, so any difference in the
    episodes is a difference in the *opponent* and nothing else. A network
    learner would put a torch generator between the claim and the evidence.
    """
    from catan.collect import mixed_caster
    from catan.selfplay import Collector, RandomPolicy

    collector = Collector(
        RandomPolicy(random.Random(seed)),
        lanes=lanes,
        fill=False,
        players=4,
        seed=seed,
        action_cap=600,
        max_offers=3,
        opponents=opponents,
        caster=mixed_caster([1.0], 4, seed),
    )
    return collector.cohort(games)


def test_a_greedy_mix_plays_the_identical_games_it_played_before_the_routing():
    """The backward-compatibility proof, end to end.

    The left side is the expression `_build` held before `--mix` went through
    `named_opponent`, written out literally here so the claim is pinned to source
    rather than to a remembered equivalence. Same learner, same seeds, same
    caster: if the routing changed the opponent by so much as a tie-break, the
    action streams diverge.
    """
    from catan.collect import greedy_opponent, mix_opponents

    was = _cast_cohort([greedy_opponent(11 + 77, 3, 2)])
    now = _cast_cohort(
        mix_opponents([("greedy", 1.0)], seed=11, max_offers=3, lanes=2)
    )

    assert len(now) == len(was) == 4
    for old, new in zip(was, now):
        assert new.index == old.index
        assert new.cast == old.cast
        assert new.outcome == old.outcome
        assert [t.action for t in new.stream()] == [t.action for t in old.stream()]
    # And it is a real test: some games were actually cast and played out.
    assert all(any(e.cast) for e in was)
    assert sum(e.outcome.actions for e in was) > 0


def test_the_greedy_mix_bot_is_field_for_field_the_one_it_always_was():
    """The same claim structurally, so a failure says *which* field moved."""
    from catan.board.board import random_base_board
    from catan.collect import greedy_opponent, mix_opponents

    board = random_base_board(random.Random(3))
    was = greedy_opponent(11 + 77, 3, 400)
    now = mix_opponents([("greedy", 1.0)], seed=11, max_offers=3, lanes=400)[0]

    old, new = was.spawn(board), now.spawn(board)
    assert type(new) is type(old)
    assert (new.depth, new.width, new.stance, new.partner_choice, new.max_offers) == (
        old.depth,
        old.width,
        old.stance,
        old.partner_choice,
        old.max_offers,
    )
    assert new.evaluator.vector == old.evaluator.vector
    # The tie-break stream too. `SearchBot` breaks ties with its rng, so a
    # different seed offset is a different bot at identical weights -- which is
    # exactly the kind of change that would move a win rate and explain nothing.
    assert new.rng.getstate() == old.rng.getstate()
    # Capacity decides when a live board's bot is evicted and respawned; 400
    # lanes rather than 2 so this is not two 256s agreeing by default.
    assert now.capacity == was.capacity == 800


def test_routing_greedy_through_the_arena_would_have_changed_the_bot():
    """Why `greedy` is reserved rather than looked up like everything else.

    `--mix greedy` takes the run's own `--max-offers`, 3 by default, so what the
    recorded runs actually played is `greedy-offers3`. The arena's `greedy` preset
    means `max_offers=None` -- the engine's whole eight-offer budget -- and greedy
    saturates that cap, so the two are different bots at different strengths.
    """
    from catan.board.board import random_base_board
    from catan.collect import RESERVED_MIX, mix_opponents, named_opponent

    assert RESERVED_MIX == ("greedy", "parent")
    board = random_base_board(random.Random(3))
    reserved = mix_opponents([("greedy", 1.0)], seed=11, max_offers=3, lanes=2)[0]
    preset = named_opponent("greedy", seed=11, lanes=2)

    assert reserved.spawn(board).max_offers == 3
    assert preset.spawn(board).max_offers is None
    # Spelled as an entrant it agrees again, which is the equivalence that holds.
    spelled = mix_opponents(
        [("greedy-offers3", 1.0)], seed=11, max_offers=3, lanes=2
    )[0]
    assert spelled.spawn(board).max_offers == 3


def test_a_mix_resolves_an_entrant_spec_the_arena_scores():
    """`search2-offers3` as a lane opponent: the entrant, not a lookalike.

    Constructed and spawned but never played -- the point is that the bot the
    arena's 400-game results describe is the bot the mix seats.
    """
    from catan.board.board import random_base_board
    from catan.collect import mix_opponents

    board = random_base_board(random.Random(3))
    bot = mix_opponents(
        [("search2-offers3", 0.1)], seed=1, max_offers=3, lanes=2
    )[0].spawn(board)
    assert (bot.depth, bot.width, bot.max_offers) == (2, 6, 3)


def test_a_mix_without_a_parent_entry_never_builds_one():
    """`parent` arrives as a thunk so a worker pays no `torch.load` for a
    checkpoint nothing in its mix casts."""
    from catan.collect import mix_opponents

    def thunk():
        raise AssertionError("built a parent for a mix that never asks for one")

    opponents = mix_opponents(
        [("greedy", 0.15), ("random", 0.1)],
        seed=1,
        max_offers=3,
        lanes=2,
        parent=thunk,
    )
    assert len(opponents) == 2


def test_the_parent_mix_opponent_is_whatever_the_thunk_returned():
    from catan.collect import mix_opponents

    sentinel = object()
    opponents = mix_opponents(
        [("greedy", 0.1), ("parent", 0.1)],
        seed=1,
        max_offers=3,
        lanes=2,
        parent=lambda: sentinel,
    )
    # Order is the caster's id space: id 2 is `mix[1]`.
    assert opponents[1] is sentinel
    with pytest.raises(ValueError, match="needs a parent"):
        mix_opponents([("parent", 0.1)], seed=1, max_offers=3, lanes=2)


def test_a_worker_casts_an_arena_entrant_and_still_records_only_the_learner():
    from catan.selfplay import owned

    collector = ParallelCollector(
        [
            spec(0, 2, mix=(("random", 1.0),)),
            spec(1, 2, mix=(("random", 1.0),)),
        ]
    )
    try:
        episodes = collector.collect(2)
    finally:
        collector.close()

    assert len(episodes) == 2
    for episode in episodes:
        assert any(pid == 1 for pid in episode.cast)
        assert episode.outcome.actions > 0
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    # `owned` is the assemble-side gate on the same fact, and the two must agree:
    # nothing an opponent did may reach an update.
    for kept, episode in zip(owned(episodes, 0), episodes):
        assert kept.trajectories == episode.trajectories


def test_a_worker_seats_three_distinct_mix_opponents_by_id():
    """The id space holds for a real diverse mix -- and one game gets one of them.

    `mixed_caster` draws a single opponent per game and gives it 2 of 4 seats, so
    three opponents at a third each is three *kinds of game*, not a table with
    three kinds of opponent. Pinned rather than merely noted, because reading a
    diverse mix as a heterogeneous table would misread every result it produces.
    """
    mix = (("random", 0.33), ("greedy-offers1", 0.33), ("random-placement", 0.33))
    collector = ParallelCollector([spec(0, 1, lanes=4, mix=mix)])
    try:
        episodes = collector.collect(24)
    finally:
        collector.close()

    seen: set[int] = set()
    for episode in episodes:
        ids = {pid for pid in episode.cast if pid}
        assert len(ids) <= 1, f"two opponents in one game: {episode.cast}"
        assert ids <= {1, 2, 3}
        seen |= ids
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    # Deterministic in the spec's seed, so this is a fixed fact and not a sample.
    assert seen == {1, 2, 3}


def test_a_worker_casts_a_table_and_records_only_the_learner_seat():
    """The seating-free geometry in the sharded collector: one learner seat,
    the rest drawn from the pool by id, opponent seats never recorded."""
    mix = (("table(random|greedy-offers1|random-placement)", 1.0),)
    collector = ParallelCollector([spec(0, 1, lanes=4, mix=mix)])
    try:
        episodes = collector.collect(24)
    finally:
        collector.close()

    seen: set[int] = set()
    for episode in episodes:
        assert episode.cast.count(0) == 1, episode.cast
        assert set(episode.cast) <= {0, 1, 2, 3}
        seen |= {pid for pid in episode.cast if pid}
        for seat, pid in enumerate(episode.cast):
            assert bool(episode.trajectories[seat]) == (pid == 0)
    assert seen == {1, 2, 3}
