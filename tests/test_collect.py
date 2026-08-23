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
