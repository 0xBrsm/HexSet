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
