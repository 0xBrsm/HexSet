# SPDX-License-Identifier: GPL-3.0-only
"""Many games in one process with their forwards batched (`hexset.lockstep`),
and `hexset.arena.compete_lanes` over it: a lane plays the game it would play
alone, every forward waiting at once shares one run per session, a lone row is
never run alone, and the run's journal keeps every game as it finishes."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from hexset import lockstep
from hexset.arena import (
    MAX_ACTIONS, Entrant, _play_one, compete, compete_lanes, lineup_from_names,
    read_lane_journal, register_entrant_kind, unregister_entrant_kind,
)
from hexset.record import to_json
from hexset.rules import STANDARD_GAME

FIXTURES = Path(__file__).parent / "clients" / "fixtures"


@pytest.fixture(params=lockstep.BACKENDS)
def backend(request):
    """Every lane test runs on both kinds of lane; greenlets where installed."""
    if request.param == "greenlet":
        pytest.importorskip("greenlet")
    return request.param


class Doubler:
    """A session stand-in: `y = 2 * x + 1` row by row, and a log of the
    batch sizes it was run on."""

    def __init__(self, fixed: bool = False) -> None:
        self.runs: list[int] = []
        self.fixed = fixed

    def get_inputs(self):
        class Declared:
            shape = [1 if self.fixed else "batch", 3]
        return [Declared()]

    def run(self, outputs, feed):
        x = feed["x"]
        self.runs.append(len(x))
        table = {"y": 2 * x + 1, "z": -x}
        return [table[name] for name in outputs]


def _rows(n: int, start: int) -> np.ndarray:
    return np.arange(start, start + 3 * n, dtype=np.float64).reshape(n, 3)


def _asks(session, sizes):
    """A job: one forward per entry of `sizes`, each checked against the rows it fed."""
    def play(job):
        for step, n in enumerate(sizes):
            x = _rows(n, 100 * job + step)
            outputs = ["y", "z"] if step % 2 else ["y"]
            got = lockstep.forward(session, outputs, {"x": x})
            assert np.array_equal(got[0], 2 * x + 1)
            if step % 2:
                assert np.array_equal(got[1], -x)
        return job
    return play


def test_outside_a_lane_a_forward_is_the_sessions_own_run():
    session = Doubler()
    (y,) = lockstep.forward(session, ["y"], {"x": _rows(1, 0)})
    assert session.runs == [1] and np.array_equal(y, 2 * _rows(1, 0) + 1)
    assert not lockstep.in_lane()


def test_lanes_share_one_run_per_session_and_get_their_own_rows(backend):
    session = Doubler()
    sizes = [1, 3, 1, 2, 5]
    done = dict(lockstep.play_lanes(_asks(session, sizes), range(6), lanes=6, backend=backend))
    assert done == {job: job for job in range(6)}
    # Six lanes asking in step: one run a step, every lane's rows in it.
    assert session.runs == [6 * n for n in sizes]


def test_more_jobs_than_lanes_refill_as_lanes_finish(backend):
    session = Doubler()
    finished = [job for job, _ in lockstep.play_lanes(_asks(session, [1, 2]), range(10), lanes=3, backend=backend)]
    assert sorted(finished) == list(range(10))
    assert max(session.runs) <= 3 * 2


def test_a_lone_row_is_run_as_two(backend):
    session = Doubler()
    seen = []

    def play(job):
        (y,) = lockstep.forward(session, ["y"], {"x": _rows(1, 7)})
        seen.append(y.shape)
        return job

    assert list(lockstep.play_lanes(play, [0], lanes=4, backend=backend)) == [(0, 0)]
    assert session.runs == [2] and seen == [(1, 3)]


def test_a_fixed_batch_graph_is_run_lane_by_lane(backend):
    session = Doubler(fixed=True)
    assert len(list(lockstep.play_lanes(_asks(session, [1, 1]), range(4), lanes=4, backend=backend))) == 4
    assert session.runs == [1] * 8


def test_a_job_that_raises_ends_the_run_and_names_its_job(backend):
    session = Doubler()
    before = threading.active_count()

    def play(job):
        lockstep.forward(session, ["y"], {"x": _rows(1, job)})
        if job == 2:
            raise KeyError("boom")
        lockstep.forward(session, ["y"], {"x": _rows(1, job)})
        return job

    with pytest.raises(KeyError) as raised:
        list(lockstep.play_lanes(play, range(5), lanes=5, backend=backend))
    assert any("2" in note for note in raised.value.__notes__)
    assert threading.active_count() == before


def test_closing_the_run_early_abandons_the_lanes_in_flight(backend):
    session = Doubler()
    played = []

    def play(job):
        for step in range(3):
            lockstep.forward(session, ["y"], {"x": _rows(1, step)})
        played.append(job)
        return job

    before = threading.active_count()
    run = lockstep.play_lanes(play, range(8), lanes=4, backend=backend)
    first = next(run)
    run.close()
    assert threading.active_count() == before
    # Only the lanes that finished their game played on past the close.
    assert first[0] in played and len(played) <= 4


def test_threaded_lanes_share_one_cpu_and_leave_the_callers_alone():
    import os

    if not hasattr(os, "sched_setaffinity") or lockstep._current_cpu() is None:
        pytest.skip("this platform does not let a thread choose its CPU")
    session = Doubler()
    allowed = os.sched_getaffinity(0)
    seen = []

    def play(job):
        for _ in range(3):
            lockstep.forward(session, ["y"], {"x": _rows(1, job)})
            seen.append(frozenset(os.sched_getaffinity(0)))
        return job

    assert len(list(lockstep.play_lanes(play, range(6), lanes=3, backend="threads"))) == 6
    assert len(set(seen)) == 1 and len(seen[0]) == 1 and seen[0] <= allowed
    assert os.sched_getaffinity(0) == allowed
    seen.clear()
    list(lockstep.play_lanes(play, range(2), lanes=2, backend="threads", pin=False))
    assert set(seen) == {frozenset(allowed)}


def test_each_lane_keeps_its_own_context(backend):
    import contextvars

    mine = contextvars.ContextVar("mine", default=None)
    session = Doubler()

    def play(job):
        assert mine.get() is None
        mine.set(job)
        for _ in range(3):
            lockstep.forward(session, ["y"], {"x": _rows(1, job)})
            assert mine.get() == job
        return job

    assert sorted(job for job, _ in lockstep.play_lanes(play, range(5), lanes=5, backend=backend)) == list(range(5))
    assert mine.get() is None


def test_an_unknown_backend_is_refused_by_name():
    with pytest.raises(ValueError, match="fibres"):
        lockstep.play_lanes(lambda job: job, [1], backend="fibres")


_POOL_SESSION = Doubler()


def _pool_job(job):
    _asks(_POOL_SESSION, [1, 2, 1])(job)
    return job * job


def test_a_pool_of_workers_plays_every_job_once(backend):
    done = list(lockstep.pool_lanes(_pool_job, range(12), workers=2, lanes=3, backend=backend))
    assert sorted(done) == [(job, job * job) for job in range(12)]


# --- compete_lanes ------------------------------------------------------------

QUICK = dict(action_cap=60)


def _jobs(lineups: dict[str, list[str]], games: int, seed: int = 4):
    built = {name: tuple(lineup_from_names(names)) for name, names in lineups.items()}
    return {(name, seed, index): (lineup, seed, index)
            for index in range(games) for name, lineup in built.items()}


def test_a_lane_game_is_the_game_compete_plays_at_its_index(tmp_path):
    lineup = lineup_from_names(["random"] * 4)
    whole = compete(lineup, 8, seed=4, records=True, **QUICK)
    lanes = compete_lanes(_jobs({"t": ["random"] * 4}, 8), tmp_path / "j.jsonl",
                          lanes=3, records=True, **QUICK)
    assert [lanes[("t", 4, i)].winner for i in range(8)] == list(whole.winners)
    assert [lanes[("t", 4, i)].points for i in range(8)] == list(whole.points)
    assert [to_json(lanes[("t", 4, i)].record) for i in range(8)] == [to_json(r) for r in whole.records]


def test_every_lane_game_is_in_the_journal_under_its_key(tmp_path):
    journal = tmp_path / "j.jsonl"
    jobs = _jobs({"a": ["random"] * 4, "b": ["random"] * 2}, 4)
    outcomes = compete_lanes(jobs, journal, lanes=2, workers=2, **QUICK)
    header, held = read_lane_journal(journal)
    assert header["lanes"] is True
    assert set(held) == set(jobs) == set(outcomes)
    for key, (entrants, seed, index, outcome) in held.items():
        assert (seed, index) == key[1:] and entrants == [e.name for e in jobs[key][0]]
        assert outcome.points == outcomes[key].points


def test_a_lane_run_stopped_part_way_resumes_to_its_own_result(tmp_path):
    jobs = _jobs({"a": ["random"] * 4}, 6)
    whole = compete_lanes(jobs, tmp_path / "whole.jsonl", lanes=4, records=True, **QUICK)
    journal = tmp_path / "j.jsonl"
    compete_lanes(jobs, journal, lanes=4, records=True, **QUICK)
    kept = journal.read_text().splitlines(keepends=True)
    assert [json.loads(k)["kind"] for k in kept[:2]] == ["header", "source"]
    journal.write_text("".join(kept[:4]) + kept[4][: len(kept[4]) // 2])
    with pytest.raises(FileExistsError):
        compete_lanes(jobs, journal, lanes=4, records=True, **QUICK)
    played = []
    resumed = compete_lanes(jobs, journal, lanes=4, records=True, resume=True,
                            progress=lambda done, total, key, outcome: played.append(key), **QUICK)
    assert set(resumed) == set(jobs) and len(played) == len(jobs)
    for key in jobs:
        assert to_json(resumed[key].record) == to_json(whole[key].record)
    _, held = read_lane_journal(journal)
    assert set(held) == set(jobs)
    games = [json.loads(line) for line in journal.read_text().splitlines()[1:]]
    assert len([g for g in games if g["kind"] == "game"]) == len(jobs)


def test_a_resumed_lane_run_refuses_another_job_under_a_held_key(tmp_path):
    journal = tmp_path / "j.jsonl"
    compete_lanes(_jobs({"a": ["random"] * 4}, 2), journal, lanes=2, **QUICK)
    moved = {("a", 4, 0): (tuple(lineup_from_names(["random"] * 4)), 4, 1)}
    with pytest.raises(ValueError, match="not the job given"):
        compete_lanes(moved, journal, lanes=2, resume=True, **QUICK)
    with pytest.raises(ValueError, match="differ"):
        compete_lanes(_jobs({"a": ["random"] * 4}, 2), journal, lanes=2, resume=True,
                      action_cap=61)


# --- with onnxruntime ---------------------------------------------------------


def _dyadic_stub(tmp_path) -> str:
    """The valued contract-6 stub with weights whose every partial sum is
    exact in float32, so its outputs are the same in any batch on any CPU:
    what a game played in lanes is compared against is then the plumbing
    alone, not a kernel's summation order."""
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    model = onnx.load(str(FIXTURES / "stub-contract6-valued.onnx"))
    for i, init in enumerate(model.graph.initializer):
        if init.name == "resource_weight":
            exact = np.array([1 / 64, -1 / 32, 1 / 128, 1 / 16, -1 / 64], dtype=np.float32)
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(exact, init.name))
    path = tmp_path / "dyadic.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_a_served_file_plays_the_same_game_in_a_lane_as_alone(tmp_path, monkeypatch, backend):
    pytest.importorskip("onnxruntime")
    from hexset.clients import onnxbot

    asked: list[tuple[bool, tuple[str, ...]]] = []

    def counted(session, outputs, feed):
        asked.append((lockstep.in_lane(), tuple(outputs)))
        return lockstep.forward(session, outputs, feed)

    monkeypatch.setattr(onnxbot, "forward", counted)
    register_entrant_kind("lane-stub", lambda e, board, rng: onnxbot.spawn(e.weights, board, rng=rng))
    try:
        stub = Entrant("lane-stub", kind="lane-stub", weights=_dyadic_stub(tmp_path))
        lineup = (stub, *lineup_from_names(["random"] * 3))
        jobs = {("t", 9, i): (lineup, 9, i) for i in range(2)}
        lanes = compete_lanes(jobs, tmp_path / "j.jsonl", lanes=2, records=True, action_cap=50,
                              turn_cap=500, backend=backend)
        # The file's value-head gate priced offers in the lanes, beside its moves.
        assert (True, ("value",)) in asked and (True, ("action_index",)) in asked
        for (_, seed, index), outcome in lanes.items():
            alone = _play_one((lineup, index, seed, 50, True, True, "round", STANDARD_GAME, 500))
            assert to_json(outcome.record) == to_json(alone.record)
            assert outcome.cleared == alone.cleared and outcome.deals == alone.deals
    finally:
        unregister_entrant_kind("lane-stub")


def _pooling_session(ort, onnx, rng):
    """A graph shaped like a served one where batching can show: a mean over
    the token axis, then a dense layer."""
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(rng.standard_normal((48, 16)).astype(np.float32), "w")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["pooled"], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["y"])],
        "pool", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 37, 48])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 16])], [weight, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def test_a_batch_invariant_graph_gives_each_lane_its_rows_whatever_shares_its_run(backend):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    rng = np.random.default_rng(0)
    session = _pooling_session(ort, onnx, rng)
    x = rng.standard_normal((40, 37, 48)).astype(np.float32)
    if not lockstep.batch_invariant(session, {"x": x}):
        pytest.skip("this onnxruntime build sums a pooled row differently by batch")
    (whole,) = session.run(["y"], {"x": x})
    for lanes in (1, 3, 13, 40):
        got = dict(lockstep.play_lanes(
            lambda i: lockstep.forward(session, ["y"], {"x": x[i:i + 1]})[0][0], range(40), lanes=lanes,
            backend=backend,
        ))
        assert all(np.array_equal(got[i], whole[i]) for i in range(40))


def test_batch_invariance_is_read_off_the_rows_a_graph_is_given():
    class RowSum:
        """Sums each row, the order depending on how many rows it is given:
        the shape of a kernel that is not batch invariant."""

        def get_outputs(self):
            class Declared:
                name = "y"
            return [Declared()]

        def run(self, outputs, feed):
            x = feed["x"].astype(np.float32)
            if len(x) % 2:
                return [x.sum(axis=1)]
            return [x[:, ::-1].cumsum(axis=1, dtype=np.float32)[:, -1]]

    x = np.array([[1e8, 1.0, -1e8, 1.0]] * 6, dtype=np.float32)
    assert not lockstep.batch_invariant(RowSum(), {"x": x})
    assert lockstep.batch_invariant(Doubler(), {"x": _rows(9, 0)}, outputs=["y"])
