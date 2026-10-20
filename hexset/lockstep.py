# SPDX-License-Identifier: GPL-3.0-only
"""Many games in one process, their network forwards batched across games.

`play_lanes(play, jobs, lanes)` runs `play(job)` for up to `lanes` jobs at
once, each in a greenlet or a thread of its own (a *lane*), with exactly one
of them running at any moment. A lane runs until it needs a network forward
or finishes, and hands over. Once no lane can run, every forward the lanes
are waiting on is answered at once: the requests on one onnxruntime session
are stacked into a single `run` over the union of the outputs they ask for,
and each lane gets its own rows back. `pool_lanes` spreads the jobs over
worker processes, each running `play_lanes`.

Greenlets (the `greenlet` package) switch on one thread, so a hand-over
stays on the core it was on and costs about a function call. Threads are
the fallback without it: the operating system may wake the next lane on
another core, which then pays for moving the interpreter's hot state and the
game's across at every forward -- more than batching saves -- so threaded
lanes are pinned to one CPU (`os.sched_setaffinity`, where the platform has
it) and `pool_lanes` gives each threaded worker a CPU of its own.

The forwards are the ones `hexset.clients.onnxbot.V2Policy` makes -- a served
file's moves, its value-head trade gate and veto, robber deals, a search's
leaves -- since every one of them goes through `forward`, and the `value`
forwards of `hexset.clients.onnxbot.DecisionPolicy` (its trade gate and veto,
robber deals); a forward that asks a decision runs on its own, since its
candidate rows index its own batch. Outside a lane `forward` is
`session.run`, so nothing else changes.

A lane plays exactly the code it would play alone: `play` is called as it is,
its random streams are its own, and because one lane runs at a time no two
games interleave inside the engine. Scheduling decides only which rows share
a `run`, and that is invisible to a game exactly when the graph is *batch
invariant*: a row's outputs do not depend on the other rows run with it.
onnxruntime does not promise it. A reduction over a `(1, n, d)` tensor can
take another kernel than over `(b, n, d)`, so one row alone and the same row
beside another can differ in the last bit; a reduction over the last axis can
sum a row differently depending on where in the batch it falls.
`batch_invariant` checks a graph on rows of its own.

So a lane never runs a lone row: a batch of one is run as two copies of it and
the copy dropped. For a graph invariant from two rows up, a lane game is then
the same game at any lane count, any worker count and any job order, and
differs from the same job played alone only where one of its single-row
forwards decides a comparison by that last bit.
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import get_context
from typing import Any, TypeVar

__all__ = [
    "BACKENDS",
    "batch_invariant",
    "forward",
    "in_lane",
    "play_lanes",
    "pool_lanes",
]

J = TypeVar("J")
R = TypeVar("R")

_here = threading.local()


def in_lane() -> bool:
    """Whether the caller is running in a lane of `play_lanes`."""
    return getattr(_here, "lane", None) is not None


def forward(session: Any, outputs: list[str], feed: dict[str, Any]) -> list[Any]:
    """`session.run(outputs, feed)`, or inside a lane, the same rows of a run
    batched with every other lane's waiting forwards. Every array in `feed`
    and every output carries the batch on its first axis."""
    lane = getattr(_here, "lane", None)
    if lane is None:
        return session.run(outputs, feed)
    return lane.wait(session, outputs, feed)


def batch_invariant(
    session: Any, feed: dict[str, Any], outputs: list[str] | None = None,
    sizes: Iterable[int] = (2, 3, 4, 5, 7, 8, 13, 16),
) -> bool:
    """Whether every row of `feed` gets the same outputs from `session` in
    batches of each of `sizes` -- `feed` cut into consecutive batches of that
    many rows, a last lone row run as two as a lane runs it -- as with all its
    rows in one batch. Lane games are independent of lane count where this
    holds (module docstring); `feed` should hold rows the graph will meet."""
    import numpy as np

    names = outputs or [declared.name for declared in session.get_outputs()]
    rows = _rows(feed)
    whole = session.run(names, feed)
    for size in sizes:
        for start in range(0, rows, size):
            part = {key: value[start:start + size] for key, value in feed.items()}
            count = _rows(part)
            if count == 1:
                part = {key: np.concatenate([value, value]) for key, value in part.items()}
            got = session.run(names, part)
            if not all(np.array_equal(g[:count], w[start:start + count]) for g, w in zip(got, whole)):
                return False
    return True


class _Abandoned(BaseException):
    """Raised inside a lane whose run was given up, the next time it is woken,
    so it unwinds without playing on.
    A `BaseException`, so game code catching `Exception` does not swallow it."""


class _Baton:
    """Who runs: the lanes able to (`runnable`), those waiting on a forward
    (`waiting`) and those finished, read and changed only by the one thread
    holding the baton. `back` is the driver's (the thread iterating
    `play_lanes`), which yields results and starts lanes."""

    def __init__(self) -> None:
        self.back = threading.Lock()
        self.back.acquire()
        self.runnable: deque[_Lane] = deque()
        self.waiting: list[_Lane] = []
        self.finished: list[_Lane] = []

    def pass_on(self, holder: "_Lane | None") -> bool:
        """Hand the baton from `holder` (a lane that has just asked for a
        forward or finished, or the driver as `None`) to whoever runs next:
        the driver while a finished lane is unreported, else the next lane
        able to run, else -- every lane waiting -- the lanes after answering
        their forwards at once, the holder first. True if that is the holder
        itself, which then runs on without a switch."""
        if self.finished and holder is not None:
            self.back.release()
            return False
        if not self.runnable:
            _answer(self.waiting)
            if holder is not None and holder in self.waiting:
                self.waiting.remove(holder)
                self.runnable.append(holder)
            self.runnable.extend(self.waiting)
            self.waiting = []
        following = self.runnable.popleft()
        if following is holder:
            return True
        following.go.release()
        return False


class _Lane:
    """One job in flight: its thread, and the forward it is waiting on."""

    def __init__(self, play: Callable[[Any], Any], job: Any, baton: _Baton, cpu: int | None) -> None:
        self.job = job
        self.play = play
        self.baton = baton
        self.cpu = cpu
        # A held lock used as a binary semaphore: the lane runs once another
        # thread releases it. A `Lock` is released and acquired in C, a
        # `Semaphore` in Python, and the baton changes hands at every forward.
        self.go = threading.Lock()
        self.go.acquire()
        self.request: tuple[Any, list[str], dict[str, Any]] | None = None
        self.reply: Any = None
        self.done = False
        self.result: Any = None
        self.error: BaseException | None = None
        self.abandoned = False
        self.thread = threading.Thread(target=self._main, daemon=True)

    def _main(self) -> None:
        if self.cpu is not None:
            os.sched_setaffinity(0, {self.cpu})  # this thread alone (Linux)
        self.go.acquire()
        _here.lane = self
        try:
            if self.abandoned:
                raise _Abandoned
            self.result = self.play(self.job)
        except BaseException as error:  # handed to the driver, which raises it
            self.error = error
        finally:
            _here.lane = None
            self.done = True
            self.baton.finished.append(self)
            self.baton.pass_on(self)

    def wait(self, session: Any, outputs: list[str], feed: dict[str, Any]) -> list[Any]:
        self.request = (session, outputs, feed)
        self.baton.waiting.append(self)
        if not self.baton.pass_on(self):
            self.go.acquire()
        if self.abandoned:
            raise _Abandoned
        reply, self.reply, self.request = self.reply, None, None
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _rows(feed: dict[str, Any]) -> int:
    return len(next(iter(feed.values())))


_FIXED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _fixed_batch(session: Any) -> bool:
    """Whether the graph declares a fixed batch size, which refuses a stack."""
    fixed = _FIXED.get(session)
    if fixed is None:
        fixed = _FIXED[session] = any(
            declared.shape and isinstance(declared.shape[0], int)
            for declared in session.get_inputs()
        )
    return fixed


def _answer(waiting: list[_Lane]) -> None:
    """Run every waiting lane's forward, one `run` per session, and leave
    each lane its own rows (or the error) as its reply."""
    import numpy as np

    groups: dict[int, list[_Lane]] = {}
    for lane in waiting:
        groups.setdefault(id(lane.request[0]), []).append(lane)
    for group in groups.values():
        session = group[0].request[0]
        if _fixed_batch(session):
            for lane in group:
                _, outputs, feed = lane.request
                try:
                    lane.reply = session.run(outputs, feed)
                except Exception as error:
                    lane.reply = error
            continue
        names: list[str] = []
        for lane in group:
            names.extend(name for name in lane.request[1] if name not in names)
        keys = list(group[0].request[2])
        sizes = [_rows(lane.request[2]) for lane in group]
        try:
            if any(sorted(lane.request[2]) != sorted(keys) for lane in group):
                raise ValueError("forwards on one session fed different inputs")
            if sum(sizes) == 1:
                # Never a lone row: see the module docstring.
                feed = {key: np.concatenate([value, value]) for key, value in group[0].request[2].items()}
            else:
                feed = {key: np.concatenate([lane.request[2][key] for lane in group]) for key in keys}
            results = dict(zip(names, session.run(names, feed)))
        except Exception as error:
            for lane in group:
                lane.reply = error
            continue
        start = 0
        for lane, size in zip(group, sizes):
            lane.reply = [results[name][start:start + size] for name in lane.request[1]]
            start += size


def _current_cpu() -> int | None:
    """The CPU the calling thread is on, where it can be read and pinned to
    (Linux); otherwise `None`."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    try:
        with open("/proc/thread-self/stat") as stat:
            # `processor` is field 39; the command name (field 2) may hold
            # spaces, so count from the parenthesis that closes it.
            cpu = int(stat.read().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError, IndexError):
        return None
    return cpu if cpu in os.sched_getaffinity(0) else None


#: What a lane can be (`play_lanes`): a greenlet, or a thread.
BACKENDS = ("greenlet", "threads")


def _backend(backend: str | None) -> str:
    """`backend`, or the default: greenlets where `greenlet` is installed."""
    if backend is None:
        try:
            import greenlet  # noqa: F401
        except ImportError:
            return "threads"
        return "greenlet"
    if backend not in BACKENDS:
        raise ValueError(f"unknown lane backend {backend!r}: {BACKENDS}")
    return backend


def play_lanes(
    play: Callable[[J], R], jobs: Iterable[J], lanes: int = 16, *,
    backend: str | None = None, pin: bool = True,
) -> Iterator[tuple[J, R]]:
    """`(job, play(job))` for every job, in the order the jobs finish, with up
    to `lanes` of them in flight and their network forwards batched (module
    docstring). `jobs` is read lazily, one job as each lane frees up.

    `backend` is what a lane is: `"greenlet"` (the default where `greenlet` is
    installed), every lane a greenlet on the calling thread, or `"threads"`,
    every lane a thread of its own. Either way one lane runs at a time.
    Threads hand over through the operating system, which may wake the next
    lane on another core: with `pin`, threaded lanes all run on the CPU the
    caller is on when the run starts. State a job keeps per lane belongs in a
    `contextvars.ContextVar`, which each lane of either kind starts empty.

    A job that raises ends the run: the lanes still in flight are abandoned
    and the job's error is raised here, its job attached as a note. Closing
    the iterator early abandons them too."""
    if lanes < 1:
        raise ValueError("lanes must be at least one")
    if _backend(backend) == "greenlet":
        return _greenlet_lanes(play, jobs, lanes)
    return _thread_lanes(play, jobs, lanes, pin)


class _Sprout:
    """One job in flight as a greenlet, and the forward it is waiting on."""

    def __init__(self, play: Callable[[Any], Any], job: Any) -> None:
        import greenlet

        self.job = job
        self.play = play
        self.request: tuple[Any, list[str], dict[str, Any]] | None = None
        self.reply: Any = None
        self.done = False
        self.result: Any = None
        self.error: BaseException | None = None
        self.abandoned = False
        self.green = greenlet.greenlet(self._main)

    def _main(self) -> None:
        try:
            self.result = self.play(self.job)
        except BaseException as error:  # handed to the driver, which raises it
            self.error = error
        self.done = True

    def wait(self, session: Any, outputs: list[str], feed: dict[str, Any]) -> list[Any]:
        self.request = (session, outputs, feed)
        self.green.parent.switch()
        if self.abandoned:
            raise _Abandoned
        reply, self.reply, self.request = self.reply, None, None
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _greenlet_lanes(play, jobs, lanes):
    pending = iter(jobs)
    active: list[_Sprout] = []
    runnable: deque[_Sprout] = deque()

    def start() -> None:
        for job in pending:
            lane = _Sprout(play, job)
            active.append(lane)
            runnable.append(lane)
            return

    try:
        for _ in range(lanes):
            start()
        while active:
            waiting: list[_Sprout] = []
            while runnable:
                lane = runnable.popleft()
                _here.lane = lane
                try:
                    lane.green.switch()
                finally:
                    _here.lane = None
                if not lane.done:
                    waiting.append(lane)
                    continue
                active.remove(lane)
                if lane.error is not None:
                    lane.error.add_note(f"in lane job {lane.job!r}")
                    raise lane.error
                yield lane.job, lane.result
                start()
            if waiting:
                _answer(waiting)
                runnable.extend(waiting)
    finally:
        for lane in active:
            if not lane.done and lane.green:
                lane.abandoned = True
                _here.lane = lane
                try:
                    lane.green.switch()
                finally:
                    _here.lane = None


def _thread_lanes(play, jobs, lanes, pin):
    baton = _Baton()
    cpu = _current_cpu() if pin else None
    pending = iter(jobs)
    exhausted = False
    active: list[_Lane] = []

    def start() -> None:
        nonlocal exhausted
        if exhausted:
            return
        try:
            job = next(pending)
        except StopIteration:
            exhausted = True
            return
        lane = _Lane(play, job, baton, cpu)
        lane.thread.start()
        active.append(lane)
        baton.runnable.append(lane)

    try:
        while len(active) < lanes and not exhausted:
            start()
        while active:
            # The driver holds the baton here: report and replace every
            # finished lane, then hand the baton on and wait for it back.
            while baton.finished:
                lane = baton.finished.pop(0)
                active.remove(lane)
                lane.thread.join()
                if lane.error is not None:
                    lane.error.add_note(f"in lane job {lane.job!r}")
                    raise lane.error
                yield lane.job, lane.result
                start()
            if active:
                baton.pass_on(None)
                baton.back.acquire()
    finally:
        for lane in active:
            if not lane.done:
                lane.abandoned = True
                lane.go.release()
                baton.back.acquire()
                lane.thread.join()


_STOP = None


def _worker(play, jobs, results, lanes, initializer, initargs, backend, cpu) -> None:
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        if initializer is not None:
            initializer(*initargs)

        def feed() -> Iterator[Any]:
            while True:
                job = jobs.get()
                if job is _STOP:
                    return
                yield job

        for job, result in play_lanes(play, feed(), lanes, backend=backend, pin=cpu is not None):
            results.put(("done", job, result))
    except BaseException:
        results.put(("error", None, traceback.format_exc()))


def pool_lanes(
    play: Callable[[J], R],
    jobs: Iterable[J],
    *,
    workers: int = 1,
    lanes: int = 16,
    initializer: Callable | None = None,
    initargs: tuple = (),
    start_method: str | None = None,
    backend: str | None = None,
    pin: bool = True,
) -> Iterator[tuple[J, R]]:
    """`play_lanes` over `workers` processes, `lanes` games in flight in each:
    `(job, play(job))` in the order they finish. Each worker takes the next
    job as one of its lanes frees up, so a slow job holds up one lane rather
    than a share of the run. `backend` is `play_lanes`'s. Where lanes are
    threads and `pin` is set, worker `k` runs on the `k`-th CPU this process
    may use (round robin past the last), where the platform lets a process
    choose; such a worker cannot move off a CPU that other work crowds, so on
    a host running more than it has CPUs, `pin=False` -- or greenlets, which
    need no pinning. `workers=1` plays here.

    `play`, the jobs and the results cross a process boundary and must
    pickle. ``initializer(*initargs)`` runs once in each worker (or here, for
    `workers=1`) before its first job, for a runtime to register its
    entrants. A job that raises, or a worker that dies, ends the run with an
    error; results already handed out stay handed out."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        yield from play_lanes(play, jobs, lanes, backend=backend, pin=pin)
        return
    backend = _backend(backend)
    jobs = list(jobs)
    if not jobs:
        return
    pinned = pin and backend == "threads" and hasattr(os, "sched_setaffinity")
    cpus = sorted(os.sched_getaffinity(0)) if pinned else []
    context = get_context(start_method)
    inbox, outbox = context.Queue(), context.Queue()
    procs = [
        context.Process(
            target=_worker, daemon=True,
            args=(play, inbox, outbox, lanes, initializer, initargs, backend,
                  None if not cpus else cpus[k % len(cpus)]),
        )
        for k in range(min(workers, len(jobs)))
    ]
    for proc in procs:
        proc.start()
    try:
        for job in jobs:
            inbox.put(job)
        for _ in procs:
            inbox.put(_STOP)
        left = len(jobs)
        while left:
            try:
                kind, job, payload = outbox.get(timeout=1.0)
            except queue.Empty:
                dead = [proc.exitcode for proc in procs if proc.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"a lane worker died (exit code {dead[0]})") from None
                continue
            if kind == "error":
                raise RuntimeError(f"a lane worker failed:\n{payload}")
            left -= 1
            yield job, payload
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join()
