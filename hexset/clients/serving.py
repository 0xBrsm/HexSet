# SPDX-License-Identifier: GPL-3.0-only
"""One process runs the ONNX forwards of many processes, joined into large
batches.

`served(device)` starts a server process that owns one onnxruntime session
per file on `device` (`"cuda"` or `"cpu"`). While the block is open, every
file `hexset.clients.onnxbot.load` loads, in this process or in any process
started from it, is a `RemoteSession` on that server: the policy, its search,
its trade gate and its robber deals run in the calling process as before, and
only the graph runs on the server.

    with served("cuda"):
        compete(entrants, games, workers=46, journal=...)

The server reads requests from every connected process and answers in rounds:
each round takes every request waiting, joins the requests for one file and
one input set into one feed (`hexset.clients.batching.join`, which moves
candidate owner rows past the rows before them), runs it once over the union
of the outputs they ask, and cuts each request's rows back out
(`hexset.clients.batching.split`). Rounds overlap: while the device runs one,
the server answers the one before and, as late as it can while still being
ready when the device comes free, joins the next from every request that has
arrived, so requests that arrive while a round runs make the next round
larger. A joined feed of one row is run as two copies and one
dropped, as `hexset.lockstep` runs it. A graph that declares a fixed batch
size runs each request on its own.

A process sends one message per round trip. Inside a `hexset.lockstep` lane
run, every forward the lanes are waiting on, decisions included, goes in one
message (`RemoteSession.joined`), so a server round carries every lane of
every process. The sending process joins its own requests for one file
first, so the server joins one feed per file and process. Arrays travel
through a memory-mapped file each side writes and the other reads (in
`/dev/shm` where it has room); the messages carry only where they are.

`resolve("auto")`, the default of `hexset.arena.compete` and `compete_lanes`,
is `"cuda"` where `cuda_available()` (onnxruntime offers its CUDA provider and
the CUDA driver reports a device), unless `HEXSET_SERVE_AUTO=0` or a server is
already open, and no server otherwise.

On CUDA, TF32 is off: every product is computed in the file's own precision.
A served forward differs from the same forward on the calling process's CPU
by the two devices' arithmetic, and a joined batch, like any batch, is not
promised to give each row the bits it gets alone (`hexset.lockstep`).
"""
from __future__ import annotations

import contextlib
import os
import queue
import secrets
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from multiprocessing import get_context
from multiprocessing.connection import Client, Listener, wait
from typing import Any

import numpy as np

from . import batching

__all__ = ["AUTO", "AUTO_ENV", "DEVICES", "ENV", "RemoteSession", "cuda_available", "current", "remote_session",
           "resolve", "served"]

#: The environment variable naming the open server (`served`) to processes
#: started from the one that opened it: its socket address and key.
ENV = "HEXSET_SERVE"

#: The devices a server runs on.
DEVICES = ("cuda", "cpu")

#: `resolve`'s choice: a server on CUDA where this host has a CUDA device
#: onnxruntime can run on, none elsewhere.
AUTO = "auto"

#: Set to 0 to make `AUTO` resolve to no server on any host.
AUTO_ENV = "HEXSET_SERVE_AUTO"

_cuda: list[bool] = []


def cuda_available() -> bool:
    """Whether this host can run a server on CUDA: onnxruntime offers its CUDA
    provider and the CUDA driver reports at least one device. Asked once per
    process."""
    if not _cuda:
        _cuda.append(_probe_cuda())
    return _cuda[0]


def _probe_cuda() -> bool:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
    except Exception:
        return False
    import ctypes

    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            driver = ctypes.CDLL(name)
        except OSError:
            continue
        count = ctypes.c_int(0)
        try:
            return driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
        except Exception:
            return False
    return False


def resolve(serve: str | None) -> str | None:
    """The device a run's `serve` setting opens a server on, or `None` for
    none. `AUTO` is `"cuda"` where `cuda_available()`, unless `AUTO_ENV` is
    0 or a server is already open for this process (its forwards already go
    there), and `None` otherwise; `None` and the `DEVICES` stand as given."""
    if serve is None or serve in DEVICES:
        return serve
    if serve != AUTO:
        raise ValueError(f"serve is {AUTO!r}, None or one of {DEVICES}, not {serve!r}")
    if os.environ.get(AUTO_ENV, "1") == "0" or current() is not None:
        return None
    return "cuda" if cuda_available() else None


def _providers(device: str) -> list:
    if device == "cuda":
        return [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"a server runs on one of {DEVICES}, not {device!r}")


class _Arg:
    """An input or output as `onnxruntime.NodeArg` describes it."""

    def __init__(self, name: str, shape: list, type: str) -> None:
        self.name, self.shape, self.type = name, shape, type


class _ModelMeta:
    def __init__(self, custom_metadata_map: dict[str, str]) -> None:
        self.custom_metadata_map = custom_metadata_map


def _describe(session: Any) -> dict:
    return {
        "inputs": [(a.name, list(a.shape) if a.shape is not None else None, a.type) for a in session.get_inputs()],
        "outputs": [(a.name, list(a.shape) if a.shape is not None else None, a.type) for a in session.get_outputs()],
        "meta": dict(session.get_modelmeta().custom_metadata_map),
        "providers": list(session.get_providers()),
    }


def _fixed(description: dict) -> bool:
    return any(shape and isinstance(shape[0], int) for _, shape, _ in description["inputs"])


class _ServerError(RuntimeError):
    """A request the server could not answer, raised in the process that sent it."""


#: How often the server's threads hand over the interpreter, in seconds.
_SWITCH_SECONDS = 0.0002
#: Arrays in a shared buffer start on this boundary.
_ALIGN = 64
#: A buffer's first size, in bytes; it grows to twice what a message needs.
_FIRST_BYTES = 4 << 20


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class _Buffer:
    """A file mapped into memory that two processes share: the writer creates
    it (`create`), the reader maps it by path (`attach`). A writer that
    needs more room makes a new, larger file under a new name, and the
    message that uses it names it."""

    def __init__(self, path: str, size: int, create: bool) -> None:
        import mmap

        self.path, self.size = path, size
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, size)
            self.map = mmap.mmap(fd, size)
        finally:
            os.close(fd)

    def write(self, arrays: list[np.ndarray]) -> list[tuple[int, str, tuple]]:
        """Copy `arrays` in, one after another; each one's (offset, dtype, shape)."""
        layout, at = [], 0
        for array in arrays:
            array = np.asarray(array)
            if array.nbytes:
                np.ndarray(array.shape, array.dtype, buffer=self.map, offset=at)[...] = array
            layout.append((at, array.dtype.str, array.shape))
            at += _aligned(array.nbytes)
        return layout

    def view(self, layout: tuple[int, str, tuple]) -> np.ndarray:
        offset, dtype, shape = layout
        return np.ndarray(shape, np.dtype(dtype), buffer=self.map, offset=offset)


def _room(arrays: list[np.ndarray]) -> int:
    return sum(_aligned(np.asarray(a).nbytes) for a in arrays)


class _Writer:
    """This side's outgoing buffer: grown, under a new name, when a message
    needs more room than it has."""

    def __init__(self, directory: str, stem: str) -> None:
        self.directory, self.stem, self.count = directory, stem, 0
        self.buffer: _Buffer | None = None

    def write(self, arrays: list[np.ndarray]) -> tuple[str, list]:
        need = _room(arrays)
        if self.buffer is None or self.buffer.size < need:
            old = self.buffer
            self.count += 1
            path = os.path.join(self.directory, f"{self.stem}-{self.count}")
            self.buffer = _Buffer(path, max(_FIRST_BYTES, 2 * need), create=True)
            if old is not None:
                with contextlib.suppress(OSError):
                    os.unlink(old.path)
        return self.buffer.path, self.buffer.write(arrays)

    def close(self) -> None:
        if self.buffer is not None:
            with contextlib.suppress(OSError):
                os.unlink(self.buffer.path)
            self.buffer = None


class _Reader:
    """The other side's buffer, mapped by the path its latest message names."""

    def __init__(self) -> None:
        self.buffer: _Buffer | None = None

    def at(self, path: str) -> _Buffer:
        if self.buffer is None or self.buffer.path != path:
            self.buffer = _Buffer(path, os.stat(path).st_size, create=False)
        return self.buffer


class _Peer:
    """One connected process, as the server sees it."""

    def __init__(self, connection: Any, directory: str, number: int) -> None:
        self.connection = connection
        self.incoming = _Reader()
        self.outgoing = _Writer(directory, f"answer-{number}")
        self.since = 0.0  # when its waiting message arrived


class _Piece:
    """One feed a message carries for one file: its arrays are views of the
    sender's buffer, valid until the sender is answered."""

    __slots__ = ("peer", "index", "path", "names", "feed", "requests", "answer")

    def __init__(self, peer: _Peer, index: int, path: str, names: list[str], feed: dict, requests: int) -> None:
        self.peer, self.index, self.path, self.names, self.feed = peer, index, path, names, feed
        self.requests = requests
        self.answer: Any = None


class _Round:
    """Pieces taken together: per (file, input names), the feed they join to."""

    def __init__(self, messages: list[tuple[_Peer, list[_Piece]]]) -> None:
        self.messages = messages
        self.groups: list[tuple[str, list[_Piece], Any]] = []
        self.outputs: list[Any] = []


def _prepare(work: _Round, described: dict) -> None:
    """Join each (file, input names)'s pieces into one feed (one run), or keep
    them apart for a file of fixed batch size."""
    by_key: dict[tuple, list[_Piece]] = {}
    for _, pieces in work.messages:
        for piece in pieces:
            by_key.setdefault((piece.path, tuple(sorted(piece.feed))), []).append(piece)
    for (path, _), pieces in by_key.items():
        description = described.get(path)
        if description is None:
            for piece in pieces:
                piece.answer = _ServerError(f"the inference server has not loaded {path}")
            continue
        if _fixed(description):
            for piece in pieces:
                work.groups.append((path, [piece], None))
            continue
        try:
            feeds = [piece.feed for piece in pieces]
            joined = batching.join(feeds if sum(batching._feed_rows(f) for f in feeds) != 1 else feeds * 2)
        except Exception as error:
            for piece in pieces:
                piece.answer = _ServerError(f"the inference server could not join feeds for {path}: "
                                            f"{type(error).__name__}: {error}")
            continue
        work.groups.append((path, pieces, joined))


def _run(work: _Round, sessions: dict, stats: dict) -> None:
    """Run every group of a prepared round: the only part that waits on the device."""
    for path, pieces, joined in work.groups:
        session = sessions[path]
        names: list[str] = []
        for piece in pieces:
            names.extend(name for name in piece.names if name not in names)
        started = time.perf_counter()
        try:
            outputs = session.run(names, joined.feed if joined is not None else pieces[0].feed)
        except Exception as error:
            outputs = _ServerError(f"the inference server could not run {path}: {type(error).__name__}: {error}")
        stats["seconds"] += time.perf_counter() - started
        stats["forwards"] += 1
        if joined is not None:
            stats["rows"] += sum(count["rows"] for count in joined.counts[:len(pieces)])
        else:
            stats["rows"] += batching._feed_rows(pieces[0].feed)
        work.outputs.append((names, outputs))


def _reply(work: _Round, stats: dict) -> None:
    """Cut each group's outputs back into its pieces, write each process's
    answers into its answer buffer and tell it where they are."""
    for (path, pieces, joined), (names, outputs) in zip(work.groups, work.outputs):
        if isinstance(outputs, BaseException):
            for piece in pieces:
                piece.answer = outputs
            continue
        parts = batching.split(names, outputs, joined) if joined is not None else [outputs]
        for piece, part in zip(pieces, parts):
            by_name = dict(zip(names, part))
            piece.answer = [by_name[name] for name in piece.names]
    stats["rounds"] += 1
    for peer, pieces in work.messages:
        stats["requests"] += sum(piece.requests for piece in pieces)
        stats["pieces"] += len(pieces)
        arrays = [a for piece in pieces if not isinstance(piece.answer, BaseException) for a in piece.answer]
        try:
            path, layout = peer.outgoing.write(arrays)
            answers, at = [], 0
            for piece in pieces:
                if isinstance(piece.answer, BaseException):
                    answers.append(piece.answer)
                else:
                    answers.append(layout[at:at + len(piece.answer)])
                    at += len(piece.answer)
            peer.connection.send(("answer", path, answers))
        except (OSError, ValueError):
            pass  # the client has gone; its connection closes on the next read


def _serve(address: str, authkey: bytes, device: str, ready: Any) -> None:
    """The server (module docstring), until a client sends "shutdown".

    The main thread reads messages; three threads pass rounds along: one
    joins the waiting pieces into feeds, one runs them (onnxruntime lets go
    of the interpreter while it runs), one cuts the outputs apart and
    answers. So the next round is joined, and the last answered, while the
    device runs this one."""
    import onnxruntime as ort

    # A thread coming back from a run waits for the interpreter until the
    # thread holding it is made to let go: 5 ms by default, longer than many
    # runs. Make it let go sooner.
    sys.setswitchinterval(_SWITCH_SECONDS)
    providers = _providers(device)
    directory = os.path.dirname(address)
    listener = Listener(address, family="AF_UNIX", authkey=authkey)
    arrived: queue.Queue = queue.Queue()
    wake_read, wake_write = get_context("spawn").Pipe(duplex=False)

    def accept() -> None:
        while True:
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            except Exception:  # a client that failed the handshake
                continue
            arrived.put(connection)
            wake_write.send_bytes(b"")

    threading.Thread(target=accept, daemon=True).start()
    sessions: dict[str, Any] = {}
    described: dict[str, dict] = {}
    stats = {"rounds": 0, "requests": 0, "pieces": 0, "forwards": 0, "rows": 0, "seconds": 0.0,
             "prepare_seconds": 0.0, "reply_seconds": 0.0, "round_trip_seconds": 0.0}
    state = threading.Condition()
    waiting: list[tuple[_Peer, list[_Piece]]] = []
    prepared: list[_Round] = []
    ran: list[_Round] = []
    # Set in turn at shutdown, each once the stage before it has ended.
    ended = {"preparer": False, "runner": False, "replier": False}

    # The next round is joined as late as it can be and still be ready when
    # the device comes free: from when the running round is expected to end,
    # less a join's expected time (both running means), so it holds every
    # request that arrives meanwhile.
    running: list[float | None] = [None]
    means = {"run": 0.0, "prepare": 0.0}

    def due() -> float | None:
        """Seconds until the next round should be taken, or None for now."""
        if running[0] is None:
            return None
        wait_for = running[0] + means["run"] - means["prepare"] - time.perf_counter()
        return wait_for if wait_for > 0 else None

    def preparer() -> None:
        while True:
            with state:
                while not (ended["preparer"] and not waiting):
                    if waiting and not prepared:
                        delay = due()
                        if delay is None:
                            break
                        state.wait(delay)
                    else:
                        state.wait()
                if not waiting:
                    return
                work = _Round(list(waiting))
                waiting.clear()
            started = time.perf_counter()
            _prepare(work, described)
            took = time.perf_counter() - started
            stats["prepare_seconds"] += took
            means["prepare"] += 0.2 * (took - means["prepare"])
            with state:
                prepared.append(work)
                state.notify_all()

    def runner() -> None:
        while True:
            with state:
                state.wait_for(lambda: prepared or ended["runner"])
                if not prepared:
                    return
                work = prepared.pop(0)
                running[0] = time.perf_counter()
                state.notify_all()
            _run(work, sessions, stats)
            with state:
                means["run"] += 0.2 * (time.perf_counter() - running[0] - means["run"])
                running[0] = None
                ran.append(work)
                state.notify_all()

    def replier() -> None:
        while True:
            with state:
                state.wait_for(lambda: ran or ended["replier"])
                if not ran:
                    return
                work = ran.pop(0)
            started = time.perf_counter()
            _reply(work, stats)
            ended_at = time.perf_counter()
            stats["reply_seconds"] += ended_at - started
            stats["round_trip_seconds"] += sum(ended_at - peer.since for peer, _ in work.messages)

    stages = [threading.Thread(target=stage, daemon=True) for stage in (preparer, runner, replier)]
    for stage in stages:
        stage.start()
    ready.send(("ready", ort.__version__))
    peers: dict[Any, _Peer] = {}
    numbered = 0
    while True:
        for connection in wait([*peers, wake_read]):
            if connection is wake_read:
                wake_read.recv_bytes()
                while not arrived.empty():
                    numbered += 1
                    arrival = arrived.get()
                    peers[arrival] = _Peer(arrival, directory, numbered)
                continue
            peer = peers[connection]
            try:
                message = connection.recv()
            except (EOFError, OSError):
                del peers[connection]
                connection.close()
                peer.outgoing.close()
                continue
            kind = message[0]
            if kind == "run":
                _, path, entries = message
                try:
                    buffer = peer.incoming.at(path)
                    pieces = [_Piece(peer, k, file, names, {key: buffer.view(at) for key, at in layout.items()},
                                     requests) for k, (file, names, layout, requests) in enumerate(entries)]
                except Exception as error:
                    connection.send(("error", f"{type(error).__name__}: {error}"))
                    continue
                peer.since = time.perf_counter()
                with state:
                    waiting.append((peer, pieces))
                    state.notify_all()
            elif kind == "load":
                path = message[1]
                try:
                    if path not in sessions:
                        options = ort.SessionOptions()
                        session = ort.InferenceSession(path, sess_options=options, providers=providers)
                        if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
                            raise RuntimeError("onnxruntime's CUDA provider did not load")
                        described[path] = _describe(session)
                        sessions[path] = session
                    connection.send(("loaded", described[path]))
                except Exception as error:
                    connection.send(("error", f"{type(error).__name__}: {error}"))
            elif kind == "stats":
                connection.send(("stats", dict(stats)))
            elif kind == "shutdown":
                for name, stage in zip(ended, stages):
                    with state:
                        ended[name] = True
                        state.notify_all()
                    stage.join()
                for each in peers.values():
                    each.outgoing.close()
                connection.send(("stats", dict(stats)))
                listener.close()
                return


class _Client:
    """This process's connection to a server: one round trip at a time. A
    message's arrays go through a buffer the two processes share; the
    message itself carries where they are."""

    def __init__(self, address: str, authkey: bytes) -> None:
        self.address, self.authkey = address, authkey
        self.pid = os.getpid()
        self.connection = Client(address, family="AF_UNIX", authkey=authkey)
        self.lock = threading.Lock()
        self.outgoing = _Writer(os.path.dirname(address), f"ask-{self.pid}-{secrets.token_hex(4)}")
        self.incoming = _Reader()

    def call(self, message: tuple) -> tuple:
        with self.lock:
            self.connection.send(message)
            return self.connection.recv()

    def run_many(self, requests: Sequence[tuple["RemoteSession", list[str], dict]]) -> list:
        """Each request's outputs, or the error the server met answering it,
        in one round trip. Requests for one file and one input set are
        joined here into one feed (`batching.join`) and cut back apart here,
        so the server handles one piece per file and process."""
        pieces: list[tuple[str, list[str], dict, list[int], Any]] = []
        by_key: dict[tuple, list[int]] = {}
        for k, (session, _, feed) in enumerate(requests):
            by_key.setdefault((session.path, tuple(sorted(feed))), []).append(k)
        for (path, _), members in by_key.items():
            session = requests[members[0]][0]
            if session.fixed or len(members) == 1:
                for k in members:
                    pieces.append((path, list(requests[k][1]), requests[k][2], [k], None))
                continue
            names: list[str] = []
            for k in members:
                names.extend(name for name in requests[k][1] if name not in names)
            joined = batching.join([requests[k][2] for k in members])
            pieces.append((path, names, joined.feed, members, joined))
        arrays = [value for _, _, feed, _, _ in pieces for value in feed.values()]
        with self.lock:
            path, layout = self.outgoing.write(arrays)
            entries, at = [], 0
            for file, names, feed, members, _ in pieces:
                entries.append((file, names, dict(zip(feed, layout[at:at + len(feed)])), len(members)))
                at += len(feed)
            self.connection.send(("run", path, entries))
            reply = self.connection.recv()
            if reply[0] != "answer":
                raise RuntimeError(f"the inference server answered {reply[0]!r}: {reply[1]}")
            _, path, answers = reply
            buffer = self.incoming.at(path) if any(not isinstance(a, BaseException) for a in answers) else None
            # Copied out: the buffer is rewritten by the next answer.
            answers = [a if isinstance(a, BaseException) else [np.array(buffer.view(at)) for at in a]
                       for a in answers]
        out: list[Any] = [None] * len(requests)
        for (_, names, _, members, joined), answer in zip(pieces, answers):
            if isinstance(answer, BaseException):
                for k in members:
                    out[k] = answer
                continue
            parts = batching.split(names, answer, joined) if joined is not None else [answer]
            for k, part in zip(members, parts):
                by_name = dict(zip(names, part))
                out[k] = [by_name[name] for name in requests[k][1]]
        return out


class RemoteSession:
    """A file on a server (`served`), as the part of
    `onnxruntime.InferenceSession` HexSet reads: `run`, `get_inputs`,
    `get_outputs`, `get_modelmeta` and `get_providers` (the server's).
    `joined` is the connection `hexset.lockstep` sends a lane run's waiting
    forwards over together (`_Client.run_many`)."""

    def __init__(self, client: _Client, path: str, description: dict) -> None:
        self.joined = client
        self.path = path
        self._inputs = [_Arg(*a) for a in description["inputs"]]
        self._outputs = [_Arg(*a) for a in description["outputs"]]
        self._meta = _ModelMeta(description["meta"])
        self._providers = description["providers"]
        self.fixed = _fixed(description)

    def get_inputs(self) -> list[_Arg]:
        return list(self._inputs)

    def get_outputs(self) -> list[_Arg]:
        return list(self._outputs)

    def get_modelmeta(self) -> _ModelMeta:
        return self._meta

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(self, outputs: list[str] | None, feed: dict[str, np.ndarray], run_options: Any = None) -> list:
        names = [a.name for a in self._outputs] if outputs is None else list(outputs)
        (answer,) = self.joined.run_many([(self, names, feed)])
        if isinstance(answer, BaseException):
            raise answer
        return answer


_clients: dict[str, _Client] = {}


def current() -> str | None:
    """The open server this process reads its files from (`ENV`), or `None`."""
    return os.environ.get(ENV) or None


def remote_session(path: str, server: str) -> RemoteSession:
    """`path` loaded on `server` (`current()`'s value), as a `RemoteSession`."""
    client = _clients.get(server)
    if client is None or client.pid != os.getpid():
        address, key = server.split("|", 1)
        client = _clients[server] = _Client(address, bytes.fromhex(key))
    kind, answer = client.call(("load", os.path.abspath(path)))
    if kind != "loaded":
        raise RuntimeError(f"the inference server could not load {path}: {answer}")
    return RemoteSession(client, os.path.abspath(path), answer)


def _shared_directory() -> str | None:
    """Memory-backed `/dev/shm` for the shared buffers where it has room
    (a container's is often 64 MB), else the default temporary directory."""
    try:
        space = os.statvfs("/dev/shm")
    except OSError:
        return None
    return "/dev/shm" if space.f_bavail * space.f_frsize >= 1 << 30 else None


@contextlib.contextmanager
def served(device: str = "cuda") -> Iterator[dict]:
    """Run every ONNX forward of this process, and of the processes it starts
    while the block is open, on one server on `device` (module docstring).
    Yields a dict that holds the server's counters once the block closes:
    `rounds`, `requests` (forwards asked), `pieces` (joined feeds the
    processes sent), `forwards` (graph runs), `rows`, `seconds` (the time
    the graph runs took), `prepare_seconds` and `reply_seconds` (joining and
    answering), and `round_trip_seconds`, summed over messages, from a
    message's arrival to its answer being sent."""
    _providers(device)
    if current() is not None:
        raise RuntimeError("a server is already open for this process")
    directory = tempfile.mkdtemp(prefix="hexset-serve-", dir=_shared_directory())
    address = os.path.join(directory, "socket")
    authkey = secrets.token_bytes(16)
    context = get_context("spawn")
    ready_read, ready_write = context.Pipe(duplex=False)
    process = context.Process(target=_serve, args=(address, authkey, device, ready_write), daemon=True)
    process.start()
    if not ready_read.poll(120):
        process.terminate()
        raise RuntimeError("the inference server did not start")
    ready_read.recv()
    token = f"{address}|{authkey.hex()}"
    os.environ[ENV] = token
    stats: dict = {}
    try:
        yield stats
    finally:
        del os.environ[ENV]
        _clients.pop(token, None)
        try:
            with contextlib.closing(Client(address, family="AF_UNIX", authkey=authkey)) as control:
                control.send(("shutdown", None))
                kind, counters = control.recv()
                stats.update(counters)
        finally:
            process.join(30)
            if process.is_alive():
                process.terminate()
            shutil.rmtree(directory, ignore_errors=True)
