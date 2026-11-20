# SPDX-License-Identifier: GPL-3.0-only
"""A seat whose decisions a decision server makes, with a local bot to fall back on.

`Server` holds bots for many games at once and answers their calls; `RemoteBot` is one game's
seat on the client side. The client keeps one connection to the server open and both ends send a
heartbeat on it every `HEARTBEAT_S`, so the client always knows whether the link is up: it is up
while frames arrive and down once the connection closes or `LINK_DOWN_S` passes without one.

A call on a `RemoteBot` (`choose(game)`, or any other method the fallback bot has) goes to the
server while the link is up, and the client waits for the answer for as long as the link stays
up. It is answered by the local `fallback` bot instead, at once, when

- the link is down when the call is made, or goes down while it waits (a seat whose first
  connection is still being made waits for it: a game's first decisions come moments after the
  seat is made);
- the server refuses the client (another code version) or the call raises there;
- the decision's clock (`set_clock`) is down to `CLOCK_MARGIN_S`;
- `max_wait` seconds pass (a backstop for a call with no clock).

`seat_at(game)` and `seen(game, actor, action, viewer)` (`hexset.bots.base.seat_at`, `watch`) go to
both bots: the fallback at once, and the server without waiting for an answer. The client keeps
every such note of the game and sends them in order; when the server makes the game's bot afresh
(a restart lost it), the client sends it every note again before any call, so the server's bot has
seen what the seat has. Non-callable attributes are read from the fallback. Every call is written to `log` (one JSON line:
the method, who answered, how long the client waited and why it fell back), and the server's
search decision logs come back with each answer and are kept in `decision_log`, where
`hexset.arena.search_logs` finds them.

The server makes a game's bot on the client's first `open` (`build(spec)`), keeps it while calls
come (`SESSION_IDLE_S`), and keeps it across reconnects, so a link that drops and returns keeps the
game's bot and its state between decisions. A bot is made as soon as its game opens, beside any
call running. One thread runs every call: the lowest `priority`
first, then in arrival order. A call already running finishes; a call the client gave up on is
dropped if it has not started.

Frames are a 4-byte big-endian length and a pickle. Both ends must run the same code: the client
sends its `code` string in its hello and the server refuses another. Pickles are read from the
peer, so the server listens on loopback only and is reached over an authenticated tunnel.
"""

from __future__ import annotations

import heapq
import itertools
import json
import os
import pickle
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

HEARTBEAT_S = 1.0          # each end sends a heartbeat this often
LINK_DOWN_S = 3.0          # the client counts the link down after this long with no frame
SERVER_SILENT_S = 30.0     # the server closes a connection silent this long
CONNECT_S = 3.0            # a connection attempt, and the server's answer to the hello
RETRY_S = 1.0              # between connection attempts
REFUSED_RETRY_S = 30.0     # between attempts after the server refused this client's code
CLOCK_MARGIN_S = 3.0       # a call falls back when the decision's clock is down to this
MAX_WAIT_S = 120.0         # a call with no clock falls back after this long
SESSION_IDLE_S = 1800.0    # the server drops a game's bot after this long without a call
MAX_FRAME = 64 << 20

URGENT, HUMAN, OTHER = 0, 1, 2   # priorities a host may use: lower is served first


def send_frame(sock: socket.socket, obj: Any, lock: threading.Lock | None = None) -> None:
    """Write one frame: a 4-byte big-endian length and the pickle of `obj`."""
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    data = struct.pack(">I", len(body)) + body
    if lock is None:
        sock.sendall(data)
    else:
        with lock:
            sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> Any:
    """Read one frame (`send_frame`). A timeout set on `sock` applies to its first byte only
    when the caller handles `socket.timeout` and calls again."""
    (size,) = struct.unpack(">I", _recv_exact(sock, 4))
    if size > MAX_FRAME:
        raise ConnectionError(f"frame of {size} bytes exceeds {MAX_FRAME}")
    return pickle.loads(_recv_exact(sock, size))


def parse_address(address: str) -> tuple[str, int]:
    host, _, port = address.rpartition(":")
    return host or "127.0.0.1", int(port)


# -- server ------------------------------------------------------------------------------


@dataclass
class _Session:
    bot: Any
    used: float
    sent: dict = field(default_factory=dict)   # id(log list) -> rows already sent


@dataclass(order=True)
class _Job:
    priority: int
    seq: int
    conn: Any = field(compare=False)
    msg: dict = field(compare=False)


class _Conn:
    def __init__(self, sock: socket.socket, peer) -> None:
        self.sock, self.peer = sock, peer
        self.lock = threading.Lock()
        self.alive = True
        self.accepted = False
        self.last_rx = time.monotonic()

    def send(self, obj) -> None:
        if not self.alive:
            return
        try:
            send_frame(self.sock, obj, self.lock)
        except OSError:
            self.close()

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class Server:
    """Serve `build(spec)`'s bots to `RemoteBot` clients on `host:port` (loopback by default).

    `build` gets the spec of a client's `open` and returns the game's bot; `code` is compared with
    each client's. `logs(bot)` returns the bot's decision log lists (`hexset.arena.search_logs` by
    default), whose new rows go back with each answer. `serve_forever` blocks; `start` runs it in
    a thread; `close` stops it."""

    def __init__(self, build: Callable[[dict], Any], *, host: str = "127.0.0.1", port: int = 0,
                 code: str = "", logs: Callable[[Any], list] | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        if logs is None:
            from .arena import search_logs as logs
        self.build, self.code, self.logs = build, code, logs
        self.on_event = on_event or (lambda event: None)
        self.sessions: dict[str, _Session] = {}
        self.listener = socket.create_server((host, port))
        self.address = self.listener.getsockname()[:2]
        self._queue: list[_Job] = []
        self._ready = threading.Condition()
        self._seq = itertools.count()
        self._cancelled: set[tuple[int, Any]] = set()
        self._stop = threading.Event()
        self._conns: set[_Conn] = set()
        self._notes_seen: dict[str, int] = {}   # notes received per session, for a reconnect
        self._opening = threading.Lock()

    # the accept loop and each connection's reader and heartbeat
    def serve_forever(self) -> None:
        threading.Thread(target=self._run_jobs, name="remote-jobs", daemon=True).start()
        threading.Thread(target=self._beat, name="remote-beat", daemon=True).start()
        self.listener.settimeout(0.5)
        while not self._stop.is_set():
            try:
                sock, peer = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn = _Conn(sock, peer)
            self._conns.add(conn)
            threading.Thread(target=self._read, args=(conn,), name="remote-read", daemon=True).start()

    def start(self) -> "Server":
        threading.Thread(target=self.serve_forever, name="remote-server", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass
        for conn in list(self._conns):
            conn.close()
        with self._ready:
            self._ready.notify_all()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            now = time.monotonic()
            for conn in list(self._conns):
                if not conn.alive:
                    self._conns.discard(conn)
                elif now - conn.last_rx > SERVER_SILENT_S:
                    self.on_event({"event": "silent", "peer": str(conn.peer)})
                    conn.close()
                    self._conns.discard(conn)
                else:
                    conn.send({"op": "hb"})
            self._reap(now)

    def _reap(self, now: float) -> None:
        for key, session in list(self.sessions.items()):
            if now - session.used > SESSION_IDLE_S:
                self.sessions.pop(key, None)
                self._notes_seen.pop(key, None)
                self.on_event({"event": "idle", "session": key})

    def _read(self, conn: _Conn) -> None:
        conn.sock.settimeout(1.0)
        try:
            while conn.alive and not self._stop.is_set():
                try:
                    msg = recv_frame(conn.sock)
                except socket.timeout:
                    continue
                conn.last_rx = time.monotonic()
                op = msg.get("op")
                if op == "hb":
                    continue
                if op == "hello":
                    conn.accepted = msg.get("code") == self.code
                    conn.send({"op": "hello", "ok": conn.accepted, "code": self.code})
                    if not conn.accepted:
                        self.on_event({"event": "refused", "peer": str(conn.peer), "code": msg.get("code")})
                        conn.close()
                    continue
                if not conn.accepted:
                    conn.close()
                    return
                if op == "cancel":
                    self._cancelled.add((msg["id"], conn))
                elif op == "close":
                    self.sessions.pop(msg["session"], None)
                    self._notes_seen.pop(msg["session"], None)
                elif op == "open":
                    # made at once, beside any search running: a game's first decision follows
                    # its open by moments, and must not queue behind other games' searches
                    threading.Thread(target=lambda m=msg, c=conn: c.send(self._answer(m)),
                                     name="remote-open", daemon=True).start()
                elif op in ("call", "note"):
                    if op == "note":
                        self._notes_seen[msg["session"]] = self._notes_seen.get(msg["session"], 0) + 1
                    with self._ready:
                        heapq.heappush(self._queue, _Job(int(msg.get("priority", OTHER)), next(self._seq), conn, msg))
                        self._ready.notify()
        except (ConnectionError, OSError, EOFError, pickle.UnpicklingError):
            pass
        finally:
            conn.close()

    # the one thread that runs every open and call
    def _run_jobs(self) -> None:
        while not self._stop.is_set():
            with self._ready:
                while not self._queue and not self._stop.is_set():
                    self._ready.wait(0.5)
                if self._stop.is_set():
                    return
                job = heapq.heappop(self._queue)
            key = (job.msg.get("id"), job.conn)
            if key in self._cancelled:
                self._cancelled.discard(key)
                continue
            if not job.conn.alive and job.msg["op"] == "call":
                continue
            answer = self._answer(job.msg)
            if answer is not None:
                job.conn.send(answer)

    def _answer(self, msg: dict) -> dict:
        reply = {"op": "result", "id": msg.get("id")}
        started = time.monotonic()
        try:
            if msg["op"] == "open":
                with self._opening:
                    session = self.sessions.get(msg["session"])
                    created = session is None
                    if created:
                        self.sessions[msg["session"]] = _Session(self.build(msg["spec"]), time.monotonic())
                        self._notes_seen[msg["session"]] = 0
                        self.on_event({"event": "open", "session": msg["session"]})
                    else:
                        session.used = time.monotonic()
                return {**reply, "ok": True, "value": None, "created": created,
                        "notes": self._notes_seen.get(msg["session"], 0)}
            session = self.sessions.get(msg["session"])
            if session is None:
                return {**reply, "ok": False, "error": "no such session", "reopen": True}
            session.used = time.monotonic()
            if msg["op"] == "note":
                from .bots.base import seat_at, watch

                (seat_at if msg["name"] == "seat_at" else watch)(session.bot, *msg.get("args", ()))
                return None
            value = getattr(session.bot, msg["name"])(*msg.get("args", ()), **msg.get("kwargs", {}))
            return {**reply, "ok": True, "value": value, "logs": self._new_rows(session),
                    "took": time.monotonic() - started}
        except Exception as exc:   # the client falls back; the server keeps serving
            if msg.get("op") == "note":
                self.on_event({"event": "error", "op": "note", "name": msg.get("name"),
                               "error": f"{type(exc).__name__}: {exc}"})
                return None
            self.on_event({"event": "error", "op": msg.get("op"), "name": msg.get("name"),
                           "error": f"{type(exc).__name__}: {exc}"})
            return {**reply, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _new_rows(self, session: _Session) -> list:
        rows = []
        for log in self.logs(session.bot):
            seen = session.sent.get(id(log), 0)
            rows.extend(log[seen:])
            session.sent[id(log)] = len(log)
        return rows


# -- client ------------------------------------------------------------------------------


class _Fallback(Exception):
    def __init__(self, why: str) -> None:
        super().__init__(why)
        self.why = why


class RemoteBot:
    """One game's seat whose calls go to a `Server` at `address`, answered by `fallback` whenever
    the server cannot answer in time (the module docstring says when).

    `session` names the game on the server, `spec` is what the server's `build` makes its bot
    from, `priority` orders its calls against other games' (`URGENT`, `HUMAN`, `OTHER`), and
    `code` must equal the server's. `log` is a path the client appends a JSON line to for each call.
    `bot` is the fallback, so code that looks inside a seat for its network (a deal answerer's
    value head) finds the fallback's, the same file's net."""

    def __init__(self, address: str, session: str, spec: dict, fallback: Any, *,
                 priority: int = OTHER, code: str = "", log: str | os.PathLike | None = None,
                 max_wait: float = MAX_WAIT_S, connect: bool = True) -> None:
        own = object.__setattr__
        own(self, "bot", fallback)
        own(self, "decision_log", [])
        own(self, "address", address)
        own(self, "session", session)
        own(self, "spec", spec)
        own(self, "priority", int(priority))
        own(self, "code", code)
        own(self, "log_path", None if log is None else os.fspath(log))
        own(self, "max_wait", float(max_wait))
        own(self, "clock_left", None)
        own(self, "counts", {"remote": 0, "local": 0})
        own(self, "_sock", None)
        own(self, "_lock", threading.Lock())          # sends
        own(self, "_state", threading.Condition())   # replies and link state
        own(self, "_replies", {})
        own(self, "_ids", itertools.count(1))
        own(self, "_last_rx", 0.0)
        own(self, "_opened", False)
        own(self, "_refused", None)
        own(self, "_closed", threading.Event())
        own(self, "_tried", threading.Event())  # set once the first connection is up or has failed
        own(self, "_notes", [])                 # every seat_at and seen of the game, in order
        own(self, "_noted", 0)                  # how many of them this connection has sent
        if connect:
            threading.Thread(target=self._keep_link, name="remote-link", daemon=True).start()

    # -- what the driver and the seat see
    def set_clock(self, clock_left: Callable[[], float | None] | None) -> None:
        """`clock_left()` gives the seconds left on the current decision's clock, or None."""
        object.__setattr__(self, "clock_left", clock_left)

    def link_up(self) -> bool:
        return (self._sock is not None and self._opened
                and time.monotonic() - self._last_rx < LINK_DOWN_S)

    def choose(self, game):
        return self._call("choose", (game,), {})

    def seat_at(self, game) -> None:
        from .bots.base import seat_at

        seat_at(self.bot, game)
        self._note_remote("seat_at", (game,))

    def seen(self, game, actor, action, viewer) -> None:
        from .bots.base import watch

        watch(self.bot, game, actor, action, viewer)
        self._note_remote("seen", (game, actor, action, viewer))

    def _note_remote(self, name: str, args: tuple) -> None:
        try:
            frame = pickle.dumps({"op": "note", "session": self.session, "priority": self.priority,
                                  "name": name, "args": args}, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return
        with self._state:
            self._notes.append(frame)
        self._flush_notes()

    def _flush_notes(self) -> None:
        """Send the notes this connection has not sent yet, in order (under the send lock, so a
        call made after a note reaches the server after it)."""
        sock = self._sock
        if sock is None or not self._opened:
            return
        with self._lock:
            while self._noted < len(self._notes):
                frame = self._notes[self._noted]
                try:
                    sock.sendall(struct.pack(">I", len(frame)) + frame)
                except OSError:
                    return
                object.__setattr__(self, "_noted", self._noted + 1)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(object.__getattribute__(self, "bot"), name)
        if not callable(attr):
            return attr
        return lambda *args, **kwargs: self._call(name, args, kwargs)

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "bot"), name, value)

    def close(self) -> None:
        self._closed.set()
        sock = self._sock
        if sock is not None:
            try:
                send_frame(sock, {"op": "close", "session": self.session}, self._lock)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    # -- a call
    def _call(self, name: str, args: tuple, kwargs: dict):
        started = time.monotonic()
        try:
            value, rows = self._remote(name, args, kwargs, started)
        except _Fallback as fell:
            value = getattr(self.bot, name)(*args, **kwargs)
            self._note(name, "local", started, fell.why)
            return value
        if rows:
            self.decision_log.extend(rows)
        self._note(name, "remote", started, None)
        return value

    def _remote(self, name, args, kwargs, started):
        while not self.link_up() and not self._tried.is_set():
            why = self._out_of_time(started)
            if why is not None:
                raise _Fallback(f"still connecting; {why}")
            self._tried.wait(0.05)
        if not self.link_up():
            raise _Fallback(self._refused or "link down")
        call_id = next(self._ids)
        self._flush_notes()
        try:
            send_frame(self._sock, {"op": "call", "id": call_id, "session": self.session,
                                    "priority": self.priority, "name": name,
                                    "args": args, "kwargs": kwargs}, self._lock)
        except Exception as exc:   # a closed socket, or an argument that does not pickle
            raise _Fallback(f"send failed: {type(exc).__name__}") from None
        with self._state:
            while call_id not in self._replies:
                why = self._give_up(started)
                if why is not None:
                    self._cancel(call_id)
                    raise _Fallback(why)
                self._state.wait(0.05)
            reply = self._replies.pop(call_id)
        if not reply.get("ok"):
            if reply.get("reopen"):
                object.__setattr__(self, "_opened", False)
            raise _Fallback(f"server: {reply.get('error')}")
        return reply.get("value"), reply.get("logs") or []

    def _give_up(self, started: float) -> str | None:
        if not self.link_up():
            return "link dropped"
        return self._out_of_time(started)

    def _out_of_time(self, started: float) -> str | None:
        clock = self.clock_left
        left = clock() if clock is not None else None
        if left is not None and left <= CLOCK_MARGIN_S:
            return f"clock at {left:.1f}s"
        if time.monotonic() - started > self.max_wait:
            return f"no answer in {self.max_wait:.0f}s"
        return None

    def _cancel(self, call_id: int) -> None:
        sock = self._sock
        if sock is not None:
            try:
                send_frame(sock, {"op": "cancel", "id": call_id}, self._lock)
            except OSError:
                pass

    def _note(self, name: str, source: str, started: float, why: str | None) -> None:
        self.counts[source] += 1
        if self.log_path is None:
            return
        row = {"t": time.time(), "call": name, "source": source,
               "wait_s": round(time.monotonic() - started, 3)}
        if why is not None:
            row["why"] = why
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass

    # -- the link
    def _keep_link(self) -> None:
        while not self._closed.is_set():
            wait = RETRY_S
            try:
                self._connect_and_read()
            except (OSError, ConnectionError, EOFError, pickle.UnpicklingError):
                pass
            finally:
                self._drop()
            if self._refused:
                wait = REFUSED_RETRY_S
            self._closed.wait(wait)

    def _connect_and_read(self) -> None:
        host, port = parse_address(self.address)
        sock = socket.create_connection((host, port), timeout=CONNECT_S)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_frame(sock, {"op": "hello", "code": self.code})
        hello = recv_frame(sock)
        if not hello.get("ok"):
            object.__setattr__(self, "_refused", "server runs other code")
            sock.close()
            return
        object.__setattr__(self, "_refused", None)
        send_frame(sock, {"op": "open", "id": 0, "session": self.session, "spec": self.spec,
                          "priority": self.priority})
        sock.settimeout(LINK_DOWN_S)
        with self._state:
            object.__setattr__(self, "_sock", sock)
            object.__setattr__(self, "_last_rx", time.monotonic())
        beat = threading.Thread(target=self._beat, args=(sock,), name="remote-beat", daemon=True)
        beat.start()
        while not self._closed.is_set():
            try:
                msg = recv_frame(sock)
            except socket.timeout:
                return                       # nothing for LINK_DOWN_S: the link is down
            with self._state:
                object.__setattr__(self, "_last_rx", time.monotonic())
                if msg.get("op") == "result":
                    if msg.get("id") == 0:
                        if msg.get("ok"):
                            # the server says how many of the game's notes it has: a bot made
                            # afresh has none, and one a dropped link cut short gets the rest
                            object.__setattr__(self, "_noted", min(int(msg.get("notes", 0)), len(self._notes)))
                        object.__setattr__(self, "_opened", bool(msg.get("ok")))
                        self._tried.set()
                    else:
                        self._replies[msg["id"]] = msg
                self._state.notify_all()
            if msg.get("op") == "result" and msg.get("id") == 0 and msg.get("ok"):
                self._flush_notes()
            if msg.get("op") == "result" and msg.get("reopen"):
                send_frame(sock, {"op": "open", "id": 0, "session": self.session, "spec": self.spec,
                                  "priority": self.priority}, self._lock)

    def _beat(self, sock: socket.socket) -> None:
        while self._sock is sock and not self._closed.is_set():
            try:
                send_frame(sock, {"op": "hb"}, self._lock)
            except OSError:
                return
            self._closed.wait(HEARTBEAT_S)

    def _drop(self) -> None:
        self._tried.set()
        with self._state:
            sock = self._sock
            object.__setattr__(self, "_sock", None)
            object.__setattr__(self, "_opened", False)
            self._state.notify_all()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


__all__ = ["URGENT", "HUMAN", "OTHER", "RemoteBot", "Server", "parse_address",
           "recv_frame", "send_frame"]
