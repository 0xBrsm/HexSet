"""The server's own bot seats, run as clients of its API.

A bot the server seats plays through the same routes every client uses: a
`BotRunner` thread reads `GET /api/state` over `LocalTransport`, which calls
`Tables.handle` in process, and submits what its brain decides through
`POST /api/action`. Only how it decides differs from an external client:
`LocalSearchBrain` hands the bot the table's `Game`, as every engine table
hands its bots the position. The bot reads that game through its own seat's
information set (`Game.state(seat)`), a search included, which roots on
worlds sampled from it. Its trades go through its own gate, in process, so
the runner owes only moves and discards.
"""

from __future__ import annotations

import random
import sys
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Protocol

from hexset.bots import Bot, seat_at
from hexset.game import Game, Phase, imagine, to_move
from hexset.server.wire import action_to_wire


# --- Transport: the API's two verbs, in process ---------------------------


class Transport(Protocol):
    def get(self, path: str, token: str) -> dict: ...
    def post(self, path: str, token: str, body: dict) -> dict: ...


@dataclass
class LocalTransport:
    """The in-process shim: calls `Tables.handle` directly, with the same two
    methods an HTTP client has. `path` carries its query string, so
    `?after=`/`?wait=` mean the same as on the wire."""

    tables: object  # api.Tables, typed loosely to avoid a hard api.py import cycle

    def get(self, path: str, token: str) -> dict:
        return self._call("GET", path, token, {})

    def post(self, path: str, token: str, body: dict) -> dict:
        return self._call("POST", path, token, body)

    def _call(self, method: str, path: str, token: str, body: dict) -> dict:
        from hexset.server.api import ApiError  # deferred: api.py imports this module

        try:
            return self.tables.handle(method, path, body, token)
        except ApiError as error:
            return {"error": str(error)}


# --- Brains: how a seat decides its move -----------------------------------


class Brain(Protocol):
    def decide(self, transport: Transport, token: str, seat: int) -> dict: ...


@dataclass
class LocalSearchBrain:
    """A bot in the server's own process, asked to `choose` on the table's
    `Game` as an engine table asks it.

    The bot is seated at the game on construction (`seat_at`), not at its
    first move: the engine's trade event can ask its gate before that, and a
    checkpoint trained for another player count is refused here, with a
    `ValueError`, rather than on every turn it is asked to play.

    `lock` is held for the whole decision: the table's other requests -- an
    undo, another seat's discard -- change the same `Game` in place, and a
    bot must not read it halfway through one."""

    bot: Bot
    game: Game
    lock: AbstractContextManager = field(default_factory=nullcontext)

    def __post_init__(self) -> None:
        seat_at(self.bot, self.game)

    def decide(self, transport: Transport, token: str, seat: int) -> dict:
        with self.lock:
            return self._decide(seat)

    def _decide(self, seat: int) -> dict:
        game = self.game
        if game.phase is not Phase.DISCARD or to_move(game) == seat:
            return action_to_wire(self.bot.choose(game))
        # Every owing seat discards at once, but `choose` answers for
        # `to_move`, the lowest one: ask on a copy where this seat alone owes,
        # then seat the bot back at the table.
        asked = imagine(game, random.Random(0), randomize_deck=False)
        asked.discard_quota = [n if s == seat else 0 for s, n in enumerate(game.discard_quota)]
        try:
            return action_to_wire(self.bot.choose(asked))
        finally:
            seat_at(self.bot, game)


# --- The runner: act while the move is this seat's, wait otherwise ---------


# How long a runner waits before retrying after an error: nothing at the
# table has to change for a failed request to start working again.
ERROR_BACKOFF = 1.0

# The duties (`api.your_move`) a runner meets by asking its brain for a move.
_MOVES = frozenset({"act", "discard"})


class Unplayable(Exception):
    """The brain refused the position with a `ValueError` -- a checkpoint
    trained for another player count, say. Asking again cannot change the
    answer, so `BotRunner.run` stops on it rather than retrying."""


@dataclass
class BotRunner:
    """One seat, driven by one brain. Nothing is paced by a clock: a turn's
    actions go out back to back, and between turns the runner parks on
    `/api/state?after=<version>` for at most `poll_interval`, which the server
    caps in its own turn."""

    seat: int
    token: str
    transport: Transport
    brain: Brain
    poll_interval: float = 10.0
    stop: threading.Event = field(default_factory=threading.Event)

    def _state(self, query: str = "") -> dict:
        view = self.transport.get(f"/api/state{query}", self.token)
        if "error" in view:
            raise RuntimeError(view["error"])
        return view

    def run_once(self) -> bool:
        """One pass: every move this seat owes now -- its turn's actions, and
        its discards while another seat is `to_move` -- then a wait; `False`
        once the game is over. `stop` is checked before each decision, so a
        closing table never gets one more move in. Raises `Unplayable` where
        the brain refuses the position."""
        view = self._state()
        while not self.stop.is_set() and view.get("your_move") in _MOVES:
            try:
                wire = self.brain.decide(self.transport, self.token, self.seat)
            except ValueError as error:
                raise Unplayable(str(error)) from error
            result = self.transport.post("/api/action", self.token, {"action": wire})
            if "error" in result:
                raise RuntimeError(result["error"])
            view = self._state()
        if view.get("your_move") == "game_over":
            return False
        if not self.stop.is_set():
            self._wait_for_change(view.get("version"))
        return True

    def _wait_for_change(self, after: int | None) -> None:
        """Parks until the table moves past `after`; a view naming no version
        can only be waited out on a timer."""
        if after is None:
            self.stop.wait(self.poll_interval)
        else:
            self._state(f"?after={after}&wait={self.poll_interval}")

    def run(self) -> None:
        """The loop the server hands to a thread. An error is logged and
        retried after `ERROR_BACKOFF` rather than killing it, except
        `Unplayable`, which is logged and stops the runner."""
        while not self.stop.is_set():
            try:
                if not self.run_once():
                    return
            except Unplayable as error:
                print(f"bot seat {self.seat} stopped: {error}", file=sys.stderr)
                self.stop.set()
                return
            except Exception as error:  # noqa: BLE001 — one bad read must not kill the runner
                print(f"bot seat {self.seat}: {error}", file=sys.stderr)
                self.stop.wait(ERROR_BACKOFF)
