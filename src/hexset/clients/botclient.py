"""Bots as peer clients of the pure API — the "on-disk ONNX" wrapper.

A bot seat is a client like any other: it joins, it polls `/api/state` for
whose move it is, and it submits its own action through the same
token-gated `/api/action` route a browser or an LLM uses. There is no
special server-side path for a bot any more (see `api.py`'s module
docstring) — what varies is only *how* a client decides its move, and
*where* it runs:

- **`RecordBrain`** plays a non-search (`NetworkBot`-shaped) contract-2
  checkpoint purely off the wire: `GET /api/record` is byte-identical to
  what `hexset.onnx_record.record_from_game` computes server-side, so there is nothing
  here to reconstruct and no way for a client to see more than an
  in-process bot would. This is what makes an external process a
  legitimate peer at all. Run as a real subprocess against a running
  server's public HTTP API (`python -m hexset.clients.botclient`), it needs
  nothing but that URL, a join code or a fresh game, and a checkpoint path.

- **`LocalSearchBrain`** is the escape hatch, and it is honestly a
  privileged one: it holds a direct reference to the live `Game` and calls
  `api.spawn_bot(...).choose(game)`, the same way every bot used to be
  driven server-side. This is the only way `search2` and an MCTS checkpoint
  can play today — a search needs the *true* state to simulate forward
  (deck contents, both dev-card piles, the setup queue, none of which are
  on the wire, deliberately), and until a proper information-set-safe
  determinizer exists (see `hexset.game.imagine`'s own note on the deck)
  that need is a documented cheat, not a solved problem. `web.py` is the
  only caller: a locally-picked bot at game creation gets a `LocalSearchBrain`
  thread regardless of whether the checkpoint actually searches, since
  running it in-process is free and `LocalSearchBrain` already covers every
  shape `api.spawn_bot` can hand back.

Either way, `BotRunner` is the same loop: poll, and when it's this seat's
turn, ask the brain to decide and submit exactly what it returns. No
cascade, no server-side "run the bot for me" endpoint — the board simply
advances on the submitted action, the same as any other seat's move.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import onnxruntime as ort

from hexset.actions import build_space
from hexset.bots import Bot

from hexset.server.constants import RECORD_CONTRACTS, TOKEN_HEADER
from hexset.server.modelmeta import search_config
from hexset.server.webplay import action_to_wire


# The two fields the graph reads as bool; every other declared input in the
# contract-2 record is int64 (see docs/onnx-contract-v2.md). Recomputed by
# `RecordBrain` from the budget-trimmed options rather than trusted from the
# wire as-is (see its own docstring for why).
_BOOL_FIELDS = frozenset({"action_mask"})

# The one record field that is not an integer count: valuations are floats
# in [-1, 1] (`hexset.trading`).
_FLOAT_FIELDS = frozenset({"valuations"})

# Fields `GET /api/record` sends alongside the record proper that are not
# themselves graph inputs.
_SIDECAR_FIELDS = frozenset({"options", "space"})


# --- Transport: the same two routes every client uses, HTTP or in-process ---


class Transport(Protocol):
    def get(self, path: str, token: str) -> dict: ...
    def post(self, path: str, token: str, body: dict) -> dict: ...


@dataclass
class HttpTransport:
    """A real client of a running server, over the same `/api/*` surface a
    browser or MCP uses — what makes an external `python -m
    hexset.clients.botclient` process a true peer rather than a special case."""

    base_url: str
    timeout: float = 30.0

    def get(self, path: str, token: str) -> dict:
        return self._request("GET", path, token, None)

    def post(self, path: str, token: str, body: dict) -> dict:
        return self._request("POST", path, token, body)

    def _request(self, method: str, path: str, token: str, body: dict | None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header(TOKEN_HEADER, token)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return json.loads(error.read().decode("utf-8"))


@dataclass
class LocalTransport:
    """The in-process shim: calls `Tables.handle` directly instead of over a
    socket. Same two methods `HttpTransport` has, so `BotRunner` cannot tell
    the difference — an embedded local bot is a peer client too, just one
    that skips the network hop (see the module docstring)."""

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


def _providers(device: str) -> list[str]:
    if not device or device == "cpu":
        return ["CPUExecutionProvider"]
    return [f"{device.upper()}ExecutionProvider", "CPUExecutionProvider"]


def _to_input(name: str, value) -> np.ndarray:
    if name in _BOOL_FIELDS:
        dtype = np.bool_
    elif name in _FLOAT_FIELDS:
        dtype = np.float32
    else:
        dtype = np.int64
    return np.asarray(value, dtype=dtype)[np.newaxis, ...]


@dataclass
class RecordBrain:
    """A non-search record-contract checkpoint, playing entirely off `GET
    /api/record` — no live `Game`, no reconstruction. Loads the ONNX session
    directly rather than through `onnxbot.load` (which wants a live
    `Topology` to fingerprint against); the wire record's own `space` field
    already carries the numbers that check needs, and `ActionSpace` only
    needs those, not the topology's adjacency."""

    session: ort.InferenceSession

    @classmethod
    def load(cls, spec: str, device: str = "cpu") -> "RecordBrain":
        session = ort.InferenceSession(spec, providers=_providers(device))
        meta = session.get_modelmeta().custom_metadata_map
        contract = meta.get("contract", "1")
        if contract not in RECORD_CONTRACTS:
            # Naming the contract actually found, rather than asserting it is
            # 1: PR #2 told the owner of a genuine contract-4 export that
            # their file "is a contract=1 checkpoint". Contract 1 itself is
            # no longer served at all — the owner dropped it 2026-09-02
            # (`docs/engine-divergence-2026-09-02.md`, B5) — so it is refused
            # here the same as any other unsupported contract, with no
            # "run it embedded instead" escape hatch left to point to.
            raise ValueError(
                f"{spec} declares contract={contract!r}; RecordBrain plays "
                f"{', '.join(sorted(RECORD_CONTRACTS))} only"
            )
        if search_config(meta).searches:
            raise ValueError(
                f"{spec} asks to be searched (metadata `search`) — a search needs the true "
                "game state to simulate forward, which no external client should have; run it "
                "as a local (embedded) bot instead (see LocalSearchBrain)"
            )
        return cls(session=session)

    def decide(self, transport: Transport, token: str, seat: int) -> dict:
        record = transport.get("/api/record", token)
        if "error" in record:
            raise RuntimeError(record["error"])
        space = build_space(**record["space"])

        # Keyed off the graph's own declared inputs rather than off every
        # field the record carries (see `onnxbot.V2Policy._run`).
        wanted = [i.name for i in self.session.get_inputs()]
        missing = [name for name in wanted if name not in record]
        if missing:
            raise RuntimeError(
                f"this checkpoint asks for {missing}, which /api/record does not carry"
            )
        inputs = {
            key: _to_input(key, record[key])
            for key in wanted
            if key not in _SIDECAR_FIELDS
        }
        (action_index,) = self.session.run(["action_index"], inputs)
        return action_to_wire(space.decode(int(action_index[0])))


@dataclass
class LocalSearchBrain:
    """In-process only, and explicitly privileged (see the module
    docstring): `bot` is whatever `api.spawn_bot` built — `search2`, a
    single-forward `NetworkBot`, or an MCTS `Search` — and `choose` is
    called against the session's own live `Game`, hidden information and
    all. Constructible only by `web.py`'s embedded runner thread; there is
    no HTTP route that could produce one."""

    bot: Bot
    game: object  # hexset.game.Game — typed loosely to avoid a game.py import cycle at module load

    def decide(self, transport: Transport, token: str, seat: int) -> dict:
        return action_to_wire(self.bot.choose(self.game))


# --- The runner: poll, decide when it's this seat's turn, submit -----------


@dataclass
class BotRunner:
    """One seat, driven by one brain, from outside the session — a bot plays
    exactly the way a human's client does: poll `/api/state`, and once it's
    this seat's turn, submit one action through `/api/action`. No cascade:
    the board advances on that submitted action like any other, and the
    next poll (this runner's, or anyone else's watching the game) sees it."""

    seat: int
    token: str
    transport: Transport
    brain: Brain
    poll_interval: float = 1.0
    stop: threading.Event = field(default_factory=threading.Event)

    def run_once(self) -> bool:
        """One iteration. Returns `False` once the game is over, so a caller
        driving this in a loop knows to stop."""
        view = self.transport.get("/api/state", self.token)
        if "error" in view:
            raise RuntimeError(view["error"])
        if view.get("game_over"):
            return False
        if view.get("to_move") != self.seat:
            return True
        wire = self.brain.decide(self.transport, self.token, self.seat)
        result = self.transport.post("/api/action", self.token, {"action": wire})
        if "error" in result:
            raise RuntimeError(result["error"])
        return True

    def run(self) -> None:
        """The loop `web.py` (embedded) or `__main__` (external) hands to a
        thread. Every error is logged and retried after `poll_interval`
        rather than killing the loop — a transient network hiccup, or one
        bad decision, shouldn't take a bot seat out of a long game."""
        while not self.stop.is_set():
            try:
                if not self.run_once():
                    return
            except Exception as error:  # noqa: BLE001 — one bad poll must not kill the runner
                print(f"botclient seat {self.seat}: {error}", file=sys.stderr)
            self.stop.wait(self.poll_interval)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of a running `python -m hexset.server.web`.")
    parser.add_argument("--game", required=True, help="The game's six-character code.")
    parser.add_argument(
        "--model", required=True, help="Path to a record-contract .onnx checkpoint (2, 3 or 4)."
    )
    parser.add_argument("--name", default=None, help="Display name for this seat.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args(argv)

    transport = HttpTransport(args.url.rstrip("/"))
    joined = transport.post("/api/join", "", {"code": args.game, "name": args.name})
    if "error" in joined:
        raise SystemExit(f"could not join {args.game}: {joined['error']}")
    token = joined["token"]
    seat = joined["seat"]
    print(f"joined {args.game} as seat {seat}", file=sys.stderr)

    brain = RecordBrain.load(args.model, device=args.device)
    runner = BotRunner(seat=seat, token=token, transport=transport, brain=brain, poll_interval=args.poll_interval)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _main()
