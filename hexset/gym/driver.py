# SPDX-License-Identifier: GPL-3.0-only
"""One batched game loop over Python or native lanes.

`Driver` plays the games a `hexset.gym.lanes.LaneEnv` deals. Each tick it
collects one request per live lane, groups the requests whose seat a caller
answers by policy id, asks each policy's `Seat` once for all of its rows, in
lane order and in the order the seats were given, and applies the answers.
Every finished game goes to `keep` before its lane is dealt again; a
per-decision hook sees each answered decision after it was applied. Bot seats
are played inside the environment, as `LaneEnv.step` plays them.

The lanes are either the environment's own Python games or native copies of
them (`hexset.gym.native`): `lanes="native"` imports each dealt game into the
native engine before its first move and refills from the same dealer in the
same order, so a game is the same function of `(seed, index)` on either kind
and both keep the same record. Native lanes need every seat to answer whole
decisions from visible buffers (`Seat.native_contract`), no bot, watcher or
trade gate, and the native extension; `lanes="auto"` takes them when that
holds and the Python lanes otherwise.

A request names its lane, game index, seat, policy id, step, decision run,
the decision (`hexset.decisions.tensors`) or atomic options, and the live
`Game` on a Python lane. `observations(requests, contract)` reads each row's
observation under a contract: the native lane's own buffer where it holds
that contract, the encoder of the `Game` otherwise. `Driver.targets()` holds
the referee's labels for the tick's rows (every seat's true points, the turn
count): never part of a request, valid until the tick's answers are applied.

A failed apply or callback stops the driver; nothing retries a move or a
chance draw.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .. import contract9, decisions as D
from ..actions import Action, space_for
from ..encoding import encode_batch
from ..game import Game
from ..victory import victory_points
from .lanes import Episode, LaneEnv

LANE_KINDS = ("python", "native", "auto")

_MASK64 = (1 << 64) - 1
_GOLDEN64 = 0x9E3779B97F4A7C15


def _mix64(x: int) -> int:
    """splitmix64's finaliser: a bijection on 64-bit integers that spreads
    every input bit over the output."""
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def draw_key(seed: int, index: int, seat: int, step: int) -> int:
    """The 64-bit key of one decision's sampling draw.

    `step` indexes the game's whole action stream, so `(index, step)` alone
    names a decision; `seat` is folded in as well so a key can never be shared
    by two seats. Pure Python integers: the same on every platform and every
    numpy version. A policy that samples from it alone plays each game as a
    pure function of `(seed, index)`, however games are batched or sharded.
    """
    key = _mix64(seed + _GOLDEN64)
    for part in (index, seat, step):
        key = _mix64((key ^ (part & _MASK64)) + _GOLDEN64)
    return key


@dataclass(frozen=True)
class Request:
    """One decision put to a seat, the same on either lane kind.

    `row` is the request's place in the tick (`Driver.targets()` rows), `lane`
    its slot and `index` its game. `decision` holds the whole decision for a
    seat asked for decisions, with empty `options`; otherwise `options` lists
    the atomic actions. `game` is the live position on a Python lane and None
    on a native one: read it, never mutate it, and encode before answering.
    `native` is a native lane's visible observation, `(contract, row)`.
    """

    row: int
    lane: int
    index: int
    seat: int
    policy: int
    step: int
    decision_id: int
    part: int
    seed: int
    decision: D.Decision | None = None
    options: tuple[Action, ...] = ()
    game: Game | None = None
    native: tuple[str, np.ndarray] | None = None

    @property
    def draw(self) -> int:
        """This decision's sampling key (`draw_key`)."""
        return draw_key(self.seed, self.index, self.seat, self.step)


@dataclass(frozen=True)
class Targets:
    """The referee's labels for one tick's rows, in `Driver.requests()` order:
    every seat's true victory points (hidden cards counted), board order, and
    the game's turn count. Labels for recording, never an input."""

    points: tuple[tuple[int, ...], ...]
    turns: tuple[int, ...]


class Answer(Protocol):
    """A seat's answer: an action, or for a decision any of its options
    (`hexset.decisions.Compound` included). `decision` locates it in the
    request's tensors, `(DecisionKind, index)`; a native lane requires it."""

    action: Action | D.Compound
    decision: tuple[int, int] | None


class Seat(Protocol):
    """Who answers one policy id: `choose_many(requests)` returns one answer
    per request, in order, and is called once a tick with all of the policy's
    rows. `native_contract`, when a seat has it, is the observation contract
    it reads from a native lane's visible buffers; a seat without one reads
    the `Game` and plays on Python lanes only."""

    def choose_many(self, requests: Sequence[Request]) -> Sequence[Answer]: ...


def observations(requests: Sequence[Request], contract: str) -> list:
    """Each request's observation under `contract`, in order.

    A contract-9 family row is `hexset.contract9.flatten`'s float32 vector, a
    contract-8 family one an `hexset.encoding.Observation`. A native lane's
    own buffer serves its contract; every other row is encoded from its game,
    one `encode_batch` for the contract-8 rows."""
    out: list = [None] * len(requests)
    games: list[int] = []
    for i, request in enumerate(requests):
        if request.native is not None and request.native[0] == contract:
            out[i] = request.native[1]
        elif request.game is not None:
            games.append(i)
        else:
            raise ValueError(f"a native lane holds no contract-{contract} observation")
    if not games:
        return out
    if contract in contract9.CONTRACTS:
        for i in games:
            out[i] = contract9.flatten(contract9.encode9(requests[i].game, requests[i].seat, contract))
    else:
        encoded = encode_batch([requests[i].game for i in games], [requests[i].seat for i in games], contract)
        for i, observation in zip(games, encoded, strict=True):
            out[i] = observation
    return out


def seat_points(game: Game) -> tuple[int, ...]:
    """Every seat's true victory points, board order, hidden cards counted."""
    state = game.state(0, hidden=False)
    return tuple(victory_points(state, seat) for seat in range(state.num_players))


class Driver:
    """Play `env`'s games with `seats` answering their policy ids.

    `keep(episode)` is required and gets each finished game, a
    `hexset.gym.lanes.Episode` with its record, before the lane is dealt
    again; it must return only once the game is retained, and what it returns
    is what `tick` and `play` return in the game's place (None keeps the
    episode). `on_decision(request, answer)`, when given, runs once per answered
    decision after its first part is applied, before the game's `keep`.
    `beat()`, when given, runs once a tick.

    `lanes` is `LANE_KINDS`' choice. Native lanes check every game against
    `planned` (the indices the run will deal; by default what the bounded
    environment has dealt and will deal) and `finished` (indices already kept,
    which are never dealt again).
    """

    def __init__(self, env: LaneEnv, seats: Mapping[int, Seat], *, keep: Callable[[Episode], object],
                 on_decision: Callable[[Request, Answer], None] | None = None, lanes: str = "python",
                 planned: Sequence[int] | None = None, finished: Sequence[int] = (),
                 beat: Callable[[], None] | None = None) -> None:
        if not callable(keep):
            raise ValueError("a driver keeps every game it plays: pass keep")
        if lanes not in LANE_KINDS:
            raise ValueError(f"lanes must be one of {LANE_KINDS}, got {lanes!r}")
        if set(seats) & set(env.bots):
            raise ValueError("a bot's policy id is played by the environment, not a seat")
        self.env, self.seats = env, dict(seats)
        self.keep, self.on_decision, self.beat = keep, on_decision, beat
        self.ticks = self.steps = self.games = 0
        self._asked: tuple[Request, ...] | None = None
        self._targets: Targets | None = None
        self._native = None
        if lanes == "auto":
            lanes = "native" if self._native_feasible() is None else "python"
        if lanes == "native":
            reason = self._native_feasible()
            if reason is not None:
                raise ValueError(f"native lanes cannot play this table: {reason}")
            self._start_native(planned, finished)
        self.lanes = lanes

    # --- choosing and starting lanes ----------------------------------------

    def native_contract(self) -> str | None:
        """The one contract every seat reads natively, or None."""
        contracts = {getattr(seat, "native_contract", None) for seat in self.seats.values()}
        return contracts.pop() if len(contracts) == 1 else None

    def _native_feasible(self) -> str | None:
        """Why native lanes cannot play this table, or None if they can."""
        from .native import available

        env = self.env
        contract = self.native_contract()
        if not available():
            return "the native extension is unavailable"
        if contract not in contract9.CONTRACTS:
            return "every seat must read one native contract"
        if env.bots or env.gates:
            return "a bot or a trade gate needs a Python game"
        if not set(self.seats) <= env.decisions:
            return "every seat must be asked for whole decisions"
        if env._stop is None:
            return "native lanes play a bounded deal"
        return None

    def _start_native(self, planned, finished) -> None:
        from .native import NativeBatch, supported

        env = self.env
        contract = self.native_contract()
        dealt = [(slot, lane) for slot, lane in enumerate(env._lanes) if lane is not None]
        if planned is None:
            planned = [lane.index for _, lane in dealt]
            planned += list(env._listed) if env._listed is not None else list(range(env._next, env._stop, env.stride))
        self._slots = len(env._lanes)
        if not dealt:
            self._batch = None
            return
        first = dealt[0][1].game
        if not supported(first, contract):
            raise ValueError("native lanes cannot hold this game")
        self.space = space_for(first)
        self.pairs = tuple(tuple(map(int, pair)) for pair in
                           contract9.incidences(first.state(0, hidden=False).board.topology))
        self.contract = contract
        lanes = [self._import(lane) for _, lane in dealt]
        # The native batch owns the positions now; the environment's slots
        # stay as they are counted, empty, so it deals refills the same way.
        for slot, _ in dealt:
            env._lanes[slot] = None
        del first, dealt
        self._batch = NativeBatch(lanes, space=self.space, pairs=self.pairs, contract=contract,
            planned_indices=planned, finished_indices=finished, keep=self._kept_native,
            on_decision=self._decided_native if self.on_decision is not None else None,
            discard_order=env.discard_order)

    def _import(self, dealt):
        from .native import NativeLane

        if any(dealt.seat_bots) or dealt.watching:
            raise ValueError("native lanes seat no scripted or watching bots")
        return NativeLane.import_game(dealt.game, index=dealt.index, seed=self.env.seed, tape=dealt.tape,
                                      action_cap=self.env.action_cap, cast=dealt.cast)

    # --- the tick --------------------------------------------------------------

    @property
    def running(self) -> bool:
        if self.lanes == "native":
            return self._batch is not None and self._batch.running
        return self.env.running

    def requests(self) -> tuple[Request, ...]:
        """This tick's requests for the seats, in lane order; the same until
        the tick is applied."""
        if self._asked is None:
            self._asked = self._native_requests() if self.lanes == "native" else self._python_requests()
        return self._asked

    def _python_requests(self) -> tuple[Request, ...]:
        seed, bots = self.env.seed, self.env.bots
        out = []
        for r in self.env.requests():
            if r.policy in bots:
                continue
            out.append(Request(len(out), r.lane, r.index, r.seat, r.policy, r.step, r.decision_id, r.part,
                               seed, r.decision, r.options, r.game))
        return tuple(out)

    def _native_requests(self) -> tuple[Request, ...]:
        seed, contract = self.env.seed, self.contract
        self._by_ids = {}
        out = []
        for row, r in enumerate(self._batch.requests()):
            ids = r.ids
            request = Request(row, ids[1], ids[0], ids[2], ids[4], ids[3], ids[5], ids[6], seed,
                              r.decision, (), None, (contract, r.observation))
            self._by_ids[ids] = request
            out.append(request)
        return tuple(out)

    def targets(self) -> Targets:
        """The referee's labels for `requests()`, row for row; valid until the
        tick's answers are applied."""
        if self._targets is None:
            asked = self.requests()
            if self.lanes == "native":
                labels = self._batch.targets()
                self._targets = Targets(tuple(tuple(map(int, row)) for row in labels.points),
                                        tuple(map(int, labels.turns)))
            else:
                self._targets = Targets(tuple(seat_points(r.game) for r in asked),
                                        tuple(r.game.turns for r in asked))
        return self._targets

    def _answers(self, asked: Sequence[Request]) -> list:
        """One `choose_many` per seat, in the order the seats were given,
        reassembled in request order."""
        shares: dict[int, list[int]] = {pid: [] for pid in self.seats}
        for i, request in enumerate(asked):
            share = shares.get(request.policy)
            if share is None:
                raise ValueError(f"lane {request.lane} seat {request.seat} is policy {request.policy}, "
                                 "which no seat answers")
            share.append(i)
        out: list = [None] * len(asked)
        for pid, share in shares.items():
            if not share:
                continue
            answers = list(self.seats[pid].choose_many([asked[i] for i in share]))
            if len(answers) != len(share):
                raise ValueError(f"policy {pid} answered {len(answers)} of {len(share)} requests")
            for i, answer in zip(share, answers):
                out[i] = answer
        return out

    def tick(self) -> list:
        """Answer and apply one request per live lane; return what `keep`
        returned for each game that finished."""
        if not self.running:
            return []
        if self.lanes == "native":
            out = self._tick_native()
        else:
            out = self._tick_python()
        self.ticks += 1
        self.games += len(out)
        if self.beat is not None:
            self.beat()
        return out

    def _tick_python(self) -> list:
        env = self.env
        outstanding = env.requests()
        if not outstanding:
            return []
        asked = self.requests()
        answers = self._answers(asked)
        if self.on_decision is not None:
            self.targets()
        # A lane left out is played by its seat's own bot (`LaneEnv.step`).
        finished = env.step({request.lane: answer.action for request, answer in zip(asked, answers)})
        self.steps += len(outstanding)
        if self.on_decision is not None:
            for request, answer in zip(asked, answers):
                self.on_decision(request, answer)
        self._asked = self._targets = None
        out = []
        for episode in finished:
            kept = self.keep(episode)
            out.append(episode if kept is None else kept)
        return out

    def _tick_native(self) -> list:
        batch = self._batch
        asked = self.requests()
        answers = self._answers(asked)
        if self.on_decision is not None:
            self.targets()
        out = list(batch.advance(answers))
        self.steps += len(asked)
        self._asked = self._targets = None
        for slot, lane in enumerate(batch.lanes):
            if lane.finished:
                fresh = self.env._fresh()
                if fresh is not None:
                    batch.replace_lane(slot, self._import(fresh))
        return out

    def _decided_native(self, native_request, answer) -> None:
        self.on_decision(self._by_ids[native_request.ids], answer)

    def _kept_native(self, completion):
        lanes_episode = Episode(completion.index, completion.seed, completion.players, completion.cast,
                                tuple(() for _ in range(completion.players)), completion.outcome,
                                record=completion.record)
        kept = self.keep(lanes_episode)
        return lanes_episode if kept is None else kept

    def play(self) -> list:
        """Tick until every dealt game is finished and kept; requires a
        bounded environment."""
        if self.env._stop is None and self.lanes == "python":
            raise ValueError("an unbounded environment never finishes; pass `deal`")
        out: list = []
        while self.running:
            out.extend(self.tick())
        return out

    def games_started(self) -> int:
        return self.env.games_started()


__all__ = ["Answer", "Driver", "LANE_KINDS", "Request", "Seat", "Targets", "draw_key", "observations",
           "seat_points"]
