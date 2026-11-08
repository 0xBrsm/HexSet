# SPDX-License-Identifier: GPL-3.0-only
"""Native no-trade lanes with visible requests and durable completion hooks.

The caller owns dealing, sampling and refill order. This bridge owns each
imported position and applies complete decisions with the original Chance and
Tape protocols. Select a backend before play; a failed apply or callback
poisons the batch and never falls back or retries a random event.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from operator import index as integer
import random

import numpy as np

from .. import contract9, decisions as D
from ..actions import Action, ActionSpace, build_space
from ..arena import game_key
from ..chance import Balanced, Recording
from ..native import NativeState, NativeUnavailable, available as _state_available, supported as _state_supported
from ..record import Record, Tape, board_fields
from .lanes import Outcome

try:
    from .._native_core import RequestBatch as _NativeRequests
except ImportError:
    _NativeRequests = None


def available():
    """Whether both owned state and batched request bindings are installed."""
    return _state_available() and _NativeRequests is not None


def supported(game, contract="9"):
    """Choose lanes only when their full binding and state are supported."""
    return available() and _state_supported(game, contract)


@dataclass(frozen=True)
class NativeRequest:
    """A deciding seat's immutable visible buffers; no Game or referee labels.

    ids are game index, lane, seat, atomic step, policy, decision id and part.
    Buffers keep their own lifetimes after the next request is packed. A
    whole-decision row holds its `observation` and `decision`; an atomic row
    (a policy in the batch's `atomic` set) holds its legal `actions`, in
    `hexset.actions.legal_actions`' order, and `view`, a read-only
    `hexset.native.playout.NativeCopy` of the lane's position that a policy
    reads as it reads a `Game`, valid until the lane moves.
    """
    epoch: int
    ids: tuple[int, ...]
    observation: np.ndarray | None
    decision: D.Decision | None
    actions: tuple[Action, ...] = ()
    view: object | None = None

    @property
    def index(self): return self.ids[0]
    @property
    def lane(self): return self.ids[1]
    @property
    def seat(self): return self.ids[2]
    @property
    def step(self): return self.ids[3]
    @property
    def policy(self): return self.ids[4]
    @property
    def decision_id(self): return self.ids[5]
    @property
    def part(self): return self.ids[6]
    @property
    def mask(self): return None if self.decision is None else self.decision.flat
    @property
    def options(self): return self.actions


@dataclass(frozen=True)
class NativeAnswer:
    """A legal action/compound and its (DecisionKind, candidate index)."""
    action: Action | D.Compound
    decision: tuple[int, int]


@dataclass(frozen=True)
class NativeTargets:
    """Referee-only recording targets, deliberately separate from requests."""
    points: np.ndarray
    turns: np.ndarray


@dataclass(frozen=True)
class NativeCompletion:
    """A whole game's result; the keep callback must durably retain it."""
    index: int
    seed: int
    players: int
    cast: tuple[int, ...]
    outcome: Outcome
    record: Record


@dataclass
class NativeLane:
    """One owned no-trade position, its complete Tape and action counters."""
    native: NativeState
    index: int
    seed: int
    cast: tuple[int, ...]
    metadata: dict
    tape: Tape
    action_cap: int
    actions: int = 0
    runs: int = 0
    running: dict[int, tuple[int, int]] = field(default_factory=dict)
    discard_rng: random.Random | None = None
    discarding: int | None = None
    finished: bool = False
    failed: BaseException | None = None

    @classmethod
    def import_game(cls, game, *, index, seed, tape, action_cap, cast=()):
        """Own a recorded fresh stream without retaining the Python Game.

        Whole-game resume belongs to the caller: import only missing games,
        with an empty Tape and the game's existing recorded Chance prefix.
        """
        if not isinstance(tape, Tape) or not isinstance(game.chance, Recording):
            raise TypeError("native lanes require a Tape and recorded Chance")
        if tape.actions or tape.trades or tape.shown or tape.actors:
            raise ValueError("import fresh streams; resume only missing whole games")
        index, seed, action_cap = integer(index), integer(seed), integer(action_cap)
        players = game.num_players
        cast = tuple(map(integer, cast)) if cast else (0,)*players
        if index < 0 or action_cap <= 0 or len(cast) != players or any(pid < 0 for pid in cast):
            raise ValueError("invalid native index, action cap or seat cast")
        metadata = dict(board_fields(game._state.board), num_players=players,
            rules=game._state.rules, seed=game_key(seed, index), first=game.first,
            balanced_dice=isinstance(game.chance.inner, Balanced),
            split_streams=getattr(game.chance.inner, "split", False), locked=tuple(sorted(game.locked)))
        return cls(NativeState(game), index, seed, cast, metadata, copy.deepcopy(tape), action_cap)

    @property
    def players(self):
        return len(self.cast)

    def record(self):
        control = self.native.core.control()
        return Record(**self.metadata, actions=tuple(self.tape.actions),
            chance=tuple(self.native._chance.events), trades=tuple(self.tape.trades),
            shown=tuple(self.tape.shown), actors=tuple(self.tape.actors),
            winner=None if control[9] < 0 else control[9], turns=control[8])


class NativeBatch:
    """Persistent native lanes with no model or sampling dependency.

    `keep(completion)` is mandatory and must return only after the whole game
    is durably retained. `on_decision(request, answer)`, if supplied, runs once
    after the decision's first successful atomic apply and Tape append. All
    answers are checked before any lane is changed. A callback/apply failure
    poisons this batch, including a partially applied compound.

    Seats whose policy is in `atomic` are asked one atomic move at a time, as
    a Python lane asks them: the row's legal actions and a view of the lane
    (`NativeRequest`), answered with one of the actions. Every other seat is
    asked whole decisions under `contract`.
    """
    def __init__(self, lanes, *, space: ActionSpace, pairs, contract, planned_indices,
                 keep, on_decision=None, discard_order="seat", finished_indices=(), atomic=()):
        if _NativeRequests is None:
            raise NativeUnavailable("the optional native extension is unavailable")
        self.lanes = list(lanes)
        if not self.lanes or not callable(keep) or on_decision is not None and not callable(on_decision):
            raise ValueError("native lanes require a durable completion callback")
        if discard_order not in ("seat", "random"):
            raise ValueError("unsupported discard order")
        self.space, self.contract = space, contract
        self.pairs = tuple(tuple(map(integer, pair)) for pair in pairs)
        self.keep, self.on_decision, self.discard_order = keep, on_decision, discard_order
        self.atomic = frozenset(map(integer, atomic))
        self._planned = set(map(integer, planned_indices))
        self._known = set(map(integer, finished_indices))
        indices = [lane.index for lane in self.lanes]
        if (len(set(indices)) != len(indices) or not set(indices) <= self._planned
                or self._known & set(indices) or not self._known <= self._planned
                or any(i < 0 for i in self._planned)
                or any(lane.players != space.num_players for lane in self.lanes)):
            raise ValueError("native lanes must be distinct missing indices in the complete plan")
        self._known.update(indices)
        # The binding checks exact topology equality across the imported lanes.
        self._native = _NativeRequests([lane.native.core for lane in self.lanes], contract)
        board = self.lanes[0].metadata
        from ..board.coords import Hex
        from ..board.topology import build
        topology = self.topology = build(Hex(*hexagon) for hexagon in board["layout"])
        if space != build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, self.lanes[0].players):
            raise ValueError("action space differs from the imported topology")
        expected_pairs = contract9.incidences(topology)
        if self.pairs != expected_pairs:
            raise ValueError("setup incidences differ from the imported topology")
        self._asked = self._targets = self._epoch = None
        self._failed = None

    def _healthy(self):
        if self._failed is not None:
            raise RuntimeError("native batch failed; do not retry a move or Chance event") from self._failed

    @property
    def running(self):
        return any(not lane.finished for lane in self.lanes)

    def replace_lane(self, slot, lane):
        self._healthy()
        slot = integer(slot)
        if self._asked is not None or not self.lanes[slot].finished:
            raise ValueError("replace only a retired lane between request epochs")
        if lane.players != self.space.num_players or lane.index not in self._planned or lane.index in self._known:
            raise ValueError("replacement must be a distinct missing game in the plan")
        self._native.replace(slot, lane.native.core)
        self.lanes[slot] = lane
        self._known.add(lane.index)

    def requests(self):
        self._healthy()
        if self._asked is not None:
            return self._asked
        try:
            return self._requests()
        except BaseException as error:
            self._failed = error
            raise

    def _requests(self):
        ids, atomic = [], []
        for slot, actors in enumerate(self._native.actors()):
            lane = self.lanes[slot]
            if lane.finished:
                continue
            if not actors:
                raise ValueError("a terminal imported lane must be retained before collection")
            seat = actors[0]
            if self.discard_order == "random" and lane.native.core.phase() == 3:
                # As `LaneEnv._actor`: the seat drawn keeps discarding until
                # its quota is met, a card at a time for an atomic seat.
                if lane.discarding is None or lane.discarding not in actors:
                    if lane.discard_rng is None:
                        lane.discard_rng = random.Random(f"discard-order:{game_key(lane.seed, lane.index)}")
                    lane.discarding = lane.discard_rng.choice(actors)
                seat = lane.discarding
            else:
                lane.discarding = None
            if seat not in lane.running:
                lane.running[seat] = lane.runs, 0
                lane.runs += 1
            run, part = lane.running[seat]
            ids.append((lane.index, slot, seat, lane.actions, lane.cast[seat], run, part))
            atomic.append(lane.cast[seat] in self.atomic)
        packet = self._native.fill([identity for identity, alone in zip(ids, atomic) if not alone])
        n, width, flat_width, setup_width = packet["shape"]
        def array(name, dtype, shape):
            value = np.frombuffer(packet[name], dtype=dtype).reshape(shape)
            if value.flags.writeable:
                raise ValueError("native request buffers must be immutable owned snapshots")
            return value
        observations = array("observation", np.float32, (n, width))
        flat = array("flat", np.bool_, (n, flat_width))
        setup = array("setup", np.bool_, (n, setup_width))
        roads = array("rb", np.int16, (-1, 2)); ro = array("rb_offsets", np.int64, (n+1,))
        discards = array("discard", np.int8, (-1, 10)); do = array("discard_offsets", np.int64, (n+1,))
        self._epoch = packet["epoch"]
        asked, row = [], 0
        # Each asked row's place in the packed whole-decision rows (None for an atomic row).
        self._filled = []
        for identity, alone in zip(ids, atomic):
            if alone:
                asked.append(self._atomic_request(identity))
                self._filled.append(None)
                continue
            self._filled.append(row)
            decision = D.Decision(flat[row], setup[row], False, roads[ro[row]:ro[row+1]],
                                  discards[do[row]:do[row+1]])
            asked.append(NativeRequest(self._epoch, identity, observations[row], decision))
            row += 1
        self._asked = tuple(asked)
        return self._asked

    def _atomic_request(self, identity):
        """An atomic row: the seat's legal actions as a Python lane lists
        them (`options_for` for the seat to move, `legal_actions(game, seat)`
        for another seat owing a discard) and a view of the lane."""
        from ..actions import Stuck
        from ..native.playout import NativeCopy
        lane, seat = self.lanes[identity[1]], identity[2]
        view = NativeCopy.viewing(lane.native, self.topology, lane.players, lane.metadata["locked"])
        actions = view.legal() if seat == view.to_move() else view.legal(seat)
        if not actions:
            raise Stuck(f"no legal action for player {seat} in lane {identity[1]}")
        return NativeRequest(self._epoch, identity, None, None, tuple(actions), view)

    def targets(self):
        """Referee recording labels; never part of any NativeRequest."""
        self.requests()
        if self._targets is None:
            labels = self._native.targets(self._epoch)
            whole = np.frombuffer(labels["points"], np.int64).reshape(-1, self.space.num_players)
            turns = np.frombuffer(labels["turns"], np.int64)
            if len(whole) == len(self._asked):
                self._targets = NativeTargets(whole, turns)
                return self._targets
            whole, turns = iter(whole), iter(turns)
            points, turned = [], []
            for request in self._asked:
                if request.view is None:
                    points.append(next(whole))
                    turned.append(next(turns))
                else:
                    core = self.lanes[request.lane].native.core
                    points.append(np.asarray(core.privileged_true_points(), np.int64))
                    turned.append(core.control()[8])
            points = np.asarray(points, np.int64).reshape(len(self._asked), self.space.num_players)
            turned = np.asarray(turned, np.int64)
            points.flags.writeable = turned.flags.writeable = False
            self._targets = NativeTargets(points, turned)
        return self._targets

    def _validate_answers(self, answers):
        asked = self.requests()
        if len(answers) != len(asked):
            raise ValueError("one answer is required per native request")
        self._native.check(self._epoch)
        actions = []
        for request, answer in zip(asked, answers):
            if request.view is not None:
                if getattr(answer, "decision", None) is not None or answer.action not in request.actions:
                    raise ValueError("answer is not one of the atomic row's legal actions")
                actions.append(answer.action)
                continue
            if answer.decision is None or len(answer.decision) != 2:
                raise ValueError("answer must name its decision candidate")
            kind, at = map(integer, answer.decision)
            decision = request.decision
            if kind == D.DecisionKind.FLAT:
                valid = 0 <= at < len(decision.flat) and decision.flat[at]
            elif kind == D.DecisionKind.SETUP:
                valid = 0 <= at < len(decision.setup) and decision.setup[at]
            elif kind == D.DecisionKind.ROAD_BUILDING:
                valid = 0 <= at < len(decision.road_building)
            elif kind == D.DecisionKind.DISCARD:
                valid = 0 <= at < len(decision.discard)
            else:
                valid = False
            if not valid:
                raise ValueError("answer names an illegal decision candidate")
            action = D.option_of(decision, kind, at, self.pairs, request.seat, self.space, self.space.num_players)
            if action != answer.action:
                raise ValueError("answer action differs from its legal candidate")
            actions.append(action)
        return actions

    def advance(self, answers):
        self._healthy()
        answers = tuple(answers)
        actions = self._validate_answers(answers)
        finished = []
        lane = None
        try:
            for row, (request, answer, option) in enumerate(zip(self._asked, answers, actions)):
                lane = self.lanes[request.lane]
                parts = option.parts if isinstance(option, D.Compound) else (option,)
                done = False
                for part, action in enumerate(parts):
                    discarding = lane.native.core.phase() == 3
                    actors = self._native.actors_at(request.lane) if discarding else ()
                    actor = request.seat if discarding and actors and actors[0] != request.seat else None
                    lane.native.apply(action, request.seat if discarding else None)
                    lane.tape.step(action, (), actor=actor)
                    lane.actions += 1
                    if part == 0 and self.on_decision is not None:
                        self.on_decision(request, answer)
                    phase = lane.native.core.phase()
                    done = phase == 6 or lane.actions >= lane.action_cap
                    if done:
                        record = lane.record()
                        outcome = Outcome(record.winner, tuple(lane.native.core.privileged_true_points()),
                            record.turns, lane.actions, phase != 6, 0, record.winner is None and phase == 6)
                        completion = NativeCompletion(lane.index, lane.seed, lane.players, lane.cast, outcome, record)
                        kept = self.keep(completion)
                        lane.finished = True
                        finished.append(completion if kept is None else kept)
                        break
                if not done:
                    types = [int(action.type) for action in parts]
                    if (self._native.lane_closes(request.lane, request.seat, types) if request.view is not None
                            else self._native.option_closes(self._filled[row], request.seat, types)):
                        del lane.running[request.seat]
                    else:
                        lane.running[request.seat] = request.decision_id, request.part + 1
            self._native.finish(self._epoch)
        except BaseException as error:
            self._failed = error
            if lane is not None:
                lane.failed = error
            raise
        self._asked = self._targets = self._epoch = None
        return tuple(finished)


__all__ = ["NativeLane", "NativeBatch", "NativeRequest", "NativeAnswer", "NativeTargets", "NativeCompletion",
           "available", "supported"]
