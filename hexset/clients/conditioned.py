"""A rollout's other seats played by a model of how each of them plays.

Paired rollouts (`hexset.rollout`) play every seat of a playout with the
searching checkpoint's own policy. A file may name another model for the
other seats: a record-contract graph that also declares an input
`condition`, one vector per row, which says how the seat to move plays. The
vector for each seat comes from a second graph, a sequence encoder, read on
that seat's own moves as the table makes them. Neither graph is the
searching seat's: it still plays its own seat in every playout, reads every
value and scores every root.

- `ConditionedPolicy`: the opponent graph, its `condition` fed per row by
  seat, its move sampled from its `prior` at temperature 1 (a model of how a
  seat plays, not a move worth scoring). Two copies of one world draw the
  same uniforms (`hexset.rollout.paired_copy`'s `paired_seed`).
- `SeatEncoder`: the sequence graph: the record from the moving seat, the
  move's action index and the seat's state in; the seat's new `condition`
  and state out.
- `Opponents`: both, for one searching seat. `seen` feeds the encoder each
  other seat's move with a choice, read in a world drawn from the searching
  seat's own view, never the table's true state; `playouts` is the policy a
  rollout plays with.
- `SeatRouted`: that policy: the searching seat's rows to its own policy,
  every other seat's moves to the opponent graph. Its `context` (the seat and
  every seat's condition) travels with the worlds a `RolloutPool` worker
  plays.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import onnxruntime as ort

from hexset.actions import Action, ActionSpace, build_space, legal_actions
from hexset.board.topology import Topology
from hexset.game import Game, imagine
from hexset.onnx_record import RECORD_FIELDS, record_from_game, record_rows
from hexset.server.constants import RECORD_CONTRACTS

__all__ = ["ConditionedPolicy", "SeatEncoder", "Opponents", "SeatRouted", "session_for"]


def session_for(path: str, threads: int | None = 1) -> ort.InferenceSession:
    """A CPU session on `path`, its pools capped at `threads`."""
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _space(session: ort.InferenceSession, topology: Topology, path: str) -> ActionSpace:
    meta = session.get_modelmeta().custom_metadata_map
    if meta.get("contract") not in RECORD_CONTRACTS:
        raise ValueError(f"{path} declares contract={meta.get('contract')!r}, not a record contract")
    if "players" not in meta:
        raise ValueError(f"{path} declares no players")
    return build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, int(meta["players"]))


def _width(session: ort.InferenceSession, name: str, path: str) -> int:
    """The width of `name`: its declared last axis, else the file's metadata
    key of that name (a traced output may carry a symbolic one)."""
    for item in [*session.get_inputs(), *session.get_outputs()]:
        if item.name == name:
            width = item.shape[-1]
            if isinstance(width, int):
                return width
            meta = session.get_modelmeta().custom_metadata_map
            if name in meta:
                return int(meta[name])
            raise ValueError(f"{path}: `{name}` has no fixed width and no metadata `{name}`")
    raise ValueError(f"{path} declares no `{name}`")


class ConditionedPolicy:
    """A record-contract graph that also reads a per-row `condition`.

    `conditions` maps a seat to its vector; a seat with none reads zeros.
    `act_rows` samples each row's move from the graph's `prior` over its
    options, temperature 1. On a rollout copy (`paired_seed`) the n-th draw a
    copy makes for a seat is the same in every copy of that world, so two
    candidates see the same opponent until their games differ; anywhere else
    the draws come from `rng`."""

    def __init__(self, path: str, topology: Topology, *, threads: int | None = 1,
                 rng: random.Random | None = None) -> None:
        self.path = path
        self.session = session_for(path, threads)
        self.space = _space(self.session, topology, path)
        self.width = _width(self.session, "condition", path)
        self.inputs = [i.name for i in self.session.get_inputs()]
        missing = [n for n in self.inputs if n != "condition" and n not in RECORD_FIELDS]
        if missing:
            raise ValueError(f"{path} asks for {missing}, which a record does not carry")
        self.conditions: dict[int, np.ndarray] = {}
        self.rng = rng or random.Random()

    def condition(self, seat: int) -> np.ndarray:
        vector = self.conditions.get(seat)
        return np.zeros(self.width, np.float32) if vector is None else vector

    def prior_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[np.ndarray]:
        """Each row's prior over its own options, in their order."""
        index = self.space.index
        indices = [[index(o) for o in options] for _, _, options in rows]
        batch = record_rows(rows, self.space, indices)
        feed = {n: batch[n] for n in self.inputs if n != "condition"}
        feed["condition"] = np.stack([self.condition(seat) for _, seat, _ in rows]).astype(np.float32)
        (prior,) = self.session.run(["prior"], feed)
        out = []
        for i, idx in enumerate(indices):
            w = prior[i][idx].astype(np.float64)
            total = w.sum()
            out.append(w / total if total > 0 else np.full(len(idx), 1.0 / len(idx)))
        return out

    def _uniform(self, game: Game, seat: int) -> float:
        seed = game.__dict__.get("paired_seed")
        if seed is None:
            return self.rng.random()
        draws = game.__dict__.setdefault("_condition_draws", {})
        n = draws.get(seat, 0)
        draws[seat] = n + 1
        return random.Random(hash((seed, seat, n)) & 0xFFFFFFFFFFFF).random()

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        if not rows:
            return []
        picked = []
        for (game, seat, options), p in zip(rows, self.prior_rows(rows)):
            u = self._uniform(game, seat)
            k = int(np.searchsorted(np.cumsum(p), u * p.sum(), side="right"))
            picked.append(options[min(k, len(options) - 1)])
        return picked


class SeatEncoder:
    """A sequence graph read one move at a time, a state per seat.

    Declared inputs: record fields (the record from the moving seat), `action`
    (the move's action index) and `state`; outputs `condition` and
    `next_state`. A seat not yet seen has a zero state and no condition."""

    def __init__(self, path: str, topology: Topology, *, threads: int | None = 1) -> None:
        self.path = path
        self.session = session_for(path, threads)
        self.space = _space(self.session, topology, path)
        self.width = _width(self.session, "state", path)
        self.inputs = [i.name for i in self.session.get_inputs()]
        missing = [n for n in self.inputs if n not in ("action", "state") and n not in RECORD_FIELDS]
        if missing:
            raise ValueError(f"{path} asks for {missing}, which a record does not carry")
        self.states: dict[int, np.ndarray] = {}

    def step(self, game: Game, seat: int, action: Action) -> np.ndarray:
        """Feed `seat`'s move `action` at `game` (the position before it); the seat's new condition."""
        record = record_from_game(game, seat, self.space)
        feed = {n: record[n][None] for n in self.inputs if n not in ("action", "state")}
        feed["action"] = np.asarray([self.space.index(action)], dtype=np.int64)
        feed["state"] = self.states.get(seat, np.zeros(self.width, np.float32))[None]
        condition, state = self.session.run(["condition", "next_state"], feed)
        self.states[seat] = state[0]
        return condition[0].astype(np.float32)


class SeatRouted:
    """A `hexset.clients.policy.Policy` for a playout: `own` for `seat`'s rows
    and for every value and root score, `others` for every other seat's
    moves."""

    def __init__(self, own, others: ConditionedPolicy, seat: int | None = None) -> None:
        self.own, self.others, self.seat = own, others, seat

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        out: list[Action | None] = [None] * len(rows)
        mine = [i for i, (_, s, _) in enumerate(rows) if s == self.seat]
        theirs = [i for i, (_, s, _) in enumerate(rows) if s != self.seat]
        for idx, policy in ((mine, self.own), (theirs, self.others)):
            if idx:
                for i, action in zip(idx, policy.act_rows([rows[i] for i in idx])):
                    out[i] = action
        return out  # type: ignore[return-value]

    def value_rows(self, rows):
        return self.own.value_rows(rows)

    def score_rows(self, rows):
        return self.own.score_rows(rows)

    def context(self) -> dict:
        """What a worker's copy needs to play as this one: the seat and every seat's condition."""
        return {"seat": self.seat, "conditions": {s: v.copy() for s, v in self.others.conditions.items()}}

    def set_context(self, context: dict) -> None:
        self.seat = context["seat"]
        self.others.conditions = dict(context["conditions"])

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "own"), name)


class Opponents:
    """The opponent graph and its encoder for one searching seat.

    `seen(game, actor, action, viewer)` is called with each move the table
    makes, before it is applied (`hexset.bots.watch`). A move by another seat
    with more than one legal option feeds that seat's encoder, read in a world
    drawn from `viewer`'s own view (`hexset.game.imagine`), and replaces the
    seat's condition. `playouts(own)` is the policy a rollout plays with."""

    def __init__(self, policy_path: str, encoder_path: str, topology: Topology, *,
                 threads: int | None = 1, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.policy = ConditionedPolicy(policy_path, topology, threads=threads,
                                        rng=random.Random(self.rng.getrandbits(64)))
        self.encoder = SeatEncoder(encoder_path, topology, threads=threads)
        if self.policy.width != _width(self.encoder.session, "condition", encoder_path):
            raise ValueError(f"{encoder_path}'s condition does not fit {policy_path}'s")
        self.moves = 0

    def seen(self, game: Game, actor: int, action: Action, viewer: int) -> None:
        if actor == viewer or actor is None:
            return
        rng = random.Random(self.rng.getrandbits(64))
        try:
            state = game.state(viewer).sample(rng)
        except ValueError:
            return                          # the record admits no deal here: nothing read
        world = imagine(game, rng, randomize_deck=False, seat=viewer)
        world.set_state(state)
        if len(legal_actions(world, actor)) < 2:
            return                          # no choice: nothing of how the seat plays
        self.policy.conditions[actor] = self.encoder.step(world, actor, action)
        self.moves += 1

    def playouts(self, own, seat: int | None = None) -> SeatRouted:
        return SeatRouted(own, self.policy, seat)
