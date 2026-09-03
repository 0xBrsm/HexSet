"""A dropped-in ONNX checkpoint as an opponent.

This module is the whole boundary between the game and a model. Everything
that knows a network exists — the record encoding, the flat action space,
masking, sampling, and how a checkpoint wants to be played — lives behind
here. The rest of the package hands over a
`Game` and gets an `Action` back, and `spawn` below is the only entry point
it needs.

`NetworkBot`, `NetworkEvaluator` and `LeafEvaluator` are thin adapters over a
policy; they only ever call methods on `self.policy` and never inspect a
network directly. `V2Policy` is an onnxruntime `InferenceSession` with its
own math in numpy — mechanical, since none of it has learned parameters.

A checkpoint's `contract` metadata names which record shape it declares
(`5` — "record in, decision out": the graph itself masks,
normalises, argmaxes and un-rotates, and `V2Policy` just reads its outputs).
`NetworkBot`, `NetworkEvaluator` and `LeafEvaluator` call
`act_rows`/`value_rows`/`score_rows`, an interface `V2Policy` presents for
every record contract, and never learn which one they are holding.

Contract 1 ("observation in, raw logits/give/want/value out", masked and
decoded here in Python against the frozen `encoding_v1` feature layout) is no
longer served — the owner dropped it 2026-09-02
(`docs/engine-divergence-2026-09-02.md`, B5). A `contract=1` file, or one
with no `contract` key at all, is refused by name at load.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np
import onnxruntime as ort

from hexset.actions import Action, ActionSpace, build_space
from hexset.board.terrain import NUM_RESOURCES
from hexset.board.topology import Topology
from hexset.game import Game, to_move
from hexset.mcts import Search
from hexset.onnx_record import record_from_game
from hexset.server.constants import RECORD_CONTRACTS
from hexset.server.modelmeta import SearchConfig, search_config
from hexset.server.rules import options_for
from hexset.state import copy_state
from hexset.trading import NO_VALUATION, VALUE_SCALE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hexset.trading import Bundle
    from hexset.view import View


def _check_players(game: Game, players: int) -> None:
    # true state: `num_players` is a fixed, public board property.
    num_players = game.state(0, hidden=False).num_players
    if num_players != players:
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {num_players}"
        )


def _providers_for(device: str) -> list[str]:
    """onnxruntime's own provider names, not torch's device strings — `cpu`
    is the only one this repo actually runs on (no GPU on the boxes it
    targets), but the `--device` flag stays functional for anyone with a
    build that has a GPU execution provider available, falling back to CPU
    if it doesn't."""
    if not device or device == "cpu":
        return ["CPUExecutionProvider"]
    return [f"{device.upper()}ExecutionProvider", "CPUExecutionProvider"]


class V2Policy:
    """A record-contract checkpoint: the graph itself masks, normalises,
    argmaxes and un-rotates, so this class only states the position and reads
    the graph's decision back. `NetworkBot`, `NetworkEvaluator` and
    `LeafEvaluator` call it through `act_rows`/`value_rows`/`score_rows`
    without knowing which of contracts 2, 3 or 4 they are holding.
    """

    def __init__(self, session: ort.InferenceSession, space: ActionSpace) -> None:
        self.session = session
        self.space = space

    def _run(
        self,
        rows: Sequence[tuple[Game, int, tuple[Action, ...]]],
        outputs: list[str],
    ) -> list[np.ndarray]:
        records = [record_from_game(game, seat, self.space, options) for game, seat, options in rows]
        return self._run_records(records, outputs)

    def _run_records(
        self, records: Sequence[dict[str, np.ndarray]], outputs: list[str]
    ) -> list[np.ndarray]:
        """`_run`'s second half, for a caller that has already built its own
        rows (`value_of`, below) rather than reading them off a live
        `(game, seat, options)` triple.

        Keyed off the graph's own input names, not off every field the
        record happens to carry. onnxruntime rejects a feed containing a
        name it does not declare, so this is exactly "give the graph the
        subset it asks for" -- and refuses loudly, below, if it asks for
        something a contract-5 record no longer carries (the four
        `offer_*` fields and `pair_mask`, gone with the offer protocol).
        """
        wanted = [i.name for i in self.session.get_inputs()]
        missing = [name for name in wanted if name not in records[0]]
        if missing:
            raise ValueError(
                f"this graph asks for {missing}, which the record does not carry — "
                "it was exported against a newer contract than this server builds"
            )
        inputs = {key: np.stack([record[key] for record in records]) for key in wanted}
        return self.session.run(outputs, inputs)

    def _batchable(self, count: int) -> bool:
        """Whether the graph's own declared input shapes allow a batch of
        `count` rows in one call.

        onnxruntime reports a dynamic batch axis as a non-`int` (a symbol
        name, or `None`); a graph traced with a fixed batch size reports a
        plain `int` there instead, and a feed of any other size is refused
        outright rather than silently rejected. `value_of` falls back to one
        call per row when this says no, rather than finding out the hard way.
        """
        for declared in self.session.get_inputs():
            shape = declared.shape
            if shape and isinstance(shape[0], int) and shape[0] != count:
                return False
        return True

    def value_of(self, records: Sequence[dict[str, np.ndarray]]) -> np.ndarray:
        """The value head alone, `(len(records), players)`, board-seat order
        -- for a caller building its own rows (`hexset.clients.onnxbot.
        NetworkBot`'s imagined hands) rather than reading them off a live
        `(game, seat)` pair the way `value_rows` does.

        One graph call if the batch dimension allows every row at once
        (`_batchable`); otherwise one call per row, in declared order. A
        fixed-batch-1 graph then pays for six forwards where a dynamic-batch
        one pays for one, which is a property of the exported graph, not of
        this method.
        """
        if not records:
            return np.zeros((0, 0), dtype=np.float32)
        if self._batchable(len(records)):
            (value,) = self._run_records(records, ["value"])
            return value
        rows = [self._run_records([record], ["value"])[0][0] for record in records]
        return np.stack(rows)

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        if not rows:
            return []
        (action_index,) = self._run(rows, ["action_index"])
        return [self.space.decode(int(a)) for a in action_index]

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """`value`, already in board-seat order — the graph un-rotates it
        itself. `options_for` stands in for this row's options, since a bare
        `(game, seat)` value query has none to offer, and an empty mask would
        leave the graph nothing legal to normalise over."""
        if not rows:
            return []
        (value,) = self._run([(game, seat, options_for(game)) for game, seat in rows], ["value"])
        return [tuple(float(v) for v in row) for row in value]

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[np.ndarray, tuple[float, ...]]]:
        prior, value = self._run(rows, ["prior", "value"])
        return [
            (
                self._combine_prior(options, prior[i]),
                tuple(float(v) for v in value[i]),
            )
            for i, (_, _, options) in enumerate(rows)
        ]

    def _combine_prior(self, options, prior: np.ndarray) -> np.ndarray:
        """A leaf's options, scored from this row's dense prior."""
        weights = np.empty(len(options))
        for i, option in enumerate(options):
            weights[i] = prior[self.space.index(option)]
        total = weights.sum()
        if total <= 0:
            return np.full(len(options), 1.0 / len(options))
        return weights / total


@dataclass(frozen=True)
class Loaded:
    """A checkpoint made playable, plus what the run it came from was doing."""

    policy: V2Policy
    space: ActionSpace
    players: int
    max_trades: int | None
    iteration: int
    search: SearchConfig = SearchConfig()


@lru_cache(maxsize=4)
def _load_cached(path: str, topology: Topology, device: str, mtime_ns: int) -> Loaded:
    session = ort.InferenceSession(str(path), providers=_providers_for(device))
    meta = session.get_modelmeta().custom_metadata_map

    players = int(meta["players"])
    # The topology fingerprint `hexnet.export_onnx` embeds: a graph traced
    # for one board shape fails silently if fed another (wrong-shaped
    # inputs, or worse, right-shaped-but-meaningless ones) rather than
    # loudly the way `net.load_state_dict` fails on a shape mismatch today
    # — so this is the loud failure standing in for that one.
    fingerprint = (
        int(meta["num_hexes"]),
        int(meta["num_vertices"]),
        int(meta["num_edges"]),
    )
    actual = (topology.num_hexes, topology.num_vertices, topology.num_edges)
    if fingerprint != actual:
        raise ValueError(
            f"{path} was exported for a board shaped {fingerprint} "
            f"(hexes, vertices, edges), not this table's {actual}"
        )

    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    contract = meta.get("contract", "1")
    if contract not in RECORD_CONTRACTS:
        # Loudly, rather than guessing at a graph shape and dying on the
        # first move with a missing-input error naming tensors nobody asked
        # about. Contract 1 (or no `contract` key at all) is refused here by
        # the same path as a genuinely unknown future contract — the owner
        # dropped it 2026-09-02, and there is no legacy path left to fall
        # back to (`docs/engine-divergence-2026-09-02.md`, B5).
        raise ValueError(
            f"{path} declares contract={contract!r}, which this server does not serve "
            f"(known: {', '.join(sorted(RECORD_CONTRACTS))})"
        )
    policy = V2Policy(session, space)
    max_trades = meta.get("max_trades") or None
    return Loaded(
        policy=policy,
        space=space,
        players=players,
        max_trades=int(max_trades) if max_trades is not None else None,
        iteration=int(meta.get("iteration", 0)),
        search=search_config(meta),
    )


def load(path: str, topology: Topology, device: str = "cpu") -> Loaded:
    """The checkpoint at `path`, ready to act on boards of this topology.

    Cache key folds in the file's mtime, unlike the training repo's loader: its
    `.pt` checkpoints under `runs/` are effectively immutable per-run
    artifacts, but hexset's whole pitch is replacing a file in `models/`
    by name — without the mtime, a same-named replacement would silently
    keep serving the old in-memory session.
    """
    return _load_cached(path, topology, device, os.stat(path).st_mtime_ns)


@dataclass
class NetworkBot:
    """A policy answering one position at a time.

    Trades off the same value head `choose` already reads, no new
    parameters: `valuation`/`accepts` mirror dev-HexNet's
    `hexnet.policy.DerivedTrader` exactly, reimplemented against the
    record-contract wire shape instead of a live `torch` forward. See both
    methods' own docstrings.
    """

    policy: V2Policy
    space: ActionSpace
    players: int
    max_trades: int | None = None
    # The game `choose` was last handed, so a trade event -- which runs
    # inside the same `apply` this bot's own choice already went through --
    # asks about the position it is actually seated at. `None` only for a
    # bot nobody has asked to move yet, which cannot happen in play (the
    # trade event runs after every seat has moved through setup) but is the
    # right answer for `valuation`/`accepts` below regardless.
    _seated: Game | None = field(default=None, repr=False, compare=False)

    def choose(self, game: Game) -> Action:
        _check_players(game, self.players)
        self._seated = game
        seat = to_move(game)
        return self.policy.act_rows([(game, seat, tuple(options_for(game)))])[0]

    def valuation(self, view: "View") -> tuple[float, ...]:
        """What this seat advertises, derived from the value head.

        `tanh(delta_V_r / VALUE_SCALE)` per resource: `delta_V_r` is the
        value head's own-row delta between this seat's hand and that hand
        holding one more card of `r`. One batched call over the hand plus
        its `NUM_RESOURCES` imagined successors (`V2Policy.value_of`) rather
        than six separate ones, mirroring `hexnet.policy.DerivedTrader.
        valuation`'s one-forward fan-out.

        Only the seat's own hand moves -- the extra card comes from another
        seat, not the bank, so nothing else about the position changes. The
        hand is read off `view.known[seat]`, exact for the perspective seat
        by construction (`hexset.view.View`).
        """
        if self.max_trades == 0 or self._seated is None:
            return NO_VALUATION
        seat = view.perspective
        hand = list(view.known[seat])
        hands = [hand]
        for resource in range(NUM_RESOURCES):
            one_more = list(hand)
            one_more[resource] += 1
            hands.append(one_more)
        values = self._own_values(seat, hands)
        return tuple(
            math.tanh((values[1 + r] - values[0]) / VALUE_SCALE)
            for r in range(NUM_RESOURCES)
        )

    def accepts(self, view: "View", received: "Bundle", counterparty: int) -> bool:
        """This seat's private gate: the value head on the concrete
        post-trade position, strictly preferred -- not the sum of the
        marginals `valuation` published, the same distinction `hexnet.
        policy.DerivedTrader.accepts` draws, and for the same reason
        (complementarity between resources lives in the joint hand, not in
        five independent one-card deltas).
        """
        del counterparty  # the joint post-trade hand is enough; who sent it is not
        if self.max_trades == 0 or self._seated is None:
            return False
        seat = view.perspective
        hand = list(view.known[seat])
        after = [n + d for n, d in zip(hand, received)]
        if any(n < 0 for n in after):
            return False
        before_value, after_value = self._own_values(seat, [hand, after])
        return after_value > before_value

    def accepts_many(
        self,
        view: "View",
        received: Sequence["Bundle"],
        counterparties: Sequence[int],
    ) -> list[bool]:
        """Batched `accepts`: one graph call over the hand plus every
        candidate's post-trade successor, instead of one call per candidate
        (`agents/reference/trading-design.md`'s post-data note, "the
        collector cost gate fails at 2.9-3.6x" -- an unbatched network gate
        asked one candidate at a time against hundreds of clearing
        candidates was the entire excess collection cost). Mirrors
        `valuation`'s own one-forward fan-out (`_own_values` ->
        `V2Policy.value_of`, which already falls back to one call per row
        when the graph's declared batch axis is not dynamic) rather than
        `accepts`'s one-candidate-at-a-time forward.

        `counterparties` is accepted for signature parity with
        `hexset.bots.Bot.accepts_many` but not read, for the same reason
        `accepts` does not read its own `counterparty`: the joint post-trade
        hand is enough to judge.
        """
        del counterparties
        if self.max_trades == 0 or self._seated is None or not received:
            return [False] * len(received)
        seat = view.perspective
        hand = list(view.known[seat])
        afters: list[list[int]] = []
        valid: list[bool] = []
        for wanted in received:
            after = [n + d for n, d in zip(hand, wanted)]
            ok = all(n >= 0 for n in after)
            valid.append(ok)
            # An uncoverable candidate still needs a row so every position
            # in `received` lines up with one in `values` below; the hand
            # itself is a safe, always-valid placeholder, and its result is
            # discarded (`ok and ...`) rather than trusted.
            afters.append(after if ok else hand)
        values = self._own_values(seat, [hand] + afters)
        before_value = values[0]
        return [ok and values[1 + i] > before_value for i, ok in enumerate(valid)]

    def _own_values(self, seat: int, hands: Sequence[Sequence[int]]) -> list[float]:
        """Each hand's value on `seat`'s own row, `seat`'s hand swapped in
        turn and everything else about the live position held fixed.

        Mirrors `hexnet.policy.DerivedTrader._own_values`: `set_state` is the
        engine's own sanctioned way to swap a hypothetical state in and back
        out, the observation is built from the game (not the bare state) for
        the same reason as there -- phase, turn count and every seat's
        published vector all live on `Game`, not `GameState` -- and the
        original is restored in a `finally` so a raised error still leaves
        the live game exactly as `choose` left it.
        """
        game = self._seated
        assert game is not None  # callers check this first
        original = game.state(seat, hidden=False)
        try:
            records = []
            for hand in hands:
                state = copy_state(original)
                state.hands[seat] = list(hand)
                game.set_state(state)
                records.append(
                    record_from_game(game, seat, self.space, options_for(game))
                )
        finally:
            game.set_state(original)
        values = self.policy.value_of(records)
        return [float(row[seat]) for row in values]


@dataclass
class NetworkEvaluator:
    """The value head as `hexset.bots.SearchBot`'s leaf evaluation."""

    policy: V2Policy
    players: int
    max_trades: int | None = None

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        _check_players(game, self.players)
        return list(self.policy.value_rows([(game, seat)])[0])


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward."""

    policy: V2Policy
    space: ActionSpace
    pad_to: int | None = None

    def __post_init__(self) -> None:
        if self.pad_to is not None and self.pad_to < 1:
            raise ValueError("pad_to must be positive")

    def evaluate(self, leaves):
        if not leaves:
            return []
        count = len(leaves)
        padded = list(leaves)
        if self.pad_to is not None and count < self.pad_to:
            padded.extend([leaves[-1]] * (self.pad_to - count))
        rows = [(leaf.game, leaf.seat, leaf.options) for leaf in padded]
        return self.policy.score_rows(rows)[:count]


def searcher(
    path: str,
    board,
    *,
    simulations: int = 128,
    wave: int = 16,
    max_trades: int | None = None,
    device: str = "cpu",
    inference_batch: int | None = None,
    rng=None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search, playing on `board`."""
    loaded = load(path, board.topology, device)
    budget = loaded.max_trades if max_trades is None else max_trades
    return Search(
        LeafEvaluator(
            policy=loaded.policy,
            space=loaded.space,
            pad_to=inference_batch,
        ),
        simulations=simulations,
        wave=wave,
        max_trades=budget,
        rng=rng,
    )


def network_evaluator(path: str, board, *, device: str = "cpu") -> NetworkEvaluator:
    """The checkpoint at `path` as a leaf evaluation for the search."""
    loaded = load(path, board.topology, device)
    return NetworkEvaluator(
        policy=loaded.policy, players=loaded.players, max_trades=loaded.max_trades
    )


def network_bot(
    path: str, board, *, max_trades: int | None = None, device: str = "cpu"
) -> NetworkBot:
    """The checkpoint at `path`, playing on `board`."""
    loaded = load(path, board.topology, device)
    return NetworkBot(
        policy=loaded.policy,
        space=loaded.space,
        players=loaded.players,
        max_trades=loaded.max_trades if max_trades is None else max_trades,
    )


def spawn(
    path: str,
    board,
    *,
    rng: random.Random | None = None,
    device: str = "cpu",
    max_trades: int | None = None,
):
    """The checkpoint at `path` as something with `.choose(game) -> Action`.

    The only entry point the rest of the package needs. Whether the file plays
    a single forward pass or a search over its own priors is the file's own
    business, declared in its metadata and read here — a caller passes a path
    and gets a bot, and never learns which it got.

    `device` is deliberately not read from metadata: it is a fact about the
    machine serving the game, not about the checkpoint, and a model file has no
    business demanding an accelerator its host may not have.
    """
    loaded = load(path, board.topology, device)
    if not loaded.search.searches:
        return network_bot(path, board, max_trades=max_trades, device=device)
    return searcher(
        path,
        board,
        simulations=loaded.search.simulations,
        wave=loaded.search.wave,
        max_trades=max_trades,
        device=device,
        rng=rng,
    )
