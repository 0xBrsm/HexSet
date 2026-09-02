"""A dropped-in ONNX checkpoint as an opponent.

This module is the whole boundary between the game and a model. Everything
that knows a network exists — the record encoding, the flat action space,
masking, sampling, the give/want pair distribution, and how a checkpoint
wants to be played — lives behind here. The rest of the package hands over a
`Game` and gets an `Action` back, and `spawn` below is the only entry point
it needs.

`NetworkBot`, `NetworkEvaluator` and `LeafEvaluator` are thin adapters over a
policy; they only ever call methods on `self.policy` and never inspect a
network directly. `V2Policy` is an onnxruntime `InferenceSession` with its
own math in numpy — mechanical, since none of it has learned parameters.

A checkpoint's `contract` metadata names which record shape it declares
(`2`, `3` or `4` — "record in, decision out": the graph itself masks,
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

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import onnxruntime as ort

from hexset.actions import (
    Action,
    ActionSpace,
    ActionType,
    build_space,
    within_offer_budget,
)
from hexset.board.terrain import NUM_RESOURCES
from hexset.board.topology import Topology
from hexset.game import Game, to_move
from hexset.mcts import Search

from .constants import RECORD_CONTRACTS
from .modelmeta import SearchConfig, search_config
from .record import build_record
from .record import pair_index as _pair_index
# The honest option list, not `hexset.bots.options_for`'s omniscient one: a
# checkpoint served here sees exactly the mask an external client or a human
# sees (`hexset_ui.rules`, and PR #2 defect 4).
from .rules import options_for


def _check_players(game: Game, players: int) -> None:
    if game.state.num_players != players:
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {game.state.num_players}"
        )


def _one_hot(resource: int) -> tuple[int, ...]:
    return tuple(1 if r == resource else 0 for r in range(NUM_RESOURCES))


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
        self.trade_slot = space.offsets[ActionType.PROPOSE_TRADE]

    def _run(
        self,
        rows: Sequence[tuple[Game, int, tuple[Action, ...]]],
        outputs: list[str],
    ) -> list[np.ndarray]:
        records = [build_record(game, seat, options, self.space) for game, seat, options in rows]
        # Keyed off the graph's own input names, not off every field the
        # record happens to carry. The record is the newest contract (29
        # fields); a contract-2 graph declares 23 of them and a contract-3
        # graph 27, and onnxruntime rejects a feed containing a name it does
        # not declare -- which is how PR #2 broke every genuine dev-hexset
        # export with `Invalid input name: offer_proposer`. Every contract so
        # far is a prefix-superset of the last, so serving an older graph is
        # exactly "give it the subset it asks for".
        wanted = [i.name for i in self.session.get_inputs()]
        missing = [name for name in wanted if name not in records[0]]
        if missing:
            raise ValueError(
                f"this graph asks for {missing}, which the record does not carry — "
                "it was exported against a newer contract than this server builds"
            )
        inputs = {key: np.stack([record[key] for record in records]) for key in wanted}
        return self.session.run(outputs, inputs)

    def _decode(self, index: int, pair: int) -> Action:
        if index == self.trade_slot:
            return Action(
                ActionType.PROPOSE_TRADE,
                give=_one_hot(pair // NUM_RESOURCES),
                want=_one_hot(pair % NUM_RESOURCES),
            )
        return self.space.decode(index)

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        if not rows:
            return []
        action_index, pair_index_out = self._run(rows, ["action_index", "pair_index"])
        return [self._decode(int(a), int(p)) for a, p in zip(action_index, pair_index_out)]

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
        prior, pair_prior, value = self._run(rows, ["prior", "pair_prior", "value"])
        return [
            (
                self._combine_prior(options, prior[i], pair_prior[i]),
                tuple(float(v) for v in value[i]),
            )
            for i, (_, _, options) in enumerate(rows)
        ]

    def _combine_prior(self, options, prior: np.ndarray, pair_prior: np.ndarray) -> np.ndarray:
        """A leaf's options, scored from this row's dense priors — plain
        multiplication, since the graph hands back linear probabilities
        rather than log-probs."""
        trade = self.trade_slot
        weights = np.empty(len(options))
        for i, option in enumerate(options):
            index = self.space.index(option)
            weights[i] = prior[index]
            if index == trade:
                weights[i] *= pair_prior[_pair_index(option.give, option.want)]
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
    max_offers: int | None
    iteration: int
    search: SearchConfig = SearchConfig()


@lru_cache(maxsize=4)
def _load_cached(path: str, topology: Topology, device: str, mtime_ns: int) -> Loaded:
    session = ort.InferenceSession(str(path), providers=_providers_for(device))
    meta = session.get_modelmeta().custom_metadata_map

    players = int(meta["players"])
    # The topology fingerprint `hexset_ui.export_onnx` embeds: a graph traced
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
    max_offers = meta.get("max_offers") or None
    return Loaded(
        policy=policy,
        space=space,
        players=players,
        max_offers=int(max_offers) if max_offers is not None else None,
        iteration=int(meta.get("iteration", 0)),
        search=search_config(meta),
    )


def load(path: str, topology: Topology, device: str = "cpu") -> Loaded:
    """The checkpoint at `path`, ready to act on boards of this topology.

    Cache key folds in the file's mtime, unlike the training repo's loader: its
    `.pt` checkpoints under `runs/` are effectively immutable per-run
    artifacts, but hexset-ui's whole pitch is replacing a file in `models/`
    by name — without the mtime, a same-named replacement would silently
    keep serving the old in-memory session.
    """
    return _load_cached(path, topology, device, os.stat(path).st_mtime_ns)


@dataclass
class NetworkBot:
    """A policy answering one position at a time."""

    policy: V2Policy
    space: ActionSpace
    players: int
    max_offers: int | None = None

    def choose(self, game: Game) -> Action:
        _check_players(game, self.players)
        seat = to_move(game)
        options = tuple(within_offer_budget(game, options_for(game), self.max_offers))
        return self.policy.act_rows([(game, seat, options)])[0]


@dataclass
class NetworkEvaluator:
    """The value head as `hexset.bots.SearchBot`'s leaf evaluation."""

    policy: V2Policy
    players: int
    max_offers: int | None = None

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
    max_offers: int | None = None,
    device: str = "cpu",
    inference_batch: int | None = None,
    rng=None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search, playing on `board`."""
    loaded = load(path, board.topology, device)
    budget = loaded.max_offers if max_offers is None else max_offers
    return Search(
        LeafEvaluator(
            policy=loaded.policy,
            space=loaded.space,
            pad_to=inference_batch,
        ),
        simulations=simulations,
        wave=wave,
        max_offers=budget,
        rng=rng,
    )


def network_evaluator(path: str, board, *, device: str = "cpu") -> NetworkEvaluator:
    """The checkpoint at `path` as a leaf evaluation for the search."""
    loaded = load(path, board.topology, device)
    return NetworkEvaluator(
        policy=loaded.policy, players=loaded.players, max_offers=loaded.max_offers
    )


def network_bot(
    path: str, board, *, max_offers: int | None = None, device: str = "cpu"
) -> NetworkBot:
    """The checkpoint at `path`, playing on `board`."""
    loaded = load(path, board.topology, device)
    return NetworkBot(
        policy=loaded.policy,
        space=loaded.space,
        players=loaded.players,
        max_offers=loaded.max_offers if max_offers is None else max_offers,
    )


def spawn(
    path: str,
    board,
    *,
    rng: random.Random | None = None,
    device: str = "cpu",
    max_offers: int | None = None,
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
        return network_bot(path, board, max_offers=max_offers, device=device)
    return searcher(
        path,
        board,
        simulations=loaded.search.simulations,
        wave=loaded.search.wave,
        max_offers=max_offers,
        device=device,
        rng=rng,
    )
