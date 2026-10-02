"""A dropped-in ONNX checkpoint as an opponent.

onnxruntime's half of the model boundary: loading a `.onnx` file, checking what
it declares, and answering `hexset.clients.policy.Policy` for it. How one is
*played* lives in `hexset.clients.netbot`; `spawn` is the only entry point the
package needs. A checkpoint's `contract` metadata names its record shape
(`hexset.onnx_record.CONTRACT_VERSION`, `6`: the graph masks, normalises,
argmaxes and un-rotates itself); anything else is refused at load.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import onnxruntime as ort

from hexset.actions import Action, ActionSpace, build_space
from hexset.board.topology import Topology
from hexset.clients.netbot import NetworkBot, bot_for, searcher_for
from hexset.game import Game
from hexset.mcts import Search
from hexset.onnx_record import record_from_game
from hexset.server.constants import RECORD_CONTRACTS
from hexset.clients._modelmeta import SearchConfig, gate_config, search_config, trader_config, trader_of
from hexset.trading import UNLIMITED, TradeParams
from hexset.actions import options_for

__all__ = [
    "DEFAULT_THREADS",
    "Loaded",
    "V2Policy",
    "load",
    "network_bot",
    "searcher",
    "spawn",
]


def _providers_for(device: str) -> list[str]:
    """onnxruntime's provider names, not torch's device strings; a non-CPU
    device falls back to CPU."""
    if not device or device == "cpu":
        return ["CPUExecutionProvider"]
    return [f"{device.upper()}ExecutionProvider", "CPUExecutionProvider"]


class V2Policy:
    """A record-contract checkpoint as a `hexset.clients.policy.Policy`
    (structurally; it inherits nothing): the graph masks, normalises, argmaxes
    and un-rotates, so this only states the position and reads the decision."""

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
        """`_run`'s second half, for pre-built record rows. Keyed off the
        graph's own input names, since onnxruntime rejects a feed carrying a
        name the graph does not declare."""
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
        """Whether the graph's declared input shapes allow a batch of `count`
        rows. onnxruntime reports a dynamic batch axis as a non-`int`; a fixed
        one reports an `int` and refuses any other size."""
        for declared in self.session.get_inputs():
            shape = declared.shape
            if shape and isinstance(shape[0], int) and shape[0] != count:
                return False
        return True

    def value_of(self, records: Sequence[dict[str, np.ndarray]]) -> np.ndarray:
        """The value head alone, `(len(records), players)`, board-seat order.
        One graph call if `_batchable`, otherwise one per row in order."""
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
        """`value`, already board-seat order. `options_for` stands in for this
        row's options, since an empty mask leaves the graph nothing legal to
        normalise over. Routed through `value_of` so a wave of rows still
        works against a fixed-batch graph."""
        if not rows:
            return []
        records = [
            record_from_game(game, seat, self.space, options_for(game))
            for game, seat in rows
        ]
        return [tuple(float(v) for v in row) for row in self.value_of(records)]

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
    """A `hexset.clients.policy.Checkpoint`, plus the search the file asks to
    be played with and its training iteration."""

    policy: V2Policy
    space: ActionSpace
    players: int
    iteration: int
    search: SearchConfig = SearchConfig()
    #: How the file asks its trade gate to bargain. `trade_params` is the name
    #: `hexset.trading.params_of` reads, so this checkpoint describes itself
    #: the same way every bot that declares its bargaining does.
    gate: TradeParams = UNLIMITED
    #: The bot the file names to answer its trades (`hexset.arena.traded`),
    #: `None` for its own gate.
    trader: str | None = None

    @property
    def trade_params(self) -> TradeParams:
        return self.gate

    @property
    def trade_floor(self) -> float:
        return self.gate.trade_floor

    @property
    def gate_plies(self) -> int:
        return self.gate.gate_plies


#: What `threads` means when nothing asks for anything else. One, because an
#: arena run is many worker processes, and onnxruntime's default is one
#: intra-op thread per visible core in each of them, so the workers would
#: oversubscribe the machine against each other. A single-process caller that
#: wants the whole box passes `threads=None` and gets onnxruntime's own choice.
DEFAULT_THREADS = 1


@lru_cache(maxsize=4)
def _load_cached(
    path: str, topology: Topology, device: str, mtime_ns: int, threads: int | None
) -> Loaded:
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=_providers_for(device)
    )
    meta = session.get_modelmeta().custom_metadata_map

    players = int(meta["players"])
    # Topology fingerprint embedded by the exporter: a graph traced for one
    # board shape would otherwise fail silently when fed another.
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
        # first move with a missing-input error.
        raise ValueError(
            f"{path} declares contract={contract!r}, which this server does not serve "
            f"(known: {', '.join(sorted(RECORD_CONTRACTS))})"
        )
    policy = V2Policy(session, space)
    # `max_trades` is the older spelling of `max_offers`; a file carrying it
    # is read rather than silently ignored, which would turn a checkpoint
    # that asked not to trade into one that does.
    legacy_offers = meta.get("max_trades")
    return Loaded(
        policy=policy,
        space=space,
        players=players,
        iteration=int(meta.get("iteration", 0)),
        search=search_config(meta),
        gate=gate_config(
            meta if not legacy_offers or "max_offers" in meta
            else {**meta, "max_offers": legacy_offers}
        ),
        trader=trader_config(meta),
    )


def load(
    path: str,
    topology: Topology,
    device: str = "cpu",
    threads: int | None = DEFAULT_THREADS,
) -> Loaded:
    """The checkpoint at `path`, ready to act on boards of this topology. The
    cache key folds in the file's mtime, so replacing a file in `models/` by
    name serves the new one.

    `threads` caps onnxruntime's intra- and inter-op pools and defaults to
    `DEFAULT_THREADS`; `threads=None` is onnxruntime's own choice, one thread
    per core.
    """
    return _load_cached(path, topology, device, os.stat(path).st_mtime_ns, threads)


def searcher(
    path: str,
    board,
    *,
    simulations: int = 128,
    wave: int = 16,
    k: int = 1,
    device: str = "cpu",
    inference_batch: int | None = None,
    rng=None,
    threads: int | None = DEFAULT_THREADS,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search on `board`, trading
    through its own value-head gate. `k` is the determinized worlds each
    decision is searched in."""
    return searcher_for(
        load(path, board.topology, device, threads),
        simulations=simulations,
        wave=wave,
        k=k,
        inference_batch=inference_batch,
        rng=rng,
    )


def network_bot(
    path: str,
    board,
    *,
    device: str = "cpu",
    threads: int | None = DEFAULT_THREADS,
    rng: random.Random | None = None,
) -> NetworkBot:
    """The checkpoint at `path`, with `rng` controlling trade belief worlds."""
    return bot_for(load(path, board.topology, device, threads), rng=rng)


def spawn(
    path: str,
    board,
    *,
    rng: random.Random | None = None,
    device: str = "cpu",
    threads: int | None = DEFAULT_THREADS,
    players: int | None = None,
):
    """The checkpoint at `path` as something with `.choose(game) -> Action`.
    Single forward or search over its own priors, and a trader to answer its
    trades, are the file's metadata to declare; `device` describes the host
    and is deliberately never read from it.

    Where the caller passes the table's `players`, a checkpoint trained for
    another player count is refused here, with the file named."""
    from hexset.arena import traded

    loaded = load(path, board.topology, device, threads)
    if players is not None and players != loaded.players:
        raise ValueError(
            f"{path} was trained for {loaded.players} players, not this table's {players}"
        )
    own = _spawn_own(path, board, loaded, rng, device, threads)
    return traded(own, trader_of(loaded), board, random.Random() if rng is None else rng)


def _spawn_own(path, board, loaded, rng, device, threads):
    """`spawn` before any trader: the file's own one forward or search."""
    if not loaded.search.searches:
        return network_bot(
            path, board, device=device, threads=threads, rng=rng
        )
    return searcher(
        path,
        board,
        simulations=loaded.search.simulations,
        wave=loaded.search.wave,
        device=device,
        rng=rng,
        threads=threads,
    )
