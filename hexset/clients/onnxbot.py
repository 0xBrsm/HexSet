"""A dropped-in ONNX checkpoint as an opponent.

onnxruntime's half of the model boundary: loading a `.onnx` file, checking what
it declares, and answering `hexset.clients.policy.Policy` for it. How one is
*played* lives in `hexset.clients.netbot`; `spawn` is the only entry point the
package needs. A checkpoint's `contract` metadata names its record shape
(`hexset.onnx_record.CONTRACT_VERSION`, `7`: the graph masks, normalises,
argmaxes and un-rotates itself). Contract 6, whose record is contract 7's
minus `roads_to_take` and `to_move`, is still served; anything else is refused
at load (`hexset.server.constants.RECORD_CONTRACTS`).
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
from hexset.clients.netbot import NetworkBot, bot_for, declared_trader, searcher_for
from hexset.game import Game
from hexset.lockstep import forward
from hexset.mcts import Search
from hexset.onnx_record import record_from_game, record_rows
from hexset.server.constants import RECORD_CONTRACTS
from hexset.clients._modelmeta import (
    SearchConfig, gate_config, search_config, trader_config, trader_veto_config, trader_veto_of,
    trader_veto_pick_config, trader_veto_pick_of,
)
from hexset.trading import UNLIMITED, TradeParams

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
        indices: Sequence[Sequence[int]] | None = None,
    ) -> list[np.ndarray]:
        """The graph on `rows`' records, built as one batch (`record_rows`);
        `indices`, each row's options' action indices, where the caller has
        them already."""
        return self._run_batch(record_rows(rows, self.space, indices), outputs)

    def _run_records(
        self, records: Sequence[dict[str, np.ndarray]], outputs: list[str]
    ) -> list[np.ndarray]:
        """`_run`'s second half, for pre-built record rows. Keyed off the
        graph's own input names, since onnxruntime rejects a feed carrying a
        name the graph does not declare."""
        return self._run_batch(
            {key: np.stack([record[key] for record in records]) for key in records[0]}, outputs
        )

    def _run_batch(self, batch: dict[str, np.ndarray], outputs: list[str]) -> list[np.ndarray]:
        """The graph on a stacked batch of record fields, fed the inputs it
        declares."""
        wanted = [i.name for i in self.session.get_inputs()]
        missing = [name for name in wanted if name not in batch]
        if missing:
            raise ValueError(
                f"this graph asks for {missing}, which the record does not carry — "
                "it was exported against a newer contract than this server builds"
            )
        inputs = {key: batch[key] for key in wanted}
        # `session.run`, or inside a `hexset.lockstep` lane, these rows of a
        # run shared with the other lanes' waiting forwards.
        return forward(self.session, outputs, inputs)

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
        """`value`, already board-seat order. Each row is masked from its own
        seat's side (`record_from_game`'s default): that seat's own options,
        or the fixed public mask where it has none, never another seat's
        legal moves. Routed through `value_of` so a wave of rows still works
        against a fixed-batch graph."""
        if not rows:
            return []
        records = [record_from_game(game, seat, self.space) for game, seat in rows]
        return [tuple(float(v) for v in row) for row in self.value_of(records)]

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[np.ndarray, tuple[float, ...]]]:
        index = self.space.index
        indices = [[index(option) for option in options] for _, _, options in rows]
        prior, value = self._run(rows, ["prior", "value"], indices)
        return [
            (
                self._combine_prior(indices[i], prior[i]),
                tuple(float(v) for v in value[i]),
            )
            for i in range(len(rows))
        ]

    def _combine_prior(self, indices: Sequence[int], prior: np.ndarray) -> np.ndarray:
        """A leaf's options, scored from this row's dense prior: `indices`
        are the options' action indices."""
        weights = prior[list(indices)].astype(np.float64)
        total = weights.sum()
        if total <= 0:
            return np.full(len(indices), 1.0 / len(indices))
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
    #: Whether the file's own gate vetoes that trader (`trader_veto`).
    trader_veto: bool = False
    #: Whether that veto picks which taker to deal with (`trader_veto_pick`).
    trader_veto_pick: bool = False

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
    # First, so a file of another contract is named as one rather than
    # failing on whichever key its own shape happens to lack.
    if "contract" not in meta:
        raise ValueError(
            f"{path} declares no contract (metadata key `contract`); this server "
            f"serves {', '.join(sorted(RECORD_CONTRACTS))}"
        )
    contract = meta["contract"]
    if contract not in RECORD_CONTRACTS:
        # Loudly, rather than guessing at a graph shape and dying on the
        # first move with a missing-input error.
        raise ValueError(
            f"{path} declares contract={contract!r}, which this server does not serve "
            f"(known: {', '.join(sorted(RECORD_CONTRACTS))})"
        )

    players = _required_int(meta, "players", path)
    # Topology fingerprint embedded by the exporter: a graph traced for one
    # board shape would otherwise fail silently when fed another.
    fingerprint = (
        _required_int(meta, "num_hexes", path),
        _required_int(meta, "num_vertices", path),
        _required_int(meta, "num_edges", path),
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
    policy = V2Policy(session, space)
    # `max_trades` is the older spelling of `max_offers`; a file carrying it
    # is read rather than silently ignored, which would turn a checkpoint
    # that asked not to trade into one that does.
    legacy_offers = meta.get("max_trades")
    return Loaded(
        policy=policy,
        space=space,
        players=players,
        iteration=_required_int(meta, "iteration", path) if "iteration" in meta else 0,
        search=search_config(meta),
        gate=gate_config(
            meta if not legacy_offers or "max_offers" in meta
            else {**meta, "max_offers": legacy_offers}
        ),
        trader=trader_config(meta),
        trader_veto=trader_veto_config(meta),
        trader_veto_pick=trader_veto_pick_config(meta),
    )


def _required_int(meta: dict[str, str], key: str, path: str) -> int:
    """`meta[key]` as an integer, or a `ValueError` naming the file and key."""
    if key not in meta:
        raise ValueError(f"{path} has no `{key}` metadata, which the contract requires")
    try:
        return int(meta[key])
    except ValueError:
        raise ValueError(f"{path}'s `{key}` metadata is not an integer: {meta[key]!r}") from None


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
    per core. A trader the file names is not resolved here, since nothing
    here trades: `spawn` resolves it.
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
    own_calibration: tuple[float, float] | None = None,
    decisive: float | None = None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search on `board`, trading
    through its own value-head gate. `k` is the determinized worlds each
    decision is searched in; `decisive`, the prior at which a root is played
    without searching (`hexset.mcts.Search`)."""
    return searcher_for(
        load(path, board.topology, device, threads),
        simulations=simulations,
        wave=wave,
        k=k,
        inference_batch=inference_batch,
        rng=rng,
        own_calibration=own_calibration,
        decisive=decisive,
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

    Everything that would fail later fails here, with the file named: a
    `trader` no loaded runtime registers (`netbot.declared_trader`) and,
    where the caller passes the table's `players`, a checkpoint trained for
    another player count."""
    from hexset.arena import traded

    loaded = load(path, board.topology, device, threads)
    if players is not None and players != loaded.players:
        raise ValueError(
            f"{path} was trained for {loaded.players} players, not this table's {players}"
        )
    trader = declared_trader(loaded, path)
    own = _spawn_own(path, board, loaded, rng, device, threads)
    return traded(own, trader, board, random.Random() if rng is None else rng, veto=trader_veto_of(loaded),
                  veto_picks=trader_veto_pick_of(loaded))


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
        own_calibration=loaded.search.own_calibration,
        decisive=loaded.search.decisive,
    )
