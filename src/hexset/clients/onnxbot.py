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
longer served — the owner dropped it 2026-09-02. A `contract=1` file, or one
with no `contract` key at all, is refused by name at load.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np
import onnxruntime as ort

from hexset.actions import Action, ActionSpace, build_space
from hexset.board.topology import Topology
from hexset.game import Game, is_over, to_move
from hexset.mcts import Search
from hexset.onnx_record import record_from_game
from hexset.server.constants import RECORD_CONTRACTS
from hexset.server.modelmeta import SearchConfig, search_config
from hexset.server.rules import options_for
from hexset.state import copy_state
from hexset.trading import NETWORK_GATE_ROWS, exchange

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
    # The topology fingerprint `hexn.export_onnx` embeds: a graph traced
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
        # back to.
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


def load(
    path: str, topology: Topology, device: str = "cpu", threads: int | None = None
) -> Loaded:
    """The checkpoint at `path`, ready to act on boards of this topology.

    Cache key folds in the file's mtime, unlike the training repo's loader: its
    `.pt` checkpoints under `runs/` are effectively immutable per-run
    artifacts, but hexset's whole pitch is replacing a file in `models/`
    by name — without the mtime, a same-named replacement would silently
    keep serving the old in-memory session.

    `threads` caps onnxruntime's intra- and inter-op pools. Left `None`,
    onnxruntime sizes them from the core count, which is right for one bot at
    one table and wrong for a caller that has already sharded the work across
    processes -- each would claim the whole machine. Such a caller passes 1.
    """
    return _load_cached(path, topology, device, os.stat(path).st_mtime_ns, threads)


@dataclass
class NetworkBot:
    """A policy answering one position at a time.

    Trades off the same value head `choose` already reads, no new
    parameters. The value head scores a candidate exchange from both
    sides at once (`_score`): the live position and the position after
    the cards move -- *both* hands, and the counterparty's ledger row, as a
    real clearing would leave them -- each encoded from this seat's own
    frame, so nothing here reads a hand this seat may not know. The head
    answers with one win probability per seat, and that vector is the
    whole gate:

    * `gains_many` -- this seat's own row, after minus before: its private
      gain in win probability, `hexset.trading.trade_event`'s gate and the
      round's own-side reading.
    * `estimate_many` -- the *counterparty's* row, after minus before: this
      seat's estimate of what the exchange does to that seat's chances,
      read by `hexset.trading.default_offer`/`default_respond` so a bot
      offers, and counters with, the candidate best for itself among those
      it believes the other seat gains from too (`agents/reference/
      trading-final.md`, "the trade round", items 1-2). By construction
      that prices the risk of handing an opponent win probability: a
      bundle that lifts this seat's row a little and the counterparty's a
      lot is a bad offer, and it reads as one.
    * `accepts`/`accepts_many` -- `gains_many` thresholded at strictly
      positive.

    `trade_floor` is `0.0`: this gate's resolution has not been measured
    (heximax's has, `hexset.bots.heximax.HEXIMAX_TRADE_FLOOR`); its gains
    are in win probability, so a paired-chance measurement would replace it.
    """

    policy: V2Policy
    space: ActionSpace
    players: int
    max_trades: int | None = None
    # This gate's clearing floor (`hexset.trading.trade_floor_of`): `accepts`
    # is a strict value-head comparison and `gains_many` reads +1/-1 off it,
    # so there is no resolution for a floor to express; strict positivity is
    # the whole gate.
    trade_floor: float = 0.0
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

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        """This seat's own private gain from every candidate at once:
        `V(after)[seat] - V(before)[seat]`, win probability, one forward.
        `-1.0` for a candidate this seat cannot cover and for any past the
        scored set (`_score`'s cap); nothing when nobody has seated this bot
        (no `choose` has run) or trading is switched off.
        """
        if self.max_trades == 0 or self._seated is None or not received:
            return [-1.0] * len(received)
        seat = view.perspective
        before, afters = self._score(seat, view, received, counterparties)
        out = [-1.0] * len(received)
        for i, after in afters.items():
            out[i] = float(after[seat] - before[seat])
        return out

    def estimate_many(
        self, view: View, candidates: Sequence[tuple[int, Bundle]]
    ) -> list[float]:
        """This seat's estimate of each `(counterparty, bundle)` candidate's
        *counterparty*-side gain: `V(after)[them] - V(before)[them]` on the
        same two positions `gains_many` scores, from this seat's own frame
        -- its own belief about the other seat's chances, standing in for
        the acceptance model the design names (`agents/reference/
        trading-final.md`, "the trade round", item 1) until one exists.
        """
        if self.max_trades == 0 or self._seated is None or not candidates:
            return [-1.0] * len(candidates)
        seat = view.perspective
        received = [b for _, b in candidates]
        thems = [c for c, _ in candidates]
        before, afters = self._score(seat, view, received, thems)
        out = [-1.0] * len(candidates)
        for i, after in afters.items():
            them = thems[i]
            out[i] = float(after[them] - before[them])
        return out

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        """`gains_many` for one candidate, thresholded at strictly positive
        -- the engine's termination argument rests on the acting seat's own
        gain strictly increasing at every step."""
        return self.gains_many(view, [received], [counterparty])[0] > 0.0

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        """`gains_many`, thresholded at strictly positive."""
        return [gain > 0.0 for gain in self.gains_many(view, received, counterparties)]

    def _score(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """The value vector of the live position and of each scored
        candidate's post-trade position, both encoded from `seat`'s frame.

        The after-state is what a real clearing leaves: `hexset.trading.
        exchange` moves both hands and `PublicLedger.apply_hand_diff`
        certifies the diff, on copies the live game never keeps. `set_state`
        and the ledger swap are the engine's sanctioned way to put a
        hypothetical position under the encoder -- phase, turn count and the
        rest live on `Game`, not `GameState` -- and both are restored in a
        `finally`, so a raised error still leaves the live game exactly as
        `choose` left it.

        Every candidate two cards or fewer a side (`_is_small`) is scored;
        the rest fill whatever is left of `NETWORK_GATE_ROWS` in the order
        the engine enumerated them -- the stated cost bound on this gate's
        own evaluation, not a claim about which candidates it favours. A
        candidate `seat` cannot cover is never scored. One "before" row plus
        one row per scored candidate is the whole fan-out, one forward.
        """
        game = self._seated
        assert game is not None  # callers check this first
        hand = list(view.known[seat])
        small = [i for i, bundle in enumerate(received) if _is_small(bundle)]
        rest = [i for i in range(len(received)) if not _is_small(received[i])]
        order = (small + rest)[: max(NETWORK_GATE_ROWS, len(small))]

        # true state: the engine is the referee for what a clearing leaves.
        original = game.state(seat, hidden=False)
        original_ledger = game.ledger
        records = [record_from_game(game, seat, self.space, options_for(game))]
        rows: dict[int, int] = {}
        try:
            for i in order:
                bundle = received[i]
                if any(n + d < 0 for n, d in zip(hand, bundle)):
                    continue
                state = copy_state(original)
                ledger = original_ledger.copy()
                hands_before = [h[:] for h in state.hands]
                exchange(state, seat, counterparties[i], bundle)
                ledger.apply_hand_diff(hands_before, state.hands)
                game.set_state(state)
                game.ledger = ledger
                rows[i] = len(records)
                records.append(record_from_game(game, seat, self.space, options_for(game)))
        finally:
            game.set_state(original)
            game.ledger = original_ledger
        values = self.policy.value_of(records)
        return values[0], {i: values[row] for i, row in rows.items()}


def _is_small(bundle: Bundle) -> bool:
    """Both sides of `bundle` move at most two cards -- always scored,
    uncapped, because two-for-one and two-for-two are the overwhelming
    majority of what a table trades (mirrors `hexn.policy._is_small`)."""
    give = sum(-n for n in bundle if n < 0)
    take = sum(n for n in bundle if n > 0)
    return give <= 2 and take <= 2


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

    def terminal(self, game: Game) -> Sequence[float]:
        """`hexset.mcts.Evaluator.terminal`: the one-hot winner, board-seat
        order.

        `RECORD_CONTRACTS` is contract 6 alone, and a contract-6 value head is
        trained on `hexn.rewards.win_loss` — a win probability, whether it came
        from `hexn.ppo` or from `hexn.distill`, which was ported to the same
        target. So every non-terminal leaf in a wave is scored on that scale,
        and returning `terminal_relative_points` here would back a points
        margin up the tree alongside them. `hexn.netbot.LeafEvaluator`, the
        torch-side twin of this class, already returns the winner for exactly
        this reason.

        Raises if `game` has not finished: `Search` only calls this on a
        terminal node, so a caller passing an unfinished game has a bug of its
        own.
        """
        if not is_over(game):
            raise ValueError("terminal() called on a game that has not finished")
        players = game.state(0, hidden=False).num_players
        winner = game.won_by
        return tuple(1.0 if seat == winner else 0.0 for seat in range(players))


class GatedSearch(Search):
    """`hexset.mcts.Search` over a checkpoint, with that checkpoint's own
    trade gate -- `gains_many`, `estimate_many`, `accepts`, `accepts_many`.

    `Search` decides moves and nothing else: it has no `accepts`,
    `accepts_many` or `gains_many`, so `hexset.trading.valued_many` priced
    every candidate at -1 for a searched checkpoint -- it never accepted an
    offer and never made one, while the same checkpoint played plainly
    (`network_bot`) traded through its value head. Found at the served table
    2026-09-08: `linear24` (exported `search: mcts`) never traded, `clio`
    (no search) did. The gate is the plain bot's, seated at the position
    `choose` was last handed, exactly as `NetworkBot.choose` seats its own.
    """

    def __init__(self, evaluator, gate: NetworkBot, **kwargs) -> None:
        super().__init__(evaluator, **kwargs)
        self.gate = gate
        self.trade_floor = gate.trade_floor

    def choose(self, game: Game) -> Action:
        self.gate._seated = game
        return super().choose(game)

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        return self.gate.accepts(view, received, counterparty)

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        return self.gate.accepts_many(view, received, counterparties)

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        return self.gate.gains_many(view, received, counterparties)

    def estimate_many(self, view: View, candidates: Sequence[tuple[int, Bundle]]) -> list[float]:
        return self.gate.estimate_many(view, candidates)


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
    threads: int | None = None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search, playing on `board`,
    trading through the checkpoint's own value-head gate (`GatedSearch`)."""
    loaded = load(path, board.topology, device, threads)
    budget = loaded.max_trades if max_trades is None else max_trades
    return GatedSearch(
        LeafEvaluator(
            policy=loaded.policy,
            space=loaded.space,
            pad_to=inference_batch,
        ),
        NetworkBot(
            policy=loaded.policy, space=loaded.space, players=loaded.players, max_trades=budget
        ),
        simulations=simulations,
        wave=wave,
        max_trades=budget,
        rng=rng,
    )


def network_evaluator(
    path: str, board, *, device: str = "cpu", threads: int | None = None
) -> NetworkEvaluator:
    """The checkpoint at `path` as a leaf evaluation for the search."""
    loaded = load(path, board.topology, device, threads)
    return NetworkEvaluator(
        policy=loaded.policy, players=loaded.players, max_trades=loaded.max_trades
    )


def network_bot(
    path: str,
    board,
    *,
    max_trades: int | None = None,
    device: str = "cpu",
    threads: int | None = None,
) -> NetworkBot:
    """The checkpoint at `path`, playing on `board`."""
    loaded = load(path, board.topology, device, threads)
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
    threads: int | None = None,
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
    loaded = load(path, board.topology, device, threads)
    if not loaded.search.searches:
        return network_bot(
            path, board, max_trades=max_trades, device=device, threads=threads
        )
    return searcher(
        path,
        board,
        simulations=loaded.search.simulations,
        wave=loaded.search.wave,
        max_trades=max_trades,
        device=device,
        rng=rng,
        threads=threads,
    )
