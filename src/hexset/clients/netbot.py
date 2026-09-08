# SPDX-License-Identifier: GPL-3.0-only
"""A checkpoint as a seat: bot, leaf evaluation, search, and trade gate.

Everything here is written against `hexset.clients.policy.Policy` and never
touches a runtime. Feed it the onnxruntime policy (`hexset.clients.onnxbot`)
and it serves a `.onnx` file; feed it a torch policy from the training repo
and it plays a `.pt` checkpoint. The two used to be separate implementations
of these same four classes, and they drifted three ways in the trade gate
alone before it was noticed (`agents/reference/hexn-boundary-audit.md`);
there is one of each now, and a runtime is a `Policy` and nothing more.

The constructors take a `Checkpoint` rather than a path for the same reason:
finding the file, reading its metadata and building a session is the
runtime's job, and it is the only part of "play this checkpoint" that differs
between runtimes. `hexset.clients.onnxbot.spawn(path, board)` is still the
one entry point a server needs.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hexset.actions import Action, ActionSpace
from hexset.clients.policy import Checkpoint, Policy
from hexset.game import Game, is_over, to_move
from hexset.mcts import Search
from hexset.server.rules import options_for
from hexset.state import copy_state
from hexset.trading import NETWORK_GATE_ROWS, exchange

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hexset.board.board import Board
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


@dataclass
class NetworkBot:
    """A policy answering one position at a time.

    Trades off the same value head `choose` already reads, no new
    parameters. The value head scores a candidate exchange from both
    sides at once (`_score`): the live position and the position after
    the cards move -- *both* hands, and the counterparty's ledger row, as a
    real clearing would leave them -- each read from this seat's own
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

    policy: Policy
    # Carried for a caller that builds a bot by hand and for symmetry with
    # `LeafEvaluator`; nothing here indexes it any more, because encoding a
    # position is the runtime's own business (`hexset.clients.policy`).
    space: ActionSpace
    players: int
    max_trades: int | None = None
    # This gate's clearing floor (`hexset.trading.trade_floor_of`): `accepts`
    # is a strict value-head comparison and `gains_many` reads +1/-1 off it,
    # so there is no resolution for a floor to express; strict positivity is
    # the whole gate.
    trade_floor: float = 0.0
    # Which seat this bot is installed at, or `None` for a bot that answers
    # whatever seat the view names (the arena spawns one bot per seat and
    # never asks it about another). When set, the view must agree: a gate
    # wired to the wrong seat would answer for somebody else's hand, and
    # that is the one failure the mechanic must never have quietly.
    seat: int | None = None
    # The game `choose` was last handed (or `seat_at` seated), so a trade
    # event -- which runs inside the same `apply` this bot's own choice
    # already went through -- asks about the position it is actually seated
    # at. `None` only for a bot nobody has seated yet, which cannot happen
    # in play (the trade event runs after every seat has moved through
    # setup) but is the right answer for the gate regardless.
    _seated: Game | None = field(default=None, repr=False, compare=False)

    def seat_at(self, game: Game) -> None:
        """Seat this bot at `game` without asking it to move: the position
        its gate is an evaluation of. `choose` does this itself; a driver
        that installs a bot as a gate only (a searched policy, a collector's
        seat) calls it in place of poking `_seated`."""
        _check_players(game, self.players)
        self._seated = game

    def choose(self, game: Game) -> Action:
        self.seat_at(game)
        seat = to_move(game)
        return self.policy.act_rows([(game, seat, tuple(options_for(game)))])[0]

    def _perspective(self, view: View) -> int:
        if self.seat is not None and view.perspective != self.seat:
            raise ValueError(
                f"a network bot seated at {self.seat} was asked for seat {view.perspective}"
            )
        return view.perspective

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
        seat = self._perspective(view)
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
        seat = self._perspective(view)
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
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
        """The value vector of the live position and of each scored
        candidate's post-trade position, both read from `seat`'s frame.

        The after-state is what a real clearing leaves: `hexset.trading.
        exchange` moves both hands and `PublicLedger.apply_hand_diff`
        certifies the diff, on copies the live game never keeps. Each
        candidate is handed to the policy as a *position* -- a shallow copy
        of the seated game carrying that state and that ledger -- rather
        than as an encoded row, so a runtime encodes however it likes and
        the live game is never mutated at all: there is nothing to restore,
        and a raised error leaves it exactly as `choose` left it. (The
        `set_state`/ledger swap is still what puts a hypothetical position
        together; it is done to the copy, which is the only reason the live
        game can stay untouched. Everything else on `Game` -- phase, turn
        count, the free-road counter -- rides along on the copy, which is
        what the encoders read and what a fresh `Game` would not have.)

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
        rows: list[tuple[Game, int]] = [(game, seat)]
        scored: dict[int, int] = {}
        for i in order:
            bundle = received[i]
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            state = copy_state(original)
            ledger = game.ledger.copy()
            hands_before = [h[:] for h in state.hands]
            exchange(state, seat, counterparties[i], bundle)
            ledger.apply_hand_diff(hands_before, state.hands)
            after = copy.copy(game)
            after.set_state(state)
            after.ledger = ledger
            scored[i] = len(rows)
            rows.append((after, seat))
        values = self.policy.value_rows(rows)
        return values[0], {i: values[row] for i, row in scored.items()}


def _is_small(bundle: Bundle) -> bool:
    """Both sides of `bundle` move at most two cards -- always scored,
    uncapped, because two-for-one and two-for-two are the overwhelming
    majority of what a table trades."""
    give = sum(-n for n in bundle if n < 0)
    take = sum(n for n in bundle if n > 0)
    return give <= 2 and take <= 2


@dataclass
class NetworkEvaluator:
    """The value head as `hexset.bots.SearchBot`'s leaf evaluation."""

    policy: Policy
    players: int
    max_trades: int | None = None

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        _check_players(game, self.players)
        return list(self.policy.value_rows([(game, seat)])[0])


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward."""

    policy: Policy
    # As on `NetworkBot`: kept because `hexset.arena.leaf_evaluator`'s
    # registered factory signature is `(policy, space, pad_to)`, and because
    # a caller with a loaded checkpoint has one to hand. The policy indexes
    # its own space.
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

        A contract-6 value head is trained on a win probability, whether it
        came from PPO or from distillation, which was ported to the same
        target. So every non-terminal leaf in a wave is scored on that scale,
        and returning `terminal_relative_points` here would back a points
        margin up the tree alongside them.

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
    (`bot_for`) traded through its value head. Found at the served table
    2026-09-08: `linear24` (exported `search: mcts`) never traded, `clio`
    (no search) did. The gate is the plain bot's, seated at the position
    `choose` was last handed, exactly as `NetworkBot.choose` seats its own.
    """

    def __init__(self, evaluator, gate: NetworkBot, **kwargs) -> None:
        super().__init__(evaluator, **kwargs)
        self.gate = gate
        self.trade_floor = gate.trade_floor

    def choose(self, game: Game) -> Action:
        self.gate.seat_at(game)
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


def bot_for(checkpoint: Checkpoint, *, max_trades: int | None = None) -> NetworkBot:
    """`checkpoint`, playing one position at a time.

    `max_trades` of `None` means the trade switch the checkpoint recorded
    training under -- the default that measures a policy on the game it
    learned. Pass `0` to seat it as its own no-trade referent.
    """
    return NetworkBot(
        policy=checkpoint.policy,
        space=checkpoint.space,
        players=checkpoint.players,
        max_trades=checkpoint.max_trades if max_trades is None else max_trades,
    )


def evaluator_for(
    checkpoint: Checkpoint, *, max_trades: int | None = None
) -> NetworkEvaluator:
    """`checkpoint`'s value head as a leaf evaluation for the handcrafted
    search (`hexset.bots.SearchBot`), rather than as a bot of its own."""
    return NetworkEvaluator(
        policy=checkpoint.policy,
        players=checkpoint.players,
        max_trades=checkpoint.max_trades if max_trades is None else max_trades,
    )


def searcher_for(
    checkpoint: Checkpoint,
    *,
    simulations: int = 128,
    wave: int = 16,
    max_trades: int | None = None,
    inference_batch: int | None = None,
    rng=None,
) -> GatedSearch:
    """`checkpoint` as a batched PUCT search, trading through its own
    value-head gate (`GatedSearch`)."""
    budget = checkpoint.max_trades if max_trades is None else max_trades
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            space=checkpoint.space,
            pad_to=inference_batch,
        ),
        bot_for(checkpoint, max_trades=budget),
        simulations=simulations,
        wave=wave,
        max_trades=budget,
        rng=rng,
    )


def _checkpoint_path(weights: object, what: str) -> str:
    """An `Entrant.weights` read as a checkpoint path.

    `weights` is fitted evaluation coefficients for a handcrafted entrant and
    a path for a network-backed one; a lineup that mixes the two up is
    refused here rather than inside a loader, where the error would name a
    numpy array.
    """
    if isinstance(weights, str):
        return weights
    raise ValueError(f"{what}'s weights is a checkpoint path")


def register_entrants(loader, *, evaluator_max_trades: int | None = None) -> None:
    """Make `hexset.arena`'s "network" and "mcts" entrant kinds -- and its
    "network" evaluator, checkpoint loader and leaf-evaluator factory --
    spawnable through `loader`.

    `loader(path, topology)` returns a `Checkpoint`; everything the arena
    then does with it is this module's, so a runtime registers itself in one
    call instead of carrying five factories of its own.

    `evaluator_max_trades` is the trade switch for the "network" *evaluator*
    (the value head under `SearchBot`, `netsearch`/`netgreedy`): `None`
    keeps each checkpoint's recorded budget; `0` switches trading off for
    every evaluator this runtime provides, which is what a handcrafted
    search over a learned value wants -- its gate scores a bare `GameState`
    no encoder can read, so it must never be asked to trade.

    Deliberately *not* called at import by any runtime in this package. The
    kinds are global names with a single owner, an entrant's `weights` is a
    path in whatever format that owner loads, and a process that has merely
    imported a module should not thereby have changed what
    `arena.spawn` does with somebody else's checkpoint. A driver that wants
    network entrants calls this itself, once, naming the runtime it means.
    """
    from hexset.arena import (
        register_checkpoint_loader,
        register_entrant_kind,
        register_evaluator_provider,
        register_leaf_evaluator_factory,
    )

    def _spawn_network(entrant, board: Board, rng) -> NetworkBot:
        path = _checkpoint_path(entrant.weights, "a network entrant")
        return bot_for(loader(path, board.topology), max_trades=entrant.max_trades)

    def _spawn_mcts(entrant, board: Board, rng) -> GatedSearch:
        return searcher_for(
            loader(_checkpoint_path(entrant.weights, "an mcts entrant"), board.topology),
            simulations=entrant.simulations,
            wave=entrant.wave,
            max_trades=entrant.max_trades,
            rng=rng,
        )

    def _spawn_evaluator(weights: object, board: Board) -> NetworkEvaluator:
        path = _checkpoint_path(weights, "a network evaluator")
        return evaluator_for(loader(path, board.topology), max_trades=evaluator_max_trades)

    def _leaf_evaluator(policy, space, pad_to=None) -> LeafEvaluator:
        return LeafEvaluator(policy=policy, space=space, pad_to=pad_to)

    register_entrant_kind("network", _spawn_network)
    register_entrant_kind("mcts", _spawn_mcts)
    register_evaluator_provider("network", _spawn_evaluator)
    register_checkpoint_loader(loader)
    register_leaf_evaluator_factory(_leaf_evaluator)
