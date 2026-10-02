# SPDX-License-Identifier: GPL-3.0-only
"""Runtime-neutral checkpoint adapters for bots, evaluation, search and trading.

Constructors take `Checkpoint` objects and use only the batched `Policy`
interface; the runtime owns loading, metadata and encoding. A gate used without
`choose` must be installed with `seat_at(game)` first, and each gate belongs to
one game and one fixed seat."""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hexset.actions import Action, ActionType, apply
from hexset.clients.policy import Checkpoint, Policy
from hexset.game import Game, imagine, is_over, to_move
from hexset.mcts import Search
from hexset.actions import options_for
from hexset.clients._modelmeta import gate_config_of, trader_of
from hexset.state import HiddenHand, copy_state, is_hidden
from hexset import trading
from hexset.trading import UNLIMITED, TradeParams, TradeProtocol, exchange
from hexset.view import View

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hexset.board.board import Board
    from hexset.trading import Bundle

__all__ = [
    "NetworkBot",
    "LeafEvaluator",
    "GatedSearch",
    "bot_for",
    "searcher_for",
    "register_entrants",
]


def _check_players(game: Game, players: int) -> None:
    # true state: `num_players` is a fixed, public board property.
    num_players = game.state(0, hidden=False).num_players
    if num_players != players:
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {num_players}"
        )


@dataclass
class NetworkBot(trading.DeclaredTrade):
    """A policy answering one position at a time, trading off its value head.

    The head's per-board-seat win probabilities are the whole valuation:
    `gains_many` is this seat's row after minus before, `estimate_many` the
    counterparty's, `accepts` those thresholded at strictly positive. Every
    coverable candidate is priced, unfiltered, in one batched forward.

    How it *bargains* over that valuation is `trade`, the same
    `hexset.trading.TradeParams` every bot that declares its bargaining
    carries: a checkpoint that asks for card caps, a responder price or the
    fragmented policy gets the shared `TradeProtocol`, installed here, rather
    than a second implementation. A checkpoint that asks for none of them --
    the default -- installs no hooks and is answered by the engine's own.
    """

    policy: Policy
    players: int
    #: Everything this gate declares about how it bargains: the off switch,
    #: the clearing floor, the card caps and direction, the responder price,
    #: the offer budget, the fragment plan and the continuation budget.
    trade: TradeParams = UNLIMITED
    # Stream behind `gate_plies > 0`'s sampled worlds; unread at `0`.
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    _salt: int | None = field(default=None, repr=False, compare=False)
    # Seat this bot is installed at, which the view must then agree with, or
    # `None` to answer whatever seat the view names.
    seat: int | None = None
    _seated: Game | None = field(default=None, repr=False, compare=False)
    _memo: tuple[tuple, dict] | None = field(default=None, repr=False, compare=False)
    _protocol: "TradeProtocol | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # The protocol installs exactly the hooks `trade` needs and shadows
        # the rest with `None`, which is how the engine reads "this gate has
        # no such hook" and falls back to its own defaults.
        # Seeded from `rng`'s state rather than a draw from it, so building
        # the protocol never disturbs the stream `_world_rng` is salted from.
        seed = int.from_bytes(
            hashlib.sha256(repr(self.rng.getstate()).encode()).digest()[:8], "big"
        )
        self._protocol = trading.install(self, TradeProtocol(self, self.trade, seed=seed))

    # Every bargaining limit -- `trade_params`, `max_offers`, `trade_floor`,
    # `gate_plies`, the offer budget -- reads through `trade`
    # (`trading.DeclaredTrade`).

    def seat_at(self, game: Game) -> None:
        """Seat this bot at `game` without asking it to move: the position its
        gate evaluates. `choose` does this itself. Raises `ValueError` for a
        game of another player count than the checkpoint's."""
        _check_players(game, self.players)
        self._seated = game
        self._memo = None

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
        """`V(after)[seat] - V(before)[seat]` per candidate, in win probability;
        `-1.0` for one this seat cannot cover, for all of them when unseated or
        when trading is off."""
        if not self.trade.trades or self._seated is None or not received:
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
        """`V(after)[them] - V(before)[them]` per `(counterparty, bundle)`, on
        the positions `gains_many` scores, from this seat's own frame."""
        if not self.trade.trades or self._seated is None or not candidates:
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
        """`gains_many` for one candidate, strictly positive, as a yes or no
        for a caller holding this bot. The engine never asks it: it prices
        through `gains_many`, which takes precedence (`trading.valued_many`).

        Where this gate's parameters price a response -- a card cap, a
        responder risk -- this is that same consent price, so the answer is
        the one automatic clearing and served execution would give."""
        if self.trade.constrains_responses:
            return self._protocol.clears(
                self._protocol.consent_gain(view, received, counterparty, role="responder")
            )
        return self.gains_many(view, [received], [counterparty])[0] > 0.0

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        """`accepts` over a batch: one forward for the whole batch where
        nothing is priced beyond the gain, candidate by candidate where the
        parameters price a response."""
        if self.trade.constrains_responses:
            return [
                self.accepts(view, r, c) for r, c in zip(received, counterparties)
            ]
        return [gain > 0.0 for gain in self.gains_many(view, received, counterparties)]

    def _score(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
        """`_evaluate`, answering a repeat of the last ask from `_memo`, keyed
        on everything it reads."""
        game = self._seated
        assert game is not None  # callers check this first
        state = view.state
        key = (
            game.turns, game.phase, game.current_player, game.free_roads,
            seat, view.perspective, tuple(view.sizes), view.signature(),
            # Road Building moves the board without spending a card, so a
            # hand-only key would go stale.
            tuple(state.edge_owner), tuple(state.vertex_building),
            tuple(received), tuple(counterparties),
        )
        memo = self._memo
        if memo is not None and memo[0] == key:
            return memo[1]
        scored = self._evaluate(seat, view, received, counterparties)
        self._memo = (key, scored)
        return scored

    def _evaluate(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
        """The live position's value vector and every coverable candidate's,
        keyed by candidate index. At `gate_plies == 0` each post-trade position
        is read from `seat`'s own frame (`_after`), never the true hand; at
        `gate_plies > 0` the rollout *acts*, so worlds are sampled from `seat`'s
        belief and certified against negative exchanges, all from one draw."""
        game = self._seated
        assert game is not None  # callers check this first
        hand = list(view.known[seat])

        coverable: list[tuple[int, int, Bundle]] = []  # (candidate index, them, bundle)
        for i, bundle in enumerate(received):
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            # An offer may ask for cards no seat is known to hold, so a
            # candidate can name a counterparty the public record says cannot
            # cover it. There is no position after that exchange to value.
            if not trading.has_room(view, counterparties[i], bundle):
                continue
            coverable.append((i, counterparties[i], bundle))

        if not coverable:
            return (), {}

        if self.gate_plies == 0:
            afters = {
                i: self._after(seat, them, hand, bundle, view)
                for i, them, bundle in coverable
            }
            rows: list[tuple[Game, int]] = [(game, seat)] + [
                (after, seat) for after in afters.values()
            ]
            values = self.policy.value_rows(rows)
            afters_values = {i: values[row] for row, i in enumerate(afters, start=1)}
            return values[0], afters_values

        # `View.certify` combines repeat counterparties by per-resource max,
        # so one world covers every candidate's bundle.
        certified = View(
            view.state, view.ledger, seat,
            certify=[(them, [max(0, n) for n in bundle]) for _, them, bundle in coverable],
        )
        rng = self._world_rng(view, seat)
        sampled = certified.sample(rng)
        # One world certifies every candidate at once, and a counterparty's
        # asks together can exceed its hand: value only what this world covers.
        coverable = [
            (i, them, bundle) for i, them, bundle in coverable
            if trading.holds(sampled, them, [max(0, n) for n in bundle])
        ]
        before = imagine(game, rng, randomize_deck=True)
        before.set_state(sampled)

        afters: dict[int, Game] = {}
        for i, them, bundle in coverable:
            after = imagine(game, rng, randomize_deck=True)
            exchanged = copy_state(sampled)
            hands_before = [h[:] for h in exchanged.hands]
            exchange(exchanged, seat, them, bundle)
            after.set_state(exchanged)
            after.ledger.apply_hand_diff(hands_before, exchanged.hands)
            afters[i] = after

        worlds = [before] + list(afters.values())
        values = self._continue(to_move(game), seat, worlds, self.gate_plies)
        afters_values = {i: values[row] for row, i in enumerate(afters, start=1)}
        return values[0], afters_values

    def _world_rng(self, view: View, seat: int) -> random.Random:
        """Seeded by the information set alone, salted once from this bot's
        `rng`, with no candidate in the key: two asks about one position agree,
        two bots draw differently."""
        if self._salt is None:
            self._salt = self.rng.getrandbits(64)
        key = (self._salt, view.perspective, tuple(view.sizes), view.signature())
        return random.Random(hash(key))

    def _continue(
        self, mover: int, seat: int, worlds: Sequence[Game], plies: int
    ) -> list[tuple[float, ...]]:
        """Each world's value vector, from `seat`'s frame, after `mover`'s best
        play from it, `plies` deep. Worlds advance in lockstep, one batched
        `act_rows` per ply, and must be belief-sampled copies, never the live
        table's own state."""
        live = [i for i, g in enumerate(worlds) if not is_over(g) and to_move(g) == mover]
        for _ in range(plies):
            if not live:
                break
            rows = [(worlds[i], mover, tuple(options_for(worlds[i]))) for i in live]
            chosen = self.policy.act_rows(rows)
            still: list[int] = []
            for i, action in zip(live, chosen):
                if action.type is ActionType.END_TURN:
                    continue
                apply(worlds[i], action)
                g = worlds[i]
                if not is_over(g) and to_move(g) == mover:
                    still.append(i)
            live = still
        out: list[tuple[float, ...] | None] = [None] * len(worlds)
        pending = []
        for i, g in enumerate(worlds):
            if is_over(g):
                out[i] = tuple(1.0 if s == g.won_by else 0.0 for s in range(self.players))
            else:
                pending.append(i)
        if pending:
            values = self.policy.value_rows([(worlds[i], seat) for i in pending])
            for i, v in zip(pending, values):
                out[i] = tuple(v)
        return out  # type: ignore[return-value]

    def _after(
        self, seat: int, them: int, hand: Sequence[int], bundle: Bundle, view: View
    ) -> Game:
        """The position `bundle` leaves, read from `seat`'s own frame, over a
        copied state *and* ledger so nothing mutates the live table. `them`'s
        known row is raised to cover what the bundle says they give before being
        moved by it, so the true hand is never read and the subtraction cannot
        go negative. The ledger must be copied too: `View` reads `them`'s known
        cards from it, not from `state.hands`."""
        game = self._seated
        assert game is not None  # callers check this first
        state = copy_state(view.state)
        state.hands[seat] = [n + d for n, d in zip(hand, bundle)]

        certified = View(
            view.state, view.ledger, seat,
            certify=[(them, [max(0, n) for n in bundle])],
        )
        their_known = [k - d for k, d in zip(certified.known[them], bundle)]

        # `View` reads `state.hands[them]` for its size alone; the known row
        # would shrink `them` by every card this seat cannot name.
        if is_hidden(state.hands[them]):
            state.hands[them] = HiddenHand(view.sizes[them] - sum(bundle))
        else:
            state.hands[them] = [max(0, n - d) for n, d in zip(state.hands[them], bundle)]

        ledger = view.ledger.copy()
        ledger.seats[them].known = their_known
        ledger.seats[them].unknown = certified.unknown[them]
        ledger.seats[seat].known = list(state.hands[seat])
        ledger.seats[seat].unknown = 0

        after = copy.copy(game)
        after.set_state(state)
        after.ledger = ledger
        return after


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward."""

    policy: Policy
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
        """`hexset.mcts.Evaluator.terminal`: the one-hot winner in board-seat
        order, the scale a wave's other leaves are scored on. Raises if `game`
        has not finished."""
        if not is_over(game):
            raise ValueError("terminal() called on a game that has not finished")
        # true state: `num_players` is a fixed, public board property.
        players = game.state(0, hidden=False).num_players
        winner = game.won_by
        return tuple(1.0 if seat == winner else 0.0 for seat in range(players))


class GatedSearch(trading.DeclaredTrade, Search):
    """Policy/value-guided MCTS with the checkpoint's own trade gate. `choose`
    binds the gate to the live game before searching; trade acceptance and
    counterparty estimates delegate to `NetworkBot`.

    This, not the `NetworkBot` behind it, is what a table seats, so it carries
    the checkpoint's bargaining parameters and installs the same protocol over
    its own delegated valuation. A searching entrant and a one-forward entrant
    from the same checkpoint therefore bargain identically."""

    def __init__(self, evaluator, gate: NetworkBot, **kwargs) -> None:
        super().__init__(evaluator, **kwargs)
        self.gate = gate
        self.trade = gate.trade
        self._protocol = trading.install(
            self, TradeProtocol(self, gate.trade, seed=gate._protocol.seed)
        )

    def seat_at(self, game: Game) -> None:
        """`NetworkBot.seat_at` on the gate behind this search."""
        self.gate.seat_at(game)

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


def bot_for(
    checkpoint: Checkpoint, *, rng: random.Random | None = None,
    trade: TradeParams | None = None,
) -> NetworkBot:
    """`checkpoint`, playing one position at a time.

    The gate is built from what the checkpoint declares about itself --
    clearing floor, continuation budget, card caps, responder price and
    fragment plan -- so a file that asks for the fragmented policy gets it and
    one that asks for nothing bargains the way every checkpoint always has.
    `trade` replaces what the checkpoint declares -- that is how a no-trade
    arm is built -- and `rng` seeds a `gate_plies > 0` checkpoint's sampled
    worlds."""
    trade = gate_config_of(checkpoint) if trade is None else trade
    return NetworkBot(
        policy=checkpoint.policy,
        players=checkpoint.players,
        trade=trade,
        rng=random.Random() if rng is None else rng,
    )


def searcher_for(
    checkpoint: Checkpoint,
    *,
    simulations: int = 128,
    wave: int = 16,
    k: int = 1,
    inference_batch: int | None = None,
    rng: random.Random | None = None,
    trade: TradeParams | None = None,
) -> GatedSearch:
    """`checkpoint` as a batched PUCT search, trading through its own value-head
    gate. `rng` seeds both the search and, derived from it, the gate's sampling
    stream, so one seed reproduces the entrant.

    `k` is how many determinizations of the seat's own belief each decision is
    searched in (`hexset.mcts.Search.worlds`); the tree is never rooted on the
    true state from here."""
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            pad_to=inference_batch,
        ),
        bot_for(checkpoint, rng=gate_rng, trade=trade),
        simulations=simulations,
        wave=wave,
        k=k,
        rng=rng,
    )


def _checkpoint_path(weights: object, what: str) -> str:
    """An `Entrant.weights` read as a checkpoint path; a lineup that passes
    anything else is refused here rather than inside a loader."""
    if isinstance(weights, str):
        return weights
    raise ValueError(f"{what}'s weights is a checkpoint path")


def register_entrants(loader) -> None:
    """Register network and MCTS arena factories for a `loader(path, topology)
    -> Checkpoint`. Call explicitly in each process that spawns entrants;
    imports do not choose a runtime."""
    from hexset.arena import register_entrant_kind, traded

    # A trader the checkpoint declares answers its trades unless the entrant
    # names its own, which `hexset.arena.spawn` seats after this returns.
    def own_trader(entrant, checkpoint) -> str | None:
        return None if entrant.trader is not None else trader_of(checkpoint)

    def spawn_network(entrant, board: Board, rng):
        checkpoint = loader(_checkpoint_path(entrant.weights, "network"), board.topology)
        bot = bot_for(checkpoint, rng=rng, trade=entrant.trade)
        return traded(bot, own_trader(entrant, checkpoint), board, rng)

    def spawn_mcts(entrant, board: Board, rng):
        checkpoint = loader(_checkpoint_path(entrant.weights, "mcts"), board.topology)
        bot = searcher_for(
            checkpoint,
            simulations=entrant.simulations, wave=entrant.wave, k=entrant.k,
            rng=rng, trade=entrant.trade,
        )
        return traded(bot, own_trader(entrant, checkpoint), board, rng)

    register_entrant_kind("network", spawn_network)
    register_entrant_kind("mcts", spawn_mcts)
