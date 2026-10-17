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
from hexset.mcts import Evaluation, Search, calibrate_own
from hexset.actions import options_for
from hexset.clients._modelmeta import (
    gate_config_of, robber_tip_of, trader_of, trader_veto_of, trader_veto_pick_of,
)
from hexset.state import HiddenHand, copy_state
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
    "tip_robber",
    "register_entrants",
    "declared_trader",
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
    # Stream behind the gate's sampled worlds: read off-turn, and on every
    # read at `gate_plies > 0`.
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    _salt: int | None = field(default=None, repr=False, compare=False)
    # Seat this bot is installed at, which the view must then agree with, or
    # `None` to answer whatever seat the view names.
    seat: int | None = None
    #: `(a, b)` for `hexset.mcts.calibrate_own`, or `None`: the seat's own
    #: chance in every value vector the gate prices is read through it, the
    #: same map its search reads positions with, so a gain and the clearing
    #: floor are on that scale.
    own_calibration: tuple[float, float] | None = None
    #: How far this seat's robber moves lean towards the leader (`tip_robber`);
    #: `0.0` plays them as the policy picks.
    robber_tip: float = 0.0
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
        options = tuple(options_for(game))
        if self.robber_tip and options[0].type is ActionType.MOVE_ROBBER:
            # The same forward `act_rows` makes, read for its prior and value.
            ((prior, value),) = self.policy.score_rows([(game, seat, options)])
            picked = max(range(len(options)), key=lambda i: prior[i])
            return options[tip_robber(options, prior, value, seat, picked, self.robber_tip)]
        return self.policy.act_rows([(game, seat, options)])[0]

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
        the positions `gains_many` scores, read in the same frames."""
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
        if self.own_calibration is not None and scored[0]:
            before, afters = scored
            scored = (
                calibrate_own(before, seat, self.own_calibration),
                {i: calibrate_own(v, seat, self.own_calibration) for i, v in afters.items()},
            )
        self._memo = (key, scored)
        return scored

    def _evaluate(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
        """The live position's value vector and every coverable candidate's,
        keyed by candidate index. At `gate_plies == 0` each post-trade
        position is read from `seat`'s own frame (`_after`), never the true
        hand: on its own turn always, and on another seat's under
        `gate_own_frame` (the default) with `gate_frames` `one`. Otherwise
        worlds are sampled from `seat`'s belief and certified against negative
        exchanges, all from one draw: a rollout *acts* on them, and on another
        seat's turn they are read from the frames `_frames` names, which need
        a whole hand (`_continue`). Every sampled world shows `seat`'s own
        hand whole (`_shown`): read from another seat's frame, only what
        `seat` cannot know is guessed."""
        game = self._seated
        assert game is not None  # callers check this first
        hand = list(view.known[seat])
        coverable = self._coverable(seat, view, received, counterparties)
        if not coverable:
            return (), {}

        to_act = to_move(game)
        if self._reads_own_frame(game, seat):
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
        # `sampled` deals its own deck from this seat's belief, so the copies
        # take it as it is: a hosted seat's deck is a count, which no shuffle
        # may read.
        before = imagine(game, rng, randomize_deck=False, seat=seat)
        before.set_state(sampled)
        _shown(before, seat)

        afters: dict[int, Game] = {}
        for i, them, bundle in coverable:
            after = imagine(game, rng, randomize_deck=False, seat=seat)
            exchanged = copy_state(sampled)
            hands_before = [h[:] for h in exchanged.hands]
            exchange(exchanged, seat, them, bundle)
            after.set_state(exchanged)
            after.ledger.apply_hand_diff(hands_before, exchanged.hands)
            _shown(after, seat)
            afters[i] = after

        worlds = [before] + list(afters.values())
        values = self._continue(to_act, seat, worlds, self.gate_plies)
        afters_values = {i: values[row] for row, i in enumerate(afters, start=1)}
        return values[0], afters_values

    def _coverable(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[tuple[int, int, Bundle]]:
        """`(candidate index, them, bundle)` for each candidate `seat`'s hand
        covers and its counterparty's public record allows."""
        hand = list(view.known[seat])
        coverable: list[tuple[int, int, Bundle]] = []
        for i, bundle in enumerate(received):
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            # An offer may ask for cards no seat is known to hold, so a
            # candidate can name a counterparty the public record says cannot
            # cover it. There is no position after that exchange to value.
            if not trading.has_room(view, counterparties[i], bundle):
                continue
            coverable.append((i, counterparties[i], bundle))
        return coverable

    def _reads_own_frame(self, game: Game, seat: int) -> bool:
        """Whether a trade at `game` is priced in `seat`'s own frame, one
        value-head read a position (`_after`), rather than over sampled worlds."""
        own = self.trade.gate_own_frame and self.trade.gate_frames == "one"
        return self.gate_plies == 0 and (own or to_move(game) == seat)

    def gains_across(
        self, seat: int, asks: Sequence[tuple[Game, Sequence[Bundle], Sequence[int]]]
    ) -> list[list[float]]:
        """`gains_many` at several positions in one forward: for each
        `(game, received, counterparties)`, `seat`'s gain from each candidate
        as if seated at that game, `-1.0` for one it cannot cover. The same
        rows `gains_many` reads, so the same numbers; a position it would
        price over sampled worlds is priced on its own. Leaves the bot
        seated where it was."""
        out: list[list[float]] = [[-1.0] * len(received) for _, received, _ in asks]
        if not self.trade.trades:
            return out
        rows: list[tuple[Game, int]] = []
        slots: list[tuple[int, int, dict[int, int]]] = []   # (ask, before row, {candidate: row})
        seated, memo = self._seated, self._memo
        try:
            for k, (game, received, counterparties) in enumerate(asks):
                if not received:
                    continue
                self.seat_at(game)
                view = game.state(seat)
                if not self._reads_own_frame(game, seat):
                    out[k] = self.gains_many(view, received, counterparties)
                    continue
                coverable = self._coverable(seat, view, received, counterparties)
                if not coverable:
                    continue
                hand = list(view.known[seat])
                before, afters = len(rows), {}
                rows.append((game, seat))
                for i, them, bundle in coverable:
                    afters[i] = len(rows)
                    rows.append((self._after(seat, them, hand, bundle, view), seat))
                slots.append((k, before, afters))
            values = self.policy.value_rows(rows) if rows else []
        finally:
            self._seated, self._memo = seated, memo
        own = (lambda v: v) if self.own_calibration is None else (
            lambda v: calibrate_own(v, seat, self.own_calibration))
        for k, before, afters in slots:
            base = own(values[before])[seat]
            for i, row in afters.items():
                out[k][i] = float(own(values[row])[seat] - base)
        return out

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
        """Each world's value vector after `mover`'s best play from it, `plies`
        deep, read from the frame of the seat to move there -- the frame a
        value head is trained in -- or from `seat`'s under `gate_own_frame`;
        where another seat than `seat` is to move, averaged over the frames
        `gate_frames` names (`_frames`). Values are in board-seat order. Worlds advance in
        lockstep, one batched `act_rows` per ply, and must be belief-sampled
        copies, never the live table's own state."""
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
            frames = {i: self._frames(seat, to_move(worlds[i])) for i in pending}
            values = self.policy.value_rows(
                [(worlds[i], f) for i in pending for f in frames[i]]
            )
            at = 0
            for i in pending:
                n = len(frames[i])
                rows = values[at:at + n]
                at += n
                out[i] = tuple(sum(v[k] for v in rows) / n for k in range(self.players))
        return out  # type: ignore[return-value]

    def _frames(self, seat: int, to_act: int) -> tuple[int, ...]:
        """The frames a position with `to_act` to move is read from for
        `seat`: its own alone when it is to move; otherwise `gate_frames`'
        choice -- the mover's, or `seat`'s under `gate_own_frame` (`one`),
        both (`pair`), or every seat's (`all`)."""
        if to_act == seat:
            return (seat,)
        frames = self.trade.gate_frames
        if frames == "all":
            return tuple(range(self.players))
        if frames == "pair":
            return (seat, to_act)
        return (seat,) if self.trade.gate_own_frame else (to_act,)

    def _after(
        self, seat: int, them: int, hand: Sequence[int], bundle: Bundle, view: View
    ) -> Game:
        """The position `bundle` leaves, read from `seat`'s own frame, over a
        copied state *and* ledger so nothing mutates the live table. `them`'s
        known row is raised to cover what the bundle says they give before being
        moved by it, so the true hand is never read and the subtraction cannot
        go negative. The ledger must be copied too: `View` reads `them`'s known
        cards from it, not from `state.hands`; the copy is `seat`'s own view,
        the only one this position is read in."""
        game = self._seated
        assert game is not None  # callers check this first
        state = copy_state(view.state)
        state.hands[seat] = [n + d for n, d in zip(hand, bundle)]

        certified = View(
            view.state, view.ledger, seat,
            certify=[(them, [max(0, n) for n in bundle])],
        )
        their_known = [k - d for k, d in zip(certified.known[them], bundle)]

        # `them`'s pile is its public size moved by the bundle, whatever the
        # live state holds: a concrete hand there is the truth, which this
        # seat may not read. The known row would shrink `them` by every card
        # this seat cannot name.
        state.hands[them] = HiddenHand(view.sizes[them] - sum(bundle))

        ledger = view.ledger.knowledge(seat)
        row = ledger.seats[them]
        row.known = their_known
        row.unknown = certified.unknown[them]
        # The bundle moves `them`'s upper bounds as it moves the cards.
        row.most = [max(0, m - d) for m, d in zip(row.most, bundle)]
        row.settle()
        ledger.seats[seat].known = list(state.hands[seat])
        ledger.seats[seat].unknown = 0
        ledger.seats[seat].most = list(state.hands[seat])

        after = copy.copy(game)
        after.set_state(state)
        after.ledger = ledger
        return after


def _shown(world: Game, seat: int) -> None:
    """Make `seat`'s hand in `world` fully known on its ledger: every card
    certified, none unknown. A world sampled from `seat`'s belief holds its
    true hand, so read from another seat's frame the encoding then carries
    all `seat` knows -- its own cards exactly -- and guesses only the rest.
    A seat whose hand the public record tracks exactly reads the same way."""
    ledger = world.ledger.seats[seat]
    hand = world.state(seat).state.hands[seat]
    ledger.known = list(hand)
    ledger.most = list(hand)
    ledger.unknown = 0


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in batched policy/value forwards."""

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
        scored = self.policy.score_rows(rows)
        if len(scored) != len(rows):
            raise ValueError(f"policy answered {len(scored)} of {len(rows)} MCTS leaves")

        # Keep the leaf mover's prior (and value) in its own frame. The value
        # head's other components are evaluated separately in each observer's
        # frame so a root's private cards are not erased on its own path, and
        # are not exposed to another mover's prior or value.
        observer_rows = []
        observer_slots = []
        values_by_observer = []
        for i, leaf in enumerate(leaves):
            prior, value = scored[i]
            value = tuple(float(v) for v in value)
            if not 0 <= leaf.seat < len(value):
                raise ValueError("leaf seat is outside its value vector")
            row = [None] * len(value)
            row[leaf.seat] = value
            values_by_observer.append(row)
            for observer in range(len(value)):
                if observer == leaf.seat:
                    continue
                observer_rows.append((leaf.game, observer))
                observer_slots.append((i, observer))

        if observer_rows:
            observer_values = self.policy.value_rows(observer_rows)
            if len(observer_values) != len(observer_rows):
                raise ValueError(
                    f"policy answered {len(observer_values)} of {len(observer_rows)} observer values"
                )
            for (i, observer), vector in zip(observer_slots, observer_values):
                values_by_observer[i][observer] = tuple(float(v) for v in vector)

        answers = []
        for i, leaf in enumerate(leaves):
            rows_for_leaf = values_by_observer[i]
            if any(row is None for row in rows_for_leaf):
                raise ValueError("policy did not return every observer value")
            prior, value = scored[i]
            answers.append(Evaluation(
                prior=prior,
                value=tuple(float(v) for v in value),
                observer_values=tuple(rows_for_leaf),
            ))
        return answers

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
        tip = self.gate.robber_tip
        if not tip:
            return super().choose(game)
        root, options, visits = self.run(game)
        if not options:
            raise ValueError("no legal action to choose from")
        picked = int(visits.argmax())
        if options[0].type is ActionType.MOVE_ROBBER:
            # The root's own prior and value: no forward beyond the search's.
            picked = tip_robber(options, root.prior, root.value, root.mover, picked, tip)
        return options[picked]

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

    def gains_across(
        self, seat: int, asks: Sequence[tuple[Game, Sequence[Bundle], Sequence[int]]]
    ) -> list[list[float]]:
        return self.gate.gains_across(seat, asks)

    def estimate_many(self, view: View, candidates: Sequence[tuple[int, Bundle]]) -> list[float]:
        return self.gate.estimate_many(view, candidates)


def tip_robber(
    options: Sequence[Action], prior: Sequence[float], value: Sequence[float],
    seat: int, picked: int, tip: float,
) -> int:
    """`picked`, or a robber move onto the leader where the policy rates it
    nearly as highly: the index of the option to play.

    The leader is the other seat `value` gives the highest win chance. A
    `picked` that already robs it stands. Otherwise the leader's best option
    by `prior` is played in its place when its prior is at least `tip` times
    `picked`'s; with none that close, `picked` stands. `tip` is in (0, 1]: `1`
    moves only an exact tie, smaller reaches further down the policy's own
    ranking. Board-seat order for `value`; `options` are robber moves, whose
    `b` names the seat robbed."""
    leader = max((s for s in range(len(value)) if s != seat), key=lambda s: value[s])
    if options[picked].b == leader:
        return picked
    robs = [i for i, option in enumerate(options) if option.b == leader]
    if not robs:
        return picked
    best = max(robs, key=lambda i: prior[i])
    return best if prior[best] >= tip * prior[picked] else picked


def bot_for(
    checkpoint: Checkpoint, *, rng: random.Random | None = None,
    trade: TradeParams | None = None,
    own_calibration: tuple[float, float] | None = None,
) -> NetworkBot:
    """`checkpoint`, playing one position at a time.

    The gate is built from what the checkpoint declares about itself --
    clearing floor, continuation budget, card caps, responder price and
    fragment plan -- so a file that asks for the fragmented policy gets it and
    one that asks for nothing bargains the way every checkpoint always has.
    `trade` replaces what the checkpoint declares -- that is how a no-trade
    arm is built -- and `rng` seeds a `gate_plies > 0` checkpoint's sampled
    worlds. `own_calibration` reads the seat's own chance through
    `hexset.mcts.calibrate_own` wherever the gate prices a trade."""
    trade = gate_config_of(checkpoint) if trade is None else trade
    return NetworkBot(
        policy=checkpoint.policy,
        players=checkpoint.players,
        trade=trade,
        rng=random.Random() if rng is None else rng,
        own_calibration=own_calibration,
        robber_tip=robber_tip_of(checkpoint),
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
    own_calibration: tuple[float, float] | None = None,
    decisive: float | None = None,
) -> GatedSearch:
    """`checkpoint` as a batched PUCT search, trading through its own value-head
    gate. `rng` seeds both the search and, derived from it, the gate's sampling
    stream, so one seed reproduces the entrant.

    `k` is how many determinizations of the seat's own belief each decision is
    searched in (`hexset.mcts.Search.worlds`); the tree is never rooted on the
    true state from here. `decisive` plays a root whose prior on one move
    reaches it without searching (`hexset.mcts.Search`)."""
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            pad_to=inference_batch,
        ),
        bot_for(checkpoint, rng=gate_rng, trade=trade, own_calibration=own_calibration),
        simulations=simulations,
        wave=wave,
        k=k,
        rng=rng,
        own_calibration=own_calibration,
        decisive=decisive,
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
    def own_trader(entrant, checkpoint, path: str) -> str | None:
        return None if entrant.trader is not None else declared_trader(checkpoint, path)

    def spawn_network(entrant, board: Board, rng):
        path = _checkpoint_path(entrant.weights, "network")
        checkpoint = loader(path, board.topology)
        trader = own_trader(entrant, checkpoint, path)
        bot = bot_for(checkpoint, rng=rng, trade=entrant.trade)
        return traded(bot, trader, board, rng, veto=trader_veto_of(checkpoint),
                      veto_picks=trader_veto_pick_of(checkpoint))

    def spawn_mcts(entrant, board: Board, rng):
        path = _checkpoint_path(entrant.weights, "mcts")
        checkpoint = loader(path, board.topology)
        trader = own_trader(entrant, checkpoint, path)
        bot = searcher_for(
            checkpoint,
            simulations=entrant.simulations, wave=entrant.wave, k=entrant.k,
            rng=rng, trade=entrant.trade, decisive=entrant.option("decisive"),
        )
        return traded(bot, trader, board, rng, veto=trader_veto_of(checkpoint),
                      veto_picks=trader_veto_pick_of(checkpoint))

    register_entrant_kind("network", spawn_network)
    register_entrant_kind("mcts", spawn_mcts)


def declared_trader(checkpoint: object, source: str) -> str | None:
    """The trader `checkpoint` declares (`modelmeta.trader_of`), resolved
    now: a name this process cannot build is refused here, naming `source`
    and the metadata key, rather than at the first trade it is asked for.
    `None` where the checkpoint trades through its own gate."""
    trader = trader_of(checkpoint)
    if trader is None:
        return None
    from hexset.arena import entrant_from_name

    try:
        entrant_from_name(trader)
    except ValueError as error:
        raise ValueError(
            f"{source}: metadata key `trader` names {trader!r}, which this process "
            f"cannot build ({error}). Load the module that registers it first "
            f"(`--runtime <module>`, or `hexset.arena.load_runtime` in code), or "
            f"remove the key to trade through the checkpoint's own gate"
        ) from None
    return trader
