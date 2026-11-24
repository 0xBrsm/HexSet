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

import numpy as np

from hexset.actions import Action, ActionType, legal_actions
from hexset.clients.policy import Checkpoint, Policy
from hexset.game import Game, is_over, to_move
from hexset.mcts import Evaluation, Search, calibrate_own, unbarred
from hexset.native.playout import NativeCopy
from hexset.rollout import TIE, PairedRollouts, RolloutPool
from hexset.actions import options_for
from hexset.clients._modelmeta import (
    closer_of, gate_config_of, hold_of, robber_hits_of, robber_tip_of, trader_of, trader_veto_of,
)
from hexset.robber import occupants
from hexset.state import HiddenHand, copy_state
from hexset import trading
from hexset.trading import UNLIMITED, TradeParams, TradeProtocol
from hexset.view import HoldReading, View

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
    "hit_robber",
    "prefer_off_own",
    "register_entrants",
    "declared_trader",
    "declared_closer",
]


def _check_players(game: Game, players: int) -> None:
    """A network counts every dealt seat or, built for fewer, the live seats
    alone (`hexset.encoding.seat_frame`): a two-seat net plays a table dealt
    four with two retired. Any other count is refused."""
    # true state: `num_players` is a fixed, public board property.
    num_players = game.state(0, hidden=False).num_players
    live = num_players - len(game.locked)
    if players not in (num_players, live):
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {num_players}" + (f" ({live} live)" if live != num_players else "")
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
    #: the offer budget and the fragment plan.
    trade: TradeParams = UNLIMITED
    # Seeds the trade protocol's stream and the sampling stream, each off its
    # state without drawing from it.
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
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
    #: Whether this seat's robber moves always hit another seat where one can
    #: be hit (`hit_robber`); `False` plays them as the policy picks.
    robber_hits: bool = False
    #: `0.0` plays the policy's top move. Above it, the seat samples its move
    #: from the policy's probabilities raised to `1 / temperature`: a whole
    #: option where the policy lists them (`option_rows`, its remaining moves
    #: played from the policy's own plan), else a move from `score_rows`. The
    #: draws come from a stream of the seat's own, seeded off `rng`'s state.
    temperature: float = 0.0
    #: The hold model the checkpoint declares for the other seats' held
    #: development cards (`hold_shape`, `hold_scale`), which `bot_for` sets,
    #: or `None`; part of the key the gate's last answer is kept under
    #: (`View.belief_key`).
    hold: HoldReading | None = None
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
        # the protocol never disturbs `rng`.
        seed = int.from_bytes(
            hashlib.sha256(repr(self.rng.getstate()).encode()).digest()[:8], "big"
        )
        self._protocol = trading.install(self, TradeProtocol(self, self.trade, seed=seed))
        if self.temperature < 0:
            raise ValueError(f"temperature must be at least 0, got {self.temperature}")
        # The sampling stream: off `rng`'s state like the protocol's seed, salted, so a draw
        # never disturbs the gate's stream and the gate's never moves a sampled move.
        self._draws = random.Random(
            hashlib.sha256((repr(self.rng.getstate()) + ":temperature").encode()).digest()
        )

    # Every bargaining limit -- `trade_params`, `max_offers`, `trade_floor`,
    # the offer budget -- reads through `trade`
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
        narrowed = hit_robber(game, options) if self.robber_hits else options
        if len(narrowed) < len(options):
            # A policy that keeps plans answers `act_rows` from the game's own
            # options; the narrowed ones are scored instead (`_scored_pick`).
            ((prior, value),) = self.policy.score_rows([(game, seat, narrowed)])
            return self._scored_pick(game, narrowed, prior, value, seat)
        if self.temperature:
            return self._sampled_many([(self, game, seat, options)])[0]
        if self.robber_tip and options[0].type is ActionType.MOVE_ROBBER:
            # The same forward `act_rows` makes, read for its prior and value.
            ((prior, value),) = self.policy.score_rows([(game, seat, options)])
            picked = max(range(len(options)), key=lambda i: prior[i])
            return options[tip_robber(options, prior, value, seat, picked, self.robber_tip)]
        move = self.policy.act_rows([(game, seat, options)])[0]
        if self.robber_hits and options[0].type is ActionType.MOVE_ROBBER:
            ((prior, _),) = self.policy.score_rows([(game, seat, options)])
            return options[_off_own_over(game, options, prior, options.index(move))]
        return move

    @classmethod
    def choose_many(cls, pairs: Sequence[tuple["NetworkBot", Game]]) -> list[Action]:
        """`choose` for several bots at once (`hexset.bots.ChoosesMany`): the
        bots sharing one policy are asked in one `act_rows` call (a robber
        move a seat tips, in one `score_rows` call), each seated at its own
        game first, so a batched table makes one forward per policy a tick
        where it made one per seat. Each answer is the one `choose` gives:
        a policy's `act_rows` answers every row as it would alone."""
        out: list[Action | None] = [None] * len(pairs)
        groups: dict[tuple[int, str], list[tuple[int, NetworkBot, Game, int, tuple[Action, ...]]]] = {}
        for i, (bot, game) in enumerate(pairs):
            bot.seat_at(game)
            seat = to_move(game)
            options = tuple(options_for(game))
            narrowed = hit_robber(game, options) if bot.robber_hits else options
            robber = options[0].type is ActionType.MOVE_ROBBER
            how = ("narrowed" if len(narrowed) < len(options) else "sampled" if bot.temperature else
                   "tipped" if bot.robber_tip and robber else "hits" if bot.robber_hits and robber else "top")
            options = narrowed
            groups.setdefault((id(bot.policy), how), []).append((i, bot, game, seat, options))
        for (_, how), members in groups.items():
            policy = members[0][1].policy
            rows = [(game, seat, options) for _, _, game, seat, options in members]
            if how == "sampled":
                answers = cls._sampled_many([(bot, game, seat, options) for _, bot, game, seat, options in members])
                for (i, *_), action in zip(members, answers):
                    out[i] = action
            elif how == "narrowed":
                for (i, bot, game, seat, options), (prior, value) in zip(members, policy.score_rows(rows)):
                    out[i] = bot._scored_pick(game, options, prior, value, seat)
            elif how == "hits":
                # `act_rows`' moves, each moved off the mover's hex only on an
                # exact tie at the top of the prior (`_off_own_over`).
                for (i, _, game, _, options), action, (prior, _) in zip(
                        members, policy.act_rows(rows), policy.score_rows(rows)):
                    out[i] = options[_off_own_over(game, options, prior, options.index(action))]
            elif how == "tipped":
                for (i, bot, _, seat, options), (prior, value) in zip(members, policy.score_rows(rows)):
                    picked = max(range(len(options)), key=lambda k: prior[k])
                    out[i] = options[tip_robber(options, prior, value, seat, picked, bot.robber_tip)]
            else:
                for (i, *_), action in zip(members, policy.act_rows(rows)):
                    out[i] = action
        return out  # type: ignore[return-value]

    def _scored_pick(self, game: Game, options: Sequence[Action], prior, value, seat: int) -> Action:
        """The move this seat plays from robber `options` it scored itself
        (`score_rows`): the prior's best, an exact tie at the top going to a
        hex the seat is not on (`prefer_off_own`), or one drawn from it above
        `temperature` 0, then tipped to the leader (`robber_tip`)."""
        picked = self._draw(prior) if self.temperature else _robber_pick(game, options, prior)
        if self.robber_tip:
            picked = tip_robber(options, prior, value, seat, picked, self.robber_tip)
        return options[picked]

    def _draw(self, probabilities) -> int:
        """An index drawn from `probabilities` raised to `1 / temperature`, off this seat's stream."""
        weights = [float(p) ** (1.0 / self.temperature) if p > 0 else 0.0 for p in probabilities]
        total = sum(weights)
        if not total > 0:
            weights, total = [1.0] * len(weights), float(len(weights))
        point, running = self._draws.random() * total, 0.0
        for index, weight in enumerate(weights):
            running += weight
            if point < running:
                return index
        return max(i for i, w in enumerate(weights) if w > 0)

    @staticmethod
    def _sampled_many(rows: Sequence[tuple["NetworkBot", Game, int, tuple[Action, ...]]]) -> list[Action]:
        """Sampled moves for rows of bots sharing one policy, one forward for the rows that draw. A
        policy that lists whole options (`option_rows`) draws one where a decision opens and plays
        its plan's next move inside it (`hexset.decisions.Plans`, the policy's `plans`); any other
        draws from `score_rows`. A position with one legal move draws nothing."""
        policy = rows[0][0].policy
        out: list[Action | None] = [None] * len(rows)
        if getattr(policy, "has_options", False):
            plans = policy.plans
            ask = [i for i, (_, game, seat, options) in enumerate(rows)
                   if len(options) > 1 and not plans.planned(game, seat, options)]
            listed = dict(zip(ask, policy.option_rows([(rows[i][1], rows[i][2]) for i in ask]) if ask else []))
            drawn: dict[int, object] = {}
            for i in ask:
                options, probabilities = listed[i]
                drawn[i] = options[rows[i][0]._draw(probabilities)]
            for i, (_, game, seat, options) in enumerate(rows):
                pick = (lambda g, s, i=i: drawn[i])
                out[i] = plans.act(game, seat, options, pick, lambda g, s, o, i=i: drawn[i])
            return out  # type: ignore[return-value]
        ask = [i for i, (_, _, _, options) in enumerate(rows) if len(options) > 1]
        scored = policy.score_rows([rows[i][1:] for i in ask]) if ask else []
        for i, (bot, _, _, options) in enumerate(rows):
            out[i] = options[0] if len(options) == 1 else None
        for i, (prior, _) in zip(ask, scored):
            out[i] = rows[i][3][rows[i][0]._draw(prior)]
        return out  # type: ignore[return-value]

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
            seat, view.perspective, tuple(view.sizes), view.belief_key(self.hold),
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
        keyed by candidate index, all read from `seat`'s own frame whoever is
        to move: the post-trade positions (`_after`) over `seat`'s own view,
        never the true hands."""
        game = self._seated
        assert game is not None  # callers check this first
        hand = list(view.known[seat])
        coverable = self._coverable(seat, view, received, counterparties)
        if not coverable:
            return (), {}
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

    def gains_across(
        self, seat: int, asks: Sequence[tuple[Game, Sequence[Bundle], Sequence[int]]]
    ) -> list[list[float]]:
        """`gains_many` at several positions in one forward: for each
        `(game, received, counterparties)`, `seat`'s gain from each candidate
        as if seated at that game, `-1.0` for one it cannot cover. The same
        rows `gains_many` reads, so the same numbers. Leaves the bot seated
        where it was."""
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


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in batched policy/value forwards."""

    policy: Policy
    pad_to: int | None = None

    def __post_init__(self) -> None:
        if self.pad_to is not None and self.pad_to < 1:
            raise ValueError("pad_to must be positive")

    @property
    def native_contract(self) -> str | None:
        """The contract the policy reads native positions in, or `None`: what
        `hexset.mcts.Search` holds its tree natively for."""
        return getattr(self.policy, "native_contract", None)

    @property
    def lists_options(self) -> bool:
        """Whether the policy lists a decision's whole options
        (`option_rows`), which a whole-decision search reads."""
        return callable(getattr(self.policy, "option_rows", None)) and bool(getattr(self.policy, "has_options", False))

    @property
    def space(self):
        """The policy's action space, which `hexset.native.playout.copyable`
        checks a native tree against."""
        return getattr(self.policy, "space", None)

    def evaluate(self, leaves):
        if not leaves:
            return []
        # A whole leaf (`hexset.mcts.Leaf.whole`) is answered with its whole
        # options and their probabilities (`option_rows`) and its seat's value;
        # every other leaf with a prior over its moves (`score_rows`).
        plain = [leaf for leaf in leaves if not leaf.whole]
        whole = [i for i, leaf in enumerate(leaves) if leaf.whole]
        scored: list = [None] * len(leaves)
        if plain:
            padded = list(plain)
            if self.pad_to is not None and len(plain) < self.pad_to:
                padded.extend([plain[-1]] * (self.pad_to - len(plain)))
            rows = [(leaf.game, leaf.seat, leaf.options) for leaf in padded]
            answered = self.policy.score_rows(rows)
            if len(answered) != len(rows):
                raise ValueError(f"policy answered {len(answered)} of {len(rows)} MCTS leaves")
            at = iter(answered)
            for i, leaf in enumerate(leaves):
                if not leaf.whole:
                    scored[i] = next(at)
        listed_at = {}
        if whole:
            asks = [(leaves[i].game, leaves[i].seat) for i in whole]
            for i, (listed, probs), value in zip(whole, self.policy.option_rows(asks), self.policy.value_rows(asks)):
                listed_at[i] = tuple(listed)
                scored[i] = (np.asarray(probs, dtype=np.float64), value)

        # Keep the leaf mover's prior (and value) in its own frame. The value
        # head's other components are evaluated separately in each observer's
        # frame so a root's private cards are not erased on its own path, and
        # are not exposed to another mover's prior or value. A retired seat
        # (`game.locked`) never moves, so the search never reads its row: it
        # takes the mover's row and is not encoded, since a seat frame needs
        # its perspective among the live seats.
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
                if observer in leaf.game.locked:
                    row[observer] = value
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
                options=listed_at.get(i),
            ))
        return answers

    def terminal(self, game: Game) -> Sequence[float]:
        """`hexset.mcts.Evaluator.terminal`: the one-hot winner in board-seat
        order, the scale a wave's other leaves are scored on. Raises if `game`
        has not finished."""
        if not is_over(game):
            raise ValueError("terminal() called on a game that has not finished")
        if type(game) is NativeCopy:
            players = game.dealt
        else:
            # true state: `num_players` is a fixed, public board property.
            players = game.state(0, hidden=False).num_players
        winner = game.won_by
        return tuple(1.0 if seat == winner else 0.0 for seat in range(players))


class _Gated:
    """The checkpoint's own trade gate behind a seat that decides its moves
    some other way than one forward: `choose` binds the gate to the live game
    first; trade acceptance and counterparty estimates delegate to the gate's
    `NetworkBot`.

    This, not the `NetworkBot` behind it, is what a table seats, so it carries
    the checkpoint's bargaining parameters and installs the same protocol over
    its own delegated valuation. A searching entrant and a one-forward entrant
    from the same checkpoint therefore bargain identically."""

    def _seat_gate(self, gate: NetworkBot) -> None:
        self.gate = gate
        self.trade = gate.trade
        self._protocol = trading.install(
            self, TradeProtocol(self, gate.trade, seed=gate._protocol.seed)
        )

    def seat_at(self, game: Game) -> None:
        """`NetworkBot.seat_at` on the gate behind this seat."""
        self.gate.seat_at(game)

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


class GatedSearch(_Gated, trading.DeclaredTrade, Search):
    """Policy/value-guided MCTS with the checkpoint's own trade gate
    (`_Gated`)."""

    def __init__(self, evaluator, gate: NetworkBot, **kwargs) -> None:
        super().__init__(evaluator, **kwargs)
        self._seat_gate(gate)

    def choose(self, game: Game) -> Action:
        self.gate.seat_at(game)
        return Search.choose(self, game)

    def _pick(self, game: Game, decision) -> int:
        picked = decision.chosen
        first = decision.options[0]
        robber = isinstance(first, Action) and first.type is ActionType.MOVE_ROBBER
        if self.gate.robber_hits and robber:
            # The tree's root holds every legal move: the most-visited one
            # that hits, where any does; level on visits at the top with
            # another that hits, one on a hex the mover is not on.
            hits = set(hit_robber(game, decision.options))
            kept = [i for i, o in enumerate(decision.options) if o in hits]
            if picked not in kept:
                picked = max(kept, key=lambda i: (decision.visits[i], -i))
            most = max(decision.visits[i] for i in kept)
            if decision.visits[picked] == most:
                level = [picked] + [i for i in kept if i != picked and decision.visits[i] == most]
                picked = prefer_off_own(game, decision.options, level)[0]
        tip = self.gate.robber_tip
        if tip and robber:
            # The root's own prior and value: no forward beyond the search's.
            picked = tip_robber(
                decision.options, decision.prior, decision.value, decision.seat, picked, tip
            )
        return unbarred(game, decision, picked)


class GatedRollouts(_Gated, trading.DeclaredTrade, PairedRollouts):
    """Paired rollouts of the checkpoint's own policy (`hexset.rollout`) with
    its own trade gate (`_Gated`)."""

    def __init__(self, policy: Policy, gate: NetworkBot, *, opponents=None, **kwargs) -> None:
        super().__init__(policy, **kwargs)
        self._seat_gate(gate)
        #: The model of the other seats its playouts play them by (`rollouts_for`'s
        #: `opponents`), or `None`: its own policy.
        self.opponents = opponents

    def seen(self, game: Game, actor: int, action: Action, viewer: int) -> None:
        """Each move the table makes, for `opponents` to read (`hexset.bots.watch`)."""
        if self.opponents is not None:
            self.opponents.seen(game, actor, action, viewer)

    def root_options(self, game: Game) -> tuple[Action, ...]:
        """Every legal move, less the robber moves `hit_robber` rules out
        where the checkpoint declares `robber_hits`."""
        options = super().root_options(game)
        return hit_robber(game, options) if self.gate.robber_hits else options

    def _own(self, game: Game, seat: int, options: tuple[Action, ...]) -> Action:
        """The policy's own move, as `PairedRollouts._own`; at a robber root
        `root_options` narrowed, the prior's best of the narrowed moves (a
        policy that keeps plans answers `act_rows` from the game's own
        options). Under `robber_hits`, an exact tie at the top of the prior
        goes to a hex the mover is not on (`_off_own_over`)."""
        if not (self.gate.robber_hits and options[0].type is ActionType.MOVE_ROBBER):
            return super()._own(game, seat, options)
        narrowed = len(options) < len(legal_actions(game))
        move = None if narrowed else super()._own(game, seat, options)
        ((prior, _),) = self.policy.score_rows([(game, seat, options)])
        if move is None:
            return options[_robber_pick(game, options, prior)]
        return options[_off_own_over(game, options, prior, options.index(move))]

    def prefer(self, game: Game, options: tuple, tied: Sequence[int]) -> tuple[int, ...]:
        """Under `robber_hits`, of tied robber moves, those onto a hex the
        mover is not on, where the tie holds both kinds (`prefer_off_own`);
        otherwise `tied` whole (`PairedRollouts.prefer`)."""
        if self.gate.robber_hits:
            return prefer_off_own(game, options, tied)
        return tuple(tied)

    def choose(self, game: Game) -> Action:
        self.gate.seat_at(game)
        if self.opponents is not None:
            self.policy.seat = to_move(game)      # this seat's rows stay its own in every playout
        decision = self.decide(game)
        tip = self.gate.robber_tip
        if tip and not decision.whole and decision.options[0].type is ActionType.MOVE_ROBBER:
            # The root's own prior and value: no forward beyond the decision's.
            picked = tip_robber(
                decision.options, decision.prior, decision.value, to_move(game), decision.chosen, tip
            )
            if picked != decision.chosen:
                self._drop(game)
                return decision.options[picked]
        return decision.move


def hit_robber(game: Game, options: Sequence[Action]) -> tuple[Action, ...]:
    """`options`, less every robber move onto a hex where no other seat has a
    building (an empty hex, the desert, or a hex the mover holds alone), when
    some robber move lands on a hex where another seat has one.

    A hex another seat shares with the mover stays. Where no legal hex holds
    another seat's building (a friendly robber shielding everyone, say),
    `options` stand whole, as they do for any decision that is not a robber
    move, so the rule only ever takes moves away. Whether a seat holds cards
    does not matter: the robber on its hex blocks its production either way.
    Buildings are public, so this reads nothing the mover cannot see.

    Among the moves that remain, a tie goes to a hex the mover is not on
    (`prefer_off_own`)."""
    options = tuple(options)
    if not options or options[0].type is not ActionType.MOVE_ROBBER:
        return options
    mover = to_move(game)
    state = game.state(mover).state  # only the board's buildings are read: public to every seat
    on = {h: occupants(state, h) for h in {o.a for o in options}}
    hits = tuple(o for o in options if any(p != mover for p in on[o.a]))
    return hits or options


def prefer_off_own(game: Game, options: Sequence, tied: Sequence[int]) -> tuple[int, ...]:
    """Of `tied`, indices into `options`, those that are robber moves onto a
    hex with none of the mover's buildings, where `tied` holds such a move
    and also one onto a hex the mover is on; otherwise `tied` whole. Order is
    kept, so the first index left is the one the caller's own tie-break
    reaches first. Options that are not robber moves are never set aside."""
    tied = tuple(tied)
    if len(tied) < 2:
        return tied
    mover = to_move(game)
    state = game.state(mover).state  # only the board's buildings are read: public to every seat
    off = tuple(
        i for i in tied
        if not (isinstance(options[i], Action) and options[i].type is ActionType.MOVE_ROBBER
                and mover in occupants(state, options[i].a))
    )
    return off if off and len(off) < len(tied) else tied


def _off_own_over(game: Game, options: Sequence[Action], prior, picked: int) -> int:
    """`picked`, an index into robber `options`; or, where it lies on a hex
    the mover is on and ties exactly at the top of `prior` with a move onto
    a hex the mover is not on, the first such move (`prefer_off_own`)."""
    top = max(float(p) for p in prior)
    tied = [i for i in range(len(options)) if float(prior[i]) == top]
    if picked not in tied:
        return picked
    kept = prefer_off_own(game, options, tied)
    return picked if picked in kept else kept[0]


def _robber_pick(game: Game, options: Sequence[Action], prior) -> int:
    """The prior's best of robber `options`, the earlier on a tie, then
    `_off_own_over`."""
    best = max(range(len(options)), key=lambda i: prior[i])
    return _off_own_over(game, options, prior, best)


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
    temperature: float = 0.0,
    hold: HoldReading | None = None,
) -> NetworkBot:
    """`checkpoint`, playing one position at a time; above `temperature` 0 it
    samples its moves (`NetworkBot.temperature`).

    The gate is built from what the checkpoint declares about itself --
    clearing floor, card caps, responder price and fragment plan -- so a file
    that asks for the fragmented policy gets it and one that asks for
    nothing bargains the way every checkpoint always has.
    `trade` replaces what the checkpoint declares -- that is how a no-trade
    arm is built -- and `rng` seeds the gate's streams (`NetworkBot.rng`).
    `own_calibration` reads the seat's own chance through
    `hexset.mcts.calibrate_own` wherever the gate prices a trade. `hold`
    replaces the hold model the checkpoint declares (`NetworkBot.hold`)."""
    trade = gate_config_of(checkpoint) if trade is None else trade
    hold = hold_of(checkpoint) if hold is None else hold
    return NetworkBot(
        policy=checkpoint.policy,
        players=checkpoint.players,
        trade=trade,
        rng=random.Random() if rng is None else rng,
        own_calibration=own_calibration,
        robber_tip=robber_tip_of(checkpoint),
        robber_hits=robber_hits_of(checkpoint),
        temperature=temperature,
        hold=hold,
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
    hold: HoldReading | None = None,
    top: int | None = None,
    stages: str | None = None,
    root: str | None = None,
    whole: bool = False,
) -> GatedSearch:
    """`checkpoint` as a batched PUCT search, trading through its own value-head
    gate. `rng` seeds both the search and, derived from it, the gate's sampling
    stream, so one seed reproduces the entrant.

    `k` is how many determinizations of the seat's own belief each decision is
    searched in (`hexset.mcts.Search.worlds`); the tree is never rooted on the
    true state from here. `decisive` plays a root whose prior on one move
    reaches it without searching (`hexset.mcts.Search`); `top` searches the
    root by sequential halving over that many moves and `stages="setup"`
    searches only the setup placements, `root="gumbel"` makes the halving
    root Gumbel MuZero's, and `whole` searches decisions of several moves as
    whole options (all `hexset.mcts.Search`). The other
    seats' held development cards are dealt in those worlds by the hold
    model the checkpoint declares (`hold_shape`,
    `hold_scale`; `hexset.holding.HoldModel`), else alike; `hold` replaces
    it. A policy that
    declares no prior to score its leaves with (`has_prior` false) is refused
    here rather than at the first leaf."""
    if not getattr(checkpoint.policy, "has_prior", True):
        raise ValueError(f"{type(checkpoint.policy).__name__} declares no `prior`: it cannot be searched")
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    hold = hold_of(checkpoint) if hold is None else hold
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            pad_to=inference_batch,
        ),
        bot_for(checkpoint, rng=gate_rng, trade=trade, own_calibration=own_calibration, hold=hold),
        simulations=simulations,
        wave=wave,
        k=k,
        hold=hold,
        rng=rng,
        own_calibration=own_calibration,
        decisive=decisive,
        top=top,
        stages=stages,
        root=root,
        whole=whole,
    )


def rollouts_for(
    checkpoint: Checkpoint,
    *,
    worlds: int = 32,
    top: int = 3,
    decisive: float | None = None,
    horizon: int | None = None,
    rng: random.Random | None = None,
    trade: TradeParams | None = None,
    full_setup: bool = False,
    pool: RolloutPool | None = None,
    own_calibration: tuple[float, float] | None = None,
    opponents=None,
    det: bool = False,
    hold: HoldReading | None = None,
    margin: float = 0.0,
    stop: float = 0.0,
    moves: bool = False,
    tie: float = TIE,
) -> GatedRollouts:
    """`checkpoint` deciding by paired rollouts of its own policy
    (`hexset.rollout.PairedRollouts`), trading through its own value-head
    gate. `rng` seeds both the rollouts and, derived from it, the gate's
    sampling stream, so one seed reproduces the entrant. `pool` plays the
    worlds out in worker processes (`hexset.rollout.RolloutPool`).
    `own_calibration` reaches the gate alone (`bot_for`): the playouts read
    the value head as it is. `opponents`, a model of the other seats, plays
    every other seat of a playout in place of the checkpoint: its
    `playouts(policy)` is the policy the playouts play with (a
    `hexset.clients.conditioned.SeatRouted`), and its `seen(game, actor,
    action, viewer)` reads each move the table makes (`hexset.bots.watch`);
    `pool` must play the same model. `det` seeds each searched decision
    from what the seat can see (`PairedRollouts.det`). `margin`
    above 0 plays another candidate than the policy's own move only where it
    clears that many standard errors (`PairedRollouts.margin`); `stop` above
    0, under a margin, stops playing a candidate that trails that move by more
    than that many standard errors at a check (`PairedRollouts.stop`); `moves`
    searches every root as a single move (`PairedRollouts.moves`); `tie`, under
    a margin, decides a root whose candidates win within that chance of the
    policy's move by the point lead (`PairedRollouts.tie`). The other
    seats' held development cards are dealt in every world by the hold
    model the checkpoint declares, else alike; `hold` replaces it
    (`searcher_for`)."""
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    hold = hold_of(checkpoint) if hold is None else hold
    return GatedRollouts(
        checkpoint.policy if opponents is None else opponents.playouts(checkpoint.policy),
        bot_for(checkpoint, rng=gate_rng, trade=trade, own_calibration=own_calibration, hold=hold),
        worlds=worlds,
        top=top,
        decisive=decisive,
        full_setup=full_setup,
        horizon=horizon,
        rng=rng,
        pool=pool,
        opponents=opponents,
        det=det,
        hold=hold,
        margin=margin,
        stop=stop,
        moves=moves,
        tie=tie,
    )


def _checkpoint_path(weights: object, what: str) -> str:
    """An `Entrant.weights` read as a checkpoint path; a lineup that passes
    anything else is refused here rather than inside a loader."""
    if isinstance(weights, str):
        return weights
    raise ValueError(f"{what}'s weights is a checkpoint path")


def register_entrants(loader) -> None:
    """Register network, MCTS and paired-rollout arena factories for a `loader(path, topology)
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
        bot = bot_for(checkpoint, rng=rng, trade=entrant.trade,
                      temperature=float(entrant.option("temperature", 0.0) or 0.0))
        return traded(bot, trader, board, rng, veto=trader_veto_of(checkpoint), hold=hold_of(checkpoint))

    def spawn_mcts(entrant, board: Board, rng):
        path = _checkpoint_path(entrant.weights, "mcts")
        checkpoint = loader(path, board.topology)
        trader = own_trader(entrant, checkpoint, path)
        bot = searcher_for(
            checkpoint,
            simulations=entrant.simulations, wave=entrant.wave, k=entrant.k,
            rng=rng, trade=entrant.trade, decisive=entrant.option("decisive"),
            top=entrant.option("top"), stages=entrant.option("stages"), root=entrant.option("root"),
            whole=bool(entrant.option("whole", False)),
        )
        return traded(bot, trader, board, rng, veto=trader_veto_of(checkpoint), hold=hold_of(checkpoint))

    def spawn_rollout(entrant, board: Board, rng):
        path = _checkpoint_path(entrant.weights, "rollout")
        checkpoint = loader(path, board.topology)
        trader = own_trader(entrant, checkpoint, path)
        bot = rollouts_for(
            checkpoint,
            worlds=entrant.simulations, top=entrant.option("top"),
            decisive=entrant.option("decisive"), horizon=entrant.option("horizon"),
            full_setup=bool(entrant.option("full_setup", False)),
            rng=rng, trade=entrant.trade, det=bool(entrant.option("det", False)),
            margin=float(entrant.option("margin", 0.0)), stop=float(entrant.option("stop", 0.0)),
            moves=bool(entrant.option("moves", False)),
            tie=float(entrant.option("tie", TIE)),
        )
        return traded(bot, trader, board, rng, veto=trader_veto_of(checkpoint), hold=hold_of(checkpoint))

    register_entrant_kind("network", spawn_network)
    register_entrant_kind("mcts", spawn_mcts)
    register_entrant_kind("rollout", spawn_rollout)


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


def declared_closer(checkpoint: object, source: str) -> str | None:
    """The closer `checkpoint` declares (`modelmeta.closer_of`), resolved
    now, as `declared_trader` resolves a trader: a name this process cannot
    build is refused here, naming `source` and the metadata key. `None`
    where the checkpoint names none."""
    closer = closer_of(checkpoint)
    if closer is None:
        return None
    from hexset.arena import entrant_from_name

    try:
        entrant_from_name(closer)
    except ValueError as error:
        raise ValueError(
            f"{source}: metadata key `closer` names {closer!r}, which this process "
            f"cannot build ({error}). Load the module that registers it first "
            f"(`--runtime <module>`, or `hexset.arena.load_runtime` in code), or "
            f"remove the key to play every move as the checkpoint's own"
        ) from None
    return closer
