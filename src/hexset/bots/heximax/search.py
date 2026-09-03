# SPDX-License-Identifier: GPL-3.0-only
"""Max^n over the honest evaluation, within a leaf budget.

`Heximax`'s search core: `choose` (the one public entry point, plus the
setup/discard shortcuts it resolves directly), iterative deepening
(`_search`/`_estimate`/`_root_values`), the tree itself
(`_value`/`_after`/`_best_of`), roll expansion (`_over_dice`), and hidden-draw
expansion (`draw_children`: a steal weighted over the victim's belief, a
dev-card buy weighted over the unseen deck). It also carries the bot's whole
trading surface -- `valuation` and `accepts`, and the marginal-value
machinery they are built from -- which is two methods and no protocol at
all now that trading is one engine event rather than an action language
(`hexset.trading`). The offer adapter this file used to inherit from
`heximax/trade.py` (candidate bundles, `score_proposal`, `accept_rule`,
`rank_partners`, `propose_actions`) is gone with the protocol it adapted to.

Cost: leaf evaluations per move are capped by `max_nodes`
(`DEFAULT_MAX_NODES`, 600). The mirror table is
`agents/scripts/heximax_cost.py` -- three four-seat games an arm, board
seeds 0/1/2, every seat the same preset, `search2-offers3` at heximax's own
three-offer budget as the control, arms interleaved seed by seed. Two rules
for reading it, both learned the hard way in the structural pass:

* **On an idle box, and paired.** heximax's per-move cost inflates faster
  than `search2`'s under contention, so identical code reads 2.6x idle and
  3.1x at load 5 -- more drift than most changes are worth. Time the before
  and after in one process instead. So paired, the pass's exact steps took
  **2.687x -> 2.357x**: -12.3% of ms/move and -13.0% of function calls per
  game (10.47M -> 9.11M, which no load can move;
  `runs/eval/heximax/structural-cost-paired-vs-810dec7.json`).
* **Beside `ratio_phase_neutral`**, which re-weights the control by
  heximax's own phase mix. Under the offer protocol `search2` booked ~20x
  more `TRADE_RESPOND` decisions over the same games (2481 against 126 over
  nine) and those cheap decisions sat in the mirror table's denominator;
  phase-neutral heximax read **2.08x** against a raw 2.36x. The phase that
  caused the skew no longer exists -- there are no trade decisions at all --
  so the raw ratio and the phase-neutral one now measure the same games,
  and the re-read belongs to the one-event readout, not to this note.

Three behaviour-changing steps were registered and none landed: a
transposition table across the iterative-deepening passes hits 0.046% of
`_value` calls (the depth-1/depth-2 redundancy is leaf *count*, not repeated
positions); a vectorised evaluator would need `_value` rewritten
breadth-first to beat a Python loop that is 2.0% of runtime; and sampling
the ply-1 roll (`EXACT_ROLL_PLIES` 2 -> 1), worth -11.9%, cleared the
trading strength gate at 67.0% but missed the no-trade gate's pre-stated
floor by 0.6 of a game, so it was reverted rather than tuned. Whether the
rest is a cost problem or a denominator problem -- the trade gate too
strict, `relative` the wrong stance for `willing`, or the ceiling owed a
protocol allowance -- is a design question, not one a performance pass
answers by loosening a gate to hit a number.

`bot.choose()`'s own choices, and the leaves it spends reaching them, are
checked on every position by
`test_choices_are_byte_identical_to_the_recorded_census`. History -- the
optimization and structural passes, with their per-change breakdowns -- is
in `agents/reference/heximax.md`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from hexset.actions import Action, ActionType, apply, legal_actions, victim_of
from hexset.board.board import Board
from hexset.board.terrain import NUM_RESOURCES
from ..search2 import STANCES, options_for
from hexset.game import ROLL_ODDS, Game, Phase, imagine, is_over, roll_dice, to_move
from hexset.ledger import PublicLedger
from hexset.mcts import draws_hidden
from hexset.placement import best as best_opening
from hexset.state import GameState, copy_state
from hexset.trading import NO_VALUATION, Bundle

from hexset.view import View
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator, Weights

# Leaf evaluations a move may spend. Chosen so the default configuration costs
# no more than twice `search2` per move (the design's ceiling): at 600 the
# mean is 1.5x and the per-move tail, which the unbounded search takes to
# ~1500 leaves, is cut at the budget. Figures in the module docstring.
DEFAULT_MAX_NODES = 600

# A roll taken this many plies or fewer below the root is expanded over all
# eleven outcomes; deeper rolls are sampled once. At the default depth of two
# every roll in the tree is exact.
EXACT_ROLL_PLIES = 2

# The common scale every seat's published valuation is squashed onto
# (`Heximax.valuation`): the mean absolute one-card marginal of the shipped
# profile. Fixed per evaluator rather than rescaled per decision, because
# the clearing rule compares two seats' surpluses and they must be in the
# same unit (the PI's ratification of decision 1,
# `agents/reference/trading-design.md`).
#
# Computed once, over the trade-free census games -- `heximax-notrade`,
# seeds 100..104, every seat the same preset -- as the mean of
# `|Eval(hand + one r) - Eval(hand)|` under `TRADING_WEIGHTS` and the
# `relative` stance, over every resource at every position the mover
# reaches (9280 marginals). Trade-free deliberately: the games that fix the
# scale must not themselves depend on it, and `max_trades=0` makes them
# independent of this constant by construction.
# `test_marginal_scale_is_the_recorded_mean` recomputes it from those same
# games and pins it to 1e-9.
MARGINAL_SCALE = 0.10231140469178995


class _Exhausted(Exception):
    """The leaf budget ran out mid-search; the caller falls back."""


class _Forced:
    """A stand-in for the game's RNG that makes `robber.steal` take one card.

    `steal` draws `randrange(total)` and walks the hand in resource order, so
    returning the index of the first card of the wanted resource makes the
    draw deterministic. Nothing else on the steal path consults the RNG.
    """

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def randrange(self, _stop: int) -> int:
        return self.index


@dataclass
class Heximax:
    """Max^n over the honest evaluation, within a leaf budget.

    `depth` counts decisions, `width` beams the branching, and `max_nodes`
    caps the leaf evaluations one `choose` may spend: the search deepens one
    ply at a time while the next ply's estimated cost fits what is left, and
    a ply that overruns is abandoned for the last completed one -- whatever
    the branching, no move costs more than `max_nodes` leaves. Opponents are
    expanded from `k` determinized worlds drawn from the belief at the root
    (`View.sample`) and the root values averaged across them (PIMC); in
    `omniscient` mode `k` is ignored and the true state is searched. Hidden
    draws are expectations, not one sample: a steal over the victim's
    expected composition, a dev-card buy over the unseen deck, each weighted
    by its probability. Rolls are exact eleven-way within `EXACT_ROLL_PLIES`
    of the root and sampled beyond.

    Opening settlements come from `placement.best` when `placement` is set;
    opening roads are searched. A discard gives up the card with the smallest
    marginal loss; a monopoly names the resource the table is expected to
    hold most of. Trading is not an action: this bot advertises a valuation
    vector and answers a gate (`valuation`/`accepts` below), and the engine
    clears the deals. `max_trades=0` publishes nothing and refuses
    everything, which is the whole of the no-trade referent.

    Every random draw comes from `rng`; the real game's stream is never read.
    """

    evaluator: HonestEvaluator
    depth: int = 2
    width: int | None = 6
    max_nodes: int = DEFAULT_MAX_NODES
    k: int = 1
    rng: random.Random = field(default_factory=random.Random)
    stance: str = "relative"
    # The trade off switch (`hexset.bots.search2.SearchBot.max_trades`): `0`
    # publishes nothing and refuses everything. Not a budget -- the engine
    # has no cap.
    max_trades: int | None = None
    placement: bool = True
    mode: str = "honest"
    exact_roll_plies: int = EXACT_ROLL_PLIES

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise ValueError(f"unknown stance: {self.stance}")
        if self.k < 1:
            raise ValueError("k must be at least one world")
        self._rank = STANCES[self.stance]
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0

    @property
    def omniscient(self) -> bool:
        """Whether this bot reads every seat's true hand (mode="omniscient")."""
        return self.evaluator.omniscient

    @property
    def nodes(self) -> int:
        """Leaf evaluations the last `choose` spent."""
        return self._spent

    # -- the decision --------------------------------------------------------

    def choose(self, game: Game) -> Action:
        """The bot's one public entry point: `Bot.choose(game) -> Action`.

        Setup and discard are resolved directly (placement's prior, the
        smallest marginal loss, or the only option); everything else
        determinizes the belief into `k` worlds, builds the root's own
        options (`_root_options`) and either returns the one option
        available or hands the rest to `_search`.
        """
        seat = to_move(game)
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0
        self.evaluator._walk_cache.clear()
        self.evaluator._belief_cache.clear()
        self.evaluator._evaluate_cache.clear()

        if game.phase is Phase.SETUP_SETTLEMENT and self.placement:
            options = options_for(game)
            # true state: opening placement scores board layout and vertex
            # ownership only (`placement.best`), both public.
            chosen = best_opening(game.state(seat, hidden=False), seat, [a.a for a in options])
            return Action(ActionType.SETUP_SETTLEMENT, chosen)
        if game.phase is Phase.DISCARD:
            options = options_for(game)
            if len(options) == 1:
                return options[0]
            view = game.state(seat)
            return min(options, key=lambda a: self._marginal_loss(view, a.a))

        worlds = self.worlds(game, seat)
        options = self._root_options(game, worlds, seat)
        if len(options) == 1:
            return options[0]
        return self._search(worlds, options, seat)

    def worlds(self, game: Game, seat: int) -> list[Game]:
        """The determinizations this decision is searched in.

        Each is an `imagine` copy whose hidden hands and cards are one draw
        from the belief; in omniscient mode, one copy of the truth.
        """
        if self.omniscient:
            return [imagine(game, self.rng)]
        belief = game.state(seat)
        out = []
        for _ in range(self.k):
            world = imagine(game, self.rng, randomize_deck=False)
            world.set_state(belief.sample(self.rng))
            out.append(world)
        return out

    def root_options(self, game: Game) -> list[Action]:
        """What the search chooses among at `game`, after the bot's own rules."""
        seat = to_move(game)
        return self._root_options(game, self.worlds(game, seat), seat)

    def _root_options(self, game: Game, worlds: list[Game], seat: int) -> list[Action]:
        # Read off the determinized worlds rather than off the truth, in
        # first-seen order: every action legal in a world is legal in the
        # truth, since builds, cards and bank trades depend only on the
        # mover's own hand, which every world reproduces exactly.
        seen: dict[Action, None] = {}
        for world in worlds:
            for action in self._options_in(world, seat):
                seen.setdefault(action, None)
        options = list(seen)
        if not options:
            options = options_for(game)

        monopolies = [a for a in options if a.type is ActionType.PLAY_MONOPOLY]
        if len(monopolies) > 1:
            belief = self.evaluator.belief_from_game(game, seat)
            target = max(
                range(NUM_RESOURCES), key=lambda r: (belief.table_holding(r), -r)
            )
            keep = Action(ActionType.PLAY_MONOPOLY, target)
            options = [a for a in options if a.type is not ActionType.PLAY_MONOPOLY or a == keep]
        return options

    def _search(self, worlds: list[Game], options: list[Action], seat: int) -> Action:
        best = options[0]
        ranked: list[tuple[float, Action]] | None = None
        costs: list[int] = []
        for depth in range(1, self.depth + 1):
            if ranked is None:
                candidates = options
            elif self.width is None:
                candidates = [a for _, a in ranked]
            else:
                candidates = [a for _, a in ranked[: self.width]]
            if depth > 1:
                estimate = self._estimate(candidates, len(options), costs)
                if self._spent + estimate > self._budget:
                    break
            before = self._spent
            partial: list[tuple[float, Action]] = []
            try:
                totals = self._root_values(worlds, candidates, depth, seat, partial)
            except _Exhausted:
                if ranked is None and partial:
                    best = max(partial, key=lambda pair: pair[0])[1]
                break
            ranked = sorted(
                ((self._rank(v, seat), a) for a, v in zip(candidates, totals)),
                key=lambda pair: -pair[0],
            )
            best = ranked[0][1]
            self.depth_reached = depth
            costs.append(self._spent - before)
        return best

    def _estimate(self, candidates: list[Action], branching: int, costs: list[int]) -> int:
        """Leaves the next ply is expected to cost, from what the last ones did."""
        if len(costs) >= 2 and costs[-2] > 0:
            return int(costs[-1] * costs[-1] / costs[-2]) + 1
        total = 0
        for action in candidates:
            if action.type is ActionType.ROLL:
                total += len(ROLL_ODDS) * branching
            elif action.type is ActionType.END_TURN:
                total += len(ROLL_ODDS)
            else:
                total += branching
        return total

    def _root_values(
        self, worlds: list[Game], candidates: list[Action], depth: int, seat: int,
        partial: list[tuple[float, Action]],
    ) -> list[list[float]]:
        share = 1.0 / len(worlds)
        totals = []
        for action in candidates:
            # true state: `num_players` is a fixed board property, same in
            # every world.
            total = [0.0] * worlds[0].state(seat, hidden=False).num_players
            for world in worlds:
                vector = self._after(world, action, depth, seat)
                for p, value in enumerate(vector):
                    total[p] += share * value
            totals.append(total)
            partial.append((self._rank(total, seat), action))
        return totals

    # -- trading (`hexset.trading`) -----------------------------------------

    def valuation(self, view: View) -> tuple[float, ...]:
        """What each resource is worth to this seat now, in [-1, 1].

        `tanh(marginal(r) / MARGINAL_SCALE)`, where `marginal(r)` is
        `Eval(hand + one r) - Eval(hand)` read on this seat's own row under
        its stance. The squash is not cosmetic: the clearing rule *compares*
        two seats' surpluses, so every seat has to publish on one common
        scale, and raw evaluation units have no ceiling to normalise
        against. `MARGINAL_SCALE` is that shared unit, fixed per evaluator
        and pinned beside its weights -- never a per-decision rescaling,
        which would make one seat's "1.0" mean something different from
        another's.

        Read through the view alone, so it is a function of the information
        set: the seat's own hand is exact there, every other seat's is the
        ledger's reconstruction, and neither the bank nor the board is
        private.
        """
        if self.max_trades == 0:
            return NO_VALUATION
        return tuple(
            math.tanh(self._marginal_gain(view, r) / MARGINAL_SCALE)
            for r in range(NUM_RESOURCES)
        )

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        """Take this exchange iff it strictly improves my own evaluation.

        The private gate of the mechanic: the public vectors say a deal is
        advertised, this says whether it is actually good, and it is read
        under the bot's stance -- so under `relative` the counterparty's
        gain is already priced in, which is what makes "not with the
        leader" expressible without a partner term. Strict: an exchange
        worth exactly nothing does not clear, which is what bounds the
        event (`hexset.trading.trade_event`).
        """
        if self.max_trades == 0:
            return False
        seat = view.perspective
        return self._delta(view, seat, seat, received, counterparty, self._rank) > 0.0

    # -- the valuation the two above are built from --------------------------

    def _vector(self, state: GameState, ledger: PublicLedger, knower: int) -> list[float]:
        """The per-seat vector for `state`, read through `knower`'s own belief
        (its hand exact, everyone else's `expected_hand`)."""
        belief = self.evaluator.belief_for(state, ledger, knower)
        return self.evaluator.evaluate(state, knower, belief)

    def _read_row(
        self, state: GameState, ledger: PublicLedger, knower: int, target: int, rank, *,
        vector: list[float] | None = None,
    ) -> float:
        """`target`'s row of `knower`'s vector, under `rank` -- how the
        valuation estimates "what would someone else make of this" without
        reading that someone's true hand."""
        if vector is None:
            vector = self._vector(state, ledger, knower)
        return rank(vector, target)

    def _marginal_gain(self, view: View, resource: int) -> float:
        """Eval(hand + one `resource`) - Eval(hand), from this seat's reading."""
        seat = view.perspective
        state = view.state
        before = self._read_row(state, view.ledger, seat, seat, self._rank)
        after = copy_state(state)
        after.hands[seat][resource] += 1
        if after.bank[resource] > 0:
            after.bank[resource] -= 1
        ledger = view.ledger.copy()
        ledger.receive(seat, resource, 1)
        return self._read_row(after, ledger, seat, seat, self._rank) - before

    def _marginal_loss(self, view: View, resource: int) -> float:
        """Eval(hand) - Eval(hand less one `resource`); zero when none is held."""
        seat = view.perspective
        state = view.state
        if state.hands[seat][resource] < 1:
            return 0.0
        before = self._read_row(state, view.ledger, seat, seat, self._rank)
        after = copy_state(state)
        after.hands[seat][resource] -= 1
        after.bank[resource] += 1
        ledger = view.ledger.copy()
        ledger.spend(seat, resource, 1)
        return before - self._read_row(after, ledger, seat, seat, self._rank)

    def _delta(
        self, view: View, knower: int, target: int, received: Bundle, counterparty: int,
        rank,
    ) -> float:
        """`target`'s row after `target` receives `received` from
        `counterparty`, less what it is now, read entirely through
        `knower`'s own information.

        Invariants, both load-bearing for honesty: the ledger is updated from
        `received` directly rather than by diffing hands, so a third party's
        hidden composition cannot leak through a clamp at zero; and hands
        move exactly for `knower` and as one total for anyone else, because
        only `knower`'s own row is ever read verbatim -- see `_move_hand`.
        """
        state = view.state
        before = self._read_row(state, view.ledger, knower, target, rank)
        after = copy_state(state)
        gains = [max(0, n) for n in received]
        losses = [max(0, -n) for n in received]
        exact = self.omniscient
        self._move_hand(after, knower, target, gains=gains, losses=losses, exact=exact)
        self._move_hand(after, knower, counterparty, gains=losses, losses=gains, exact=exact)
        ledger = view.ledger.copy()
        for r in range(NUM_RESOURCES):
            if losses[r]:
                ledger.spend(target, r, losses[r])
                ledger.receive(counterparty, r, losses[r])
            if gains[r]:
                ledger.spend(counterparty, r, gains[r])
                ledger.receive(target, r, gains[r])
        return self._read_row(after, ledger, knower, target, rank) - before

    @staticmethod
    def _move_hand(
        state: GameState, knower: int, seat: int, *,
        gains: list[int], losses: list[int], exact: bool = False,
    ) -> None:
        """`seat`'s hand after gaining `gains` and losing `losses`.

        Exact, per resource, when `seat == knower` -- its own hand is read
        verbatim, so it must reflect the exchange precisely. Otherwise only
        the total moves, folded into one resource slot: a non-knower's hand
        reaches the honest evaluation only through `View.expected_hand`,
        which reads `known`/`unknown`/`pool` off the ledger and the bank,
        and the one thing it takes from `state.hands` is the *size* -- which
        the fold preserves and a per-resource move, clamped at zero when the
        seat cannot cover `losses`, would not.

        `exact` forces the per-resource move for every seat, and the
        omniscient bot passes it. Under omniscience the reasoning above is
        void: `known` *is* `state.hands`, every row is scored on the real
        cards, and folding would price an all-one-resource fiction whose
        `progress`, `diversity` and `scarce` terms are nothing like the
        position's.
        """
        hand = state.hands[seat]
        if seat == knower or exact:
            for r in range(len(hand)):
                hand[r] += gains[r] - losses[r]
                if hand[r] < 0:
                    hand[r] = 0
        else:
            net = sum(gains) - sum(losses)
            state.hands[seat] = [max(0, sum(hand) + net)] + [0] * (len(hand) - 1)

    # -- the tree ------------------------------------------------------------

    def _leaf(self, game: Game, knower: int) -> list[float]:
        if self._spent >= self._budget:
            raise _Exhausted
        self._spent += 1
        return self.evaluator.evaluate_game(game, knower)

    def _after(
        self, game: Game, action: Action, depth: int, knower: int, ply: int = 0,
    ) -> list[float]:
        """Value of the position `action` leads to, with `depth - 1` plies left."""
        if action.type is ActionType.ROLL:
            return self._over_dice(game, depth, knower, ply)
        if draws_hidden(game, action):
            # true state: `num_players` is a fixed board property.
            total = [0.0] * game.state(knower, hidden=False).num_players
            for weight, child in self.draw_children(game, action, knower):
                for p, value in enumerate(self._value(child, depth - 1, knower, ply + 1)):
                    total[p] += weight * value
            return total
        return self._value(self._plain_child(game, action), depth - 1, knower, ply + 1)

    def _over_dice(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
        # Unreached at every shipped preset (depth=2 <= exact_roll_plies=2, so
        # `ply` never gets this high) -- kept because `depth` is a public
        # field a deeper search may raise, and this is what bounds its cost.
        if ply >= self.exact_roll_plies:
            child = imagine(game, self.rng)
            roll_dice(child)
            return self._value(child, depth - 1, knower, ply + 1)
        # true state: `num_players` is a fixed board property.
        total = [0.0] * game.state(knower, hidden=False).num_players
        for roll, weight in ROLL_ODDS:
            child = imagine(game, self.rng)
            roll_dice(child, roll)
            for p, value in enumerate(self._value(child, depth - 1, knower, ply + 1)):
                total[p] += weight * value
        return total

    def draw_children(
        self, game: Game, action: Action, knower: int,
    ) -> list[tuple[float, Game]]:
        """Every outcome of a hidden draw, with its probability under the belief.

        A steal is resolved to each resource the victim might hold, weighted by
        the knower's belief about that hand; a purchase to each card type,
        weighted by the unseen deck composition. The child for an outcome is
        built in the world the tree is playing in: if that world happens not to
        hold the outcome (a sampled hand without the resource, a shuffled deck
        without the card), one untyped card is swapped so that it does --
        conditioning the determinization on the outcome rather than discarding
        it. Only cards the record has not certified are ever swapped.
        """
        # `omniscient` can be True here, and `Game.state(seat, hidden=True)`
        # is never omniscient -- so this builds the `View` directly rather
        # than through `game.state(knower)`, same as `View.from_game` always
        # did.
        belief = View.from_game(game, knower, omniscient=self.omniscient)
        if action.type is ActionType.BUY_DEV_CARD:
            odds = belief.deck_odds()
            children = []
            for card, weight in enumerate(odds):
                if weight <= 0:
                    continue
                child = imagine(game, self.rng)
                # true state: the search owns `child` outright (a fresh
                # `imagine` copy) and mutates its deck to build this outcome.
                _put_on_top(child.state(knower, hidden=False).deck, card)
                apply(child, action)
                children.append((weight, child))
            return children or [(1.0, self._plain_child(game, action))]

        victim = victim_of(game, action.b)
        assert victim is not None
        odds = belief.steal_odds(victim)
        children = []
        for resource, weight in enumerate(odds):
            if weight <= 0:
                continue
            child = imagine(game, self.rng)
            # true state: same as the deck mutation above -- `child` is the
            # search's own copy, mutated in place to build this outcome.
            hand = child.state(knower, hidden=False).hands[victim]
            if hand[resource] == 0:
                donor = _donor(hand, belief.known[victim])
                if donor is None:
                    continue
                hand[donor] -= 1
                hand[resource] += 1
            child.rng = _Forced(sum(hand[:resource]))  # type: ignore[assignment]
            apply(child, action)
            child.rng = self.rng
            children.append((weight, child))
        if not children:
            return [(1.0, self._plain_child(game, action))]
        scale = 1.0 / sum(weight for weight, _ in children)
        return [(weight * scale, child) for weight, child in children]

    def _plain_child(self, game: Game, action: Action) -> Game:
        child = imagine(game, self.rng)
        apply(child, action)
        return child

    def _options_in(self, world: Game, knower: int) -> list[Action]:
        """`legal_actions` in a determinized world.

        Nothing to filter any more: with trading an engine event rather than
        an action (`hexset.trading`), no legal action's legality depends on
        another seat's hand, so every action a world offers is one the truth
        offers too.
        """
        del knower
        return legal_actions(world)

    def _value(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
        if depth <= 0 or is_over(game):
            return self._leaf(game, knower)
        options = self._options_in(game, knower)
        if not options:
            return self._leaf(game, knower)
        mover = to_move(game)

        if depth == 1 or self.width is None or len(options) <= self.width:
            return self._best_of(game, options, depth, mover, knower, ply)
        ranked = sorted(
            ((self._rank(self._after(game, a, 1, knower, ply), mover), a) for a in options),
            key=lambda pair: -pair[0],
        )
        beam = [a for _, a in ranked[: self.width]]
        return self._best_of(game, beam, depth, mover, knower, ply)

    def _best_of(
        self, game: Game, options: list[Action], depth: int, mover: int, knower: int, ply: int,
    ) -> list[float]:
        best: list[float] | None = None
        best_rank = 0.0
        for action in options:
            vector = self._after(game, action, depth, knower, ply)
            rank = self._rank(vector, mover)
            if best is None or rank > best_rank:
                best, best_rank = vector, rank
        assert best is not None
        return best


def _put_on_top(deck: list[int], card: int) -> None:
    """Make `card` the next draw (`devcards.buy` pops the end of the deck)."""
    if not deck:
        return
    for index in range(len(deck) - 1, -1, -1):
        if deck[index] == card:
            deck[index], deck[-1] = deck[-1], deck[index]
            return
    deck[-1] = card


def _donor(hand: list[int], known: list[int]) -> int | None:
    """A resource the hand holds beyond what the record certifies, if any."""
    for r in range(NUM_RESOURCES):
        if hand[r] > known[r]:
            return r
    for r in range(NUM_RESOURCES):
        if hand[r] > 0:
            return r
    return None


MODES = ("honest", "omniscient", "notrade")

# Sentinel for `heximax(max_trades=...)`: "whatever the mode's own setting is".
BY_MODE: int = object()  # type: ignore[assignment]


def heximax(
    board: Board, rng: random.Random | None = None, *, mode: str = "honest", depth: int = 2,
    width: int | None = 6, max_trades: int | None = BY_MODE,  # type: ignore[assignment]
    max_nodes: int = DEFAULT_MAX_NODES, k: int = 1, stance: str = "relative",
    placement: bool = True, exact_progress_samples: int = 0, weights: Weights | None = None,
) -> Heximax:
    """The three shipped configurations, by `mode`.

    `honest` reads the ledger and the trading-table weights; `omniscient`
    reads every true hand with the same weights; `notrade` is honest with the
    no-trade weights. Left at `BY_MODE`, trading is on for the first two and
    off (`max_trades=0`) for `notrade`; any explicit value, `None` included,
    is taken as given.

    `weights` overrides the mode's own profile (`TRADING_WEIGHTS` or
    `NO_TRADE_WEIGHTS`) with the given vector, leaving everything else about
    the mode -- the trade switch, `omniscient` -- unchanged. This is the hook
    `hexset.tuning` fits through: a candidate and the incumbent are otherwise
    identical heximax bots, differing only in this vector.
    """
    if mode not in MODES:
        raise ValueError(f"unknown heximax mode: {mode}")
    if max_trades is BY_MODE:
        max_trades = 0 if mode == "notrade" else None
    if weights is None:
        weights = NO_TRADE_WEIGHTS if mode == "notrade" else TRADING_WEIGHTS
    evaluator = HonestEvaluator(
        board,
        weights,
        omniscient=(mode == "omniscient"),
        exact_progress_samples=exact_progress_samples,
    )
    return Heximax(
        evaluator,
        depth=depth,
        width=width,
        max_nodes=max_nodes,
        k=k,
        rng=rng or random.Random(),
        stance=stance,
        max_trades=max_trades,
        placement=placement,
        mode=mode,
    )
