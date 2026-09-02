# SPDX-License-Identifier: GPL-3.0-only
"""Max^n over the honest evaluation, within a leaf budget.

`Heximax`'s search core: `choose` (the one public entry point, plus the
setup/no-trade/discard shortcuts it resolves directly), iterative deepening
(`_search`/`_estimate`/`_root_values`), the tree itself
(`_value`/`_after`/`_best_of`), roll expansion (`_over_dice`), and hidden-draw
expansion (`draw_children`: a steal weighted over the victim's belief, a
dev-card buy weighted over the unseen deck). `Heximax` also inherits
`trade._TradeMixin`, so `_options_in`'s `ACCEPT_TRADE` gate reaches
`self.accept_rule` there by the same attribute lookup as everything declared
in this file -- the split changes no lookup, only which file a definition
lives in.

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
  heximax's own phase mix. `search2` books ~20x more `TRADE_RESPOND`
  decisions over the same games (2481 against 126 over nine), because the
  engine's naive one-for-one `PROPOSE_TRADE` sample makes offers
  `score_proposal`'s crisp `willing` gate does not, and those cheap
  decisions sit in the mirror table's denominator. Phase-neutral heximax
  reads **2.08x** -- at the ceiling per decision of the kind it actually
  faces -- MAIN 2.18x, ROBBER **1.94x**, ROLL the one bucket over at
  **3.34x**.

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

import random
from dataclasses import dataclass, field

from hexset.actions import (
    Action,
    ActionType,
    apply,
    legal_actions,
    victim_of,
    within_offer_budget,
)
from hexset.board.board import Board
from hexset.board.terrain import NUM_RESOURCES
from ..search2 import STANCES, options_for
from hexset.game import ROLL_ODDS, Game, Phase, imagine, is_over, roll_dice, to_move
from hexset.mcts import draws_hidden
from hexset.placement import best as best_opening
from hexset.trading import can_accept

from hexset.view import View
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator, Weights
from .trade import _TradeMixin

# Leaf evaluations a move may spend. Chosen so the default configuration costs
# no more than twice `search2` per move (the design's ceiling): at 600 the
# mean is 1.5x and the per-move tail, which the unbounded search takes to
# ~1500 leaves, is cut at the budget. Figures in the module docstring.
DEFAULT_MAX_NODES = 600

# A roll taken this many plies or fewer below the root is expanded over all
# eleven outcomes; deeper rolls are sampled once. At the default depth of two
# every roll in the tree is exact.
EXACT_ROLL_PLIES = 2


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
class Heximax(_TradeMixin):
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
    hold most of. `max_offers` is the bot's own budget below the engine's; at
    zero it never proposes and always declines. The trade adapter has two
    touch points: `propose_actions` supplies the `PROPOSE_TRADE` root options
    (top `propose_top_n` candidates over `propose_margin`), and `_options_in`
    gates every `TRADE_RESPOND` node's `ACCEPT_TRADE` on `accept_rule` at
    `accept_margin`. Both margins are unfitted; `0.0` accepts or proposes
    whenever the valuation itself is positive.

    Every random draw comes from `rng`; the real game's stream is never read.
    """

    evaluator: HonestEvaluator
    depth: int = 2
    width: int | None = 6
    max_nodes: int = DEFAULT_MAX_NODES
    k: int = 1
    rng: random.Random = field(default_factory=random.Random)
    stance: str = "relative"
    max_offers: int | None = 3
    placement: bool = True
    mode: str = "honest"
    exact_roll_plies: int = EXACT_ROLL_PLIES
    # The trade adapter's own knobs (unfitted): how many of
    # `candidate_bundles`' scored proposals become root options, and the
    # margins below which a proposal is not offered or an offer not
    # accepted. See the class docstring's trade-adapter paragraph.
    propose_top_n: int = 3
    propose_margin: float = 0.0
    accept_margin: float = 0.0

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

        Setup, a no-trade bot's forced decline, and discard are resolved
        directly (placement's prior, `marginal_loss`, or the only option);
        everything else determinizes the belief into `k` worlds, builds the
        root's own options (`_root_options` -- engine legality plus, in
        MAIN, the trade adapter's `propose_actions`), and either returns the
        one option available or hands the rest to `_search`.
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
        if game.phase is Phase.TRADE_RESPOND and self.max_offers == 0:
            return Action(ActionType.DECLINE_TRADE)
        if game.phase is Phase.DISCARD:
            options = options_for(game)
            if len(options) == 1:
                return options[0]
            return min(options, key=lambda a: self.marginal_loss(game, seat, a.a))

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
        # The engine's offer sample is built from the opponents' true hands
        # (who could cover what), so the root's options are read off the
        # worlds instead, in first-seen order. Every action legal in a world
        # is legal in the truth: builds, cards and bank trades depend only on
        # the mover's hand, and an offer needs only what the proposer holds.
        seen: dict[Action, None] = {}
        for world in worlds:
            for action in self._options_in(world, seat):
                seen.setdefault(action, None)
        options = list(seen)
        if game.phase is Phase.MAIN:
            # The engine's one-for-one sample (`actions._offer_actions`,
            # already folded into `seen` above) is replaced wholesale by
            # `propose_actions`. Guarded by the same budget test
            # `within_offer_budget` applies below, so a bot with no offers
            # left never pays for the candidate search.
            options = [a for a in options if a.type is not ActionType.PROPOSE_TRADE]
            if self.max_offers is None or game.offers_made < self.max_offers:
                options.extend(self.propose_actions(game, seat))
        options = within_offer_budget(game, options, self.max_offers)
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
        """`legal_actions` in a determinized world, minus an ACCEPT it cannot
        honour or one `accept_rule` would refuse.

        The engine builds its responder list from the true hands, so a world
        sampled without that knowledge may have dealt the seat being asked a
        hand that cannot cover the offer: that hard constraint drops ACCEPT
        outright. The soft one runs at every `TRADE_RESPOND` node, not only
        the root, and `knower` there is always the search's root seat, never
        the responder -- so a simulated opponent's row is read off `knower`'s
        belief (`_partner_delta`), never that world's sampled truth, and its
        accept cannot depend on which of the `k` worlds it was sampled into.
        """
        options = legal_actions(world)
        if world.phase is Phase.TRADE_RESPOND and world.offer is not None:
            responder = to_move(world)
            # true state: `world` is a determinization already, so the hard
            # constraint below (can `responder` cover it in THIS sampled
            # world) is read off `world`'s own sampled truth, not the honest
            # view -- see this method's docstring.
            if not can_accept(
                world.state(responder, hidden=False), world.offer, responder
            ) or not self.accept_rule(
                world, responder, world.offer, self.accept_margin, knower=knower
            ):
                options = [a for a in options if a.type is not ActionType.ACCEPT_TRADE]
        return options

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

# Sentinel for `heximax(max_offers=...)`: "whatever the mode's own budget is".
BY_MODE: int = object()  # type: ignore[assignment]


def heximax(
    board: Board, rng: random.Random | None = None, *, mode: str = "honest", depth: int = 2,
    width: int | None = 6, max_offers: int | None = BY_MODE,  # type: ignore[assignment]
    max_nodes: int = DEFAULT_MAX_NODES, k: int = 1, stance: str = "relative",
    placement: bool = True, exact_progress_samples: int = 0, weights: Weights | None = None,
    propose_top_n: int = 3, propose_margin: float = 0.0, accept_margin: float = 0.0,
) -> Heximax:
    """The three shipped configurations, by `mode`.

    `honest` reads the ledger and the trading-table weights; `omniscient`
    reads every true hand with the same weights; `notrade` is honest with the
    no-trade weights. Left at `BY_MODE`, the offer budget is three for the
    first two and zero for `notrade`; any explicit value, `None` included,
    is taken as given. `propose_top_n`, `propose_margin` and `accept_margin`
    are the trade adapter's own knobs, unfitted -- see `Heximax`'s docstring.

    `weights` overrides the mode's own profile (`TRADING_WEIGHTS` or
    `NO_TRADE_WEIGHTS`) with the given vector, leaving everything else about
    the mode -- the offer budget, `omniscient` -- unchanged. This is the hook
    `hexset.tuning` fits through: a candidate and the incumbent are otherwise
    identical heximax bots, differing only in this vector.
    """
    if mode not in MODES:
        raise ValueError(f"unknown heximax mode: {mode}")
    if max_offers is BY_MODE:
        max_offers = 0 if mode == "notrade" else 3
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
        max_offers=max_offers,
        placement=placement,
        mode=mode,
        propose_top_n=propose_top_n,
        propose_margin=propose_margin,
        accept_margin=accept_margin,
    )
