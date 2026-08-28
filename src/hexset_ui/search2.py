"""The handcrafted opponent, whole: a position evaluation and the max^n search
that reads it.

This is the one bot that needs no checkpoint, which is why it exists — a fresh
clone with an empty `models/` still has something to play against. It learns
nothing and loads nothing; every number in it was fitted once and written down
here.

Self-contained on purpose. A learned opponent lives entirely behind
`hexset_ui.onnxbot`, and the two share only the engine and the small bot
primitives in `hexset_ui.actions`, so neither can quietly acquire a dependency on
how the other works.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from typing import NamedTuple, Sequence

from .actions import (
    Action,
    ActionType,
    apply,
    legal_actions,
    options_for,
    within_offer_budget,
)
from .board.board import Board, pips, scarce_resources
from .board.ports import BASE_TRADE_RATIO
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from .economy import COSTS, Purchase
from .game import ROLL_ODDS, Game, imagine, is_over, roll_dice, to_move
from .robber import DISCARD_THRESHOLD
from .state import GameState
from .trading import Offer, execute as execute_trade, responders
from .victory import WINNING_POINTS, award_points, card_points

ROLLS = 36
WIN_SCORE = 100.0

# Roads are left out of the build-progress term. At one wood and one brick,
# nearly any hand is half way to a road, so including it would score almost
# every hand alike and tell the search nothing.
PROGRESS_PURCHASES = (Purchase.SETTLEMENT, Purchase.CITY, Purchase.DEV_CARD)


class Survey(NamedTuple):
    """What one walk over a player's vertices yields."""

    rate: float
    kinds: int
    scarce: int
    buildings: int
    port_gain: int


@dataclass(frozen=True)
class Weights:
    """Scoring weights in units of victory points.

    Fitted by `hexset_ui.tuning`, not guessed, and refitted once trading existed:
    60 hill-climb rounds at 400 games a duel, confirmed over 2000 games at
    56.2% (95% CI 54.1-58.4%) against the pre-trading values. Fitted for the
    one-ply bot; the deeper search has not been refitted.

    `victory_point` is pinned at 1.0 because scaling every weight alike cannot
    change which position the search prefers, so the unit has to come from
    somewhere. Still a dataclass so the terms stay ablatable.

    What trading changed is the interesting part. Production went up two and a
    half times, and progress towards a specific build collapsed to almost
    nothing. Both follow from the same thing: when you can trade, what matters
    is how much you produce, not whether you happen to hold the right cards for
    the build in front of you. The hand sorts itself out.

    Ablation under plain max^n said only four of these earn their keep, but
    that finding did not survive the relative stance, which is the default.
    Read relatively, seven of nine are load-bearing: `diversity` (45.0%
    [41.1, 49.0] when zeroed), `card` and `surplus_card` all came back, because
    a bot judging its position against the table cares again about not
    producing a single resource and about not being caught over seven. Only
    `knight` earns nothing outright and `port` cannot be told from nothing.
    Do not quote the older four-term reading — see `agents/status.md`.

    There is no reachable-production term. There was, and ablation showed it was
    actively harmful — zeroing it beat keeping it 77.2% over 1000 games. The
    likely reason is that settling a junction occupies it and blocks its
    neighbours, so any term rewarding open junctions nearby quietly argues
    against the expansion that actually scores. Do not reintroduce it without
    a formulation that does not penalise building.
    """

    victory_point: float = 1.0
    production: float = 7.067
    diversity: float = 0.358
    # How many *scarce* resources the seat reaches — the ones with fewer hexes
    # than the commonest, so brick and ore on the base map.
    #
    # The only weight here not fitted against the engine. It comes from the
    # human corpus, where the conditional logit over openings put it at 0.91
    # pips; `CORPUS_SCARCE` converts that at `production / ROLLS` VP per pip and
    # a test pins this default to it. Adopted untuned, at the corpus's own
    # exchange rate, and it won anyway: 51.62% [50.75, 52.48] over 12,800 games
    # against the same evaluation without it.
    #
    # Every duel result recorded before 2026-08-15 was measured with this at
    # zero, so it moved the baseline for all of them.
    scarce: float = 0.1786
    progress: float = 0.01843
    road: float = 0.1209
    knight: float = 0.1041
    card: float = 0.005406
    surplus_card: float = -0.3891
    port: float = 0.007373


TERM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Weights))

# The corpus's scarcity weight in this evaluation's units: 0.91 pips, as fitted
# against the human opening corpus, at `production / ROLLS` victory points per pip. Derived
# rather than typed so it follows a refit of `production` instead of silently
# meaning something else afterwards.
CORPUS_SCARCE: float = 0.91 * Weights.production / ROLLS


class Evaluator:
    """Handcrafted per-seat scoring of a position.

    Returns one score per seat rather than a scalar, matching both the planned
    value head and the max^n backup the search needs. Per-vertex yields are
    precomputed from the board, which never changes during a game.
    """

    def __init__(self, board: Board, weights: Weights | None = None) -> None:
        self.weights = weights or Weights()
        self.vector = tuple(getattr(self.weights, name) for name in TERM_NAMES)
        self.scarce = scarce_resources(board)
        topology = board.topology
        self.yields = tuple(
            tuple(
                (h, TERRAIN_RESOURCE[board.terrain[h]], pips(board.tokens[h]))
                for h in topology.vertex_hexes[v]
                if pips(board.tokens[h])
            )
            for v in range(topology.num_vertices)
        )
        # Ports indexed by vertex rather than the board's port list, so the
        # walk below can ask "what does this vertex give me" instead of asking
        # every port whether this player owns it. A tuple because nothing stops
        # a vertex touching two ports.
        self.ports_at = tuple(
            tuple(
                (port.resource, port.ratio)
                for port in board.ports
                if vertex in port.vertices
            )
            for vertex in range(topology.num_vertices)
        )

    @classmethod
    def for_game(cls, game: Game, weights: Weights | None = None) -> Evaluator:
        return cls(game.state.board, weights)

    def survey(self, state: GameState, player: int) -> Survey:
        """Everything the player's own vertices are worth, in one walk.

        Production, diversity, building points and port rates all want the same
        vertices, and each used to walk all of them separately — `production`
        here, `victory.building_points` there, and `economy.trade_ratios` asking
        every port on the board whether this player owned it. Profiling put
        those three at about 38% of all runtime between them, `trade_ratios`
        alone at 19%, which is what a triple walk costs when it happens at
        every leaf of a search.

        The arithmetic is unchanged, and `test_the_survey_agrees_with_the_rules`
        pins it to the canonical functions so this cannot quietly drift from
        them.

        Gold counts toward the rate but not toward diversity: it pays the
        holder's choice, so it has no fixed resource to be distinct from.
        """
        total = 0
        kinds: set[int] = set()
        buildings = 0
        generic = BASE_TRADE_RATIO
        specific = [BASE_TRADE_RATIO] * NUM_RESOURCES

        robber = state.robber
        vertex_building = state.vertex_building
        yields = self.yields
        ports_at = self.ports_at

        for vertex, owner in enumerate(state.vertex_owner):
            if owner != player:
                continue
            built = vertex_building[vertex]
            buildings += built
            for hex_index, resource, p in yields[vertex]:
                if hex_index == robber:
                    continue
                total += built * p
                if resource is not None:
                    kinds.add(resource)
            for resource, ratio in ports_at[vertex]:
                if resource is None:
                    if ratio < generic:
                        generic = ratio
                elif ratio < specific[resource]:
                    specific[resource] = ratio

        port_gain = sum(
            BASE_TRADE_RATIO - (generic if generic < best else best)
            for best in specific
        )
        return Survey(
            rate=total / ROLLS,
            kinds=len(kinds),
            scarce=len(kinds & self.scarce),
            buildings=buildings,
            port_gain=port_gain,
        )

    def production(self, state: GameState, player: int) -> tuple[float, int]:
        """Expected cards per turn, and how many distinct resources supply them."""
        walk = self.survey(state, player)
        return walk.rate, walk.kinds

    def progress(self, state: GameState, player: int) -> float:
        """How far the hand has got towards the purchase it is closest to.

        Two wheat and three ore is a city; five wood is a quarter of a
        settlement. Card count alone cannot tell those apart.
        """
        hand = state.hands[player]
        return max(
            sum(min(hand[r], n) for r, n in enumerate(COSTS[purchase]) if n)
            / sum(COSTS[purchase])
            for purchase in PROGRESS_PURCHASES
        )

    def terms(
        self, state: GameState, player: int, *, knower: int | None = None
    ) -> tuple[float, ...]:
        """The raw term values, in `TERM_NAMES` order.

        These are exactly what the weights multiply, so anything learning
        coefficients from data is fitting the same model the search uses. The
        alternative — a separate feature extractor — drifts silently.
        """
        walk = self.survey(state, player)
        held = sum(state.hands[player])
        points = walk.buildings + award_points(state, player)
        if player == knower:
            points += card_points(state, player)

        return (
            points,
            walk.rate,
            walk.kinds,
            walk.scarce,
            self.progress(state, player),
            sum(1 for owner in state.edge_owner if owner == player),
            state.knights_played[player],
            held,
            max(0, held - DISCARD_THRESHOLD),
            walk.port_gain,
        )

    def score(
        self, state: GameState, player: int, *, knower: int | None = None
    ) -> float:
        values = self.terms(state, player, knower=knower)
        total = 0.0
        for weight, value in zip(self.vector, values):
            total += weight * value
        if values[0] >= WINNING_POINTS:
            total += WIN_SCORE
        return total

    def evaluate(self, state: GameState, knower: int | None = None) -> list[float]:
        """Score every seat. `knower` is the seat whose hidden cards may be counted."""
        return [
            self.score(state, p, knower=knower) for p in range(state.num_players)
        ]


def _own(vector: Sequence[float], seat: int) -> float:
    """Plain max^n: each seat wants its own score high and ignores the rest."""
    return vector[seat]


def _relative(vector: Sequence[float], seat: int) -> float:
    """Own score less the average of everyone else's.

    A constant-sum reading of the vector. HexSet has exactly one winner, so a
    position is only worth what it is worth *compared to* the table, and an
    action that lifts everyone equally has achieved nothing.
    """
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - sum(others) / len(others)


def _paranoid(vector: Sequence[float], seat: int) -> float:
    """Own score less the best opponent's. The leader is the only rival."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - max(others)


# How a seat turns the per-seat evaluation vector into the one number it
# maximises. The evaluation is unchanged; only the reading of it differs.
STANCES = {"own": _own, "relative": _relative, "paranoid": _paranoid}


@dataclass
class SearchBot:
    """Max^n search over the handcrafted evaluation.

    `depth` counts decisions, not turns. A HexSet turn contains many actions, so
    depth two plans a pair of the mover's own actions rather than reaching an
    opponent; passing the turn is itself one of the actions searched.

    Each seat maximises its own component of the evaluation vector, which is
    max^n rather than minimax — with more than two players there is no single
    opponent to minimise, and assuming everyone ganged up on the mover would
    model the table badly.

    A roll is a chance node expanded over all eleven outcomes and weighted by
    probability rather than sampled, so the value is not noisy. `width` beams
    the branching, since the main phase can offer sixty-odd actions.

    `stance` is how a seat reads the per-seat vector — see `STANCES`. Every
    seat in the tree reads it the same way, so a relative stance models a table
    that all thinks relatively, not one bot that has noticed something. It
    defaults to `relative`, which beat plain max^n 53.6% over 2000 games: the
    baseline exists to be beaten, so it should be the best one available.
    """

    evaluator: Evaluator
    depth: int = 2
    width: int | None = 6
    rng: random.Random = field(default_factory=random.Random)
    stance: str = "relative"
    partner_choice: bool = False
    # How many offers this bot will propose in a turn, below whatever the engine
    # allows. `None` spends the engine's whole budget. Kept here rather than in
    # the engine because a cap every seat receives cannot be duelled against
    # itself: only a bot that declines an action its opponent still has can say
    # what the action was worth.
    max_offers: int | None = None

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise ValueError(f"unknown stance: {self.stance}")
        self._rank = STANCES[self.stance]
        # A leaf evaluation that wants the whole `Game` rather than the state
        # says so by offering `evaluate_game`. The learned one does, because the
        # encoder reads the phase, the turn count and the free-road counter, and
        # none of those are on `GameState`. The handcrafted evaluations need
        # only the state and keep the cheaper call — this is resolved once here
        # rather than tested at every leaf.
        self._leaf = getattr(self.evaluator, "evaluate_game", None) or self._from_state

    def _from_state(self, game: Game, seat: int) -> list[float]:
        return self.evaluator.evaluate(game.state, seat)

    def choose(self, game: Game) -> Action:
        options = within_offer_budget(game, options_for(game), self.max_offers)
        if len(options) == 1:
            return self._addressed(game, options[0], to_move(game))
        # Only the seat to move may count its own hidden cards, and it stays the
        # perspective for the whole search: deeper nodes are still this seat's
        # reasoning about the game, not somebody else's.
        seat = to_move(game)
        candidates = self._beam(game, options, seat, seat)
        best = max(
            candidates,
            key=lambda a: self._rank(self._after(game, a, self.depth, seat), seat),
        )
        return self._addressed(game, best, seat)

    def _addressed(self, game: Game, action: Action, seat: int) -> Action:
        """Name who the proposer would rather have take the offer, best first.

        Only worth computing when more than one player could cover it. The
        search valued the offer under the engine's neutral order, so ordering it
        afterwards can only improve on what was searched, never contradict it.
        """
        if not self.partner_choice or action.type is not ActionType.PROPOSE_TRADE:
            return action
        offer = Offer(proposer=seat, give=action.give, want=action.want)
        willing = responders(game.state, offer)
        if len(willing) < 2:
            return action

        def value(responder: int) -> float:
            child = imagine(game, self.rng)
            execute_trade(child.state, offer, responder)
            return self._rank(self._leaf(child, seat), seat)

        return action._replace(ask=tuple(sorted(willing, key=value, reverse=True)))

    def _beam(
        self, game: Game, options: list[Action], mover: int, knower: int
    ) -> list[Action]:
        if self.width is None or len(options) <= self.width:
            return options
        ranked = sorted(
            options,
            key=lambda a: -self._rank(self._after(game, a, 1, knower), mover),
        )
        return ranked[: self.width]

    def _after(self, game: Game, action: Action, depth: int, knower: int) -> list[float]:
        """Value of the position `action` leads to, with `depth - 1` plies left."""
        if action.type is ActionType.ROLL:
            return self._over_dice(game, depth, knower)
        child = imagine(game, self.rng)
        apply(child, action)
        return self._value(child, depth - 1, knower)

    def _over_dice(self, game: Game, depth: int, knower: int) -> list[float]:
        total = [0.0] * game.state.num_players
        for roll, weight in ROLL_ODDS:
            child = imagine(game, self.rng)
            roll_dice(child, roll)
            for p, value in enumerate(self._value(child, depth - 1, knower)):
                total[p] += weight * value
        return total

    def _value(self, game: Game, depth: int, knower: int) -> list[float]:
        if depth <= 0 or is_over(game):
            return self._leaf(game, knower)
        options = legal_actions(game)
        if not options:
            return self._leaf(game, knower)

        mover = to_move(game)
        best: list[float] | None = None
        best_rank = 0.0
        for action in self._beam(game, options, mover, knower):
            vector = self._after(game, action, depth, knower)
            rank = self._rank(vector, mover)
            if best is None or rank > best_rank:
                best, best_rank = vector, rank
        assert best is not None
        return best


def greedy(
    evaluator: Evaluator,
    rng: random.Random | None = None,
    stance: str = "relative",
    partner_choice: bool = False,
    max_offers: int | None = None,
) -> SearchBot:
    """One ply: take the action with the best position after it.

    Cheap enough to run tens of thousands of games, and the reference the deeper
    search has to beat before depth is worth paying for.
    """
    return SearchBot(
        evaluator,
        depth=1,
        width=None,
        rng=rng or random.Random(),
        stance=stance,
        partner_choice=partner_choice,
        max_offers=max_offers,
    )


def search2(
    board: Board,
    rng: random.Random | None = None,
    *,
    max_offers: int | None = None,
) -> SearchBot:
    """The handcrafted opponent the picker offers by name.

    Depth two over the fitted evaluation, read relatively. These were the
    settings that won, and they are fixed here rather than exposed: this is
    meant to be one known opponent, not a family of them.
    """
    return SearchBot(
        Evaluator(board),
        depth=2,
        width=6,
        rng=rng or random.Random(),
        stance="relative",
        max_offers=max_offers,
    )
