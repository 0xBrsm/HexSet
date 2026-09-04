# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Mapping, NamedTuple, Sequence

from ..board.board import Board, pips, scarce_resources
from ..board.ports import BASE_TRADE_RATIO
from ..board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from ..economy import COSTS, Purchase
from ..game import Game
from ..robber import DISCARD_THRESHOLD
from ..state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, Building, GameState
from ..victory import WINNING_POINTS, award_points, card_points

ROLLS = 36
WIN_SCORE = 100.0
SEVEN_ODDS = 6 / 36

# What one purchase is worth to a seat that can still make it, relative to a
# victory point -- the multiplier on how far the hand has got towards it.
# Settlement and city are each a point outright. A development card is a
# point one time in five and a knight most of the rest, so it is worth
# noticeably less than a building but far more than nothing. A road scores
# only `Weights.road`, which is what this evaluation already pays for one --
# and it is only ever counted when the seat has no legal settlement spot, so
# what it prices is a road that opens one, not a road as a place to put two
# cards. These are the shape of the term, not fitted coefficients: the one
# fitted number is `Weights.buy_progress`, the scale over all four.
PURCHASE_VALUE: dict[Purchase, float] = {
    Purchase.SETTLEMENT: 1.0,
    Purchase.CITY: 1.0,
    Purchase.DEV_CARD: 0.4,
    Purchase.ROAD: 0.35,
}

# `COSTS[purchase]` with the zero entries dropped, and its own divisor:
# `hand_terms`' inner loop, run four times per seat per leaf.
PURCHASE_COST: dict[Purchase, tuple[tuple[tuple[int, int], ...], int]] = {
    purchase: (
        tuple((r, n) for r, n in enumerate(COSTS[purchase]) if n),
        sum(COSTS[purchase]),
    )
    for purchase in PURCHASE_VALUE
}


class Survey(NamedTuple):
    """What one walk over a player's vertices and edges yields."""

    rate: float
    kinds: int
    scarce: int
    buildings: int
    port_gain: int
    settlements: int
    cities: int
    roads: int
    # Vertices this seat could legally settle right now, ignoring the piece
    # supply (`hand_terms` applies that): connected to one of its roads,
    # empty, and clear of the distance rule.
    spots: int
    # The cheapest rate this seat can trade each resource at, ports included.
    ratios: tuple[int, ...]


def seven_before_next_turn(num_players: int) -> float:
    """The chance a 7 is rolled before this seat rolls again."""
    return 1.0 - (1.0 - SEVEN_ODDS) ** max(0, num_players - 1)


def _affordable(purchase: Purchase, walk: Survey, deck_left: int) -> bool:
    """Whether this purchase is still worth saving for, from board facts alone."""
    if purchase is Purchase.SETTLEMENT:
        return walk.spots > 0 and walk.settlements < MAX_SETTLEMENTS
    if purchase is Purchase.CITY:
        return walk.settlements > 0 and walk.cities < MAX_CITIES
    if purchase is Purchase.DEV_CARD:
        return deck_left > 0
    # A road is priced only as the way to open a settlement spot, so it counts
    # for nothing while the seat already has one to build on.
    return walk.spots == 0 and walk.roads < MAX_ROADS and walk.settlements < MAX_SETTLEMENTS


def hand_terms(
    hand: Sequence[float], walk: Survey, *, num_players: int, deck_left: int
) -> tuple[float, float, float]:
    """The three hand terms: purchase progress, spare cards, robber exposure.

    A pure function of the hand and this seat's own board facts, so the trade
    gate that recomputes it from a post-trade hand gets bit-identically what
    `Evaluator.terms` would have got from the same hand.
    """
    best = 0.0
    best_cost: tuple[int, ...] | None = None
    for purchase, value in PURCHASE_VALUE.items():
        if not _affordable(purchase, walk, deck_left):
            continue
        needed, total = PURCHASE_COST[purchase]
        scored = value * sum([min(hand[r], n) for r, n in needed]) / total
        if scored > best:
            best = scored
            best_cost = COSTS[purchase]

    ratios = walk.ratios
    # Every card is either committed to the purchase the hand is closest to or
    # spare; a spare card is worth what the bank or a port will give for it,
    # which is a quarter of a card, or a half where the seat holds the port.
    if best_cost is None:
        spare = sum([hand[r] / ratios[r] for r in range(NUM_RESOURCES)])
    else:
        spare = sum(
            [max(0.0, hand[r] - best_cost[r]) / ratios[r] for r in range(NUM_RESOURCES)]
        )

    # Robber exposure, not a cliff: the chance of a 7 before this seat plays
    # again, times the half-hand it would discard, times what those cards are
    # worth. `ramp` is the discard rule made continuous in hand size -- exactly
    # zero at seven cards and below, exactly a half hand at eight and above --
    # so an expected hand that crosses the threshold does not jump.
    held = sum(hand)
    over = held - DISCARD_THRESHOLD
    ramp = 0.0 if over <= 0.0 else (1.0 if over >= 1.0 else over)
    if not ramp:
        return best, spare, 0.0
    bank_value = sum([hand[r] / ratios[r] for r in range(NUM_RESOURCES)])
    return best, spare, seven_before_next_turn(num_players) * 0.5 * ramp * bank_value


@dataclass(frozen=True)
class Weights:
    """Scoring weights in units of victory points.

    Fitted by `hexset.tuning`, not guessed, and refitted once trading existed:
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
    [41.1, 49.0] when zeroed), and the hand terms came back, because a bot
    judging its position against the table cares again about not producing a
    single resource and about not being caught over seven. Only `knight`
    earns nothing outright and `port` cannot be told from nothing. Do not
    quote the older four-term reading.

    The three hand terms are the 2026-09-04 redesign (`hand_terms`, readout
    `docs/readouts/hand-valuation/`): the flat `card`/`progress`/
    `surplus_card` trio priced a card at 0.005 VP and being one card over
    seven at -0.39, so dumping six cards for one read as two victory points
    and any lopsided trade cleared. They are refit here, the rest of the
    vector left at its trading fit.

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
    # The only weight here not fitted against the engine. It comes from the same
    # conditional logit over openings that produced `hexset.placement`, which put
    # it at 0.91 pips; `FITTED_SCARCE` converts that at `production / ROLLS` VP
    # per pip and a test pins this default to it. Adopted untuned, at the fit's
    # own exchange rate, and it won anyway: 51.62% [50.75, 52.48] over 12,800
    # games against the same evaluation without it.
    #
    # Every arena number recorded before 2026-08-15 was measured with this at
    # zero, so it moved the baseline for all of them.
    scarce: float = 0.1786
    # How far the hand has got towards the best purchase still open to this
    # seat, each purchase weighted by `PURCHASE_VALUE`. Replaces `progress`.
    buy_progress: float = 0.45
    road: float = 0.1209
    knight: float = 0.1041
    # Per card the best purchase does not need, counted at the seat's own
    # bank or port rate. Replaces the flat per-card `card`.
    spare_card: float = 0.15
    # Per bank-equivalent card expected to be lost to a 7 before this seat
    # plays again. Replaces the `surplus_card` cliff.
    robber_risk: float = -0.15
    port: float = 0.007373


TERM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Weights))

# The fitted scarcity weight in this evaluation's units: 0.91 pips, as fitted in
# `hexset.placement`, at `production / ROLLS` victory points per pip. Derived
# rather than typed so it follows a refit of `production` instead of silently
# meaning something else afterwards.
FITTED_SCARCE: float = 0.91 * Weights.production / ROLLS


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
        # true state: board layout is public.
        return cls(game.state(0, hidden=False).board, weights)

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
        settlements = cities = 0
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
            if built == Building.SETTLEMENT:
                settlements += 1
            elif built == Building.CITY:
                cities += 1
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

        # The seat's own edges, once: how many roads it has built, and which
        # of their endpoints it could legally settle -- `state.can_place_
        # settlement`'s three conditions bar the piece supply, which
        # `hand_terms` applies from `settlements`. Walking the seat's roads
        # rather than every vertex is the cheap way round: a connected spot
        # is by definition an endpoint of one of them.
        roads = spots = 0
        topology = state.board.topology
        edges = topology.edges
        neighbours = topology.vertex_neighbors
        seen: set[int] = set()
        for edge, owner in enumerate(state.edge_owner):
            if owner != player:
                continue
            roads += 1
            for vertex in edges[edge]:
                if vertex in seen:
                    continue
                seen.add(vertex)
                if vertex_building[vertex] != Building.NONE:
                    continue
                if any(vertex_building[n] != Building.NONE for n in neighbours[vertex]):
                    continue
                spots += 1

        ratios = tuple(
            generic if generic < best else best for best in specific
        )
        port_gain = sum(BASE_TRADE_RATIO - ratio for ratio in ratios)
        return Survey(
            rate=total / ROLLS,
            kinds=len(kinds),
            scarce=len(kinds & self.scarce),
            buildings=buildings,
            port_gain=port_gain,
            settlements=settlements,
            cities=cities,
            roads=roads,
            spots=spots,
            ratios=ratios,
        )

    def production(self, state: GameState, player: int) -> tuple[float, int]:
        """Expected cards per turn, and how many distinct resources supply them."""
        walk = self.survey(state, player)
        return walk.rate, walk.kinds

    def hand_terms(
        self, state: GameState, player: int, walk: Survey
    ) -> tuple[float, float, float]:
        """`evaluate.hand_terms` for this seat's real hand and board facts."""
        return hand_terms(
            state.hands[player],
            walk,
            num_players=state.num_players,
            deck_left=len(state.deck),
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
        points = walk.buildings + award_points(state, player)
        if player == knower:
            points += card_points(state, player)
        progress, spare, risk = self.hand_terms(state, player, walk)

        return (
            points,
            walk.rate,
            walk.kinds,
            walk.scarce,
            progress,
            walk.roads,
            state.knights_played[player],
            spare,
            risk,
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


def hand_shifted(state: GameState, changes: Mapping[int, Sequence[int]]) -> GameState:
    """`state` with `changes[seat]` added elementwise to that seat's hand, for
    every seat named in `changes` -- everything else, including every other
    seat's hand, shared by reference rather than copied.

    A trade gate pricing a candidate exchange needs the position it would
    lead to, but only two hands ever move -- the board, the bank, the deck
    and every dev-card pile are exactly what they were. `state.copy_state`
    does not know that and clones all of it; this shares it instead, so a
    private gate can price a candidate without paying for a board it never
    reads differently. Never mutates `state` itself, so the same `state` is
    safe to keep pricing further candidates against.
    """
    hands = list(state.hands)
    for seat, delta in changes.items():
        hands[seat] = [h + d for h, d in zip(state.hands[seat], delta)]
    return replace(state, hands=hands)
