from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

from .board.board import Board, pips, scarce_resources
from .board.ports import BASE_TRADE_RATIO
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from .economy import COSTS, Purchase
from .game import Game
from .robber import DISCARD_THRESHOLD
from .state import GameState
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
    # Every arena number recorded before 2026-08-15 was measured with this at
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
# in `hexset_ui.placement`, at `production / ROLLS` victory points per pip. Derived
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
