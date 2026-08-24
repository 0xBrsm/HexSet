"""A second evaluation, kept for comparison. `catan.evaluate` is the default.

The term set is reimplemented from the design used by catanatron's value
function, described rather than copied — it is GPLv3 and this project needs to
stay unencumbered.

**It is not the default, and why is the useful part.** Tuned, it plays the
default evaluation to a dead heat: 51.3% over 1000 games, 95% CI [48.2%, 54.4%].
It also generates data at 40 games/sec against the default's 84, so it loses on
the one requirement that separated them.

That two substantially different feature sets — nine blended terms there, twelve
tiered ones with relative opponent production here — land at identical strength
once each is tuned is evidence that the ceiling is the approach and not the
features. A handcrafted linear evaluation at one ply is what it is, and more
feature work has a predictably poor return. This file is kept as the control
that says so, and as a second baseline to measure the network against.

Two structural ideas came from that reading and are the reason this is not just
the default file with extra terms.

*Weights are a priority order, not a blend.* Victory points dominate everything,
then production, then reachable production, and so on down. Encoding a hierarchy
in the magnitudes says "never trade a point for a pip", which comparable-sized
coefficients cannot express — and a search over blended weights had already
saturated with nothing left to find.

*Some terms are relative.* Winning is being ahead, not being good. Own
production is scored against the strongest opponent's at the same magnitude,
so a position that helps everyone equally is correctly worth nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .board.board import Board, pips
from .board.terrain import TERRAIN_RESOURCE
from .devcards import holdings
from .economy import COSTS, Purchase
from .game import Game
from .robber import DISCARD_THRESHOLD
from .roads import longest_road
from .state import NO_OWNER, GameState
from .victory import card_points, public_victory_points

ROLLS = 36

# How far the road network is searched for places worth expanding to. Two steps
# is one road beyond what is already buildable.
REACH_STEPS = 2

# A resource you produce at all is worth roughly this many pips of a resource
# you already have, because trading away a surplus costs at least two for one.
VARIETY_PIPS = 4


@dataclass(frozen=True)
class Weights:
    """Scoring weights as magnitude tiers, coarsest first.

    The spacing is the point. A tier is chosen so that no amount of the terms
    below it can outweigh one unit of it: a victory point beats any production
    advantage, production beats any amount of reachable production, and so on.
    Hand-designed rather than fitted, because a search cannot discover a
    hierarchy — it can only tune within one, which is what `catan.tuning` is
    for once these are in place.
    """

    # Tier 1: the only thing that actually ends a game. Twelve points times this
    # has to stay well inside float64's exact range, or the tie-break tiers stop
    # being representable at all once a leader is close to winning.
    victory_point: float = 1e12

    # Tier 2: what is being produced now, and by the strongest opponent. The
    # fit cut the opponent term to a third of the tier, so a rival's production
    # matters, but less than a rival's is worth giving up your own for.
    production: float = 1e8
    enemy_production: float = -3.085e7

    # Tier 3: production that expansion would unlock.
    reachable_production_1: float = 1.131e4

    # Tier 4: whether there is anywhere left to build at all.
    buildable_nodes: float = 902.9

    # Tier 5: the hand, valued by what it is close to buying.
    hand_synergy: float = 34.68

    # Tier 6: slow accumulations.
    longest_road: float = 8.218
    hand_devs: float = 11.85
    army_size: float = 14.21

    # Tier 7: tie-breaks.
    num_tiles: float = 1.0
    hand_resources: float = 1.14
    discard_penalty: float = -6.55


TERM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Weights))

# Longest road only earns its weight once there is nowhere left to settle.
# Otherwise expanding is worth more than extending, and a bot that values road
# length all game long builds roads instead of settlements.
BOXED_IN_ROAD_WEIGHT = 0.1


@dataclass(frozen=True)
class Snapshot:
    """Per-seat quantities that are cheaper to compute for the whole table at once."""

    production: list[float]
    bare_production: list[float]
    sources: list[set[int]]
    roads: list[int]


class Evaluator:
    """Scores every seat, not a scalar, matching the planned value head.

    Per-vertex yields are precomputed from the board, which never changes.
    """

    def __init__(self, board: Board, weights: Weights | None = None) -> None:
        self.weights = weights or Weights()
        self.vector = tuple(getattr(self.weights, name) for name in TERM_NAMES)
        topology = board.topology
        self.yields = tuple(
            tuple(
                (h, TERRAIN_RESOURCE[board.terrain[h]], pips(board.tokens[h]))
                for h in topology.vertex_hexes[v]
                if pips(board.tokens[h])
            )
            for v in range(topology.num_vertices)
        )

    @classmethod
    def for_game(cls, game: Game, weights: Weights | None = None) -> Evaluator:
        return cls(game.state.board, weights)

    def snapshot(self, state: GameState) -> Snapshot:
        """Everything every seat needs, in one pass over the board.

        Scoring four seats used to walk all vertices and edges once per seat,
        and the enemy-production term walked them once per opponent per seat —
        quadratic in players over the whole board. This is the hot loop of every
        search node, so it is worth the extra structure.
        """
        players = state.num_players
        pips_of = [0] * players
        kinds: list[set[int]] = [set() for _ in range(players)]
        sources: list[set[int]] = [set() for _ in range(players)]
        roads = [0] * players
        robber = state.robber

        for v, owner in enumerate(state.vertex_owner):
            if owner == NO_OWNER:
                continue
            sources[owner].add(v)
            buildings = state.vertex_building[v]
            for h, resource, p in self.yields[v]:
                if h == robber:
                    continue
                pips_of[owner] += buildings * p
                if resource is not None:
                    kinds[owner].add(resource)

        edges = state.board.topology.edges
        for e, owner in enumerate(state.edge_owner):
            if owner != NO_OWNER:
                roads[owner] += 1
                sources[owner].update(edges[e])

        return Snapshot(
            production=[
                (pips_of[p] + len(kinds[p]) * VARIETY_PIPS) / ROLLS
                for p in range(players)
            ],
            bare_production=[pips_of[p] / ROLLS for p in range(players)],
            sources=sources,
            roads=roads,
        )

    def reachable(self, state: GameState, player: int, sources: set[int]) -> tuple[float, int]:
        """Production reachable within `REACH_STEPS` roads, and settleable spots.

        Own buildings stay in the reachable set. This is the trap an earlier
        version of this file fell into: if settling a junction removed it from
        your own reachable production, the score argued against the expansion
        that scores points, and ablation showed the term was worse than useless.
        Keeping what you have built means building can only ever add.
        """
        neighbours = state.board.topology.vertex_neighbors
        owners = state.vertex_owner
        building = state.vertex_building

        counted = set(sources)
        frontier = list(sources)
        spots = 0
        total = 0.0

        for step in range(REACH_STEPS + 1):
            following = []
            for v in frontier:
                if owners[v] == player:
                    # Already ours: its production is counted by `production`.
                    pass
                elif not building[v] and not any(building[n] for n in neighbours[v]):
                    spots += 1
                    total += sum(p for _, _, p in self.yields[v])
                if step == REACH_STEPS:
                    continue
                for w in neighbours[v]:
                    if w in counted:
                        continue
                    # An opponent's building stops a road passing through.
                    if owners[w] != NO_OWNER and owners[w] != player:
                        continue
                    counted.add(w)
                    following.append(w)
            frontier = following

        return total / ROLLS, spots

    def hand_synergy(self, state: GameState, player: int) -> float:
        """How close the hand is to buying a settlement and a city at once.

        Two targets rather than the nearest one, so a hand is rewarded for
        being broadly useful instead of for hoarding towards a single build.
        """
        hand = state.hands[player]
        together = 0.0
        for purchase in (Purchase.SETTLEMENT, Purchase.CITY):
            cost = COSTS[purchase]
            have = sum(min(hand[r], n) for r, n in enumerate(cost) if n)
            together += have / sum(cost)
        return together / 2.0

    def touched_tiles(self, state: GameState, player: int) -> int:
        """Distinct hexes the player has a building on.

        Breadth of footprint rather than yield: sitting on many hexes is what
        makes a player hard to shut out with the robber and hard to block.
        """
        topology = state.board.topology
        touched = set()
        for v, owner in enumerate(state.vertex_owner):
            if owner == player:
                touched.update(topology.vertex_hexes[v])
        return len(touched)

    def terms(
        self,
        state: GameState,
        player: int,
        *,
        knower: int | None = None,
        shared: Snapshot | None = None,
    ) -> tuple[float, ...]:
        """The raw term values, in `TERM_NAMES` order.

        These are exactly what the weights multiply, so anything fitting
        coefficients is fitting the model the search actually uses.
        """
        shared = shared or self.snapshot(state)
        reach, spots = self.reachable(state, player, shared.sources[player])
        held = sum(state.hands[player])

        points = public_victory_points(state, player)
        devs = 0
        if player == knower:
            points += card_points(state, player)
            devs = sum(holdings(state, player))

        # The true longest trail is an exhaustive search, and it only earns its
        # tier once there is nowhere left to settle. While expansion is still
        # open the term is scaled down to a tie-break anyway, so the far cheaper
        # road count stands in for it there.
        if spots == 0:
            road_value = float(longest_road(state, player))
        else:
            road_value = BOXED_IN_ROAD_WEIGHT * shared.roads[player]

        return (
            points,
            shared.production[player],
            max(
                (shared.bare_production[p] for p in range(state.num_players) if p != player),
                default=0.0,
            ),
            reach,
            spots,
            self.hand_synergy(state, player),
            road_value,
            devs,
            state.knights_played[player],
            self.touched_tiles(state, player),
            held,
            1.0 if held > DISCARD_THRESHOLD else 0.0,
        )

    def score(
        self,
        state: GameState,
        player: int,
        *,
        knower: int | None = None,
        shared: Snapshot | None = None,
    ) -> float:
        values = self.terms(state, player, knower=knower, shared=shared)
        total = 0.0
        for weight, value in zip(self.vector, values):
            total += weight * value
        return total

    def evaluate(self, state: GameState, knower: int | None = None) -> list[float]:
        """Score every seat. `knower` is the seat whose hidden cards may be counted."""
        shared = self.snapshot(state)
        return [
            self.score(state, p, knower=knower, shared=shared)
            for p in range(state.num_players)
        ]
