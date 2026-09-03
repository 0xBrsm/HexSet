# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from .cards import DevCard
from .devcards import holdings
from .roads import MIN_LONGEST_ROAD, road_lengths
from .state import NO_OWNER, GameState

MIN_LARGEST_ARMY = 3
LONGEST_ROAD_VP = 2
LARGEST_ARMY_VP = 2
WINNING_POINTS = 10


def _award(counts: list[int], holder: int, minimum: int) -> int:
    """Who holds a "most of X" card, given who holds it now.

    A challenger must beat the holder outright, not match them. If the holder
    falls behind and several challengers tie, the card leaves play until one of
    them pulls ahead.
    """
    best = max(counts)
    if best < minimum:
        return NO_OWNER
    if holder != NO_OWNER and counts[holder] == best:
        return holder
    leaders = [p for p, n in enumerate(counts) if n == best]
    return leaders[0] if len(leaders) == 1 else NO_OWNER


def update_longest_road(state: GameState) -> int:
    state.longest_road_holder = _award(
        road_lengths(state), state.longest_road_holder, MIN_LONGEST_ROAD
    )
    return state.longest_road_holder


def update_largest_army(state: GameState) -> int:
    state.largest_army_holder = _award(
        state.knights_played, state.largest_army_holder, MIN_LARGEST_ARMY
    )
    return state.largest_army_holder


def building_points(state: GameState, player: int) -> int:
    # Building values are 1 for a settlement and 2 for a city, which is also
    # what each is worth.
    return sum(
        state.vertex_building[v]
        for v, owner in enumerate(state.vertex_owner)
        if owner == player
    )


def card_points(state: GameState, player: int) -> int:
    """Victory point cards score the moment they are drawn, so they can win a game."""
    return holdings(state, player)[DevCard.VICTORY_POINT]


def award_points(state: GameState, player: int) -> int:
    points = 0
    if state.longest_road_holder == player:
        points += LONGEST_ROAD_VP
    if state.largest_army_holder == player:
        points += LARGEST_ARMY_VP
    return points


def victory_points(state: GameState, player: int) -> int:
    return (
        building_points(state, player)
        + card_points(state, player)
        + award_points(state, player)
    )


def public_victory_points(state: GameState, player: int) -> int:
    """What opponents can see. Victory point cards stay hidden until they win."""
    return building_points(state, player) + award_points(state, player)


def winner(state: GameState) -> int | None:
    for player in range(state.num_players):
        if victory_points(state, player) >= WINNING_POINTS:
            return player
    return None


def relative_points(points: tuple[int, ...]) -> tuple[float, ...]:
    """Each seat's terminal points less the mean of the others, over 10.

    Exactly zero-sum: the per-seat values sum to zero for any input, since
    subtracting the mean of the others is an affine transform whose total
    cancels. That is the property worth having — it says in the reward what the
    game already says, that Catan has one winner and a position is only worth
    what it is worth compared to the table. An action that lifts every seat
    equally earns nothing, which is the whole reason for reading points this
    way rather than absolutely.

    Scaled by the 10 points that win a game, so a seat's reward lands in about
    [-1, +1] and a value head does not have to learn the units.

    **Do not discount this.** With a zero-sum reward roughly half of terminal
    values are negative, and γ < 1 makes a negative terminal cheaper the later
    it arrives — which pays a losing policy to stall. Trading in circles was
    precisely that move, and it is why the action cap exists; a policy cannot
    stall that way any more, since trading is not an action at all
    (`hexset.trading`). Horizon control belongs in what is measured, not in a
    discount factor that quietly changes the objective.

    Lives here rather than in `hexset.mcts` or `hexnet.rewards` (both of which
    use it) because it is a pure function of terminal points and `WINNING_POINTS`
    -- the engine side of the hexset/hexnet boundary, with nothing else pulled
    in. `hexnet.rewards.relative_points` re-exports this definition rather than
    keeping a second one, so this docstring's warning still travels with the
    one function anything trains against.
    """
    seats = len(points)
    if seats < 2:
        raise ValueError("a relative reward needs at least two seats")
    total = sum(points)
    return tuple(
        (own - (total - own) / (seats - 1)) / WINNING_POINTS for own in points
    )
