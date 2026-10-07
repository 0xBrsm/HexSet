# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations_with_replacement
from collections.abc import Sequence
from typing import NamedTuple

from .board.terrain import NUM_RESOURCES, Resource
from .cards import DevCard
from .devcards import can_buy
from .economy import Purchase, can_afford, trade_ratios
from .game import (
    Game,
    Phase,
    build_city,
    build_road,
    build_settlement,
    buy_development_card,
    discard_one,
    end_turn,
    legal_initial_roads,
    may_act,
    move_robber_to,
    place_initial_road,
    place_initial_settlement,
    play_knight_card,
    play_monopoly_card,
    play_road_building_card,
    pending_free_roads,
    play_year_of_plenty_card,
    players_owing_discards,
    roll_dice,
    run_trade_event,
    to_move,
    trade_with_bank,
)
from .robber import allowed_targets, may_stand, victims
from .state import (
    MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, can_place_settlement, city_count,
    city_upgradeable, road_count, road_placeable, settlement_count, settlement_placeable,
)

__all__ = [
    "YEAR_OF_PLENTY_PAIRS",
    "ActionType",
    "Action",
    "ActionSpace",
    "build_space",
    "space_for",
    "legal_actions",
    "mask_of",
    "legal_mask",
    "apply",
    "victim_of",
    "Stuck",
    "options_for",
]


YEAR_OF_PLENTY_PAIRS: tuple[tuple[int, int], ...] = tuple(
    combinations_with_replacement(range(NUM_RESOURCES), 2)
)


class ActionType(IntEnum):
    """What an `Action` does; `a` and `b` are its operands."""

    ROLL = 0
    END_TURN = 1
    BUY_DEV_CARD = 2
    PLAY_ROAD_BUILDING = 3
    SETUP_SETTLEMENT = 4
    SETUP_ROAD = 5
    BUILD_ROAD = 6
    BUILD_SETTLEMENT = 7
    BUILD_CITY = 8
    MOVE_ROBBER = 9
    PLAY_KNIGHT = 10
    PLAY_MONOPOLY = 11
    PLAY_YEAR_OF_PLENTY = 12
    BANK_TRADE = 13
    DISCARD = 14


class Action(NamedTuple):
    """`a` and `b` carry the operands: a board index, a resource, or a victim.
    Two operands and nothing else, so every action fits in a flat index.
    Player-to-player trading is not an action; see `hexset.trading`."""

    type: ActionType
    a: int = 0
    b: int = 0


@dataclass(frozen=True)
class ActionSpace:
    """A flat index over typed actions, sized from the board rather than fixed:
    board-local actions get one slot per node, so a graph model reads the policy
    straight off vertex, edge and hex embeddings."""

    num_vertices: int
    num_edges: int
    num_hexes: int
    num_players: int
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    size: int

    def index(self, action: Action) -> int:
        stride = self._stride(action.type)
        return self.offsets[action.type] + action.a * stride + action.b

    def decode(self, index: int) -> Action:
        """The action at `index`. Exactly invertible with `index`."""
        for kind in reversed(ActionType):
            start = self.offsets[kind]
            if index >= start:
                stride = self._stride(kind)
                local = index - start
                return Action(kind, local // stride, local % stride)
        raise ValueError(f"no action at index {index}")

    def _stride(self, kind: ActionType) -> int:
        if kind is ActionType.MOVE_ROBBER:
            # One slot per (hex, victim); the last victim slot is "nobody".
            return self.num_players + 1
        if kind is ActionType.BANK_TRADE:
            return NUM_RESOURCES
        return 1


def build_space(num_vertices: int, num_edges: int, num_hexes: int, players: int) -> ActionSpace:
    """The `ActionSpace` for a board of this many vertices, edges and hexes
    seating `players`, its blocks laid out in `ActionType` order."""
    robber = num_hexes * (players + 1)
    sizes = {
        ActionType.ROLL: 1,
        ActionType.END_TURN: 1,
        ActionType.BUY_DEV_CARD: 1,
        ActionType.PLAY_ROAD_BUILDING: 1,
        ActionType.SETUP_SETTLEMENT: num_vertices,
        ActionType.SETUP_ROAD: num_edges,
        ActionType.BUILD_ROAD: num_edges,
        ActionType.BUILD_SETTLEMENT: num_vertices,
        ActionType.BUILD_CITY: num_vertices,
        ActionType.MOVE_ROBBER: robber,
        # Operand-less: a knight spends the card and enters the robber phase,
        # where `MOVE_ROBBER` names the hex and victim.
        ActionType.PLAY_KNIGHT: 1,
        ActionType.PLAY_MONOPOLY: NUM_RESOURCES,
        ActionType.PLAY_YEAR_OF_PLENTY: len(YEAR_OF_PLENTY_PAIRS),
        ActionType.BANK_TRADE: NUM_RESOURCES * NUM_RESOURCES,
        ActionType.DISCARD: NUM_RESOURCES,
    }
    ordered = tuple(sizes[kind] for kind in ActionType)
    offsets = []
    running = 0
    for count in ordered:
        offsets.append(running)
        running += count
    return ActionSpace(
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_hexes=num_hexes,
        num_players=players,
        sizes=ordered,
        offsets=tuple(offsets),
        size=running,
    )


def space_for(game: Game) -> ActionSpace:
    """The `ActionSpace` for `game`'s board and seat count."""
    topology = game._state.board.topology
    return build_space(
        topology.num_vertices,
        topology.num_edges,
        topology.num_hexes,
        game._state.num_players,
    )


def _robber_targets(game: Game, kind: ActionType) -> list[Action]:
    state = game._state
    # A host's rule for this move, else the game type's (friendly robber).
    allowed = allowed_targets(state, game.current_player, game.robber_allowed)
    out = []
    for h in range(state.board.num_hexes):
        if not may_stand(state, h) or (allowed is not None and h not in allowed):
            continue
        reachable = victims(state, h, game.current_player)
        if reachable:
            out.extend(Action(kind, h, v) for v in reachable)
        else:
            out.append(Action(kind, h, state.num_players))
    return out


def _building_actions(game: Game) -> list[Action]:
    state = game._state
    player = game.current_player
    topology = state.board.topology
    out: list[Action] = []

    # Each piece limit is read once here rather than rescanned inside every
    # per-node predicate, which would make enumeration quadratic;
    # `road_placeable` and friends are the public predicates without the check.
    if (game.free_roads > 0 or can_afford(state, player, Purchase.ROAD)) and (
        road_count(state, player) < MAX_ROADS
    ):
        out.extend(
            Action(ActionType.BUILD_ROAD, e)
            for e in range(topology.num_edges)
            if road_placeable(state, player, e)
        )
    if (
        can_afford(state, player, Purchase.SETTLEMENT)
        and settlement_count(state, player) < MAX_SETTLEMENTS
    ):
        out.extend(
            Action(ActionType.BUILD_SETTLEMENT, v)
            for v in range(topology.num_vertices)
            if settlement_placeable(state, player, v)
        )
    if can_afford(state, player, Purchase.CITY) and city_count(state, player) < MAX_CITIES:
        out.extend(
            Action(ActionType.BUILD_CITY, v)
            for v in range(topology.num_vertices)
            if city_upgradeable(state, player, v)
        )
    return out


def _card_actions(game: Game) -> list[Action]:
    state = game._state
    player = game.current_player
    out: list[Action] = []
    if game.dev_card_played:
        return out

    held = state.dev_cards[player]
    if held[DevCard.KNIGHT]:
        out.append(Action(ActionType.PLAY_KNIGHT))
    if held[DevCard.ROAD_BUILDING] and road_count(state, player) < MAX_ROADS:
        # With no road pieces left the card is unplayable, not a free pass;
        # with one, it places one road (the printed rule).
        out.append(Action(ActionType.PLAY_ROAD_BUILDING))
    if held[DevCard.MONOPOLY]:
        out.extend(Action(ActionType.PLAY_MONOPOLY, r) for r in range(NUM_RESOURCES))
    if held[DevCard.YEAR_OF_PLENTY]:
        out.extend(
            Action(ActionType.PLAY_YEAR_OF_PLENTY, i)
            for i, pair in enumerate(YEAR_OF_PLENTY_PAIRS)
            if all(state.bank[r] >= pair.count(r) for r in set(pair))
        )
    return out


def _trade_actions(game: Game) -> list[Action]:
    state = game._state
    ratios = trade_ratios(state, game.current_player)
    hand = state.hands[game.current_player]
    return [
        Action(ActionType.BANK_TRADE, give, receive)
        for give in range(NUM_RESOURCES)
        for receive in range(NUM_RESOURCES)
        if give != receive and hand[give] >= ratios[give] and state.bank[receive] > 0
    ]


def legal_actions(game: Game, seat: int | None = None) -> list[Action]:
    """What may be played right now, optionally as a named `seat`.

    `seat is None` asks for the engine's single actor (`hexset.game.to_move`); a
    live table passes the seat actually asking, because `Phase.DISCARD` entitles
    several at once. A seat that may not act gets an empty list."""
    state = game._state
    player = game.current_player

    if game.phase is Phase.GAME_OVER:
        return []

    if seat is not None and not may_act(game, seat):
        return []

    if game.phase is Phase.SETUP_SETTLEMENT:
        return [
            Action(ActionType.SETUP_SETTLEMENT, v)
            for v in range(state.board.topology.num_vertices)
            if can_place_settlement(state, player, v, connected=False)
        ]

    if game.phase is Phase.SETUP_ROAD:
        return [Action(ActionType.SETUP_ROAD, e) for e in legal_initial_roads(game)]

    if game.phase is Phase.ROLL:
        roads = pending_free_roads(game)
        if roads:
            return [Action(ActionType.BUILD_ROAD, edge) for edge in roads]
        # Rulebook: any development card may be played before rolling, not only
        # the knight; building, buying and trading stay MAIN-only.
        return [Action(ActionType.ROLL)] + _card_actions(game)

    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if not owing:
            return []
        # One card at a time, so the space stays linear in resources rather than
        # combinatorial in hand size. `seat is None` means `to_move`'s
        # lowest-indexed owing seat; any owing seat asking for itself gets its
        # own options, making the round simultaneous.
        discarding = owing[0] if seat is None else seat
        return [
            Action(ActionType.DISCARD, r)
            for r in range(NUM_RESOURCES)
            if state.hands[discarding][r] > 0
        ]

    if game.phase is Phase.ROBBER:
        return _robber_targets(game, ActionType.MOVE_ROBBER)

    if game.free_roads > 0:
        # A Road Building card resolves when played: while roads are owed no
        # other action is legal, and with nowhere to put them the credit lapses.
        roads = pending_free_roads(game)
        if roads:
            return [Action(ActionType.BUILD_ROAD, edge) for edge in roads]

    building = _building_actions(game)
    out = building + _card_actions(game) + _trade_actions(game)
    if can_buy(state, player):
        out.append(Action(ActionType.BUY_DEV_CARD))
    out.append(Action(ActionType.END_TURN))
    return out


def mask_of(space: ActionSpace, options: Sequence[Action]) -> list[bool]:
    """The flat legality mask for options already enumerated, for a caller that
    must not run `legal_actions` twice."""
    mask = [False] * space.size
    for action in options:
        mask[space.index(action)] = True
    return mask


def legal_mask(
    game: Game, space: ActionSpace | None = None, seat: int | None = None
) -> list[bool]:
    """The flat legality mask of `legal_actions(game, seat)` over `space`,
    `space_for(game)` when none is given."""
    return mask_of(space or space_for(game), legal_actions(game, seat))


def apply(game: Game, action: Action, seat: int | None = None) -> None:
    """Execute `action`, optionally on behalf of a named `seat`.

    `seat` only ever names the discarding seat: `DISCARD` is the one action whose
    actor the position does not fix. `None` means the lowest-indexed owing seat.

    After a main-phase action that leaves the game in `MAIN` (and not in the
    middle of a Road Building card's free roads, which a credit with nowhere
    left to go is not), the turn's trade window is offered again:
    `run_trade_event` no-ops once the event is used, and only a gate
    declaring `trade_now` can still have it open."""
    kind = action.type
    was_main = game.phase is Phase.MAIN
    if kind is ActionType.ROLL:
        roll_dice(game)
    elif kind is ActionType.END_TURN:
        end_turn(game)
    elif kind is ActionType.BUY_DEV_CARD:
        buy_development_card(game)
    elif kind is ActionType.PLAY_ROAD_BUILDING:
        play_road_building_card(game)
    elif kind is ActionType.SETUP_SETTLEMENT:
        place_initial_settlement(game, action.a)
    elif kind is ActionType.SETUP_ROAD:
        place_initial_road(game, action.a)
    elif kind is ActionType.BUILD_ROAD:
        build_road(game, action.a)
    elif kind is ActionType.BUILD_SETTLEMENT:
        build_settlement(game, action.a)
    elif kind is ActionType.BUILD_CITY:
        build_city(game, action.a)
    elif kind is ActionType.MOVE_ROBBER:
        move_robber_to(game, action.a, victim_of(game, action.b))
    elif kind is ActionType.PLAY_KNIGHT:
        play_knight_card(game)
    elif kind is ActionType.PLAY_MONOPOLY:
        play_monopoly_card(game, Resource(action.a))
    elif kind is ActionType.PLAY_YEAR_OF_PLENTY:
        pair = YEAR_OF_PLENTY_PAIRS[action.a]
        play_year_of_plenty_card(game, [Resource(r) for r in pair])
    elif kind is ActionType.BANK_TRADE:
        trade_with_bank(game, Resource(action.a), Resource(action.b))
    elif kind is ActionType.DISCARD:
        discarding = players_owing_discards(game)[0] if seat is None else seat
        discard_one(game, discarding, Resource(action.a))
    else:
        raise ValueError(f"unhandled action {action}")
    if was_main and game.phase is Phase.MAIN and not pending_free_roads(game):
        run_trade_event(game)


def victim_of(game: Game, slot: int) -> int | None:
    """Who a robber or knight action steals from, or `None` for nobody. Public
    because `hexset.mcts` needs it to tell a chance edge from an ordinary one."""
    return None if slot >= game._state.num_players else slot


class Stuck(RuntimeError):
    """A live game has no legal action for the active seat."""


def options_for(game: Game) -> list[Action]:
    """Return legal actions, raising Stuck if the active seat cannot move."""
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {to_move(game)}")
    return options
