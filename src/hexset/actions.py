# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations_with_replacement
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
    move_robber_to,
    place_initial_road,
    place_initial_settlement,
    play_knight_card,
    play_monopoly_card,
    play_road_building_card,
    play_year_of_plenty_card,
    players_owing_discards,
    roll_dice,
    run_pending_event,
    trade_with_bank,
)
from .robber import victims
from .state import can_place_road, can_place_settlement, can_upgrade_to_city


YEAR_OF_PLENTY_PAIRS: tuple[tuple[int, int], ...] = tuple(
    combinations_with_replacement(range(NUM_RESOURCES), 2)
)


class ActionType(IntEnum):
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

    Two operands and nothing else: every action in this space fits in a flat
    index. Player-to-player trading used to be the exception -- an offer is
    ten numbers, so a propose action carried `give`/`want`/`ask` alongside
    its index -- and it is no longer an action at all. Trades clear once a turn
    from the seats' published valuation vectors (`hexset.trading`), which are
    observation, not action.
    """

    type: ActionType
    a: int = 0
    b: int = 0


@dataclass(frozen=True)
class ActionSpace:
    """A flat index over typed actions, sized from the board rather than fixed.

    Board-local actions are laid out one slot per node, so a graph model can
    read the policy straight off vertex, edge and hex embeddings, and a larger
    Seafarers map widens the space without changing any of this code.
    """

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
        if kind in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
            # One slot per (hex, victim), with the last victim slot meaning
            # "nobody to rob".
            return self.num_players + 1
        if kind is ActionType.BANK_TRADE:
            return NUM_RESOURCES
        return 1


def build_space(num_vertices: int, num_edges: int, num_hexes: int, players: int) -> ActionSpace:
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
        ActionType.PLAY_KNIGHT: robber,
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
    topology = game._state.board.topology
    return build_space(
        topology.num_vertices,
        topology.num_edges,
        topology.num_hexes,
        game._state.num_players,
    )


def _robber_targets(game: Game, kind: ActionType) -> list[Action]:
    state = game._state
    out = []
    for h in range(state.board.num_hexes):
        if h == state.robber:
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

    if game.free_roads > 0 or can_afford(state, player, Purchase.ROAD):
        out.extend(
            Action(ActionType.BUILD_ROAD, e)
            for e in range(topology.num_edges)
            if can_place_road(state, player, e)
        )
    if can_afford(state, player, Purchase.SETTLEMENT):
        out.extend(
            Action(ActionType.BUILD_SETTLEMENT, v)
            for v in range(topology.num_vertices)
            if can_place_settlement(state, player, v)
        )
    if can_afford(state, player, Purchase.CITY):
        out.extend(
            Action(ActionType.BUILD_CITY, v)
            for v in range(topology.num_vertices)
            if can_upgrade_to_city(state, player, v)
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
        out.extend(_robber_targets(game, ActionType.PLAY_KNIGHT))
    if held[DevCard.ROAD_BUILDING]:
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


def legal_actions(game: Game) -> list[Action]:
    # One of the three event-trigger points (`Game.event_pending`'s
    # docstring): the current player's first `legal_actions` call of the
    # turn fires this turn's pending trade event, if one is still
    # outstanding, before the options below are computed off the
    # (possibly now different) hand -- the PI amendment "publish points
    # and the event trigger".
    if game.phase is Phase.MAIN:
        run_pending_event(game)

    state = game._state
    player = game.current_player

    if game.phase is Phase.GAME_OVER:
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
        out = [Action(ActionType.ROLL)]
        if not game.dev_card_played and state.dev_cards[player][DevCard.KNIGHT]:
            out.extend(_robber_targets(game, ActionType.PLAY_KNIGHT))
        return out

    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if not owing:
            return []
        # One card at a time, so the space stays linear in resources rather
        # than combinatorial in hand size.
        discarding = owing[0]
        return [
            Action(ActionType.DISCARD, r)
            for r in range(NUM_RESOURCES)
            if state.hands[discarding][r] > 0
        ]

    if game.phase is Phase.ROBBER:
        return _robber_targets(game, ActionType.MOVE_ROBBER)

    building = _building_actions(game)
    out = building + _card_actions(game) + _trade_actions(game)
    if can_buy(state, player):
        out.append(Action(ActionType.BUY_DEV_CARD))

    # Free roads must be placed before ending the turn, unless there is nowhere
    # legal to put them — otherwise the player would have no legal action at all.
    owed_roads = game.free_roads > 0 and any(
        a.type is ActionType.BUILD_ROAD for a in building
    )
    if not owed_roads:
        out.append(Action(ActionType.END_TURN))
    return out


def legal_mask(game: Game, space: ActionSpace | None = None) -> list[bool]:
    space = space or space_for(game)
    mask = [False] * space.size
    for action in legal_actions(game):
        mask[space.index(action)] = True
    return mask


def apply(game: Game, action: Action) -> None:
    kind = action.type
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
        play_knight_card(game, action.a, victim_of(game, action.b))
    elif kind is ActionType.PLAY_MONOPOLY:
        play_monopoly_card(game, Resource(action.a))
    elif kind is ActionType.PLAY_YEAR_OF_PLENTY:
        pair = YEAR_OF_PLENTY_PAIRS[action.a]
        play_year_of_plenty_card(game, [Resource(r) for r in pair])
    elif kind is ActionType.BANK_TRADE:
        trade_with_bank(game, Resource(action.a), Resource(action.b))
    elif kind is ActionType.DISCARD:
        discard_one(game, players_owing_discards(game)[0], Resource(action.a))
    else:
        raise ValueError(f"unhandled action {action}")


def victim_of(game: Game, slot: int) -> int | None:
    """Who a robber or knight action steals from, or `None` for nobody.

    Public because the rule decides whether the action draws a hidden card at
    all, and `hexset.mcts` has to know that to tell a chance edge from an
    ordinary one. Two copies of it would drift.
    """
    return None if slot >= game._state.num_players else slot
