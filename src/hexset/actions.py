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
    trade_with_bank,
)
from .robber import victims
from .state import (
    MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, can_place_settlement, city_count,
    city_upgradeable, road_count, road_placeable, settlement_count, settlement_placeable,
)

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
    its index -- and it is no longer an action at all. The engine clears
    trades once a turn, on the transition into `MAIN`, by asking each
    seat's own private gate (`hexset.trading`) -- nothing about it rides in
    the action space.
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
        if kind is ActionType.MOVE_ROBBER:
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
        # Operand-less: playing a knight no longer names a target or victim
        # (`hexset.game.play_knight_card`) -- it spends the card and enters
        # the same robber phase a seven does, where `MOVE_ROBBER` names the
        # hex and victim instead.
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

    # Each piece limit is a property of the player, so it is read once here
    # rather than rescanned inside every per-edge/per-vertex predicate --
    # `state.road_placeable`/`settlement_placeable`/`city_upgradeable` are
    # exactly the public predicates without that check. Enumerating a board
    # was quadratic in it, which the trade gate's continuation rollouts paid
    # tens of millions of times a game.
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
        # A player with no road pieces left has nothing to place: the card
        # is unplayable, not a free pass. colonist refuses it outright
        # (2026-09-12, three live games at fifteen roads).
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

    `seat is None` asks for the engine's own single actor
    (`hexset.game.to_move`) and is what every offline caller -- the arena, a
    bot's search, the AEC environment -- uses. A live table passes the seat
    that is actually asking, because `Phase.DISCARD` entitles several seats
    to act at once (see `to_move`/`may_act`): seat 3's options there are its
    own hand's, whether or not seat 0 has discarded yet. A seat that may not
    act at all gets an empty list rather than somebody else's options.
    """
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
        # Production phase (rulebook): "You may play a development card
        # before rolling dice" -- every playable card, not only the knight,
        # so this reuses the same `_card_actions` MAIN uses below (it
        # already respects `dev_card_played` and "not one built this turn").
        # Only the card-play actions are offered here: building, buying and
        # trading are Action-phase activities and stay MAIN-only.
        return [Action(ActionType.ROLL)] + _card_actions(game)

    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if not owing:
            return []
        # One card at a time, so the space stays linear in resources rather
        # than combinatorial in hand size.
        #
        # `seat is None` means "whichever seat the engine would serialize to"
        # -- `to_move`'s lowest-indexed owing seat. Any owing seat asking for
        # itself gets its own hand's options instead, which is what makes the
        # round simultaneous rather than queued (the `may_act` guard above
        # has already refused a seat that owes nothing).
        discarding = owing[0] if seat is None else seat
        return [
            Action(ActionType.DISCARD, r)
            for r in range(NUM_RESOURCES)
            if state.hands[discarding][r] > 0
        ]

    if game.phase is Phase.ROBBER:
        return _robber_targets(game, ActionType.MOVE_ROBBER)

    if game.free_roads > 0:
        # A Road Building card resolves when played: both roads are placed
        # before anything else, exactly as before the roll. While roads are
        # owed no other action is legal -- colonist's server takes none, and
        # two live games were lost sending a settlement and a dev-card buy
        # past the debt. With nowhere legal to put them the credit is
        # unusable and the turn continues as normal.
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
    """The flat legality mask for options already enumerated -- what a
    batched consumer stacks per decision without running `legal_actions` a
    second time. `legal_mask` is this over `legal_actions(game)`."""
    mask = [False] * space.size
    for action in options:
        mask[space.index(action)] = True
    return mask


def legal_mask(
    game: Game, space: ActionSpace | None = None, seat: int | None = None
) -> list[bool]:
    return mask_of(space or space_for(game), legal_actions(game, seat))


def apply(game: Game, action: Action, seat: int | None = None) -> None:
    """Execute `action`, optionally on behalf of a named `seat`.

    `seat` only ever names the discarding seat: `DISCARD` is the one action
    whose actor is not fixed by the position (`Phase.DISCARD` owes cards from
    several seats at once and takes them in any order), and the flat action
    space has no operand left to carry it -- an `Action` is a type and two
    ints, and `DISCARD`'s one operand is the resource. Every other action
    belongs to `to_move` by construction and ignores this argument. `None`
    keeps the historical behaviour: the lowest-indexed seat still owing.
    """
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


def victim_of(game: Game, slot: int) -> int | None:
    """Who a robber or knight action steals from, or `None` for nobody.

    Public because the rule decides whether the action draws a hidden card at
    all, and `hexset.mcts` has to know that to tell a chance edge from an
    ordinary one. Two copies of it would drift.
    """
    return None if slot >= game._state.num_players else slot


class Stuck(RuntimeError):
    """A live game has no legal action for the active seat."""


def options_for(game: Game) -> list[Action]:
    """Return legal actions, raising Stuck if the active seat cannot move."""
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {to_move(game)}")
    return options
