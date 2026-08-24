from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations_with_replacement
from typing import NamedTuple, Sequence

from .board.terrain import NUM_RESOURCES, Resource
from .cards import DevCard
from .devcards import can_buy
from .economy import Purchase, can_afford, trade_ratios
from .game import (
    MAX_OFFERS_PER_TURN,
    Game,
    Phase,
    accept_trade,
    decline_trade,
    propose_trade,
    to_move,
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
    trade_with_bank,
)
from .robber import victims
from .state import can_place_road, can_place_settlement, can_upgrade_to_city
from .trading import Offer, can_propose


YEAR_OF_PLENTY_PAIRS: tuple[tuple[int, int], ...] = tuple(
    combinations_with_replacement(range(NUM_RESOURCES), 2)
)
ONE_RESOURCE: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(r == resource) for r in range(NUM_RESOURCES))
    for resource in range(NUM_RESOURCES)
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
    ACCEPT_TRADE = 15
    DECLINE_TRADE = 16
    PROPOSE_TRADE = 17


class Action(NamedTuple):
    """`a` and `b` carry the operands: a board index, a resource, or a victim.

    `give` and `want` carry a trade offer, and are the one operand that will not
    fit in an index. An offer is ten numbers, so `PROPOSE_TRADE` occupies a
    single slot in the flat space meaning "an offer is available", and the
    bundles ride alongside it. A network emits them from their own heads rather
    than by choosing among enumerated offers.

    `ask` is who the proposer would rather have take it, best first. An offer
    stops at the first player to accept, so the order is worth something and it
    is the proposer's to choose — preferring the player it costs least to feed
    is a tactic, not a rule, and imposing it in `trading.responders` would bind
    every bot to one opinion. Empty means no preference, and the engine's
    neutral order stands.
    """

    type: ActionType
    a: int = 0
    b: int = 0
    give: tuple[int, ...] = ()
    want: tuple[int, ...] = ()
    ask: tuple[int, ...] = ()


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
        """The action at `index`.

        Lossy for `PROPOSE_TRADE`, which comes back with empty bundles: the
        offer is not recoverable from an index because it was never in one.
        """
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
        ActionType.ACCEPT_TRADE: 1,
        ActionType.DECLINE_TRADE: 1,
        # One slot, not one per offer. Offers are uncapped, so they cannot be
        # enumerated; this bit says only that proposing is available now.
        ActionType.PROPOSE_TRADE: 1,
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
    topology = game.state.board.topology
    return build_space(
        topology.num_vertices,
        topology.num_edges,
        topology.num_hexes,
        game.state.num_players,
    )


def _robber_targets(game: Game, kind: ActionType) -> list[Action]:
    state = game.state
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
    state = game.state
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
    state = game.state
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
    state = game.state
    ratios = trade_ratios(state, game.current_player)
    hand = state.hands[game.current_player]
    return [
        Action(ActionType.BANK_TRADE, give, receive)
        for give in range(NUM_RESOURCES)
        for receive in range(NUM_RESOURCES)
        if give != receive and hand[give] >= ratios[give] and state.bank[receive] > 0
    ]


def _offer_actions(game: Game) -> list[Action]:
    """A representative sample of offers, not every legal one.

    Offers are uncapped, so enumerating them is not possible and this is the
    one place `legal_actions` is a sample rather than the whole set. It covers
    one-for-one trades, which are the overwhelming majority of what gets traded,
    and skips any offer nobody at the table could cover. `apply` will still
    carry out any well-formed offer a stronger policy comes up with.
    """
    state = game.state
    player = game.current_player
    if game.offers_made >= MAX_OFFERS_PER_TURN:
        return []

    # For a one-for-one offer, responder eligibility depends only on what is
    # wanted. Compute that once per resource instead of rebuilding two bundles
    # and walking every opponent for every (give, want) pair.
    wanted_available = tuple(
        any(
            responder != player and state.hands[responder][wanted] > 0
            for responder in range(state.num_players)
        )
        for wanted in range(NUM_RESOURCES)
    )

    out: list[Action] = []
    for given in range(NUM_RESOURCES):
        if not state.hands[player][given]:
            continue
        for wanted in range(NUM_RESOURCES):
            if wanted == given or not wanted_available[wanted]:
                continue
            out.append(
                Action(
                    ActionType.PROPOSE_TRADE,
                    give=ONE_RESOURCE[given],
                    want=ONE_RESOURCE[wanted],
                )
            )
    return out


def is_legal(game: Game, action: Action, options: Sequence[Action]) -> bool:
    """Whether `action` is one of `options` (normally `legal_actions(game)`),
    except for `PROPOSE_TRADE`, checked against `can_propose` instead.

    `legal_actions` only *samples* coverable (give, want) pairs, to keep
    enumeration small — but a proposal nobody can currently cover is still a
    legal move (`propose_trade` has a defensive path for exactly that,
    concluding the offer with no takers rather than raising), so gating it
    on sample membership would reject a legal offer for a reason that was
    never a real rule. Checking `can_propose` directly also sidesteps `ask`
    entirely: it only reorders who gets asked, never who's eligible, so it
    was never part of what made an offer legal in the first place.
    """
    if action.type is ActionType.PROPOSE_TRADE:
        offer = Offer(proposer=game.current_player, give=action.give, want=action.want)
        return can_propose(game.state, offer)
    return action in options


def legal_actions(game: Game) -> list[Action]:
    state = game.state
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

    if game.phase is Phase.TRADE_RESPOND:
        # Only players who can cover the offer are asked, so accepting is
        # always available to whoever is being asked.
        return [Action(ActionType.ACCEPT_TRADE), Action(ActionType.DECLINE_TRADE)]

    if game.phase is Phase.ROBBER:
        return _robber_targets(game, ActionType.MOVE_ROBBER)

    building = _building_actions(game)
    out = (
        building
        + _card_actions(game)
        + _trade_actions(game)
        + _offer_actions(game)
    )
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


def within_offer_budget(
    game: Game, options: Sequence[Action], budget: int | None
) -> list[Action]:
    """Drop proposals once a player has spent an offer budget of its own.

    A budget below `MAX_OFFERS_PER_TURN` is a choice a player makes, not a rule
    the engine enforces, and it has to stay that way: a cap in the engine
    reaches every entrant at once, and a mirror duel cannot see a capability
    everyone receives. Three offers a turn cost about 0.1 victory points and cut
    a game from 2225 actions to 950, which is why a training run wants one.

    Shared by `bots.SearchBot` and `selfplay.Collector` because the two must
    agree. If a policy trained under one budget were evaluated under another,
    the horizon the training assumed would be quietly wrong.
    """
    if budget is None or game.offers_made < budget:
        return list(options)
    kept = [a for a in options if a.type is not ActionType.PROPOSE_TRADE]
    return kept or list(options)


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
        move_robber_to(game, action.a, _victim(game, action.b))
    elif kind is ActionType.PLAY_KNIGHT:
        play_knight_card(game, action.a, _victim(game, action.b))
    elif kind is ActionType.PLAY_MONOPOLY:
        play_monopoly_card(game, Resource(action.a))
    elif kind is ActionType.PLAY_YEAR_OF_PLENTY:
        pair = YEAR_OF_PLENTY_PAIRS[action.a]
        play_year_of_plenty_card(game, [Resource(r) for r in pair])
    elif kind is ActionType.BANK_TRADE:
        trade_with_bank(game, Resource(action.a), Resource(action.b))
    elif kind is ActionType.DISCARD:
        discard_one(game, players_owing_discards(game)[0], Resource(action.a))
    elif kind is ActionType.ACCEPT_TRADE:
        accept_trade(game, to_move(game))
    elif kind is ActionType.DECLINE_TRADE:
        decline_trade(game, to_move(game))
    elif kind is ActionType.PROPOSE_TRADE:
        propose_trade(game, action.give, action.want, ask=action.ask)
    else:
        raise ValueError(f"unhandled action {action}")


def _victim(game: Game, slot: int) -> int | None:
    return None if slot >= game.state.num_players else slot
