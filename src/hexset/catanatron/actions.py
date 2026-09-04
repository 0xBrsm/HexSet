# SPDX-License-Identifier: GPL-3.0-only
"""Maps a dev-catan `Action` (the bot's choice) back onto one of catanatron's
`playable_actions` for the same decision.

Every case here searches `playable_actions` for a match rather than
constructing a catanatron `Action` from scratch. That is deliberate: a
constructed action that happens to be wrong would either be silently accepted
(if it coincidentally validates) or raise deep inside catanatron with no
indication of which translation was at fault. Searching means a missing match
fails right here, at the boundary, with the actual mismatch visible.

`PLAY_KNIGHT` is the one case not handled by a single lookup, because dev-catan
bundles "play the knight" and "move the robber" into one decision where
catanatron asks them as two: it maps to whichever half catanatron is offering
right now. `player.py` (our bot inside catanatron) plays the first half and
replays its own choice into the second; `bot.py` (a catanatron bot at our
table) lets catanatron take both halves and reassembles them into one
`PLAY_KNIGHT`. Both directions match against the same two entries here.
"""

from __future__ import annotations

from hexset.actions import Action as OurAction, ActionType as OurActionType, YEAR_OF_PLENTY_PAIRS
from hexset.game import Game as OurGame

from catanatron.models.enums import Action as TheirAction, ActionType as TheirActionType

from .board import BoardMapping
from .names import RESOURCE_NAMES
from .state import Seating

_NO_VALUE = {
    OurActionType.ROLL: TheirActionType.ROLL,
    OurActionType.END_TURN: TheirActionType.END_TURN,
    OurActionType.BUY_DEV_CARD: TheirActionType.BUY_DEVELOPMENT_CARD,
    OurActionType.PLAY_ROAD_BUILDING: TheirActionType.PLAY_ROAD_BUILDING,
}


def find(playable_actions, predicate, description: str) -> TheirAction:
    for action in playable_actions:
        if predicate(action):
            return action
    raise ValueError(f"no catanatron playable_action matched: {description}")


def move_robber(
    hex_index: int,
    victim_seat: int,
    num_players: int,
    mapping: BoardMapping,
    seats: Seating,
    playable_actions,
) -> TheirAction:
    coord = mapping.coord_of[hex_index]
    victim_color = None if victim_seat >= num_players else seats.color_of[victim_seat]

    def matches(action: TheirAction) -> bool:
        if action.action_type is not TheirActionType.MOVE_ROBBER:
            return False
        their_coord, their_victim = action.value
        return their_coord == coord and their_victim == victim_color

    return find(playable_actions, matches, f"MOVE_ROBBER {coord} victim={victim_color}")


def to_catanatron(
    our_action: OurAction,
    our_game: OurGame,
    mapping: BoardMapping,
    seats: Seating,
    playable_actions,
) -> TheirAction:
    kind = our_action.type

    if kind in _NO_VALUE:
        their_kind = _NO_VALUE[kind]
        return find(
            playable_actions,
            lambda a: a.action_type is their_kind,
            their_kind.name,
        )

    if kind in (OurActionType.SETUP_SETTLEMENT, OurActionType.BUILD_SETTLEMENT):
        node = mapping.node_of[our_action.a]
        return find(
            playable_actions,
            lambda a: a.action_type is TheirActionType.BUILD_SETTLEMENT and a.value == node,
            f"BUILD_SETTLEMENT {node}",
        )

    if kind is OurActionType.BUILD_CITY:
        node = mapping.node_of[our_action.a]
        return find(
            playable_actions,
            lambda a: a.action_type is TheirActionType.BUILD_CITY and a.value == node,
            f"BUILD_CITY {node}",
        )

    if kind in (OurActionType.SETUP_ROAD, OurActionType.BUILD_ROAD):
        node_a, node_b = mapping.catanatron_edge_of[our_action.a]
        wanted = {node_a, node_b}
        return find(
            playable_actions,
            lambda a: a.action_type is TheirActionType.BUILD_ROAD and set(a.value) == wanted,
            f"BUILD_ROAD {wanted}",
        )

    if kind in (OurActionType.MOVE_ROBBER, OurActionType.PLAY_KNIGHT):
        # `PLAY_KNIGHT` is two catanatron decisions and one of ours (module
        # docstring), so it maps to whichever half is on the table right now
        # and otherwise resolves exactly as a bare robber move does.
        knight = next(
            (a for a in playable_actions if a.action_type is TheirActionType.PLAY_KNIGHT_CARD),
            None,
        )
        if kind is OurActionType.PLAY_KNIGHT and knight is not None:
            return knight
        # true state: whichever direction this is translating for rebuilds the
        # position fresh every decision, so this is the sanctioned read.
        num_players = our_game.state(0, hidden=False).num_players
        return move_robber(
            our_action.a, our_action.b, num_players, mapping, seats, playable_actions
        )

    if kind is OurActionType.PLAY_MONOPOLY:
        resource = RESOURCE_NAMES[our_action.a]
        return find(
            playable_actions,
            lambda a: a.action_type is TheirActionType.PLAY_MONOPOLY and a.value == resource,
            f"PLAY_MONOPOLY {resource}",
        )

    if kind is OurActionType.PLAY_YEAR_OF_PLENTY:
        r1, r2 = YEAR_OF_PLENTY_PAIRS[our_action.a]
        wanted = tuple(sorted((RESOURCE_NAMES[r1], RESOURCE_NAMES[r2])))
        return find(
            playable_actions,
            lambda a: (
                a.action_type is TheirActionType.PLAY_YEAR_OF_PLENTY
                and len(a.value) == 2
                and tuple(sorted(a.value)) == wanted
            ),
            f"PLAY_YEAR_OF_PLENTY {wanted}",
        )

    if kind is OurActionType.BANK_TRADE:
        give, receive = RESOURCE_NAMES[our_action.a], RESOURCE_NAMES[our_action.b]

        def matches_trade(a: TheirAction) -> bool:
            if a.action_type is not TheirActionType.MARITIME_TRADE:
                return False
            given, asked = a.value[:4], a.value[4]
            return asked == receive and give in given and all(g in (give, None) for g in given)

        return find(playable_actions, matches_trade, f"MARITIME_TRADE {give}->{receive}")

    if kind is OurActionType.DISCARD:
        resource = RESOURCE_NAMES[our_action.a]
        return find(
            playable_actions,
            lambda a: a.action_type is TheirActionType.DISCARD_RESOURCE and a.value == resource,
            f"DISCARD_RESOURCE {resource}",
        )

    raise NotImplementedError(
        f"{kind.name} is out of scope for this bridge (player-to-player trading; see state.py)"
    )
