# SPDX-License-Identifier: GPL-3.0-only
"""A live catanatron `Game` read back into a hexset `Game` -- the round-trip
oracle for `hexset.catanatron._state.to_catanatron`.

Mirroring a position out and back confirms the two engines agree on
everything both represent; nothing outside the tests reads a catanatron
position into hexset.
"""

from __future__ import annotations

import random

from hexset.board.terrain import NUM_RESOURCES
from hexset.cards import PLAYABLE, DevCard, NUM_DEV_CARDS
from hexset.chance import Live
from hexset.game import Game, Phase
from hexset.ledger import PublicLedger, SeatLedger
from hexset.roads import road_lengths
from hexset.rules import Rules
from hexset.state import NO_OWNER, Building, GameState, pile_size

from catanatron.models.enums import SETTLEMENT
from catanatron.state_functions import (
    get_largest_army,
    get_longest_road_color,
    player_has_rolled,
)

from hexset.catanatron._board import BoardMapping
from hexset.catanatron._names import DEV_CARD_NAMES, NAME_TO_DEV_CARD, RESOURCE_NAMES
from hexset.catanatron._state import Seating, seating

_PROMPT_TO_PHASE = {
    "BUILD_INITIAL_SETTLEMENT": Phase.SETUP_SETTLEMENT,
    "BUILD_INITIAL_ROAD": Phase.SETUP_ROAD,
    "MOVE_ROBBER": Phase.ROBBER,
    "DISCARD": Phase.DISCARD,
}


def _phase(catanatron_state, current_color) -> Phase:
    """Dispatches on `current_prompt`, as `generate_playable_actions` does, so
    it cannot disagree with what catanatron is offering. Not on the boolean
    indicators: `is_moving_knight` is never cleared after the first robber
    move."""
    prompt_name = catanatron_state.current_prompt.name
    if prompt_name in _PROMPT_TO_PHASE:
        return _PROMPT_TO_PHASE[prompt_name]
    if prompt_name == "PLAY_TURN":
        return Phase.MAIN if player_has_rolled(catanatron_state, current_color) else Phase.ROLL
    raise NotImplementedError(
        f"{prompt_name} is out of scope for this bridge "
        "(player-to-player trading; see `hexset.catanatron._state`)"
    )


def _setup_step(catanatron_state, phase: Phase, num_players: int) -> tuple[list[int], int]:
    queue = list(range(num_players)) + list(range(num_players))[::-1]
    total_settlements = sum(
        len(catanatron_state.buildings_by_color[c][SETTLEMENT])
        for c in catanatron_state.colors
    )
    step = total_settlements if phase is Phase.SETUP_SETTLEMENT else total_settlements - 1
    return queue, step


def translate(catanatron_game, mapping: BoardMapping, rng: random.Random) -> tuple[Game, Seating]:
    """`to_catanatron` backwards: the hexset `Game` a catanatron position is."""
    cstate = catanatron_game.state
    seats = seating(cstate.colors)
    n = len(cstate.colors)
    board = mapping.board
    topology = board.topology

    vertex_owner = [NO_OWNER] * topology.num_vertices
    vertex_building = [Building.NONE] * topology.num_vertices
    for node_id, (color, kind) in cstate.board.buildings.items():
        v = mapping.vertex_of[node_id]
        vertex_owner[v] = seats.seat_of[color]
        vertex_building[v] = Building.SETTLEMENT if kind == SETTLEMENT else Building.CITY

    edge_owner = [NO_OWNER] * topology.num_edges
    for (a, b), color in cstate.board.roads.items():
        key = (min(a, b), max(a, b))
        edge_index = mapping.edge_of.get(key)
        if edge_index is not None:
            edge_owner[edge_index] = seats.seat_of[color]

    robber = mapping.hex_of[cstate.board.robber_coordinate]

    hands = [[0] * NUM_RESOURCES for _ in range(n)]
    bank = [cstate.resource_freqdeck[i] for i in range(NUM_RESOURCES)]
    dev_cards = [[0] * NUM_DEV_CARDS for _ in range(n)]
    new_dev_cards = [[0] * NUM_DEV_CARDS for _ in range(n)]
    knights_played = [0] * n
    # Table-wide non-knight plays, summed from catanatron's per-seat
    # `PLAYED_{name}` counters (see `hexset.view.View.unseen_dev_cards`).
    dev_cards_played = [0] * NUM_DEV_CARDS

    for seat in range(n):
        color = seats.color_of[seat]
        key = f"P{cstate.color_to_index[color]}"
        for r, name in enumerate(RESOURCE_NAMES):
            hands[seat][r] = cstate.player_state[f"{key}_{name}_IN_HAND"]
        for card, name in DEV_CARD_NAMES.items():
            total = cstate.player_state[f"{key}_{name}_IN_HAND"]
            # catanatron's maturity is a per-type boolean, not hexset's
            # per-copy count, so matured-only-when-set is a conservative
            # subset: never more than catanatron allows, sometimes fewer.
            owned_at_start = cstate.player_state.get(f"{key}_{name}_OWNED_AT_START", False)
            matured = total if owned_at_start else 0
            dev_cards[seat][card] = matured
            new_dev_cards[seat][card] = total - matured
            if card in PLAYABLE and card != DevCard.KNIGHT:
                dev_cards_played[card] += cstate.player_state[f"{key}_PLAYED_{name}"]
        knights_played[seat] = cstate.player_state[f"{key}_PLAYED_KNIGHT"]

    deck = [
        NAME_TO_DEV_CARD[card_name]
        for card_name in cstate.development_listdeck
    ]

    longest_road_color = get_longest_road_color(cstate)
    largest_army_color, _ = get_largest_army(cstate)

    state = GameState(
        board=board,
        num_players=n,
        vertex_owner=vertex_owner,
        vertex_building=vertex_building,
        edge_owner=edge_owner,
        robber=robber,
        hands=hands,
        bank=bank,
        deck=deck,
        dev_cards=dev_cards,
        new_dev_cards=new_dev_cards,
        knights_played=knights_played,
        dev_cards_played=dev_cards_played,
        longest_road_holder=seats.seat_of[longest_road_color]
        if longest_road_color is not None
        else NO_OWNER,
        largest_army_holder=seats.seat_of[largest_army_color]
        if largest_army_color is not None
        else NO_OWNER,
        # Read off the live catanatron game: a 15VP/9-discard table must not
        # be evaluated as a 10VP/7-discard one.
        rules=Rules(
            winning_points=catanatron_game.vps_to_win,
            discard_limit=catanatron_game.state.discard_limit,
        ),
    )
    # From our own topology, not catanatron's `board.road_lengths`, to stay
    # independent of any difference in the two route algorithms.
    state.road_lengths = road_lengths(state)

    current_color = cstate.current_color()
    phase = _phase(cstate, current_color)
    turn_color = cstate.colors[cstate.current_turn_index]
    turn_seat = seats.seat_of[turn_color]
    turn_key = f"P{cstate.color_to_index[turn_color]}"

    # A snapshot has no history to reconstruct hexset's public-knowledge
    # ledger from, so this is a memoryless one: nothing certified, the whole
    # hand `unknown`. It keeps the invariant (`total() == true hand size`,
    # `known[r] <= true[r]`) and understates rather than overstates.
    ledger = PublicLedger(
        seats=[
            SeatLedger(known=[0] * NUM_RESOURCES, unknown=pile_size(hands[seat]))
            for seat in range(n)
        ]
    )

    game = Game(
        _state=state,
        rng=rng,
        # catanatron owns the roll/steal/deck being translated, so `game`
        # never resolves a chance event of its own.
        chance=Live(rng),
        ledger=ledger,
        phase=phase,
        current_player=turn_seat,
        dev_card_played=cstate.player_state[f"{turn_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
        discard_quota=[
            cstate.discard_counts[cstate.color_to_index[seats.color_of[seat]]]
            for seat in range(n)
        ],
        free_roads=cstate.free_roads_available if cstate.is_road_building else 0,
        turns=cstate.num_turns,
    )

    if phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        queue, step = _setup_step(cstate, phase, n)
        game.setup_queue = queue
        game.setup_step = step
        if phase is Phase.SETUP_ROAD:
            last_node = cstate.buildings_by_color[current_color][SETTLEMENT][-1]
            game.last_settlement = mapping.vertex_of[last_node]

    return game, seats
