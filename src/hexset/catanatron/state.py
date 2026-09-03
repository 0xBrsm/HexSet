# SPDX-License-Identifier: GPL-3.0-only
"""Snapshots a live catanatron `Game` into a dev-catan `Game`/`GameState`.

Rebuilt fresh on every decision rather than mirrored incrementally: the two
engines' turn machines differ in enough small ways (see `Phase`, below) that
keeping them in lockstep move-by-move would be far more fragile than reading
catanatron's own state directly each time and asking "what does dev-catan's
engine call this position".

Not translated, by deliberate v1 scope: player-to-player trading. catanatron's
own move generator (`generate_playable_actions`) never emits `OFFER_TRADE`
during a normal turn -- it exists in the rules but no bundled player, ours
included, can reach it without constructing an action outside
`playable_actions`. `Phase.TRADE_RESPOND` is therefore unreachable here and
`player.py` forces `max_trades=0` on every dev-catan entrant so our own bot
never tries to propose one either.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import DevCard, NUM_DEV_CARDS
from hexset.game import Game, Phase
from hexset.ledger import PublicLedger, SeatLedger
from hexset.state import NO_OWNER, Building, GameState

from catanatron.models.enums import SETTLEMENT
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_largest_army,
    get_longest_road_color,
    player_has_rolled,
)

from .board import BoardMapping
from .names import DEV_CARD_NAMES, NAME_TO_DEV_CARD, RESOURCE_NAMES


@dataclass(frozen=True)
class Seating:
    """catanatron plays by `Color`; dev-catan plays by seat index. One order."""

    color_of: dict[int, Color]
    seat_of: dict[Color, int]


def seating(colors: tuple[Color, ...]) -> Seating:
    seat_of = {c: i for i, c in enumerate(colors)}
    color_of = {i: c for c, i in seat_of.items()}
    return Seating(color_of=color_of, seat_of=seat_of)


_PROMPT_TO_PHASE = {
    "BUILD_INITIAL_SETTLEMENT": Phase.SETUP_SETTLEMENT,
    "BUILD_INITIAL_ROAD": Phase.SETUP_ROAD,
    "MOVE_ROBBER": Phase.ROBBER,
    "DISCARD": Phase.DISCARD,
}


def _phase(catanatron_state, current_color) -> Phase:
    """Dispatches on `current_prompt`, exactly as `generate_playable_actions` does.

    Not on the boolean indicators (`is_moving_knight` and friends) the `State`
    docstring recommends instead: `is_moving_knight` is set on rolling a seven
    or finishing a discard sequence, but `apply_move_robber` never clears it
    back to `False` -- it is stale (stuck `True`) for the rest of the game
    after the first robber move. `current_prompt` is what `generate_
    playable_actions` itself reads, so it cannot disagree with the actions
    catanatron is actually offering.
    """
    prompt_name = catanatron_state.current_prompt.name
    if prompt_name in _PROMPT_TO_PHASE:
        return _PROMPT_TO_PHASE[prompt_name]
    if prompt_name == "PLAY_TURN":
        return Phase.MAIN if player_has_rolled(catanatron_state, current_color) else Phase.ROLL
    raise NotImplementedError(
        f"{prompt_name} is out of scope for this bridge "
        "(player-to-player trading; see module docstring)"
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

    for seat in range(n):
        color = seats.color_of[seat]
        key = f"P{cstate.color_to_index[color]}"
        for r, name in enumerate(RESOURCE_NAMES):
            hands[seat][r] = cstate.player_state[f"{key}_{name}_IN_HAND"]
        for card, name in DEV_CARD_NAMES.items():
            total = cstate.player_state[f"{key}_{name}_IN_HAND"]
            # catanatron's maturity rule is a per-type boolean ("owned any at
            # the start of this turn"), not a per-copy count -- slightly more
            # lenient than dev-catan's "this exact copy wasn't bought this
            # turn". Treating everything as matured only when that boolean is
            # set, and immature otherwise, is a conservative subset: dev-catan
            # will never offer to play a card catanatron would call illegal,
            # it may occasionally offer fewer than catanatron would allow.
            owned_at_start = cstate.player_state.get(f"{key}_{name}_OWNED_AT_START", False)
            matured = total if owned_at_start else 0
            dev_cards[seat][card] = matured
            new_dev_cards[seat][card] = total - matured
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
        longest_road_holder=seats.seat_of[longest_road_color]
        if longest_road_color is not None
        else NO_OWNER,
        largest_army_holder=seats.seat_of[largest_army_color]
        if largest_army_color is not None
        else NO_OWNER,
    )

    current_color = cstate.current_color()
    phase = _phase(cstate, current_color)
    turn_color = cstate.colors[cstate.current_turn_index]
    turn_seat = seats.seat_of[turn_color]
    turn_key = f"P{cstate.color_to_index[turn_color]}"

    # `Game.ledger` (hexset >= 0.14, the public-knowledge hand ledger) is a
    # required field. hexset's own engine builds it incrementally, routing
    # every hand mutation through it as the game is played; this bridge
    # instead re-translates catanatron's state from scratch on every
    # `decide()`, so there is no history here to reconstruct one from. The
    # honest choice is a *memoryless* ledger: for every seat, nothing
    # certified by type and the whole hand counted as `unknown` -- exactly
    # what a bystander with no memory of the game could still vouch for.
    # It keeps the ledger's one invariant (`total() == true hand size`,
    # `known[r] <= true[r]`) trivially, and it is a lower bound on what an
    # information-set-honest bot could know: a stateful ledger would need the
    # bridge player to carry a `PublicLedger` across decisions and route hand
    # diffs and steals through it, which is out of scope here.
    # TODO: carry a `PublicLedger` across decisions in `player.py` and route
    # hand diffs and steals through it, so the bot sees what the public log
    # actually certifies rather than this floor.
    ledger = PublicLedger(
        seats=[
            SeatLedger(known=[0] * NUM_RESOURCES, unknown=sum(hands[seat]))
            for seat in range(n)
        ]
    )

    game = Game(
        _state=state,
        rng=rng,
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
        # All-zero, matching `hexset.game.new_game`: trading is out of scope
        # for this bridge (module docstring), but `hexset.encoding` reads one
        # valuation vector per seat unconditionally, so it must be sized to
        # `n` rather than left at `Game.valuations`'s empty default.
        valuations=[(0.0,) * NUM_RESOURCES for _ in range(n)],
        # Same sizing requirement as `valuations`, for the same reason:
        # `Game.published_post_roll`/`published_end_turn` (PI correction
        # "two publish points, not one") default to an empty list, and
        # `end_turn`/`enter_main` index them by seat unconditionally --
        # reached whenever a bot doing lookahead (`search2`, `heximax`)
        # simulates a `ROLL` or `END_TURN` action through this bridge's own
        # `Game`, trading or not. `True` (nothing due) throughout, matching
        # a fresh `start()` before setup completes: this bridge never seats
        # gates, so `publish_due` is never acted on here regardless.
        published_post_roll=[True] * n,
        published_end_turn=[True] * n,
    )

    if phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        queue, step = _setup_step(cstate, phase, n)
        game.setup_queue = queue
        game.setup_step = step
        if phase is Phase.SETUP_ROAD:
            last_node = cstate.buildings_by_color[current_color][SETTLEMENT][-1]
            game.last_settlement = mapping.vertex_of[last_node]

    return game, seats