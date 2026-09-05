# SPDX-License-Identifier: GPL-3.0-only
"""Snapshots a live catanatron `Game` into a dev-catan `Game`/`GameState`, and back.

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
never tries to propose one either, and `bot.py`'s catanatron entrant declines
every exchange for the same reason.

`to_catanatron` is the mirror image, for a catanatron bot sitting at a dev-catan
table (`bot.py`): the catanatron `Game` a catanatron `Player` would see at this
decision, rebuilt fresh the same way and off the same tables. It writes the
`State` fields directly rather than replaying a game into them, because
`State.__init__` reseats the players at random and reseeds the global `random`
module -- neither of which a mirror may do.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random

from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import DevCard, NUM_DEV_CARDS
from hexset.chance import Live
from hexset.game import Game, Phase, to_move
from hexset.ledger import PublicLedger, SeatLedger
from hexset.robber import DISCARD_THRESHOLD
from hexset.state import NO_OWNER, Building, GameState
from hexset.victory import WINNING_POINTS

from catanatron.game import Game as CatanatronGame
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import (
    STATIC_GRAPH,
    Board as CatanatronBoard,
    longest_acyclic_path,
)
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, ActionPrompt
from catanatron.models.player import Color, Player
from catanatron.state import PLAYER_INITIAL_STATE, State
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

# The same table backwards, plus the one merge it is not a bijection over:
# catanatron asks `PLAY_TURN` both before and after the roll and tells the two
# apart by `HAS_ROLLED`, where dev-catan has a phase for each.
_PHASE_TO_PROMPT = {phase: prompt for prompt, phase in _PROMPT_TO_PHASE.items()} | {
    Phase.ROLL: "PLAY_TURN",
    Phase.MAIN: "PLAY_TURN",
}

_SETUP = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


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
        # This bridge never lets `game` itself resolve a chance event --
        # catanatron owns the roll/steal/deck it is translating from, and
        # `Game.chance`'s docstring on why `imagine` never inherits a real
        # game's chance source applies here too: `Live(rng)` is the same
        # inert default `hexset.game.start` would build from this `rng`.
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


def _catanatron_board(state: GameState, mapping: BoardMapping, seats: Seating):
    """catanatron's `Board` holding this position, caches and all.

    The caches are what make this more than a transcription: `connected_
    components` (a colour's road network, walked out to the enemy nodes that
    close it) and `road_lengths` are maintained move by move there and
    recomputed here from the position, through catanatron's own `dfs_walk` and
    `longest_acyclic_path` so the answer is theirs, not a second one.
    """
    board = CatanatronBoard(mapping.catan_map)
    for vertex, owner in enumerate(state.vertex_owner):
        if owner == NO_OWNER:
            continue
        node = mapping.node_of[vertex]
        kind = SETTLEMENT if state.vertex_building[vertex] == Building.SETTLEMENT else CITY
        board.buildings[node] = (seats.color_of[owner], kind)
        board.board_buildable_ids.discard(node)
        board.board_buildable_ids.difference_update(STATIC_GRAPH.neighbors(node))
    for edge, owner in enumerate(state.edge_owner):
        if owner == NO_OWNER:
            continue
        a, b = mapping.catanatron_edge_of[edge]
        board.roads[(a, b)] = board.roads[(b, a)] = seats.color_of[owner]
    board.robber_coordinate = mapping.coord_of[state.robber]

    for seat in range(state.num_players):
        color = seats.color_of[seat]
        seeds = {node for node, (c, _) in board.buildings.items() if c == color}
        seeds.update(
            node for edge, c in board.roads.items() if c == color
            for node in edge if not board.is_enemy_node(node, color)
        )
        components: list[set[int]] = []
        for seed in sorted(seeds):
            if not any(seed in component for component in components):
                components.append(board.dfs_walk(seed, color))
        board.connected_components[color] = components
        board.road_lengths[color] = max(
            (len(longest_acyclic_path(board, c, color)) for c in components), default=0
        )

    # Who *holds* longest road is dev-catan's answer, not a recomputation:
    # both engines award it to the incumbent on a tie, so only the position's
    # own history can say who that is.
    if state.longest_road_holder != NO_OWNER:
        board.road_color = seats.color_of[state.longest_road_holder]
        board.road_length = board.road_lengths[board.road_color]
    return board


def _player_state(game: Game, state: GameState, seats: Seating, board) -> dict:
    """catanatron's flat per-seat feature dictionary for this position."""
    out = {}
    for seat in range(state.num_players):
        color = seats.color_of[seat]
        pieces = Counter(
            state.vertex_building[v]
            for v, owner in enumerate(state.vertex_owner)
            if owner == seat
        )
        settlements, cities = pieces[Building.SETTLEMENT], pieces[Building.CITY]
        values = dict(PLAYER_INITIAL_STATE)
        for r, name in enumerate(RESOURCE_NAMES):
            values[f"{name}_IN_HAND"] = state.hands[seat][r]
        for card, name in DEV_CARD_NAMES.items():
            matured = state.dev_cards[seat][card]
            values[f"{name}_IN_HAND"] = matured + state.new_dev_cards[seat][card]
            # `translate`'s reading of the same rule, inverted: catanatron's
            # maturity is one boolean per type, so any matured copy sets it and
            # a seat holding only fresh ones cannot play them.
            if f"{name}_OWNED_AT_START" in values:
                values[f"{name}_OWNED_AT_START"] = matured > 0
        values["PLAYED_KNIGHT"] = state.knights_played[seat]
        values["ROADS_AVAILABLE"] -= state.edge_owner.count(seat)
        values["SETTLEMENTS_AVAILABLE"] -= settlements
        values["CITIES_AVAILABLE"] -= cities
        values["HAS_ROAD"] = state.longest_road_holder == seat
        values["HAS_ARMY"] = state.largest_army_holder == seat
        values["LONGEST_ROAD_LENGTH"] = board.road_lengths[color]
        mine = seat == game.current_player
        values["HAS_ROLLED"] = mine and game.phase not in (*_SETUP, Phase.ROLL)
        values["HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = mine and game.dev_card_played
        values["VICTORY_POINTS"] = (
            settlements + 2 * cities + 2 * values["HAS_ROAD"] + 2 * values["HAS_ARMY"]
        )
        values["ACTUAL_VICTORY_POINTS"] = (
            values["VICTORY_POINTS"] + values["VICTORY_POINT_IN_HAND"]
        )
        out.update({f"P{seat}_{field}": value for field, value in values.items()})
    return out


def to_catanatron(game: Game, mapping: BoardMapping, seats: Seating) -> CatanatronGame:
    """`translate` backwards: the catanatron `Game` mirroring `game` right now."""
    # true state: a catanatron `Player` reads the whole table, so this adapter
    # reads the true state through the sanctioned path (`Game.state`'s
    # docstring names it as one of the three callers) rather than a `View`.
    state = game.state(0, hidden=False)
    colors = tuple(seats.color_of[seat] for seat in range(state.num_players))

    cstate = State([], None, initialize=False)
    cstate.players = [Player(color) for color in colors]
    cstate.colors = colors
    cstate.color_to_index = {color: seat for seat, color in enumerate(colors)}
    cstate.discard_limit = DISCARD_THRESHOLD
    cstate.friendly_robber = False
    cstate.board = _catanatron_board(state, mapping, seats)
    cstate.player_state = _player_state(game, state, seats, cstate.board)
    cstate.resource_freqdeck = list(state.bank)
    cstate.development_listdeck = [DEV_CARD_NAMES[DevCard(c)] for c in state.deck]
    cstate.action_records = []
    cstate.num_turns = game.turns

    cstate.buildings_by_color = {color: defaultdict(list) for color in colors}
    for node, (color, kind) in cstate.board.buildings.items():
        cstate.buildings_by_color[color][kind].append(node)
    for edge, color in cstate.board.roads.items():
        if edge[0] < edge[1]:
            cstate.buildings_by_color[color][ROAD].append(edge)
    if game.phase is Phase.SETUP_ROAD:
        # `initial_road_possibilities` reads the *last* settlement in this
        # list, which is the one dev-catan is holding in `last_settlement`;
        # everywhere else the order is immaterial.
        settlements = cstate.buildings_by_color[seats.color_of[game.current_player]][SETTLEMENT]
        last = mapping.node_of[game.last_settlement]
        settlements.remove(last)
        settlements.append(last)

    cstate.current_player_index = to_move(game)
    cstate.current_turn_index = game.current_player
    cstate.current_prompt = ActionPrompt[_PHASE_TO_PROMPT[game.phase]]
    cstate.is_initial_build_phase = game.phase in _SETUP
    cstate.is_discarding = game.phase is Phase.DISCARD
    cstate.discard_counts = list(game.discard_quota)
    cstate.is_moving_knight = game.phase is Phase.ROBBER
    cstate.is_road_building = game.free_roads > 0
    cstate.free_roads_available = game.free_roads
    cstate.is_resolving_trade = False
    cstate.current_trade = (0,) * 11
    cstate.acceptees = tuple(False for _ in colors)

    cgame = CatanatronGame([], initialize=False)
    cgame.seed = 0
    cgame.id = ""
    cgame.vps_to_win = WINNING_POINTS
    cgame.friendly_robber = False
    cgame.state = cstate
    cgame.playable_actions = generate_playable_actions(cstate)
    return cgame
