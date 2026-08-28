from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from hexset_ui.actions import Action, ActionType, apply, legal_actions
from hexset_ui.board.board import random_base_board
from hexset_ui.board.coords import BASE_LAYOUT, MINI_LAYOUT, Hex, hexagon

from hexset_ui.board.topology import build as build_topology
from conftest import RandomBot
from hexset_ui.game import Phase, is_over, start, to_move
from hexset_ui.journal import (
    DEFAULT_DIR,
    ENV_DIR,
    configured_dir,
    open_journal,
    replayable,
)
from hexset_ui.journal import RESOURCE_NAMES as JOURNAL_RESOURCE_NAMES
from hexset_ui.victory import victory_points
from hexset_ui.webplay import (
    RESOURCE_NAMES,
    SQRT3,
    GameSession,
    action_to_wire,
    board_layout,
    hex_center,
    hex_corner,
    vertex_pixels,
    wire_to_action,
)

def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)

# --- Hex-to-pixel layout ------------------------------------------------------

def test_hex_center_places_the_origin_hex_at_the_origin():
    assert hex_center(Hex(0, 0, 0), size=42.0) == (0.0, 0.0)

def test_neighbouring_hex_centers_are_size_times_sqrt3_apart():
    topology = build_topology(hexagon(1))
    size = 25.0
    center = topology.hex_index[Hex(0, 0, 0)]
    origin = hex_center(Hex(0, 0, 0), size)
    for neighbor in topology.hex_neighbors[center]:
        other = hex_center(topology.hexes[neighbor], size)
        distance = math.dist(origin, other)
        assert distance == pytest.approx(size * SQRT3)

def test_hex_corners_match_the_pointy_top_formula():
    """`Topology.hex_vertices[h][i]` sits at the textbook angle 60*i - 30."""
    topology = build_topology([Hex(0, 0, 0)])
    size = 10.0
    vpix = vertex_pixels(topology, size)
    for i, v in enumerate(topology.hex_vertices[0]):
        expected = hex_corner((0.0, 0.0), i, size)
        assert vpix[v] == pytest.approx(expected)

@pytest.mark.parametrize("layout", [MINI_LAYOUT, BASE_LAYOUT])
def test_every_edge_measures_exactly_size_between_its_two_vertices(layout):
    """A regular hexagon's edge length equals its circumradius (`size`), so this
    holds for every edge on any board built from hex coordinates, regardless of
    shape — a broad, cheap check on the whole layout at once."""
    topology = build_topology(layout)
    size = 37.0
    vpix = vertex_pixels(topology, size)
    for a, b in topology.edges:
        assert math.dist(vpix[a], vpix[b]) == pytest.approx(size)

def test_vertex_pixels_covers_every_vertex_exactly_once():
    topology = build_topology(BASE_LAYOUT)
    vpix = vertex_pixels(topology, 50.0)
    assert len(vpix) == topology.num_vertices
    assert all(isinstance(p, tuple) and len(p) == 2 for p in vpix)

def test_board_layout_ids_are_internally_consistent():
    board = random_base_board(random.Random(7))
    layout = board_layout(board)
    num_vertices = len(layout["vertices"])
    num_hexes = len(layout["hexes"])

    for hex_entry in layout["hexes"]:
        assert len(hex_entry["vertex_ids"]) == 6
        assert all(0 <= v < num_vertices for v in hex_entry["vertex_ids"])

    for edge in layout["edges"]:
        assert 0 <= edge["v0"] < num_vertices
        assert 0 <= edge["v1"] < num_vertices

    edge_endpoints = {e["id"]: {e["v0"], e["v1"]} for e in layout["edges"]}
    for port in layout["ports"]:
        assert 0 <= port["edge"] < len(layout["edges"])
        assert set(port["vertices"]) == edge_endpoints[port["edge"]]
        assert port["resource"] is None or port["resource"] in layout["resources"]

    assert len(layout["resources"]) == 5
    assert len(layout["year_of_plenty_pairs"]) == 15
    assert num_hexes == board.topology.num_hexes

# --- Wire format ---------------------------------------------------------------

def test_wire_round_trips_across_a_played_out_game():
    game = a_game(seed=11)
    rng = random.Random(99)
    steps = 0
    while not is_over(game) and steps < 500:
        options = legal_actions(game)
        for action in options:
            assert wire_to_action(action_to_wire(action)) == action
        apply(game, rng.choice(options))
        steps += 1
    assert steps > 50  # sanity: the loop actually exercised many phases

def test_action_to_wire_uses_json_friendly_types():
    action = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 1, 0, 0, 0), ask=(2, 3)
    )
    wire = action_to_wire(action)
    assert wire == {
        "type": "PROPOSE_TRADE",
        "a": 0,
        "b": 0,
        "give": [1, 0, 0, 0, 0],
        "want": [0, 1, 0, 0, 0],
        "ask": [2, 3],
    }

def test_wire_to_action_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        wire_to_action({"type": "TELEPORT", "a": 0, "b": 0})

def test_wire_to_action_rejects_a_malformed_payload():
    with pytest.raises(ValueError):
        wire_to_action({"type": "BUILD_ROAD", "a": "not-a-number"})

# --- GameSession ---------------------------------------------------------------

def test_session_rejects_an_action_not_currently_legal():
    game = a_game(seed=2)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    # ROLL is never legal during setup placement.
    forged = action_to_wire(Action(ActionType.ROLL))
    with pytest.raises(ValueError):
        session.apply_human_action(forged)
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert all(owner == -1 for owner in game.state.vertex_owner)

def test_session_rejects_an_out_of_range_target():
    game = a_game(seed=3)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    forged = action_to_wire(Action(ActionType.SETUP_SETTLEMENT, a=999999))
    with pytest.raises(ValueError):
        session.apply_human_action(forged)

def test_session_rejects_when_it_is_not_the_humans_turn():
    game = a_game(seed=4)
    mover = to_move(game)
    other = (mover + 1) % game.state.num_players
    session = GameSession(game=game, human_seat=other, bot=RandomBot())

    # A perfectly legal action for whoever is actually on the move.
    legal_for_mover = action_to_wire(legal_actions(game)[0])
    with pytest.raises(ValueError):
        session.apply_human_action(legal_for_mover)

def test_legal_wire_actions_offers_every_held_resource_regardless_of_who_could_cover_it():
    """HexSet hands are private: the human must not be able to learn what an
    opponent holds by noticing that proposing to trade for it is or isn't
    offered. `hexset_ui.actions.legal_actions`'s own PROPOSE_TRADE sample
    filters to pairs some opponent could currently cover — correct for a
    bot with full-state access, but exactly the leak a human-facing wire
    payload must not repeat. See GameSession._proposable_options."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=19)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    # No opponent holds anything at all: under the omniscient sample this
    # give would offer zero PROPOSE_TRADE wants. The public-info version
    # must still offer all four regardless.
    for seat in range(1, state.num_players):
        for r in range(len(state.hands[seat])):
            state.hands[seat][r] = 0

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    proposals = [a for a in session.legal_wire_actions() if a["type"] == "PROPOSE_TRADE"]
    wanted_for_wood = {
        r for a in proposals if a["give"][Resource.WOOD] == 1
        for r, n in enumerate(a["want"]) if n
    }

    assert wanted_for_wood == {r for r in range(len(state.hands[0])) if r != Resource.WOOD}

def test_nothing_is_proposable_before_the_roll():
    """Trading is a Main-phase act. Offering pairs in Roll made the hand
    clickable and opened the trade modal on a turn where the bank half of it
    could not be there — BANK_TRADE only exists in Main — so a port the human
    could plainly afford showed up dimmed."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=19)
    game.phase = Phase.ROLL
    game.current_player = 0
    game.state.hands[0][Resource.WHEAT] += 6

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    kinds = {a["type"] for a in session.legal_wire_actions()}

    assert "PROPOSE_TRADE" not in kinds
    assert "BANK_TRADE" not in kinds
    assert "ROLL" in kinds

def test_a_human_trade_with_no_ask_defaults_to_lowest_vp_first():
    """GameSession's own addition on top of hexset_ui.game.propose_trade's
    neutral ask=() default (clockwise seat order) — favours whoever's
    behind rather than strict seat order. See GameSession._default_ask_order.
    """
    from hexset_ui.board.terrain import Resource
    from hexset_ui.state import Building

    game = a_game(seed=7)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    # Seat 0 (the proposer) can give wood; seats 1, 2, 3 can all cover the
    # ore ask, so all three are eligible and the ordering is what's under
    # test, not who can respond at all.
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    # Seat 1: a city (2 VP). Seat 3: a settlement (1 VP). Seat 2: nothing (0
    # VP) — expect lowest first: [2, 3, 1].
    state.vertex_owner[0] = 1
    state.vertex_building[0] = Building.CITY
    state.vertex_owner[1] = 3
    state.vertex_building[1] = Building.SETTLEMENT

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session.apply_human_action(action_to_wire(offer))

    assert game.pending_responders == [2, 3, 1]

def test_a_human_trade_honours_an_explicit_ask_instead_of_the_default():
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=8)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(3, 2, 1)
    )
    session.apply_human_action(action_to_wire(offer))

    assert game.pending_responders == [3, 2, 1]

def test_a_bot_trade_also_defaults_to_asking_the_lowest_vp_player_first():
    """_apply's ask-defaulting isn't human-only — every seat's proposal goes
    through the same choke point, bot or human (see _apply's own comment)."""
    from hexset_ui.board.terrain import Resource
    from hexset_ui.state import Building

    class _AlwaysProposes:
        def __init__(self, give, want):
            self._action = Action(ActionType.PROPOSE_TRADE, give=give, want=want)

        def choose(self, game):
            return self._action

    game = a_game(seed=9)
    game.phase = Phase.MAIN
    game.current_player = 1  # the bot seat proposing; seat 0 is the human
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[1][Resource.WOOD] += 1
    for seat in (0, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    # Seat 0 (the human) has the lowest VP of the three eligible responders,
    # so it lands first in the queue and advance_bots() stops right there —
    # no need to also script a TRADE_RESPOND answer for this bot.
    state.vertex_owner[0] = 2
    state.vertex_building[0] = Building.CITY
    state.vertex_owner[1] = 3
    state.vertex_building[1] = Building.SETTLEMENT

    bot = _AlwaysProposes(give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session = GameSession(game=game, human_seat=0, bot=bot)
    session.advance_bots()

    assert game.pending_responders == [0, 3, 2]

# --- Log summarizing: builds and trades collapse into one entry -----------

def test_setup_settlement_and_road_collapse_into_one_log_line():
    """Successive builds by the same actor merge into one entry, not one
    each — see GameSession._log_action's _BUILD_KIND streak."""
    game = a_game(seed=10)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    settlement = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT)
    session.apply_human_action(action_to_wire(settlement))
    assert len(session.log) == 1

    road = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_ROAD)
    session.apply_human_action(action_to_wire(road))

    assert len(session.log) == 1  # rewritten, not appended to
    text = session.log[0]
    assert "settlement" in text and "road" in text
    assert text.count("placed") == 1  # one merged sentence, not two

def test_a_build_streak_breaks_on_a_different_actor():
    game = a_game(seed=11)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    settlement = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT)
    session.apply_human_action(action_to_wire(settlement))
    road = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_ROAD)
    session.apply_human_action(action_to_wire(road))
    assert len(session.log) == 1  # human's merged settlement+road

    session.advance_bots()  # the next seat(s) in the snake place too

    assert len(session.log) >= 2  # human's line, plus at least the next seat's
    assert session._run["key"][0] != human_seat  # the run moved on

def test_list_with_counts_pluralises_repeats_but_not_singles():
    from hexset_ui.webplay import _list_with_counts

    assert _list_with_counts(["settlement", "road"]) == "a settlement and a road"
    assert _list_with_counts(["road", "road"]) == "2 roads"
    assert _list_with_counts(["road", "road", "city"]) == "2 roads and a city"
    assert _list_with_counts(["road"]) == "a road"

def test_a_trade_that_gets_accepted_summarizes_into_one_line():
    """The proposal and the eventual outcome fold into one entry, only
    reaching `log` once the offer concludes — see GameSession._log_action.
    A decline along the way (seat 1, below) must NOT be individually named:
    only who's eligible to cover an offer is ever asked, so naming a
    decliner would tell a human they hold the wanted resource — hidden
    information a real board never gives up."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=13)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(1, 2)
    )
    session.apply_human_action(action_to_wire(offer))
    assert session.log == []  # held back until the offer concludes

    session._apply(1, Action(ActionType.DECLINE_TRADE))
    assert session.log == []  # still pending — seat 2 hasn't answered yet

    session._apply(2, Action(ActionType.ACCEPT_TRADE))

    assert len(session.log) == 1
    text = session.log[0]
    assert "offered" in text and "accepted" in text
    assert "Player 3" in text  # seat 2, who actually accepted
    # Not "declined": seat 1's decline never gets named — see the docstring.
    assert "declined" not in text
    assert "Player 2" not in text  # the decliner isn't named at all
    assert session._trade_buffer is None

def test_a_trade_nobody_can_cover_is_still_legal_and_reads_as_declined():
    """propose_trade() concludes an uncoverable offer on the spot — no
    DECLINE_TRADE/ACCEPT_TRADE is ever coming to flush a held-back buffer,
    so this must not wait for one. And it must be reachable through
    apply_human_action itself, not just a direct _apply: legal_actions()
    only *samples* coverable (give, want) pairs (see its PROPOSE_TRADE
    enumerator's own docstring), but a proposal nobody can cover is still a
    legal move — see is_legal's docstring for why it's checked against
    can_propose instead of sample membership."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=14)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    # Nobody else holds any ore, so nobody is eligible to respond.

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    # The server must actually have offered this — see the module docstring's
    # "never build an action the engine did not offer" rule the frontend
    # leans on — not merely tolerate it when submitted directly.
    assert action_to_wire(offer) in session.legal_wire_actions()
    session.apply_human_action(action_to_wire(offer))

    assert len(session.log) == 1
    assert "offered" in session.log[0]
    assert "declined" in session.log[0]
    # Deliberately not "nobody could cover it": that would state opponent
    # hand contents as fact. HexSet hands are private.
    assert "cover" not in session.log[0].lower()
    assert session._trade_buffer is None

def test_a_trade_everyone_declines_summarizes_into_one_line():
    """Reads as a single generic 'Everyone declined.' — not one line per
    decliner and not a count — regardless of how many opponents were
    actually asked, so it can't be distinguished from an offer nobody was
    eligible to take at all (see test_a_trade_nobody_can_cover...)."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=15)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(1, 2)
    )
    session.apply_human_action(action_to_wire(offer))
    session._apply(1, Action(ActionType.DECLINE_TRADE))
    assert session.log == []

    session._apply(2, Action(ActionType.DECLINE_TRADE))

    assert len(session.log) == 1
    assert session.log[0].count("declined") == 1  # "Everyone declined.", not one per seat
    assert "Player 2" not in session.log[0] and "Player 3" not in session.log[0]
    assert session._trade_buffer is None

def _discard_all(session: GameSession, seat: int) -> None:
    """Run every DISCARD the engine asks `seat` for, one at a time."""
    while (
        session.game.phase is Phase.DISCARD
        and to_move(session.game) == seat
    ):
        action = next(a for a in legal_actions(session.game) if a.type is ActionType.DISCARD)
        session._apply(seat, action)

def _owing_game(seed: int, seat: int, hand: list[int]) -> GameSession:
    """A game parked in Phase.DISCARD with `seat` owing half of `hand`."""
    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = seat
    game.state.hands[seat] = list(hand)
    game.discard_quota = [0] * game.state.num_players
    game.discard_quota[seat] = sum(hand) // 2
    return game

def test_a_discard_collapses_into_one_line_however_many_cards():
    """The engine takes discards one card at a time (legal_actions under
    Phase.DISCARD keeps the action space linear in resources rather than
    combinatorial in hand size), so a seven can cost one seat half a dozen
    steps in a row. The log is one line."""
    game = _owing_game(seed=20, seat=0, hand=[4, 4, 0, 0, 0])
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    _discard_all(session, 0)

    assert game.discard_quota[0] == 0  # four cards actually went
    assert len(session.log) == 1
    assert session.log[0].count("discarded") == 1

def test_a_humans_discard_line_names_the_resources_with_counts():
    game = _owing_game(seed=21, seat=0, hand=[4, 4, 0, 0, 0])
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    _discard_all(session, 0)

    text = session.log[0]
    # Counted, not repeated — and never pluralised (see _resource_counts).
    assert "4 Wood." in text or "4 Brick." in text or "2 Wood, 2 Brick" in text
    assert "Woods" not in text and "Bricks" not in text

def test_a_bots_discard_line_is_a_bare_count_never_the_resources():
    """A collapsed line is exactly where a whole hidden hand would leak at
    once — the same rule _describe applied to a single bot discard."""
    game = _owing_game(seed=22, seat=1, hand=[4, 4, 0, 0, 0])
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    _discard_all(session, 1)

    text = session.log[0]
    assert "discarded 4 cards" in text
    assert not any(r in text for r in RESOURCE_NAMES)

def test_two_seats_discarding_get_a_line_each():
    game = _owing_game(seed=23, seat=0, hand=[4, 4, 0, 0, 0])
    game.state.hands[1] = [4, 4, 0, 0, 0]
    game.discard_quota[1] = 4
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    _discard_all(session, 0)
    _discard_all(session, 1)

    assert len(session.log) == 2  # not merged across actors

def test_consecutive_bank_trades_of_the_same_pair_sum_into_one_line():
    game = a_game(seed=24)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [8, 0, 0, 0, 0]
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    trade = next(
        a for a in legal_actions(game)
        if a.type is ActionType.BANK_TRADE and a.a == 0 and a.b == 4
    )
    session._apply(0, trade)
    session._apply(0, trade)

    assert len(session.log) == 1
    assert "8 Wood" in session.log[0] and "2 Ore" in session.log[0]

def test_a_different_bank_pair_starts_its_own_line():
    game = a_game(seed=25)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [4, 4, 0, 0, 0]
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    for give, want in ((0, 4), (1, 4)):
        action = next(
            a for a in legal_actions(session.game)
            if a.type is ActionType.BANK_TRADE and a.a == give and a.b == want
        )
        session._apply(0, action)

    assert len(session.log) == 2

def test_a_roll_between_two_discards_keeps_them_apart():
    """Only one run is ever open, so nothing can reach back across an
    intervening line to join something older."""
    game = _owing_game(seed=26, seat=0, hand=[4, 4, 0, 0, 0])
    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    _discard_all(session, 0)
    assert len(session.log) == 1

    session.game.phase = Phase.ROLL
    session.game.current_player = 0
    session._apply(0, Action(ActionType.ROLL))
    assert len(session.log) == 2

    session.game.phase = Phase.DISCARD
    session.game.state.hands[0] = [2, 0, 0, 0, 0]
    session.game.discard_quota = [1, 0, 0, 0]
    _discard_all(session, 0)

    assert len(session.log) == 3  # a fresh discard line, not a swollen one

def test_ending_a_turn_writes_no_log_line():
    """Every turn ends eventually and the next line already implies it — a
    dedicated line for each one was pure noise, not information."""
    game = a_game(seed=16)
    game.phase = Phase.MAIN
    game.current_player = 0
    session = GameSession(game=game, human_seat=0, bot=RandomBot())

    session._apply(0, Action(ActionType.END_TURN))

    assert session.log == []

def test_ending_a_turn_closes_an_open_run():
    game = a_game(seed=17)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    settlement = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT)
    session.apply_human_action(action_to_wire(settlement))
    assert session._run is not None

    session.game.phase = Phase.MAIN
    session.game.current_player = human_seat
    session._apply(human_seat, Action(ActionType.END_TURN))

    assert session._run is None

def test_advance_bots_always_stops_at_the_human_seat_or_game_over():
    game = a_game(seed=5)
    human_seat = 1
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot(rng=random.Random(0)))
    rng = random.Random(6)

    session.advance_bots()
    steps = 0
    while not is_over(session.game) and steps < 300:
        assert to_move(session.game) == human_seat
        options = legal_actions(session.game)
        wire = action_to_wire(rng.choice(options))
        session.apply_human_action(wire)
        session.advance_bots()
        steps += 1
        assert is_over(session.game) or to_move(session.game) == human_seat

def test_state_view_hides_opponent_hands_but_not_the_humans():
    game = a_game(seed=8)
    human_seat = to_move(game)
    other = (human_seat + 1) % game.state.num_players
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())

    game.state.hands[human_seat][0] = 3
    game.state.hands[other][0] = 5

    view = session.state_view()
    players = {p["seat"]: p for p in view["players"]}
    assert "hand" in players[human_seat]
    assert players[human_seat]["hand"]["Wood"] == 3
    assert "hand" not in players[other]
    assert players[other]["hand_size"] == 5

def test_state_view_reveals_every_hand_once_the_game_is_over():
    game = a_game(seed=9)
    human_seat = to_move(game)
    session = GameSession(game=game, human_seat=human_seat, bot=RandomBot())
    game.won_by = (human_seat + 1) % game.state.num_players
    game.phase = Phase.GAME_OVER

    view = session.state_view()
    assert all("hand" in p for p in view["players"])
    assert view["legal_actions"] == []

def test_state_view_does_not_expose_who_is_eligible_to_respond_to_an_offer():
    """`game.pending_responders` is exactly who's eligible to cover the open
    offer, in ask order — sending it to the client before anyone has
    actually responded would leak the same hidden hand information the log
    (see _log_action's "Everyone declined." handling) is built to hide,
    just earlier and over a different channel."""
    from hexset_ui.board.terrain import Resource

    game = a_game(seed=18)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = GameSession(game=game, human_seat=0, bot=RandomBot())
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session.apply_human_action(action_to_wire(offer))
    assert game.pending_responders  # the offer really is pending on someone

    view = session.state_view()
    assert "responders" not in view["offer"]

# --- Recording and journalling ----------------------------------------------

SEED = 42

@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """One game played out in full, journalled to its own directory.

    Module-scoped because playing a whole game is by far the slowest thing in
    this file: every test below reads the same finished game rather than
    dealing another one of its own.
    """
    directory = tmp_path_factory.mktemp("games")
    # Two independent random.Random(SEED) instances, matching what
    # `webserver._new_session` and `_resume_session` both do: the board spends
    # one stream and `start` gets a fresh one, so the game's own rng must begin
    # from the same untouched state here too.
    board = random_base_board(random.Random(SEED))
    game = start(board, 4, random.Random(SEED))
    session = GameSession(
        game=game,
        human_seat=to_move(game),
        bot=RandomBot(rng=random.Random(1)),
        seed=SEED,
        journal=open_journal(SEED, str(directory)),
    )

    human_rng = random.Random(2)
    session.advance_bots()
    steps = 0
    while not is_over(session.game) and steps < 4000:
        options = legal_actions(session.game)
        session.apply_human_action(action_to_wire(human_rng.choice(options)))
        session.advance_bots()
        steps += 1
    assert is_over(session.game)
    return session, directory

def journal_events(directory) -> list[dict]:
    """The one per-game journal in `directory`, parsed."""
    files = list(Path(directory).glob("*.jsonl"))
    assert len(files) == 1, f"expected one game journal, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines()]

def test_a_journalled_game_replays_clean(played):
    """The strongest check there is on the journal: fed back through the
    engine, its actions have to be legal in order and end the same game.

    Deliberately goes through `replayable` and `restore` — the same two calls
    `webserver._resume_session` makes — rather than a replay written for the
    test. A journal that replays clean here is one a returning player would
    actually get their game back from.
    """
    session, directory = played
    events = journal_events(directory)
    header = events[0]
    assert header["seed"] == SEED

    board = random_base_board(random.Random(SEED))
    resumed = GameSession(
        game=start(board, header["num_players"], random.Random(SEED)),
        human_seat=header["human_seat"],
        bot=RandomBot(rng=random.Random(1)),
        seed=SEED,
    )
    resumed.restore(replayable(events))  # raises ResumeError if it doesn't

    assert resumed.game.won_by == session.game.won_by
    assert resumed.game.turns == session.game.turns

def test_the_journal_holds_one_line_per_action_in_order(played):
    session, directory = played
    events = journal_events(directory)
    assert events[0]["kind"] == "game"
    assert events[-1]["kind"] == "result"

    actions = [e for e in events if e["kind"] == "action"]
    assert [e["step"] for e in actions] == list(range(len(actions)))
    assert len(actions) == session._steps

def test_the_journal_states_the_dice_and_the_hands_they_paid(played):
    session, directory = played
    events = journal_events(directory)
    rolls = [e for e in events if e["kind"] == "action" and e["type"] == "ROLL"]
    assert rolls, "a game this long rolled dice"
    assert all(2 <= e["roll"] <= 12 for e in rolls)
    # Hands are absolute and cover every seat, so production is readable off
    # the line itself rather than inferred from the board.
    assert all(len(e["hands"]) == 4 for e in rolls)

    actions = [e for e in events if e["kind"] == "action"]
    assert actions[-1]["hands"] == [list(h) for h in session.game.state.hands]

def test_the_journal_names_every_development_card_in_deck_order(played):
    """The header's deck is the whole point: every card bought later has to be
    the one the shuffle put there, so the game's hidden cards are known from
    the file alone without re-running the engine."""
    _, directory = played
    events = journal_events(directory)
    deck = events[0]["deck"]
    assert len(deck) == 25  # the standard development deck

    drawn = [e["drew"] for e in events if e["kind"] == "action" and e["type"] == "BUY_DEV_CARD"]
    assert drawn, "a game this long bought development cards"
    # `devcards.buy` pops off the end, so purchases run backwards through the
    # deck as written.
    assert drawn == deck[::-1][: len(drawn)]

def test_the_journal_names_the_card_every_steal_took(played):
    """The sidebar hides this when neither side is the human (see `_describe`);
    the journal never does."""
    _, directory = played
    events = journal_events(directory)
    steals = [e["stole"] for e in events if e.get("stole")]
    assert steals, "a game this long moved the robber onto someone"
    for steal in steals:
        assert steal["from"] in range(4)
        # None only where the victim's hand was empty and nothing moved.
        assert steal["resource"] in (None, *JOURNAL_RESOURCE_NAMES)

def test_the_journal_ends_with_the_result(played):
    session, directory = played
    result = journal_events(directory)[-1]
    assert result == {
        "kind": "result",
        "at": result["at"],
        "winner": session.game.won_by,
        "turns": session.game.turns,
        "points": [
            victory_points(session.game.state, p)
            for p in range(session.game.state.num_players)
        ],
    }

def test_journalling_is_on_unless_it_is_switched_off(monkeypatch):
    """On by default, so a server started with no arguments still keeps an
    account of every game it deals."""
    monkeypatch.delenv(ENV_DIR, raising=False)
    assert configured_dir() == DEFAULT_DIR
    monkeypatch.setenv(ENV_DIR, "/somewhere/else")
    assert configured_dir() == "/somewhere/else"
    monkeypatch.setenv(ENV_DIR, "")
    assert configured_dir() is None
    assert open_journal(seed=1) is None

def test_a_session_without_a_journal_still_plays():
    game = a_game(seed=13)
    session = GameSession(game=game, human_seat=to_move(game), bot=RandomBot())
    session._apply(session.human_seat, legal_actions(game)[0])
    assert session.journal is None  # nothing to have written to

def test_an_unwritable_directory_costs_the_journal_and_not_the_game(tmp_path):
    """A player mid-turn should not lose a game to a full disk or a mount that
    came up read-only."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")  # mkdir under a regular file cannot succeed
    game = a_game(seed=7)
    session = GameSession(
        game=game,
        human_seat=to_move(game),
        bot=RandomBot(rng=random.Random(3)),
        journal=open_journal(1, str(blocked / "games")),
    )
    session._apply(session.human_seat, legal_actions(game)[0])
    assert session.journal._off

def test_an_undone_placement_is_written_down_not_erased(tmp_path):
    """The journal is append-only and read forwards, so a step number that
    quietly came round twice would leave a reader unable to say which of the
    two actions counted."""
    game = a_game(seed=5)
    human_seat = to_move(game)
    session = GameSession(
        game=game,
        human_seat=human_seat,
        bot=RandomBot(rng=random.Random(4)),
        journal=open_journal(5, str(tmp_path)),
    )
    settlement = next(
        a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT
    )
    session.apply_human_action(action_to_wire(settlement))
    session.undo_last_build()

    events = journal_events(tmp_path)
    assert [e["kind"] for e in events] == ["game", "action", "undo"]
    assert events[1]["type"] == "SETUP_SETTLEMENT"
    assert events[2]["back_to"] == 0  # everything from step 0 did not happen
