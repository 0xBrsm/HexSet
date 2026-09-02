from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.coords import Hex, hexagon
from hexset.board.maps import BASE_LAYOUT, MINI_LAYOUT

from hexset.board.topology import build as build_topology
from conftest import RandomBot
from hexset.game import Phase, is_over, to_move
from hexset_ui.seating import start_at
from hexset_ui.journal import (
    DEFAULT_DIR,
    ENV_DIR,
    configured_dir,
    open_journal,
    replayable,
)
from hexset_ui.journal import RESOURCE_NAMES as JOURNAL_RESOURCE_NAMES
from hexset.victory import victory_points
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
    return start_at(random_base_board(rng), players, rng, first=0)

def a_session(game, claimed, **kwargs) -> GameSession:
    """A `GameSession` over `claimed` seats — every claimed seat submits its
    own actions through `submit` the same way now, human or "bot" (there is
    no `bot:` field any more; a seat played by a bot in these tests is just
    driven directly, via `_apply` or `submit`, exactly like any other seat —
    see `botclient.py` for how a real bot runner does the same from outside
    a session entirely)."""
    return GameSession(game=game, claimed_seats=set(claimed), **kwargs)

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
    seat = to_move(game)
    session = a_session(game, {seat})

    # ROLL is never legal during setup placement.
    forged = action_to_wire(Action(ActionType.ROLL))
    with pytest.raises(ValueError):
        session.submit(seat, forged)
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert all(owner == -1 for owner in game.state.vertex_owner)

def test_session_rejects_an_out_of_range_target():
    game = a_game(seed=3)
    seat = to_move(game)
    session = a_session(game, {seat})

    forged = action_to_wire(Action(ActionType.SETUP_SETTLEMENT, a=999999))
    with pytest.raises(ValueError):
        session.submit(seat, forged)

def test_session_rejects_an_action_from_a_seat_that_has_not_claimed_it():
    game = a_game(seed=4)
    mover = to_move(game)
    other = (mover + 1) % game.state.num_players
    session = a_session(game, {other})

    # A perfectly legal action for whoever is actually on the move.
    legal_for_mover = action_to_wire(legal_actions(game)[0])
    with pytest.raises(ValueError):
        session.submit(other, legal_for_mover)

def test_legal_wire_actions_offers_every_held_resource_regardless_of_who_could_cover_it():
    """HexSet hands are private: no client — human, LLM, or bot — must be
    able to learn what an opponent holds by noticing that proposing to
    trade for it is or isn't offered. `hexset_ui.actions.legal_actions`'s
    own PROPOSE_TRADE sample filters to pairs some opponent could currently
    cover — correct for a search that already sees the true state, but
    exactly the leak a wire-facing payload must not repeat. See
    `webplay.fair_legal_actions`/`_proposable_options`."""
    from hexset.board.terrain import Resource

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

    session = a_session(game, {0})
    proposals = [a for a in session.legal_wire_actions(0) if a["type"] == "PROPOSE_TRADE"]
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
    from hexset.board.terrain import Resource

    game = a_game(seed=19)
    game.phase = Phase.ROLL
    game.current_player = 0
    game.state.hands[0][Resource.WHEAT] += 6

    session = a_session(game, {0})
    kinds = {a["type"] for a in session.legal_wire_actions(0)}

    assert "PROPOSE_TRADE" not in kinds
    assert "BANK_TRADE" not in kinds
    assert "ROLL" in kinds

def test_an_explicit_ask_is_honoured():
    from hexset.board.terrain import Resource

    game = a_game(seed=8)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2, 3):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = a_session(game, {0})
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(3, 2, 1)
    )
    session.submit(0, action_to_wire(offer))

    assert game.pending_responders == [3, 2, 1]

# --- Log summarizing: builds and trades collapse into one entry -----------

def test_setup_settlement_and_road_collapse_into_one_log_line():
    """Successive builds by the same actor merge into one entry, not one
    each — see GameSession._log_action's _BUILD_KIND streak."""
    game = a_game(seed=10)
    seat = to_move(game)
    session = a_session(game, {seat})

    settlement = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT)
    session.submit(seat, action_to_wire(settlement))
    assert len(session.log_for(seat)) == 1

    road = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_ROAD)
    session.submit(seat, action_to_wire(road))

    assert len(session.log_for(seat)) == 1  # rewritten, not appended to
    text = session.log_for(seat)[0]
    assert "settlement" in text and "road" in text
    assert text.count("placed") == 1  # one merged sentence, not two

def test_a_build_streak_breaks_on_a_different_actor():
    game = a_game(seed=11)
    seat = to_move(game)
    session = a_session(game, {0, 1, 2, 3})

    settlement = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT)
    session.submit(seat, action_to_wire(settlement))
    road = next(a for a in legal_actions(game) if a.type is ActionType.SETUP_ROAD)
    session.submit(seat, action_to_wire(road))
    mine = session.log_for(seat)[0]
    assert len(session.log_for(seat)) == 1  # this seat's merged settlement+road

    # The next seat in the snake places too.
    next_seat = to_move(game)
    assert next_seat != seat
    for _ in range(2):
        action = next(
            a for a in legal_actions(game)
            if a.type in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD)
        )
        session.submit(next_seat, action_to_wire(action))

    lines = session.log_for(seat)
    assert len(lines) >= 2  # this seat's line, plus at least the next seat's
    assert lines[0] == mine  # the other seat's placements started their own, not this one

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
    from hexset.board.terrain import Resource

    game = a_game(seed=13)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state

    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = a_session(game, {0})
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(1, 2)
    )
    session.submit(0, action_to_wire(offer))
    assert session.log_for(0) == []  # held back until the offer concludes

    session._apply(1, Action(ActionType.DECLINE_TRADE))
    assert session.log_for(0) == []  # still pending — seat 2 hasn't answered yet

    session._apply(2, Action(ActionType.ACCEPT_TRADE))

    assert len(session.log_for(0)) == 1
    text = session.log_for(0)[0]
    assert "offered" in text and "accepted" in text
    assert "Player 3" in text  # seat 2, who actually accepted
    # Not "declined": seat 1's decline never gets named — see the docstring.
    assert "declined" not in text
    assert "Player 2" not in text  # the decliner isn't named at all

def test_a_trade_nobody_can_cover_is_still_legal_and_reads_as_declined():
    """propose_trade() concludes an uncoverable offer on the spot — no
    DECLINE_TRADE/ACCEPT_TRADE is ever coming to flush a held-back buffer,
    so this must not wait for one. And it must be reachable through
    submit itself, not just a direct _apply: legal_actions()
    only *samples* coverable (give, want) pairs (see its PROPOSE_TRADE
    enumerator's own docstring), but a proposal nobody can cover is still a
    legal move — see is_legal's docstring for why it's checked against
    can_propose instead of sample membership."""
    from hexset.board.terrain import Resource

    game = a_game(seed=14)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    # Nobody else holds any ore, so nobody is eligible to respond.

    session = a_session(game, {0})
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    # The server must actually have offered this — see the module docstring's
    # "never build an action the engine did not offer" rule the frontend
    # leans on — not merely tolerate it when submitted directly.
    assert action_to_wire(offer) in session.legal_wire_actions(0)
    session.submit(0, action_to_wire(offer))

    assert len(session.log_for(0)) == 1
    assert "offered" in session.log_for(0)[0]
    assert "declined" in session.log_for(0)[0]
    # Deliberately not "nobody could cover it": that would state opponent
    # hand contents as fact. HexSet hands are private.
    assert "cover" not in session.log_for(0)[0].lower()

def test_a_trade_everyone_declines_summarizes_into_one_line():
    """Reads as a single generic 'Everyone declined.' — not one line per
    decliner and not a count — regardless of how many opponents were
    actually asked, so it can't be distinguished from an offer nobody was
    eligible to take at all (see test_a_trade_nobody_can_cover...)."""
    from hexset.board.terrain import Resource

    game = a_game(seed=15)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = a_session(game, {0})
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(1, 2)
    )
    session.submit(0, action_to_wire(offer))
    session._apply(1, Action(ActionType.DECLINE_TRADE))
    assert session.log_for(0) == []

    session._apply(2, Action(ActionType.DECLINE_TRADE))

    assert len(session.log_for(0)) == 1
    assert session.log_for(0)[0].count("declined") == 1  # "Everyone declined.", not one per seat
    assert "Player 2" not in session.log_for(0)[0] and "Player 3" not in session.log_for(0)[0]

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
    session = a_session(game, {0})

    _discard_all(session, 0)

    assert game.discard_quota[0] == 0  # four cards actually went
    assert len(session.log_for(0)) == 1
    assert session.log_for(0)[0].count("discarded") == 1

def test_a_humans_discard_line_names_the_resources_with_counts():
    game = _owing_game(seed=21, seat=0, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0})

    _discard_all(session, 0)

    text = session.log_for(0)[0]
    # Counted, not repeated — and never pluralised (see _resource_counts).
    assert "4 Wood." in text or "4 Brick." in text or "2 Wood, 2 Brick" in text
    assert "Woods" not in text and "Bricks" not in text

def test_another_seats_discard_line_is_a_bare_count_never_the_resources():
    """A collapsed line is exactly where a whole hidden hand would leak at
    once — the same rule _describe applies to any seat but the reader's."""
    game = _owing_game(seed=22, seat=1, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0, 1})

    _discard_all(session, 1)

    text = session.log_for(0)[0]
    assert "discarded 4 cards" in text
    assert not any(r in text for r in RESOURCE_NAMES)

def test_two_seats_discarding_get_a_line_each():
    game = _owing_game(seed=23, seat=0, hand=[4, 4, 0, 0, 0])
    game.state.hands[1] = [4, 4, 0, 0, 0]
    game.discard_quota[1] = 4
    session = a_session(game, {0, 1})

    _discard_all(session, 0)
    _discard_all(session, 1)

    assert len(session.log_for(0)) == 2  # not merged across actors

def test_two_seats_at_one_table_are_owed_two_different_transcripts():
    """The whole reason the log is a fold over stored events rather than a
    list of sentences: one shared transcript cannot say two things at once,
    and a discard spells out the cards only for the seat that lost them."""
    game = _owing_game(seed=27, seat=0, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0, 1})

    _discard_all(session, 0)

    mine, theirs = session.log_for(0)[0], session.log_for(1)[0]
    assert any(r in mine for r in RESOURCE_NAMES)  # named, to the seat that paid
    assert "discarded 4 cards" in theirs
    assert not any(r in theirs for r in RESOURCE_NAMES)

def test_a_spectator_is_owed_the_least_of_anyone():
    game = _owing_game(seed=28, seat=0, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0})

    _discard_all(session, 0)

    assert "discarded 4 cards" in session.log_for(None)[0]

def test_consecutive_bank_trades_of_the_same_pair_sum_into_one_line():
    game = a_game(seed=24)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [8, 0, 0, 0, 0]
    session = a_session(game, {0})

    trade = next(
        a for a in legal_actions(game)
        if a.type is ActionType.BANK_TRADE and a.a == 0 and a.b == 4
    )
    session._apply(0, trade)
    session._apply(0, trade)

    assert len(session.log_for(0)) == 1
    assert "8 Wood" in session.log_for(0)[0] and "2 Ore" in session.log_for(0)[0]

def test_undoing_a_bank_trade_refunds_the_hand_and_drops_the_line():
    game = a_game(seed=24)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [8, 0, 0, 0, 0]
    session = a_session(game, {0})

    trade = next(
        a for a in legal_actions(game)
        if a.type is ActionType.BANK_TRADE and a.a == 0 and a.b == 4
    )
    before_bank = list(session.game.state.bank)
    session.submit(0, action_to_wire(trade))
    assert session.game.state.hands[0] != [8, 0, 0, 0, 0]
    assert session.log_for(0)

    session.undo_last_build(0)

    assert session.game.state.hands[0] == [8, 0, 0, 0, 0]
    assert session.game.state.bank == before_bank
    assert session.log_for(0) == []
    assert session._undo is None

def test_a_different_bank_pair_starts_its_own_line():
    game = a_game(seed=25)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [4, 4, 0, 0, 0]
    session = a_session(game, {0})

    for give, want in ((0, 4), (1, 4)):
        action = next(
            a for a in legal_actions(session.game)
            if a.type is ActionType.BANK_TRADE and a.a == give and a.b == want
        )
        session._apply(0, action)

    assert len(session.log_for(0)) == 2

def test_a_roll_between_two_discards_keeps_them_apart():
    """Only one run is ever open, so nothing can reach back across an
    intervening line to join something older."""
    game = _owing_game(seed=26, seat=0, hand=[4, 4, 0, 0, 0])
    session = a_session(game, {0})
    _discard_all(session, 0)
    assert len(session.log_for(0)) == 1

    session.game.phase = Phase.ROLL
    session.game.current_player = 0
    session._apply(0, Action(ActionType.ROLL))
    assert len(session.log_for(0)) == 2

    session.game.phase = Phase.DISCARD
    session.game.state.hands[0] = [2, 0, 0, 0, 0]
    session.game.discard_quota = [1, 0, 0, 0]
    _discard_all(session, 0)

    assert len(session.log_for(0)) == 3  # a fresh discard line, not a swollen one

def test_ending_a_turn_writes_no_log_line():
    """Every turn ends eventually and the next line already implies it — a
    dedicated line for each one was pure noise, not information."""
    game = a_game(seed=16)
    game.phase = Phase.MAIN
    game.current_player = 0
    session = a_session(game, {0})

    session._apply(0, Action(ActionType.END_TURN))

    assert session.log_for(0) == []

def test_ending_a_turn_closes_an_open_run():
    """END_TURN writes no line of its own but still breaks the streak: an
    identical trade afterwards is a second trade, not more of the first, even
    though actor, pair and round number all still match."""
    game = a_game(seed=17)
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [8, 0, 0, 0, 0]
    session = a_session(game, {0})

    trade = next(
        a for a in legal_actions(game)
        if a.type is ActionType.BANK_TRADE and a.a == 0 and a.b == 4
    )
    session._apply(0, trade)
    session._apply(0, Action(ActionType.END_TURN))
    assert len(session.log_for(0)) == 1  # END_TURN itself wrote nothing

    session.game.phase = Phase.MAIN
    session.game.current_player = 0
    session._apply(0, trade)

    assert len(session.log_for(0)) == 2
    assert session.round == 1  # the run's own key never changed; END_TURN broke it

def test_undo_is_available_to_any_claimed_seat_not_just_a_person():
    """`_UNDOABLE_BUILDS` no longer special-cases who the actor is — any
    claimed seat's own qualifying build is its own to take back, the same
    choke point regardless of what's driving that seat (see `_apply`)."""
    game = a_game(seed=5)
    game.phase = Phase.MAIN
    game.current_player = 1
    game.state.vertex_owner[0] = 1  # something of seat 1's own to build from
    game.state.hands[1] = [1, 1, 0, 0, 0]  # a road's cost (see economy.Purchase.ROAD)
    session = a_session(game, {0, 1})

    road = next(a for a in legal_actions(game) if a.type is ActionType.BUILD_ROAD)
    session.submit(1, action_to_wire(road))

    assert session._undo is not None
    assert session._undo.actor == 1
    session.undo_last_build(1)  # raises if seat 1 weren't allowed to

def test_state_view_hides_opponent_hands_but_reveals_the_viewers():
    game = a_game(seed=8)
    seat = to_move(game)
    other = (seat + 1) % game.state.num_players
    session = a_session(game, {seat})

    game.state.hands[seat][0] = 3
    game.state.hands[other][0] = 5

    view = session.state_view(seat)
    players = {p["seat"]: p for p in view["players"]}
    assert "hand" in players[seat]
    assert players[seat]["hand"]["Wood"] == 3
    assert "hand" not in players[other]
    assert players[other]["hand_size"] == 5

def test_state_view_reveals_every_hand_once_the_game_is_over():
    game = a_game(seed=9)
    seat = to_move(game)
    session = a_session(game, {seat})
    game.won_by = (seat + 1) % game.state.num_players
    game.phase = Phase.GAME_OVER

    view = session.state_view(seat)
    assert all("hand" in p for p in view["players"])
    assert view["legal_actions"] == []

def test_state_view_carries_the_public_ledger_for_every_seat():
    """Resource *counting* is public knowledge in this game — only a
    steal's identity and dev-card types are hidden (see `hexset_ui.ledger`)
    — so `known`/`unknown` show up for every seat, reveal or not, unlike
    `hand`."""
    game = a_game(seed=8)
    seat = to_move(game)
    other = (seat + 1) % game.state.num_players
    session = a_session(game, {seat})

    game.ledger.receive(other, 0, 2)

    view = session.state_view(seat)
    players = {p["seat"]: p for p in view["players"]}
    assert players[other]["known"]["Wood"] == 2
    assert players[other]["unknown"] == 0
    assert "hand" not in players[other]

def test_state_view_does_not_expose_who_is_eligible_to_respond_to_an_offer():
    """`game.pending_responders` is exactly who's eligible to cover the open
    offer, in ask order — sending it to the client before anyone has
    actually responded would leak the same hidden hand information the log
    (see _log_action's "Everyone declined." handling) is built to hide,
    just earlier and over a different channel."""
    from hexset.board.terrain import Resource

    game = a_game(seed=18)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = a_session(game, {0})
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session.submit(0, action_to_wire(offer))
    assert game.pending_responders  # the offer really is pending on someone

    view = session.state_view(0)
    assert "responders" not in view["offer"]

def test_state_view_offer_answered_is_the_proposers_own_information_only():
    """`offer_answered`'s wire counterpart: filtered exactly the way
    `record.py:build_record` filters it — all-zero (absent, here) unless
    `viewer == offer.proposer`, since a responder must not condition on an
    earlier decline."""
    from hexset.board.terrain import Resource

    game = a_game(seed=18)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    state.bank[Resource.WOOD] -= 1
    state.hands[0][Resource.WOOD] += 1
    for seat in (1, 2):
        state.bank[Resource.ORE] -= 1
        state.hands[seat][Resource.ORE] += 1

    session = a_session(game, {0, 1, 2})
    offer = Action(
        ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1), ask=(1, 2)
    )
    session.submit(0, action_to_wire(offer))
    session._apply(1, Action(ActionType.DECLINE_TRADE))

    assert session.state_view(0)["offer"]["answered"] == [1]
    assert "answered" not in session.state_view(2)["offer"]
    assert "answered" not in session.state_view(None)["offer"]

def test_state_view_reports_locked_seats():
    from hexset_ui.seating import lock_seat

    game = a_game(seed=8)
    lock_seat(game, 2)
    session = a_session(game, {0})

    assert session.state_view(0)["locked"] == [2]

# --- Recording and journalling ----------------------------------------------

SEED = 42

@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """One game played out in full, journalled to its own directory — every
    seat driven the same way, through `submit`, since there is no more
    distinction between "the human seat" and "the bot's seat" at this
    layer (see `botclient.py` for where that distinction now lives, one
    layer up).

    Module-scoped because playing a whole game is by far the slowest thing in
    this file: every test below reads the same finished game rather than
    dealing another one of its own.
    """
    directory = tmp_path_factory.mktemp("games")
    # Two independent random.Random(SEED) instances, matching what
    # `api.build_session` does: the board spends one stream and `start` gets
    # a fresh one, so the game's own rng must begin from the same untouched
    # state here too.
    board = random_base_board(random.Random(SEED))
    game = start_at(board, 4, random.Random(SEED), first=0)
    session = GameSession(
        game=game,
        claimed_seats={0, 1, 2, 3},
        seed=SEED,
        journal=open_journal(SEED, str(directory)),
    )

    driver = RandomBot(rng=random.Random(2))
    steps = 0
    while not is_over(session.game) and steps < 4000:
        seat = to_move(session.game)
        action = driver.choose(session.game)
        session.submit(seat, action_to_wire(action))
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
    `api.resume_session` makes — rather than a replay written for the
    test. A journal that replays clean here is one a returning player would
    actually get their game back from.
    """
    session, directory = played
    events = journal_events(directory)
    header = events[0]
    assert header["seed"] == SEED
    assert header["first"] == 0

    board = random_base_board(random.Random(SEED))
    resumed = GameSession(
        game=start_at(board, header["num_players"], random.Random(SEED), first=header["first"]),
        claimed_seats=set(header["human_seats"]),
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
    """The sidebar hides this from anyone but the thief and the victim (see
    `_describe`); the journal never does."""
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
    seat = to_move(game)
    session = a_session(game, {seat})
    session._apply(seat, legal_actions(game)[0])
    assert session.journal is None  # nothing to have written to

def test_an_unwritable_directory_costs_the_journal_and_not_the_game(tmp_path):
    """A player mid-turn should not lose a game to a full disk or a mount that
    came up read-only."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")  # mkdir under a regular file cannot succeed
    game = a_game(seed=7)
    seat = to_move(game)
    session = a_session(game, {seat}, journal=open_journal(1, str(blocked / "games")))
    session._apply(seat, legal_actions(game)[0])
    assert session.journal._off

def test_an_undone_placement_is_written_down_not_erased(tmp_path):
    """The journal is append-only and read forwards, so a step number that
    quietly came round twice would leave a reader unable to say which of the
    two actions counted."""
    game = a_game(seed=5)
    seat = to_move(game)
    session = a_session(game, {seat}, journal=open_journal(5, str(tmp_path)))
    settlement = next(
        a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT
    )
    session.submit(seat, action_to_wire(settlement))
    session.undo_last_build(seat)

    events = journal_events(tmp_path)
    assert [e["kind"] for e in events] == ["game", "action", "undo"]
    assert events[1]["type"] == "SETUP_SETTLEMENT"
    assert events[2]["back_to"] == 0  # everything from step 0 did not happen


def test_to_move_does_not_reveal_who_can_cover_an_offer():
    """PR #2 defect 3. `to_move` during `TRADE_RESPOND` is
    `pending_responders[0]`, and `pending_responders` is
    `trading.responders(...)` -- the seats that *can cover the offer*. Sending
    it to every poller says which seats hold the wanted card and which were
    skipped: a hand's composition, on the wire, to bystanders and to the
    token-free observer, at exactly the moment the offer block is careful to
    omit `responders` and the log is careful to say a uniform "Everyone
    declined."

    Rigged so only seat 2 holds ore: seat 0 offers wood for ore, so seat 2 is
    the only responder, and a leak is unambiguous rather than probabilistic.
    """
    from hexset.board.terrain import Resource

    game = a_game(seed=18)
    game.phase = Phase.MAIN
    game.current_player = 0
    state = game.state
    for hand in state.hands:
        hand[Resource.ORE] = 0
    state.hands[0][Resource.WOOD] += 1
    state.hands[2][Resource.ORE] = 1

    session = a_session(game, {0, 1, 2, 3})
    offer = Action(ActionType.PROPOSE_TRADE, give=(1, 0, 0, 0, 0), want=(0, 0, 0, 0, 1))
    session.submit(0, action_to_wire(offer))
    assert game.phase is Phase.TRADE_RESPOND
    assert game.pending_responders == [2]

    # The one seat actually being asked is told so -- it has to act.
    assert session.state_view(2)["to_move"] == 2

    # Nobody else is. They are told the proposer, which the offer block
    # already names, so no client loses its "am I on move?" test.
    for viewer in (0, 1, 3, None):
        view = session.state_view(viewer)
        assert view["to_move"] == 0, viewer
        assert view["to_move"] != 2, viewer

    # And `current_player` was never the leak: during TRADE_RESPOND the turn
    # is still the proposer's, only the decision moved.
    assert session.state_view(None)["current_player"] == 0


def test_to_move_is_unfiltered_in_every_phase_but_trade_respond():
    """The filter is narrow on purpose. Discarding also hands the decision to
    somebody other than the current player, but who owes a discard is public
    (hand sizes are, and `discard_quota` is served), so filtering there would
    cost every client its turn indicator for nothing."""
    from hexset.board.terrain import Resource

    game = a_game(seed=31)
    session = a_session(game, {0, 1, 2, 3})
    for _ in range(40):
        if is_over(game):
            break
        if game.phase is Phase.TRADE_RESPOND:
            continue
        seat = to_move(game)
        for viewer in (None, 0, 1, 2, 3):
            assert session.state_view(viewer)["to_move"] == seat, (game.phase, viewer)
        session.submit(seat, action_to_wire(legal_actions(game)[0]))

    game.phase = Phase.DISCARD
    game.current_player = 0
    game.state.hands[2] = [8, 0, 0, 0, 0]
    game.discard_quota = [0, 0, 4, 0]
    assert to_move(game) == 2
    for viewer in (None, 0, 1, 2, 3):
        assert session.state_view(viewer)["to_move"] == 2, viewer
