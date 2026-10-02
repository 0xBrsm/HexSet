# SPDX-License-Identifier: GPL-3.0-only
"""A catanatron `Player` at a HexSet table -- `hexset.catanatron.bot`.

The mirroring direction, so the guard is that the mirror is the *same
position*: the board translation run both ways lands on the board it started
from, and a mirrored position read back agrees with the original on
everything both engines represent.
"""

from __future__ import annotations

import random

import pytest

# A submodule, not bare "catanatron": this directory is named `catanatron`
# too, so with `tests/` on sys.path a bare import can resolve to it as an
# empty namespace package and skip nothing. `catanatron.game` exists only in
# the real distribution.
pytest.importorskip("catanatron.game")

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.arena import entrant_from_name, spawn
from hexset.board.board import random_base_board
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.play import step_randomly

from catanatron.models.player import Color

from hexset.catanatron._board import catanatron_map, translate_board
from hexset.catanatron.bot import CatanatronBot
from hexset.catanatron._state import seating, to_catanatron
from tests.catanatron.oracle import translate


def _positions(seeds, seats=4, every=7):
    """A snapshot every `every` plies of a random game per seed. Snapshots, not
    the game: collecting the live object collected one position per seed,
    however many times it was listed.
    """
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        board = random_base_board(rng)
        game = start(board, seats, rng)
        for ply in range(140):
            if is_over(game):
                break
            if ply % every == 0:
                out.append(imagine(game, random.Random(seed), randomize_deck=False))
            step_randomly(game, rng)
    return out


@pytest.fixture(scope="module")
def positions():
    return _positions(range(3))


def test_the_map_is_the_boards_own_translation_run_backwards():
    """Ports could be wrong and still look right: HexSet spaces them evenly
    around the coast, off the template's own port edges.
    """
    for seed in range(3):
        board = random_base_board(random.Random(seed))
        mirrored = translate_board(catanatron_map(board)).board
        assert mirrored.topology == board.topology
        assert mirrored.terrain == board.terrain
        assert mirrored.tokens == board.tokens
        assert {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in mirrored.ports
        } == {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in board.ports
        }


def test_a_position_survives_the_round_trip(positions):
    assert len(positions) > 50
    for game in positions:
        state = game.state(0, hidden=False)
        seats = seating(tuple(list(Color)[: state.num_players]))
        mapping = translate_board(catanatron_map(state.board))
        mirror = to_catanatron(game, mapping, seats)
        back, back_seats = translate(mirror, mapping, random.Random(0))
        again = back.state(0, hidden=False)

        assert back_seats == seats
        assert again.hands == state.hands
        assert again.bank == state.bank
        assert again.vertex_owner == state.vertex_owner
        assert again.vertex_building == state.vertex_building
        assert again.edge_owner == state.edge_owner
        assert again.robber == state.robber
        assert again.deck == state.deck
        assert again.knights_played == state.knights_played
        assert again.longest_road_holder == state.longest_road_holder
        assert again.largest_army_holder == state.largest_army_holder
        assert [
            [held + fresh for held, fresh in zip(*seat)]
            for seat in zip(again.dev_cards, again.new_dev_cards)
        ] == [
            [held + fresh for held, fresh in zip(*seat)]
            for seat in zip(state.dev_cards, state.new_dev_cards)
        ]
        assert back.current_player == game.current_player
        assert back.phase is game.phase
        assert back.turns == game.turns
        assert back.discard_quota == game.discard_quota
        assert back.free_roads == game.free_roads
        assert back.dev_card_played == game.dev_card_played
        if game.phase is Phase.SETUP_ROAD:
            assert back.last_settlement == game.last_settlement


def test_every_offered_action_maps_back_to_exactly_one_of_ours(positions):
    """The forward translation must be injective over the legal set: two of ours
    collapsing onto one of catanatron's would silently drop a move.
    """
    bot = CatanatronBot()
    checked = 0
    for game in positions:
        state = game.state(0, hidden=False)
        bot._mapping = translate_board(catanatron_map(state.board))
        bot._seats = seating(tuple(list(Color)[: state.num_players]))
        mirror = to_catanatron(game, bot._mapping, bot._seats)
        raw = list(mirror.playable_actions)
        offered = bot._offer(game, mirror)

        ours = legal_actions(game)
        assert len(set(offered.values())) == len(offered), "two keys, one of our actions"
        for their, our in offered.items():
            assert their in raw, "offered an action catanatron never had"
            assert our in ours
        assert set(offered.values()) == set(ours), [a for a in ours if a not in offered.values()]
        checked += 1
    assert checked > 50


def test_a_knight_resolves_as_two_separate_decisions():
    from hexset.cards import DevCard

    rng = random.Random(3)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        step_randomly(game, rng)
    state = game.state(0, hidden=False)
    state.dev_cards[game.current_player][DevCard.KNIGHT] = 1
    assert game.phase is Phase.ROLL

    bot = CatanatronBot()
    action = bot.choose(game)
    assert action == Action(ActionType.PLAY_KNIGHT)
    assert action in legal_actions(game)

    apply(game, action)
    assert game.phase is Phase.ROBBER

    move = bot.choose(game)
    assert move.type is ActionType.MOVE_ROBBER
    assert move in legal_actions(game)
    assert move.a != state.robber


@pytest.mark.slow
def test_a_catanatron_seat_plays_out_full_games():
    from hexset.arena import MAX_ACTIONS

    import hexset.catanatron.bot  # noqa: F401 -- registers the preset

    games = 2
    for seed in range(games):
        rng = random.Random(1000 + seed)
        board = random_base_board(rng)
        game = start(board, 4, rng)
        seat = seed % 4
        bots = [
            spawn(
                entrant_from_name("catanatron" if i == seat else "test-trader"),
                board,
                random.Random(seed * 4 + i),
            )
            for i in range(4)
        ]
        for _ in range(MAX_ACTIONS):
            if is_over(game):
                break
            action = bots[to_move(game)].choose(game)
            assert action in legal_actions(game)
            apply(game, action)
        else:
            pytest.fail("adapter game exceeded the arena action cap")


# --- Seated at the served table -----------------------------------------------


def test_the_picker_offers_catanatron_and_a_seat_takes_it():
    from conftest import new_tables

    registry = new_tables()
    models = registry.handle("GET", "/api/models", {}, None)["models"]
    assert models[:2] == ["test-trader", "catanatron"]

    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    table = registry.get(code)
    open_seats = [i for i, s in enumerate(table.seats) if s.kind.name == "EMPTY"]
    seated = registry.handle(
        "POST", "/api/bot", {"seat": open_seats[0], "model": "catanatron"}, token
    )
    assert seated["seats"][open_seats[0]]["name"] == "catanatron"


def test_catanatron_can_spawn_without_parent_process_registration():
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = """
import random
from hexset.arena import Entrant, spawn, deal_board
bot = spawn(Entrant('catanatron', kind='catanatron'), deal_board(1, 0), random.Random(1))
assert type(bot).__name__ == 'CatanatronBot'
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert completed.returncode == 0, completed.stderr


def test_seats_share_map_and_release_table_mirrors():
    import gc
    import weakref
    from catanatron.models.player import Player
    from hexset.catanatron.bot import _TABLE_MIRRORS

    class First(Player):
        def decide(self, game, playable_actions):
            return playable_actions[0]

    rng = random.Random(22)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    bots = [CatanatronBot(First) for _ in range(4)]
    for bot in bots:
        assert bot.choose(game) in legal_actions(game)
    assert all(bot._mapping is bots[0]._mapping for bot in bots)
    table_ref = weakref.ref(bots[0]._table)
    key = (id(board), 4)
    other = start(random_base_board(random.Random(23)), 3, random.Random(23))
    assert bots[0].choose(other) in legal_actions(other)
    assert bots[0]._mapping is not bots[1]._mapping
    assert len(bots[0]._seats.color_of) == 3
    del bot, bots
    gc.collect()
    assert table_ref() is None
    assert key not in _TABLE_MIRRORS


def test_cached_board_matches_fresh_and_isolates_mutation(positions):
    from hexset.catanatron._state import BoardMirrorCache
    from hexset.state import copy_state

    for game in positions[::20]:
        state = game.state(0, hidden=False)
        mapping = translate_board(catanatron_map(state.board))
        seats = seating(tuple(list(Color)[:state.num_players]))
        cache = BoardMirrorCache(mapping, seats)
        # Same occupancy, then robber, building, road and holder changes.
        snapshots = [copy_state(state) for _ in range(6)]
        snapshots[1].robber = (state.robber + 1) % state.board.num_hexes
        for vertex, kind in enumerate(snapshots[2].vertex_building):
            if kind.name == "SETTLEMENT":
                snapshots[2].vertex_building[vertex] = type(kind).CITY
                break
        for edge, owner in enumerate(snapshots[3].edge_owner):
            if owner >= 0:
                snapshots[3].edge_owner[edge] = -1
                break
        snapshots[4].longest_road_holder = 0 if state.longest_road_holder != 0 else -1
        for vertex, owner in enumerate(snapshots[5].vertex_owner):
            if owner >= 0:
                snapshots[5].vertex_owner[vertex] = (owner + 1) % state.num_players
                break
        for snapshot in snapshots:
            from hexset.catanatron._state import _catanatron_board
            expected = _catanatron_board(snapshot, mapping, seats)
            for _ in range(2):
                actual = cache.board(snapshot)
                for field in vars(expected):
                    if field == "map":
                        assert actual.map is expected.map
                    elif field == "buildable_subgraph":
                        assert list(actual.buildable_subgraph.nodes) == list(expected.buildable_subgraph.nodes)
                        assert list(actual.buildable_subgraph.edges) == list(expected.buildable_subgraph.edges)
                    else:
                        assert getattr(actual, field) == getattr(expected, field), field
                actual.buildings.clear()
                actual.roads.clear()
                actual.road_lengths.clear()
                actual.board_buildable_ids.clear()
                actual.buildable_edges_cache[Color.RED] = [(100, 101)]
                actual.player_port_resources_cache[Color.RED] = {"ORE"}
                for components in actual.connected_components.values():
                    for component in components:
                        component.clear()
        fresh = to_catanatron(game, mapping, seats)
        cached = to_catanatron(game, mapping, seats, board_cache=cache)
        assert cached.state.player_state == fresh.state.player_state
        assert cached.state.buildings_by_color == fresh.state.buildings_by_color
        assert cached.playable_actions == fresh.playable_actions


def test_stranded_road_building_credit_does_not_deadlock_reference():
    from hexset.state import can_place_road, place_road, place_settlement
    from hexset.game import pending_free_roads

    board = random_base_board(random.Random(3))
    game = start(board, 4, random.Random(4))
    place_settlement(game._state, 0, 0, connected=False)
    for _ in range(15):
        edge = next(e for e in range(len(game._state.edge_owner))
                    if can_place_road(game._state, 0, e))
        place_road(game._state, 0, edge)
    game.phase = Phase.ROLL  # the credit outlived the turn it was dealt in
    game.current_player = 0
    game.free_roads = 1
    game.dev_card_played = True
    assert not pending_free_roads(game)
    seats = seating(tuple(list(Color)[:4]))
    mapping = translate_board(catanatron_map(board))
    mirror = to_catanatron(game, mapping, seats)
    assert not mirror.state.is_road_building
    assert mirror.state.free_roads_available == 0
    reference = spawn(entrant_from_name("catanatron"), board, random.Random(5))
    assert reference.choose(game) == Action(ActionType.ROLL)
    assert game.free_roads == 1


def test_mirror_rng_is_picklable_shared_by_copies_and_isolated_from_live_game():
    import pickle
    from hexset.arena import deal_game
    game = deal_game(145, 0, 4)
    board = game.state(0, hidden=False).board
    mapping = translate_board(catanatron_map(board))
    live_before, global_before = game.rng.getstate(), random.getstate()
    mirror = to_catanatron(game, mapping, seating(tuple(Color)))
    copied = mirror.copy()
    assert copied.random is copied.state.random is mirror.random is mirror.state.random
    assert mirror.random is not game.rng
    copied.random.random()
    restored = pickle.loads(pickle.dumps(mirror))
    assert restored.random is restored.state.random
    assert restored.random.getstate() == mirror.random.getstate()
    assert game.rng.getstate() == live_before
    assert random.getstate() == global_before


def test_a_mirror_without_a_stream_never_draws_the_live_games_dice():
    """A mirror's stream is its own: a copy of the live chance stream would
    have a sampling catanatron player roll the real game's next dice."""
    from catanatron.models.player import Player
    from hexset.arena import deal_game

    game = deal_game(145, 0, 4)
    board = game.state(0, hidden=False).board
    mapping = translate_board(catanatron_map(board))
    live_next = random.Random()
    live_next.setstate(game.rng.getstate())
    live_next = live_next.random()

    mirror = to_catanatron(game, mapping, seating(tuple(Color)))
    assert mirror.random.random() != live_next

    draws = []

    class RecordingPlayer(Player):
        def decide(self, mirror, actions):
            draws.append(mirror.copy().random.random())
            return actions[0]

    before = game.rng.getstate()
    CatanatronBot(RecordingPlayer).choose(game)
    assert draws and draws[0] != live_next
    assert game.rng.getstate() == before


def test_a_catanatron_spec_names_its_depth_and_world_vote(monkeypatch):
    import hexset.catanatron.bot as module
    from hexset.arena import deal_board

    entrant = entrant_from_name("catanatron:worlds=8:select=sample:temperature=0.5:depth=3")
    assert entrant.kind == "catanatron" and entrant.depth == 3
    assert entrant.options == (("select", "sample"), ("temperature", 0.5), ("worlds", 8))
    built = {}

    def recording(player, **kwargs):
        built.update(kwargs)
        return "bot"

    monkeypatch.setattr(module, "CatanatronBot", recording)
    spawn(entrant, deal_board(1, 0), random.Random(1))
    assert (built["worlds"], built["select"], built["temperature"]) == (8, "sample", 0.5)
    spawn(entrant_from_name("catanatron"), deal_board(1, 0), random.Random(1))
    assert (built["worlds"], built["select"], built["temperature"]) == (0, "argmax", 0.0)

    for bad in ("catanatron:depth=0", "catanatron:worlds=-1", "catanatron:select=best",
                "catanatron:width=3", "catanatron:depth=2:depth=3", "catanatron:"):
        with pytest.raises(ValueError, match="catanatron:"):
            entrant_from_name(bad)


def test_arena_reference_bots_use_independent_seeded_search_streams():
    from catanatron.models.player import Player
    from hexset.arena import Entrant, deal_game
    game = deal_game(145, 0, 4)
    board = game.state(0, hidden=False).board
    draws = []

    class RecordingPlayer(Player):
        def decide(self, mirror, actions):
            draws.append(mirror.copy().random.random())
            return actions[0]

    bot = spawn(Entrant('catanatron', kind='catanatron'), board, random.Random(145))
    other = spawn(Entrant('catanatron', kind='catanatron'), board, random.Random(99))
    bot.player = other.player = RecordingPlayer
    live_before, global_before = game.rng.getstate(), random.getstate()
    bot.choose(game)
    other.choose(game)
    bot.choose(game)
    expected = random.Random(145)
    assert draws == [expected.random(), random.Random(99).random(), expected.random()]
    assert game.rng.getstate() == live_before
    assert random.getstate() == global_before


def test_a_world_vote_reuses_reference_answers_per_decision():
    from catanatron.models.player import Player
    from hexset.arena import deal_game
    from hexset.bots.determinized import holdings_signature
    game = deal_game(96, 0, 4)
    calls = []
    class FirstPlayer(Player):
        def decide(self, mirror, actions):
            calls.append(mirror)
            return actions[0]
    rng = random.Random(7)
    before = rng.getstate(), game.rng.getstate()
    bot = CatanatronBot(FirstPlayer, worlds=40, rng=rng,
                       world_key=holdings_signature)
    assert bot.choose(game) in legal_actions(game)
    assert bot.choose(game) in legal_actions(game)
    assert len(calls) == 2, "forty worlds, but one reference answer per decision"
    assert (rng.getstate(), game.rng.getstate()) == before


def test_reference_world_vote_uses_current_pinned_ab2_and_reproduces(positions):
    game = positions[-1]
    before = game.rng.getstate()
    first = CatanatronBot(worlds=3, rng=random.Random(7)).choose(game)
    again = CatanatronBot(worlds=3, rng=random.Random(7)).choose(game)
    assert first == again
    assert first in legal_actions(game)
    assert game.rng.getstate() == before


def test_the_mirror_holds_the_seats_that_can_still_act():
    """A four-seat table with two seats closed is a two-player game. Mirroring
    the closed seats as colours that never move would have the reference
    engine search against opponents that do not exist."""
    from hexset.catanatron.bot import live_seats
    from hexset.catanatron._state import mirrored_seats, to_catanatron
    from hexset.catanatron.bot import _TableMirror
    from hexset.game import lock_seat
    from hexset.arena import deal_game

    game = deal_game(90000, 0, 4)
    for seat in (2, 3):
        lock_seat(game, seat)
    state = game.state(0, hidden=False)

    live = live_seats(game, state)
    assert live == (0, 1)

    table = _TableMirror(state.board, live)
    assert mirrored_seats(table.seats) == (0, 1)
    mirror = to_catanatron(game, table.mapping, table.seats, rng=random.Random(0))
    assert len(mirror.state.colors) == 2
    assert mirror.state.current_player_index < 2
    # catanatron indexes its own colour tuple, not the hexset seat number
    assert "P0_ACTUAL_VICTORY_POINTS" in mirror.state.player_state
    assert "P2_ACTUAL_VICTORY_POINTS" not in mirror.state.player_state


def test_a_seat_that_retires_with_pieces_stays_in_the_mirror():
    """Its buildings and roads still occupy the board, so its colour has to
    exist for the reference engine to see them."""
    from hexset.catanatron.bot import live_seats
    from hexset.game import lock_seat
    from hexset.arena import deal_game
    from hexset.state import Building

    game = deal_game(90000, 0, 4)
    state = game.state(0, hidden=False)
    vertex = next(v for v, o in enumerate(state.vertex_owner) if o == -1)
    state.vertex_owner[vertex] = 3
    state.vertex_building[vertex] = Building.SETTLEMENT
    lock_seat(game, 3)

    assert 3 in live_seats(game, state)
