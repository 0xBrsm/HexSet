# SPDX-License-Identifier: GPL-3.0-only
"""A catanatron `Player` sitting at a HexSet table -- `hexset.catanatron.bot`.

The direction the rest of this suite covers reads a live catanatron game;
this one mirrors a live HexSet game back into catanatron, so the guard has to
be that the mirror is the *same position*: the board translation run backwards
and forwards must land on the board it started from, and a position mirrored
into catanatron and read back out through `state.translate` must agree with
the original on everything both engines represent.
"""

from __future__ import annotations

import random

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

import hexset.bots  # noqa: F401 -- registers the "heximax" presets
from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.arena import entrant_from_name, spawn
from hexset.board.board import random_base_board
from hexset.game import Phase, is_over, start, to_move

from catanatron.models.player import Color

from hexset.catanatron.board import catanatron_map, translate_board
from hexset.catanatron.bot import CatanatronBot
from hexset.catanatron.state import seating, to_catanatron, translate


def _positions(seeds, seats=4):
    """Real positions from heximax self-play, sampled at every decision."""
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        board = random_base_board(rng)
        game = start(board, seats, rng)
        bots = [
            spawn(entrant_from_name("heximax"), board, random.Random(seed * 10 + i))
            for i in range(seats)
        ]
        depth = random.Random(seed).randrange(20, 140)
        for _ in range(depth):
            if is_over(game):
                break
            out.append(game)
            apply(game, bots[to_move(game)].choose(game))
    return out


@pytest.fixture(scope="module")
def positions():
    return _positions(range(3))


def test_the_map_is_the_boards_own_translation_run_backwards():
    """`catanatron_map` is `translate_board`'s inverse, ports included.

    Ports are the half that could plausibly be wrong and still look right:
    HexSet spaces them evenly around the coast rather than at catanatron's
    official positions, so every one of them is re-seated on a coastal edge
    the template has no port at.
    """
    for seed in range(3):
        board = random_base_board(random.Random(seed))
        mirrored = translate_board(catanatron_map(board)).board
        assert mirrored.topology == board.topology
        assert mirrored.terrain == board.terrain
        assert mirrored.tokens == board.tokens
        # `vertices` is an unordered pair on either side; everything else is
        # the port itself.
        assert {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in mirrored.ports
        } == {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in board.ports
        }


def test_a_position_survives_the_round_trip(positions):
    """Mirrored into catanatron and read back out, a position is unchanged.

    The one field that cannot round-trip exactly is the split between matured
    and freshly-bought development cards: catanatron records maturity as one
    boolean per card type where HexSet counts copies, so only the per-type
    *total* is preserved (see `state.to_catanatron`). Everything else is
    compared as-is.
    """
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
    """The offer is a bijection between the two engines' action sets.

    `CatanatronBot` never hands catanatron's own `playable_actions` to the
    player: it offers HexSet's legal actions translated forwards, so the
    inverse is a dict lookup. What this pins is that the translation forwards
    is injective over that legal set -- two of our actions collapsing onto one
    of catanatron's would silently drop a move. `PLAY_KNIGHT` used to be the
    one documented exception here (catanatron asks it as two decisions, so it
    offered one `PLAY_KNIGHT_CARD` however many robber targets we had); the
    knight two-step fix drops its operand, so it maps one-to-one now exactly
    like every other simple action, and no exception remains.
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
    """`PLAY_KNIGHT` no longer folds catanatron's two decisions into one: the
    bridge is asked for a bare `PLAY_KNIGHT` (no operand), applying it enters
    `Phase.ROBBER` on both sides, and the bridge is asked again for the
    `MOVE_ROBBER` that follows -- the same one hexset decision either engine
    would ask for after a seven.

    Reached by handing the seat a matured knight in `Phase.ROLL` -- the phase
    where HexSet offers the roll and the knight and nothing else, and
    catanatron's single `PLAY_TURN` prompt would offer every dev card.
    """
    from hexset.cards import DevCard

    rng = random.Random(3)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    bots = [spawn(entrant_from_name("heximax"), board, random.Random(i)) for i in range(4)]
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bots[to_move(game)].choose(game))
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
    """Two full games check adapter legality and bounded termination."""
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
                entrant_from_name("catanatron" if i == seat else "heximax"),
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
    """`/api/models` and `POST /api/bot`, the two the browser actually uses.

    The order is the picker's order: `heximax` first (the default opponent),
    then `catanatron`, then whatever checkpoints are on disk.
    """
    from conftest import new_tables

    registry = new_tables()
    models = registry.handle("GET", "/api/models", {}, None)["models"]
    assert models[:2] == ["heximax", "catanatron"]

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
    # Reusing a bot with another board must replace both translation and player.
    other = start(random_base_board(random.Random(23)), 3, random.Random(23))
    assert bots[0].choose(other) in legal_actions(other)
    assert bots[0]._mapping is not bots[1]._mapping
    assert len(bots[0]._seats.color_of) == 3
    del bot, bots
    gc.collect()
    assert table_ref() is None
    assert key not in _TABLE_MIRRORS


@pytest.mark.parametrize("mode", ["off", "fast"])
def test_cached_board_matches_fresh_and_isolates_mutation(positions, mode):
    from hexset.catanatron.state import BoardMirrorCache
    from hexset.state import copy_state
    from hexset.catanatron.speedups import catanatron_speedups

    with catanatron_speedups(mode):
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
                from hexset.catanatron.state import _catanatron_board
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


@pytest.mark.parametrize("phase", [Phase.MAIN, Phase.ROLL])
def test_stranded_road_building_credit_does_not_deadlock_reference(phase):
    from hexset.state import can_place_road, place_road, place_settlement
    from hexset.game import pending_free_roads

    board = random_base_board(random.Random(3))
    game = start(board, 4, random.Random(4))
    place_settlement(game._state, 0, 0, connected=False)
    for _ in range(15):
        edge = next(e for e in range(len(game._state.edge_owner))
                    if can_place_road(game._state, 0, e))
        place_road(game._state, 0, edge)
    game.phase = phase
    game.current_player = 0
    game.free_roads = 1
    game.dev_card_played = True
    assert not pending_free_roads(game)
    seats = seating(tuple(list(Color)[:4]))
    mapping = translate_board(catanatron_map(board))
    mirror = to_catanatron(game, mapping, seats)
    assert not mirror.state.is_road_building
    assert mirror.state.free_roads_available == 0
    expected = ActionType.END_TURN if phase is Phase.MAIN else ActionType.ROLL
    reference = spawn(entrant_from_name("catanatron"), board, random.Random(5))
    assert reference.choose(game) == Action(expected)
    assert game.free_roads == 1  # the mirror never mutates the real game
