# SPDX-License-Identifier: GPL-3.0-only
"""Exercises the state translator against real, randomly-played catanatron games.

Random play is deliberately the stress test here rather than a scripted one:
it is cheap to run hundreds of decisions through, and it touches every phase
(setup, roll, discard, robber, main, dev cards) without hand-picking the
positions that reach them.
"""

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

from hexset.board.terrain import NUM_RESOURCES
from hexset.cards import NUM_DEV_CARDS
from hexset.game import Phase
from hexset.state import NO_OWNER, Building

from catanatron.game import Game as CatanatronGame
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import get_played_dev_cards

from hexset.catanatron.board import translate_board
from hexset.catanatron.state import translate


def _play_and_snapshot(num_ticks: int, seed: int):
    random.seed(seed)
    players = [RandomPlayer(c) for c in Color]
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    game = CatanatronGame(players, catan_map=catan_map)
    mapping = translate_board(catan_map)
    rng = random.Random(seed)

    snapshots = []
    for _ in range(num_ticks):
        if game.winning_color() is not None:
            break
        our_game, seats = translate(game, mapping, rng)
        played = sum(get_played_dev_cards(game.state, c) for c in game.state.colors)
        snapshots.append((our_game, seats, played))

        action = game.state.current_player().decide(game, game.playable_actions)
        # RandomPlayer.decide is defined on Player already; construct like Game.play does
        game.execute(action)
    return snapshots, mapping


@pytest.mark.parametrize("seed", range(20))
def test_translated_snapshots_conserve_resources_and_occupancy(seed):
    snapshots, mapping = _play_and_snapshot(600, seed=seed)
    assert len(snapshots) > 50  # the game actually progressed

    phases_seen = {s[0].phase for s in snapshots}
    assert Phase.DISCARD in phases_seen or Phase.ROBBER in phases_seen, (
        "a 600-tick random game never rolled a 7 -- widen the run, this test "
        "is meant to exercise those phases"
    )

    for our_game, seats, played in snapshots:
        state = our_game.state
        total_resources = sum(sum(hand) for hand in state.hands) + sum(state.bank)
        assert total_resources == NUM_RESOURCES * 19

        total_dev = sum(
            sum(state.dev_cards[p]) + sum(state.new_dev_cards[p])
            for p in range(state.num_players)
        ) + len(state.deck)
        assert total_dev + played == 25  # standard 25-card development deck

        # Every seat has 0-4 non-negative counts and phase is one we support.
        assert our_game.phase in (
            Phase.SETUP_SETTLEMENT,
            Phase.SETUP_ROAD,
            Phase.ROLL,
            Phase.DISCARD,
            Phase.ROBBER,
            Phase.MAIN,
        )
        assert 0 <= our_game.current_player < state.num_players

        occupied_vertices = sum(1 for b in state.vertex_building if b != Building.NONE)
        occupied_edges = sum(1 for o in state.edge_owner if o != NO_OWNER)
        assert occupied_vertices <= 54
        assert occupied_edges <= 72


def test_first_snapshot_is_setup_settlement_with_nothing_built():
    snapshots, _ = _play_and_snapshot(1, seed=11)
    our_game, seats, _ = snapshots[0]
    assert our_game.phase is Phase.SETUP_SETTLEMENT
    assert all(b == Building.NONE for b in our_game.state.vertex_building)
    assert all(o == NO_OWNER for o in our_game.state.edge_owner)
