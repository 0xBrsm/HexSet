# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import numpy as np
import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import DECK_SIZE
from hexset.encoding import (
    BANK_SCALE,
    HAND_SCALE,
    HEX_FEATURES,
    NUM_BUILDINGS,
    TURN_SCALE,
    _building_points,
    _seat,
    edge_features,
    encode,
    encode_batch,
    from_frame,
    global_columns,
    global_features,
    to_frame,
    vertex_features,
)
from hexset.game import Phase, is_over, start
from hexset.play import step_randomly
from hexset.state import NO_OWNER, Building
from hexset.victory import building_points, public_victory_points


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def arrays(obs):
    return (obs.hexes, obs.vertices, obs.edges, obs.globals)


def test_batched_encoding_is_byte_identical_to_the_canonical_path():
    rng = random.Random(31)
    games = [start(random_base_board(rng), 4, rng) for _ in range(8)]

    checked = 0
    for _ in range(80):
        for game in games:
            if not is_over(game):
                step_randomly(game, rng)

        live = [game for game in games if not is_over(game)]
        perspectives = [rng.randrange(game._state.num_players) for game in live]
        fast = encode_batch(live, perspectives)
        canonical = [
            encode(game, perspective)
            for game, perspective in zip(live, perspectives, strict=True)
        ]

        for got, want in zip(fast, canonical, strict=True):
            for got_array, want_array in zip(arrays(got), arrays(want), strict=True):
                assert np.array_equal(got_array, want_array)
                assert got_array.dtype == want_array.dtype == np.float32
            assert got.graph is want.graph
            checked += 1

    assert checked > 300


def test_serializing_one_batched_observation_does_not_carry_the_whole_tick():
    games = [a_game(seed=seed, steps=60 + seed) for seed in range(8)]
    observations = encode_batch(games, [game.current_player for game in games])
    assert all(o._packed is observations[0]._packed for o in observations)
    assert [o._row for o in observations] == list(range(8))

    restored = pickle.loads(pickle.dumps(observations[3]))

    assert restored._packed is None
    canonical = encode(games[3], games[3].current_player)
    for got, want in zip(arrays(restored), arrays(canonical), strict=True):
        assert np.array_equal(got, want)
    restored.hexes.fill(7.0)
    assert not np.array_equal(restored.hexes, observations[3].hexes)


@pytest.mark.parametrize("players", [4])
def test_shapes_match_the_declared_widths(players):
    obs = encode(a_game(players=players))
    assert obs.hexes.shape == (19, HEX_FEATURES)
    assert obs.vertices.shape == (54, vertex_features(players))
    assert obs.edges.shape == (72, edge_features(players))
    assert obs.globals.shape == (global_features(players),)


def test_everything_is_finite_and_bounded():
    obs = encode(a_game())
    for array in arrays(obs):
        assert np.isfinite(array).all()
        assert array.min() >= 0.0
        assert array.max() <= 1.0


def test_observations_on_one_board_do_not_share_memory():
    """A caller receiving the cached template uncopied would corrupt every later position."""
    game = a_game()
    first = encode(game)
    second = encode(game)
    assert first.hexes is not second.hexes
    assert first.vertices is not second.vertices

    first.hexes.fill(7.0)
    first.vertices.fill(7.0)
    third = encode(game)
    assert np.array_equal(third.hexes, second.hexes)
    assert np.array_equal(third.vertices, second.vertices)


def test_the_robber_is_marked_on_exactly_one_hex():
    game = a_game()
    obs = encode(game)
    flags = obs.hexes[:, HEX_FEATURES - 1]
    assert flags.sum() == 1.0
    assert flags[game._state.robber] == 1.0


def test_opponent_hand_contents_do_not_leak():
    game = a_game(players=3)
    state = game._state
    for player in (1, 2):
        for resource in range(5):
            state.bank[resource] += state.hands[player][resource]
            state.hands[player][resource] = 0

    state.hands[1][Resource.WOOD] = 1
    state.hands[2][Resource.ORE] = 1
    state.bank[Resource.WOOD] -= 1
    state.bank[Resource.ORE] -= 1
    before = encode(game, perspective=0)

    state.hands[1] = [0, 0, 0, 0, 1]
    state.hands[2] = [1, 0, 0, 0, 0]
    after = encode(game, perspective=0)

    for lhs, rhs in zip(arrays(before), arrays(after)):
        assert np.array_equal(lhs, rhs)


def test_opponent_development_cards_show_only_as_a_count():
    from hexset.cards import DevCard

    game = a_game(players=3)
    state = game._state
    state.dev_cards[1][DevCard.KNIGHT] = 2
    before = encode(game, perspective=0)

    state.dev_cards[1][DevCard.KNIGHT] = 0
    state.dev_cards[1][DevCard.MONOPOLY] = 2
    after = encode(game, perspective=0)

    for lhs, rhs in zip(arrays(before), arrays(after)):
        assert np.array_equal(lhs, rhs)


def _canonical_vertex_block(state, perspective):
    """`encode` reaches the same rows by table lookup on a combined key."""
    players = state.num_players
    width = NUM_BUILDINGS + players + 1
    out = np.zeros((state.board.topology.num_vertices, width), dtype=np.float32)
    for v in range(out.shape[0]):
        out[v, int(state.vertex_building[v])] = 1.0
        owner = state.vertex_owner[v]
        slot = players if owner == NO_OWNER else _seat(owner, perspective, players)
        out[v, NUM_BUILDINGS + slot] = 1.0
    return out


def _canonical_edges(state, perspective):
    players = state.num_players
    out = np.zeros(
        (state.board.topology.num_edges, edge_features(players)), dtype=np.float32
    )
    for e in range(out.shape[0]):
        owner = state.edge_owner[e]
        slot = players if owner == NO_OWNER else _seat(owner, perspective, players)
        out[e, slot] = 1.0
    return out


def _check_blocks(game, players):
    state = game._state
    for perspective in range(players):
        obs = encode(game, perspective=perspective)
        block = obs.vertices[:, : NUM_BUILDINGS + players + 1]
        assert np.array_equal(block, _canonical_vertex_block(state, perspective))
        assert np.array_equal(obs.edges, _canonical_edges(state, perspective))


@pytest.mark.parametrize("players", [3])
def test_the_table_lookups_agree_with_the_loops(players):
    rng = random.Random(11 + players)
    game = start(random_base_board(rng), players, rng)

    positions = 0
    while not is_over(game) and game.turns < 40:
        step_randomly(game, rng)
        _check_blocks(game, players)
        positions += players
    assert positions > 100

    state = game._state
    for owner in range(players):
        state.vertex_building[owner] = Building.CITY
        state.vertex_owner[owner] = owner
        state.vertex_building[players + owner] = Building.SETTLEMENT
        state.vertex_owner[players + owner] = owner
        state.edge_owner[owner] = owner
    _check_blocks(game, players)


@pytest.mark.parametrize("players", [4])
def test_building_points_agree_with_the_rules(players):
    rng = random.Random(21 + players)
    game = start(random_base_board(rng), players, rng)
    scored = 0

    while not is_over(game) and game.turns < 40:
        step_randomly(game, rng)
        state = game._state
        for perspective in range(players):
            obs = encode(game, perspective=perspective)
            points = _building_points(obs.vertices, players)
            for i in range(players):
                seat = (perspective + i) % players
                assert points[i] == building_points(state, seat)
                scored += points[i] > 0

    assert scored > 0




def _main_phase_game(seed: int = 5):
    from hexset.game import Phase

    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(400):
        if game.phase is Phase.MAIN:
            return game
        step_randomly(game, rng)
    raise AssertionError("no MAIN phase reached in 400 steps")


def _set_hand(game, player: int, resource, n: int) -> None:
    """Keeps `game.ledger` in sync, so a later spend reads it as certain."""
    game._state.hands[player][resource] = n
    game.ledger.seats[player].known[resource] = n


def _ledger_width(players: int = 4) -> int:
    return (players - 1) * (NUM_RESOURCES + 1)




def _ledger_tail(obs, players: int = 4):
    """Each opponent's known[5] + unknown, seat-relative, own seat excluded."""
    tail = obs.globals[-_ledger_width(players):]
    known = [tail[i * 6 : i * 6 + 5] for i in range(players - 1)]
    unknown = [tail[i * 6 + 5] for i in range(players - 1)]
    return known, unknown


def test_global_features_counts_the_ledger_block():
    """Pinned so a layout change has to own the new width."""
    assert global_features(4) == 67
    assert global_features(4) - global_features(3) == (
        1 + 1 + 1 + 1 + 2 + (NUM_RESOURCES + 1)
    )


def test_a_steal_shows_up_as_unknown_in_the_encoding():
    """The one-resource hand makes the outcome deterministic whatever was drawn."""
    from hexset.game import Phase, move_robber_to
    from hexset.ledger import SeatLedger

    game = _main_phase_game()
    game.phase = Phase.ROBBER
    thief, victim, bystander = 0, 1, 2
    game.current_player = thief
    game.ledger.seats[thief] = SeatLedger()
    game.ledger.seats[victim] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=0)
    game._state.hands[victim] = [1, 0, 0, 0, 0]

    topology = game._state.board.topology
    target = next(
        h
        for h in range(game._state.board.num_hexes)
        if h != game._state.robber
        and any(game._state.vertex_owner[v] == victim for v in topology.hex_vertices[h])
    )

    move_robber_to(game, target, victim)

    assert game.ledger.seats[thief].unknown == 1
    assert game.ledger.seats[victim].known == [0, 0, 0, 0, 0]
    assert game.ledger.seats[victim].unknown == 0

    known, unknown = _ledger_tail(encode(game, bystander))
    thief_slot = _seat(thief, bystander, 4) - 1
    victim_slot = _seat(victim, bystander, 4) - 1
    assert unknown[thief_slot] == pytest.approx(0.1)
    assert not known[victim_slot].any()
    assert unknown[victim_slot] == pytest.approx(0.0)


#
# The convention is "the perspective seat is slot 0, others follow in turn
# order". The two functions below are copies of `hexn`'s exact formulas, so a
# change of direction or modulus fails here though nothing here imports hexn.


def _hexn_ppo_rotate(rewards, seat):
    """`hexn.ppo.rotate`'s exact formula: board order -> perspective frame."""
    players = len(rewards)
    return tuple(rewards[(seat + i) % players] for i in range(players))


def _hexn_netbot_board_order(value, seat):
    """`hexn.netbot._board_order`'s formula: perspective frame -> board order."""
    players = len(value)
    return tuple(
        float(value[(board_seat - seat) % players]) for board_seat in range(players)
    )


@pytest.mark.parametrize("players", [4])
def test_to_frame_matches_the_arithmetic_hexn_uses(players):
    values = [float(i) for i in range(players)]
    for seat in range(players):
        assert to_frame(values, seat) == _hexn_ppo_rotate(values, seat)


@pytest.mark.parametrize("players", [3])
def test_from_frame_matches_the_arithmetic_hexn_uses(players):
    values = [float(i) for i in range(players)]
    for seat in range(players):
        assert from_frame(values, seat) == _hexn_netbot_board_order(values, seat)


@pytest.mark.parametrize("players", [3])
def test_global_columns_tile_the_vector_exactly(players):
    from itertools import pairwise

    columns = global_columns(players)
    spans = sorted(columns.values(), key=lambda s: s.start)

    assert spans[0].start == 0
    assert spans[-1].stop == global_features(players)
    for earlier, later in pairwise(spans):
        assert earlier.stop == later.start
    covered = sum(span.stop - span.start for span in spans)
    assert covered == global_features(players)


def test_global_columns_read_the_value_the_encoder_wrote():
    from hexset.ledger import SeatLedger

    players = 4
    perspective = 1
    game = _main_phase_game()

    for player in range(players):
        _set_hand(game, player, Resource.WOOD, player + 1)
    game._state.bank = [3, 4, 5, 6, 7]
    for player in range(players):
        game._state.dev_cards[player] = [1, 0, 0, 0, player]
        game._state.new_dev_cards[player] = [0, 0, 1, 0, 0]
    game._state.knights_played = [0, 2, 1, 3]
    game._state.longest_road_holder = 2
    game._state.largest_army_holder = 0
    game.phase = Phase.MAIN
    game.free_roads = 1
    game._state.deck = game._state.deck[:7]
    game.turns = 42
    game.ledger.seats[0] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=0)
    game.ledger.seats[2] = SeatLedger(known=[0, 2, 0, 0, 0], unknown=1)
    game.ledger.seats[3] = SeatLedger(known=[0, 0, 0, 3, 0], unknown=0)

    obs = encode(game, perspective)
    columns = global_columns(players)
    seats = to_frame(range(players), perspective)

    own_hand = obs.globals[columns["own_hand"]]
    assert own_hand == pytest.approx(
        np.array(game._state.hands[perspective]) / HAND_SCALE
    )

    opponent_hand_sizes = obs.globals[columns["opponent_hand_sizes"]]
    for i, seat in enumerate(seats[1:]):
        assert opponent_hand_sizes[i] == pytest.approx(
            sum(game._state.hands[seat]) / HAND_SCALE
        )

    assert obs.globals[columns["bank"]] == pytest.approx(
        np.array(game._state.bank) / BANK_SCALE
    )

    own_dev = [
        h + f
        for h, f in zip(
            game._state.dev_cards[perspective], game._state.new_dev_cards[perspective]
        )
    ]
    assert obs.globals[columns["own_dev_cards"]] == pytest.approx(
        np.array(own_dev) / 5.0
    )

    opponent_dev_counts = obs.globals[columns["opponent_dev_card_counts"]]
    for i, seat in enumerate(seats[1:]):
        expected = sum(game._state.dev_cards[seat]) + sum(
            game._state.new_dev_cards[seat]
        )
        assert opponent_dev_counts[i] == pytest.approx(expected / 5.0)

    knights = obs.globals[columns["knights_played"]]
    for i, seat in enumerate(seats):
        assert knights[i] == pytest.approx(game._state.knights_played[seat] / 5.0)

    vp = obs.globals[columns["victory_points"]]
    for i, seat in enumerate(seats):
        assert vp[i] == pytest.approx(public_victory_points(game._state, seat) / 10.0)

    longest = obs.globals[columns["longest_road_holder"]]
    assert longest[_seat(game._state.longest_road_holder, perspective, players)] == 1.0
    assert longest.sum() == 1.0

    largest = obs.globals[columns["largest_army_holder"]]
    assert largest[_seat(game._state.largest_army_holder, perspective, players)] == 1.0
    assert largest.sum() == 1.0

    phase_block = obs.globals[columns["phase"]]
    assert phase_block[int(game.phase)] == 1.0
    assert phase_block.sum() == 1.0

    assert obs.globals[columns["free_roads"]][0] == pytest.approx(
        game.free_roads / 2.0
    )
    assert obs.globals[columns["deck_size"]][0] == pytest.approx(
        len(game._state.deck) / DECK_SIZE
    )
    assert obs.globals[columns["turn"]][0] == pytest.approx(
        min(game.turns / TURN_SCALE, 1.0)
    )

    ledger_block = obs.globals[columns["ledger"]]
    for i, seat in enumerate(seats[1:]):
        entry = game.ledger.seats[seat]
        known = ledger_block[i * 6 : i * 6 + 5]
        unknown = ledger_block[i * 6 + 5]
        assert known == pytest.approx(np.array(entry.known) / HAND_SCALE)
        assert unknown == pytest.approx(entry.unknown / HAND_SCALE)


def test_default_encoding_uses_discarding_seat_instead_of_turn_owner():
    game = a_game(steps=0)
    game.phase = Phase.DISCARD
    game.current_player = 0
    game.discard_quota = [0, 0, 4, 0]
    game._state.hands[0] = [1, 0, 0, 0, 0]
    game._state.hands[2] = [0, 8, 0, 0, 0]
    default, discarder, owner = encode(game), encode(game, 2), encode(game, 0)
    for actual, expected in zip(arrays(default), arrays(discarder), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(default.globals, owner.globals)
