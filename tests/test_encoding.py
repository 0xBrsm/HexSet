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
    AGE_FILL,
    AGE_SCALE,
    BANK_SCALE,
    DEV_SCALE,
    HAND_SCALE,
    HEX_FEATURES,
    PLAYABLE_KINDS,
    POINTS_SCALE,
    RULE_BLOCKS,
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
from hexset.game import ROLL_ODDS, Phase, is_over, observe, roll_odds, start
from hexset.rules import DUEL_VARIANT, DUEL_VARIANT_GAME
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


def test_a_seats_observed_position_batch_encodes_as_it_encodes_alone():
    """Every other seat's hand and development cards are hidden there, so
    the batch reads them by size, as `encode` does, and matches the true
    position's encoding from that seat."""
    games = [a_game(seed=seed, steps=90 + 7 * seed) for seed in range(6)]
    seats = [seed % 4 for seed in range(6)]
    seen = [observe(game, seat) for game, seat in zip(games, seats, strict=True)]
    for got, alone, true in zip(
        encode_batch(seen, seats),
        (encode(game, seat) for game, seat in zip(seen, seats, strict=True)),
        (encode(game, seat) for game, seat in zip(games, seats, strict=True)),
        strict=True,
    ):
        for got_array, alone_array, true_array in zip(arrays(got), arrays(alone), arrays(true), strict=True):
            assert np.array_equal(got_array, alone_array)
            assert np.array_equal(got_array, true_array)


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
    """Non-negative; the one-hots and the closeness reads in [0, 1]. Counts and ages are not capped: a hand of 12
    reads 1.2, as it does."""
    players = 4
    for seed in range(6):
        obs = encode(a_game(seed=seed, steps=400))
        for array in arrays(obs):
            assert np.isfinite(array).all()
        buildings = obs.vertices[:, :players]
        assert buildings.min() >= 0.0 and buildings.max() <= 2.0
        assert obs.vertices[:, players:].min() >= 0.0 and obs.vertices[:, players:].max() <= 1.0
        for array in (obs.hexes, obs.edges):
            assert array.min() >= 0.0 and array.max() <= 1.0
        columns = global_columns(players)
        assert obs.globals.min() >= 0.0
        for name in ("seat_longest_road", "seat_largest_army", "opponent_to_move", "dev_odds", "phase"):
            assert obs.globals[columns[name]].max() <= 1.0


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
    assert HEX_FEATURES == 7
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
    out = np.zeros((state.board.topology.num_vertices, players), dtype=np.float32)
    for v in range(out.shape[0]):
        owner = state.vertex_owner[v]
        if owner != NO_OWNER:
            out[v, _seat(owner, perspective, players)] = float(state.vertex_building[v])
    return out


def _canonical_edges(state, perspective):
    players = state.num_players
    out = np.zeros(
        (state.board.topology.num_edges, edge_features(players)), dtype=np.float32
    )
    for e in range(out.shape[0]):
        owner = state.edge_owner[e]
        if owner != NO_OWNER:
            out[e, _seat(owner, perspective, players)] = 1.0
    return out


def _check_blocks(game, players):
    state = game._state
    for perspective in range(players):
        obs = encode(game, perspective=perspective)
        block = obs.vertices[:, :players]
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


def _opponent_hand(obs, players: int = 4):
    """Each opponent's expected count of each resource, seat-relative."""
    block = obs.globals[global_columns(players)["opponent_hand"]]
    return block.reshape(players - 1, NUM_RESOURCES) * HAND_SCALE


def test_the_widths_are_pinned():
    """Pinned so a layout change has to own the new width."""
    assert global_features(4) == 90
    assert HEX_FEATURES == 7
    assert vertex_features(4) == 14
    assert edge_features(4) == 4
    # Per opponent: a hand, a card count, three ages; per seat: points,
    # Longest Road and Largest Army; per opponent, its turn.
    assert global_features(4) - global_features(3) == (NUM_RESOURCES + 1 + 3) + 3 + 1


def test_the_blocks_are_in_contract_order():
    assert list(global_columns(4)) == [
        "own_hand", "own_dev", "own_dev_playable", "opponent_hand",
        "opponent_dev_count", "opponent_dev_ages", "seat_points", "seat_longest_road", "seat_largest_army",
        "opponent_to_move", "bank", "deck", "dev_odds", "phase",
        "free_roads", "turn", "rules", "roll_odds",
    ]


def _roads_position(seed: int = 5):
    """A played-out position with Longest Road held by someone: a real
    holder where random play reaches one, otherwise the card handed to a seat
    (`roads_to_take` reads the holder off the state, as the encoder does)."""
    for attempt in range(40):
        game = a_game(seed=seed + attempt, steps=400)
        if not is_over(game) and game._state.longest_road_holder != NO_OWNER:
            return game
    game = a_game(seed=seed, steps=200)
    game._state.longest_road_holder = 2
    return game


def test_longest_road_and_the_mover_read_seat_relative():
    from hexset.game import to_move
    from hexset.roads import roads_to_take_all

    players = 4
    game = _roads_position()
    holder = game._state.longest_road_holder
    assert holder != NO_OWNER
    counts = roads_to_take_all(game._state)
    assert counts[holder] == 0
    mover = to_move(game)
    columns = global_columns(players)
    for perspective in range(players):
        obs = encode(game, perspective)
        seats = to_frame(range(players), perspective)
        roads = obs.globals[columns["seat_longest_road"]]
        assert list(roads) == pytest.approx([1.0 / (1.0 + counts[s]) for s in seats])
        assert roads[_seat(holder, perspective, players)] == 1.0
        assert all(0.2 <= r <= 1.0 for r in roads)
        moving = obs.globals[columns["opponent_to_move"]]
        slot = _seat(mover, perspective, players)
        assert moving.sum() == (0.0 if slot == 0 else 1.0)
        if slot:
            assert moving[slot - 1] == 1.0

    # The batched path, from every seat, byte for byte.
    batch = encode_batch([game] * players, list(range(players)))
    for perspective, got in enumerate(batch):
        assert np.array_equal(got.globals, encode(game, perspective).globals)


def test_the_mover_block_follows_a_discard_not_the_turn():
    game = a_game(steps=0)
    game.phase = Phase.DISCARD
    game.current_player = 0
    game.discard_quota = [0, 0, 4, 0]
    game._state.hands[2] = [0, 8, 0, 0, 0]
    moving = encode(game, 1).globals[global_columns(4)["opponent_to_move"]]
    assert list(moving) == [1.0, 0.0, 0.0]  # seat 2, one after seat 1


def test_a_steal_from_a_certified_one_resource_hand_is_typed_for_the_table():
    """The one-resource hand makes the outcome deterministic whatever was
    drawn, and certified, every seat can name the card."""
    from hexset.game import Phase, move_robber_to
    from hexset.ledger import SeatLedger

    game = _main_phase_game()
    game.phase = Phase.ROBBER
    thief, victim, bystander = 0, 1, 2
    game.current_player = thief
    game._state.hands[thief] = [0, 0, 0, 0, 0]
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

    assert game.ledger.seats[thief].unknown == 0
    assert game.ledger.seats[thief].known == [1, 0, 0, 0, 0]

    hands = _opponent_hand(encode(game, bystander))
    assert hands[_seat(thief, bystander, 4) - 1] == pytest.approx([1, 0, 0, 0, 0])
    assert hands[_seat(victim, bystander, 4) - 1] == pytest.approx([0, 0, 0, 0, 0])


def _brute_expected(known, hidden, room, pool):
    """Every split of `hidden` cards within `room`, weighed by its
    hypergeometric chance from `pool`."""
    from itertools import product
    from math import comb

    total, sums = 0, [0] * NUM_RESOURCES
    for split in product(*(range(r + 1) for r in room)):
        if sum(split) != hidden:
            continue
        w = 1
        for n, p in zip(split, pool):
            w *= comb(p, n)
        total += w
        sums = [a + w * n for a, n in zip(sums, split)]
    return [k + a / total for k, a in zip(known, sums)]


def test_opponent_hands_are_the_exact_expected_counts_within_the_bounds():
    from hexset.view import View

    for seed in range(40):
        game = a_game(seed=seed, steps=300)
        if is_over(game):
            continue
        for perspective in range(4):
            view = View.from_game(game, perspective)
            hands = _opponent_hand(encode(game, perspective))
            for i, seat in enumerate(to_frame(range(4), perspective)[1:]):
                expected = view.expected_counts(seat)
                assert hands[i] == pytest.approx(expected, abs=1e-5)
                assert sum(expected) == pytest.approx(view.sizes[seat])
                if view.unknown[seat]:
                    assert expected == pytest.approx(_brute_expected(
                        view.known[seat], view.unknown[seat],
                        [min(r, p) for r, p in zip(view.room[seat], view.pool)], view.pool))


def test_a_binding_bound_is_enumerated_not_spread():
    """Two hidden cards, room for at most one wood: the closed form would put
    more than one there."""
    from hexset.view import View

    game = a_game(steps=0)
    view = View.from_game(game, 0)
    view.known[1] = [0, 0, 0, 0, 0]
    view.unknown[1] = 2
    view.room[1] = [1, 2, 2, 2, 2]
    view.bounded[1] = True
    view.pool = [12, 1, 1, 1, 1]
    view.pool_size = 16
    got = view.expected_counts(1)
    assert got == pytest.approx(_brute_expected([0] * 5, 2, [1, 1, 1, 1, 1], view.pool))
    assert got[0] <= 1.0
    assert sum(got) == pytest.approx(2.0)


def _settled_position(seed: int = 3):
    for attempt in range(40):
        game = a_game(seed=seed + attempt, steps=200)
        if not is_over(game) and game.phase is not Phase.SETUP_SETTLEMENT:
            return game
    raise AssertionError("no position past setup")


def test_reach_is_zero_distance_on_the_network_and_grows_one_road_a_step():
    from hexset.roads import reach_all

    game = _settled_position()
    state = game._state
    topology = state.board.topology
    reach = reach_all(state)
    for seat in range(4):
        for v, owner in enumerate(state.vertex_owner):
            if owner == seat:
                assert reach[seat][v] == 0
        for v, d in enumerate(reach[seat]):
            if d > 0:
                # One road from a vertex one step nearer, through a junction
                # no opponent building blocks, on an edge holding no road.
                assert any(
                    reach[seat][w] == d - 1 and state.vertex_owner[w] in (NO_OWNER, seat)
                    and state.edge_owner[e] == NO_OWNER
                    for e in topology.vertex_edges[v]
                    for w in topology.edges[e] if w != v
                )
    obs = encode(game, 1)
    block = obs.vertices[:, 4 + 1 + NUM_RESOURCES:]
    for i, seat in enumerate(to_frame(range(4), 1)):
        want = [0.0 if d < 0 else 1.0 / (1.0 + d) for d in reach[seat]]
        assert list(block[:, i]) == pytest.approx(want)


def test_a_seat_without_a_building_reaches_nothing():
    from hexset.roads import reach_all

    game = a_game(steps=0)
    assert all(d == -1 for row in reach_all(game._state) for d in row)
    assert not encode(game).vertices[:, 4 + 1 + NUM_RESOURCES:].any()


def test_reach_stops_at_an_opponent_building():
    from hexset.roads import reach_all

    game = _settled_position()
    state = game._state
    topology = state.board.topology
    seat = 0
    mine = next(v for v, o in enumerate(state.vertex_owner) if o == seat)
    # An opponent settles every neighbour: the seat reaches them, and no
    # further from here.
    for w in topology.vertex_neighbors[mine]:
        if state.vertex_owner[w] == NO_OWNER:
            state.vertex_owner[w] = 1
            state.vertex_building[w] = Building.SETTLEMENT
    reach = reach_all(state)[seat]
    for w in topology.vertex_neighbors[mine]:
        if state.vertex_owner[w] == 1:
            assert reach[w] in (0, 1)


def test_largest_army_counts_down_and_reads_zero_out_of_reach():
    from hexset.cards import DevCard

    players, perspective = 4, 0
    game = _main_phase_game()
    state = game._state
    state.knights_played = [1, 3, 0, 0]
    state.largest_army_holder = 1
    state.dev_cards[perspective][DevCard.KNIGHT] = 2
    columns = global_columns(players)
    army = encode(game, perspective).globals[columns["seat_largest_army"]]
    assert army[_seat(1, perspective, players)] == 1.0
    # Seat 0 must play 3 more to beat 3: it holds 2 and has not seen the rest.
    assert army[0] == pytest.approx(1.0 / 4.0)
    assert army[_seat(2, perspective, players)] == pytest.approx(1.0 / 5.0)
    # Every unseen knight accounted for: seat 2 needs 4, more than are left.
    from hexset.cards import DECK_COMPOSITION
    state.dev_cards_played[DevCard.KNIGHT] = 0
    state.knights_played = [1, 3, 0, 0]
    total = DECK_COMPOSITION[DevCard.KNIGHT]
    state.dev_cards[perspective][DevCard.KNIGHT] = total - 4 - 3
    army = encode(game, perspective).globals[columns["seat_largest_army"]]
    assert army[_seat(2, perspective, players)] == 0.0
    assert army[_seat(3, perspective, players)] == 0.0


def test_only_matured_cards_are_playable_and_none_after_a_play():
    from hexset.cards import DevCard

    game = _main_phase_game()
    state = game._state
    me = 0
    state.dev_cards[me] = [1, 1, 0, 1, 0]
    state.new_dev_cards[me] = [1, 0, 1, 0, 0]
    columns = global_columns(4)
    obs = encode(game, me)
    assert list(obs.globals[columns["own_dev"]]) == pytest.approx([2 / DEV_SCALE, 1 / DEV_SCALE, 1 / DEV_SCALE, 1 / DEV_SCALE, 0])
    assert list(obs.globals[columns["own_dev_playable"]]) == pytest.approx(
        [state.dev_cards[me][k] / DEV_SCALE for k in PLAYABLE_KINDS])
    assert PLAYABLE_KINDS[0] == DevCard.KNIGHT
    game.dev_card_played = True
    assert not encode(game, me).globals[columns["own_dev_playable"]].any()


def test_card_ages_read_oldest_first_and_fill_where_unknown():
    game = _main_phase_game()
    state = game._state
    perspective, seat = 0, 2
    state.dev_cards[seat] = [2, 0, 1, 0, 1]
    state.new_dev_cards[seat] = [0, 0, 0, 0, 0]
    state.dev_ages[seat] = [1, 9, 4, 2]
    columns = global_columns(4)
    slot = _seat(seat, perspective, 4) - 1
    ages = encode(game, perspective).globals[columns["opponent_dev_ages"]].reshape(3, 3)[slot]
    assert list(ages) == pytest.approx([10 / AGE_SCALE, 5 / AGE_SCALE, 3 / AGE_SCALE])
    state.dev_cards[seat] = [1, 0, 0, 0, 1]
    state.dev_ages[seat] = None
    ages = encode(game, perspective).globals[columns["opponent_dev_ages"]].reshape(3, 3)[slot]
    assert list(ages) == pytest.approx([(AGE_FILL[0] + 1) / AGE_SCALE, (AGE_FILL[1] + 1) / AGE_SCALE, 0.0])


def test_nothing_of_trading_is_read():
    """Shown wants and spares stay with the trade gate: they move nothing."""
    game = _main_phase_game()
    before = encode(game, 0)
    rows = game.ledger.seats
    rows[3].want = [0, 2, 0, 0, 0]
    rows[3].waste = [0, 0, 0, 3, 0]
    after = encode(game, 0)
    for lhs, rhs in zip(arrays(before), arrays(after)):
        assert np.array_equal(lhs, rhs)


def test_dev_odds_are_the_unseen_cards_by_kind():
    from hexset.view import View

    game = a_game(steps=300)
    for perspective in range(4):
        odds = encode(game, perspective).globals[global_columns(4)["dev_odds"]]
        unseen = View.from_game(game, perspective).unseen_dev_cards()
        assert odds.sum() == pytest.approx(1.0)
        assert list(odds) == pytest.approx([n / sum(unseen) for n in unseen])


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
    game._state.largest_army_holder = 3
    game.phase = Phase.MAIN
    game.free_roads = 1
    game._state.deck = game._state.deck[:7]
    game.turns = 42
    for seat in (0, 2, 3):
        game.ledger.seats[seat] = SeatLedger(known=list(game._state.hands[seat]), unknown=0)

    obs = encode(game, perspective)
    columns = global_columns(players)
    seats = to_frame(range(players), perspective)

    assert obs.globals[columns["own_hand"]] == pytest.approx(
        np.array(game._state.hands[perspective]) / HAND_SCALE
    )
    own_dev = [
        h + f
        for h, f in zip(
            game._state.dev_cards[perspective], game._state.new_dev_cards[perspective]
        )
    ]
    assert obs.globals[columns["own_dev"]] == pytest.approx(np.array(own_dev) / DEV_SCALE)
    hands = _opponent_hand(obs)
    for i, seat in enumerate(seats[1:]):
        assert hands[i] == pytest.approx(game._state.hands[seat])

    counts = obs.globals[columns["opponent_dev_count"]]
    for i, seat in enumerate(seats[1:]):
        expected = sum(game._state.dev_cards[seat]) + sum(game._state.new_dev_cards[seat])
        assert counts[i] == pytest.approx(expected / DEV_SCALE)

    points = obs.globals[columns["seat_points"]]
    for i, seat in enumerate(seats):
        assert points[i] == pytest.approx(public_victory_points(game._state, seat) / 10.0)

    assert obs.globals[columns["seat_longest_road"]][_seat(2, perspective, players)] == 1.0
    assert obs.globals[columns["seat_largest_army"]][_seat(3, perspective, players)] == 1.0

    assert obs.globals[columns["bank"]] == pytest.approx(np.array(game._state.bank) / BANK_SCALE)
    phase_block = obs.globals[columns["phase"]]
    assert len(phase_block) == len(Phase) - 1
    assert phase_block[int(game.phase)] == 1.0
    assert phase_block.sum() == 1.0
    assert obs.globals[columns["free_roads"]][0] == pytest.approx(game.free_roads / 2.0)
    assert obs.globals[columns["deck"]][0] == pytest.approx(len(game._state.deck) / DECK_SIZE)
    assert obs.globals[columns["turn"]][0] == pytest.approx(min(game.turns / TURN_SCALE, 1.0))


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


def a_duel(seed: int = 0, steps: int = 400):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 2, rng, game_type=DUEL_VARIANT_GAME)
    for _ in range(steps):
        if is_over(game):
            break
        step_randomly(game, rng)
    return game


def test_contract_8_1_appends_its_blocks_after_every_contract_8_column():
    columns = global_columns(4)
    assert list(columns)[-len(RULE_BLOCKS):] == list(RULE_BLOCKS)
    assert columns[RULE_BLOCKS[-1]].stop == global_features(4)


def test_the_rules_block_states_the_tables_rules():
    standard = encode(a_game(steps=0))
    assert standard.globals[global_columns(4)["rules"]].tolist() == pytest.approx([1.0, 0.7, 0.0, 0.0])
    duel = encode(a_duel(steps=0))
    assert duel.globals[global_columns(2)["rules"]].tolist() == pytest.approx(
        [DUEL_VARIANT.winning_points / POINTS_SCALE, DUEL_VARIANT.discard_limit / HAND_SCALE, 1.0, 1.0]
    )


def test_points_are_read_against_the_points_the_game_is_played_to():
    for game, players in ((a_game(), 4), (a_duel(), 2)):
        state = game._state
        obs = encode(game, 0)
        expected = [public_victory_points(state, s) / state.rules.winning_points for s in range(players)]
        assert obs.globals[global_columns(players)["seat_points"]].tolist() == pytest.approx(expected)


def test_the_roll_odds_are_the_next_rolls():
    plain = encode(a_game()).globals[global_columns(4)["roll_odds"]]
    assert plain.tolist() == pytest.approx([chance for _, chance in ROLL_ODDS])
    moved = False
    for seed in range(4):
        game = a_duel(seed)
        odds = dict(roll_odds(game))
        block = encode(game).globals[global_columns(2)["roll_odds"]]
        assert block.tolist() == pytest.approx([odds.get(total, 0.0) for total in range(2, 13)])
        assert block.sum() == pytest.approx(1.0)
        moved |= not np.allclose(block, plain)
    assert moved, "the balanced dice deck never moved off the plain odds"
