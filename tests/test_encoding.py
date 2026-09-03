# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import numpy as np
import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.board.topology import build as build_topology
from hexset.board.maps import BASE_LAYOUT, MINI_LAYOUT
from hexset.encoding import (
    HEX_FEATURES,
    NUM_BUILDINGS,
    _building_points,
    _seat,
    edge_features,
    encode,
    encode_batch,
    global_features,
    static_graph,
    vertex_features,
)
from hexset.game import is_over, start
from hexset.play import step_randomly
from hexset.state import NO_OWNER, Building
from hexset.victory import building_points


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def arrays(obs):
    return (obs.hexes, obs.vertices, obs.edges, obs.globals)


def test_batched_encoding_is_byte_identical_to_the_canonical_path():
    """The collector fast path must remain only a different way to write bytes."""
    rng = random.Random(31)
    games = [start(random_base_board(rng), 4, rng) for _ in range(24)]

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

    assert checked > 1000


def test_serializing_one_batched_observation_does_not_carry_the_whole_tick():
    games = [a_game(seed=seed, steps=60 + seed) for seed in range(8)]
    observations = encode_batch(games, [game.current_player for game in games])

    restored = pickle.loads(pickle.dumps(observations[3]))

    assert restored._packed is None
    canonical = encode(games[3], games[3].current_player)
    for got, want in zip(arrays(restored), arrays(canonical), strict=True):
        assert np.array_equal(got, want)
    # The restored arrays are independent, so mutating one transition cannot
    # corrupt any sibling transition after it crosses a collector pipe.
    restored.hexes.fill(7.0)
    assert not np.array_equal(restored.hexes, observations[3].hexes)


@pytest.mark.parametrize("players", [2, 3, 4])
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


def test_adjacency_matches_the_topology():
    topology = build_topology(BASE_LAYOUT)
    graph = static_graph(topology)

    assert graph.hex_vertex.shape == (2, 19 * 6)
    assert graph.vertex_edge.shape == (2, 2 * 72)
    assert graph.hex_hex.shape[1] == sum(len(n) for n in topology.hex_neighbors)
    for h, v in graph.hex_vertex.T:
        assert v in topology.hex_vertices[h]


def test_the_graph_is_cached_per_board():
    topology = build_topology(MINI_LAYOUT)
    assert static_graph(topology) is static_graph(build_topology(MINI_LAYOUT))


def test_observations_on_one_board_do_not_share_memory():
    """The board-static template is cached and handed to every encode.

    If a caller ever received the cached array itself rather than a copy, the
    corruption would be silent and would spread to every later position on the
    board, so this pins the copy rather than trusting it.
    """
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


def test_ownership_is_one_hot():
    obs = encode(a_game())
    players = 4
    owner_slice = obs.vertices[:, 3 : 3 + players + 1]
    assert np.allclose(owner_slice.sum(axis=1), 1.0)
    assert np.allclose(obs.edges.sum(axis=1), 1.0)


def test_the_robber_is_marked_on_exactly_one_hex():
    game = a_game()
    obs = encode(game)
    flags = obs.hexes[:, HEX_FEATURES - 1]
    assert flags.sum() == 1.0
    assert flags[game._state.robber] == 1.0


def test_the_mover_is_always_seat_zero():
    game = a_game()
    topology = game._state.board.topology
    owned = [v for v, o in enumerate(game._state.vertex_owner) if o == game.current_player]
    assert owned, "expected the mover to hold something by now"

    obs = encode(game)
    for v in owned:
        assert obs.vertices[v, 3] == 1.0, "own buildings belong in the first owner slot"
    assert topology.num_vertices == obs.vertices.shape[0]


def test_perspective_changes_what_is_seen():
    game = a_game()
    a = encode(game, perspective=0)
    b = encode(game, perspective=1)
    assert not np.array_equal(a.globals, b.globals)
    assert not np.array_equal(a.vertices, b.vertices)


def test_own_hand_is_encoded_exactly():
    game = a_game()
    state = game._state
    before = encode(game, perspective=0).globals.copy()

    state.bank[Resource.ORE] -= 1
    state.hands[0][Resource.ORE] += 1

    assert not np.array_equal(encode(game, perspective=0).globals, before)


def test_opponent_hand_contents_do_not_leak():
    """Swapping cards between two opponents must be invisible to a third player."""
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


def test_opponent_hand_sizes_are_visible():
    game = a_game(players=3)
    state = game._state
    before = encode(game, perspective=0).globals.copy()

    state.bank[Resource.WHEAT] -= 1
    state.hands[1][Resource.WHEAT] += 1

    assert not np.array_equal(encode(game, perspective=0).globals, before)


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


def test_ports_are_marked_on_both_of_their_vertices():
    game = a_game()
    obs = encode(game)
    players = game._state.num_players
    port_base = 3 + players + 1

    for port in game._state.board.ports:
        for v in port.vertices:
            flags = obs.vertices[v, port_base : port_base + 6]
            assert flags.sum() >= 1.0


def test_an_unknown_perspective_is_rejected():
    game = a_game(players=3)
    with pytest.raises(ValueError):
        encode(game, perspective=3)


def test_encoding_holds_up_across_a_whole_game():
    rng = random.Random(4)
    game = start(random_base_board(rng), 4, rng)
    while not game.won_by and game.turns < 60:
        step_randomly(game, rng)
        for seat in range(game._state.num_players):
            obs = encode(game, perspective=seat)
            for array in arrays(obs):
                assert np.isfinite(array).all()


def _canonical_vertex_block(state, perspective):
    """Building one-hot then owner one-hot, written the plain way.

    `encode` reaches these same rows by table lookup on a combined key, which
    is faster and is not obviously the same thing, so this is what it is
    pinned to.
    """
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


@pytest.mark.parametrize("players", [2, 3, 4])
def test_the_table_lookups_agree_with_the_loops(players):
    rng = random.Random(11 + players)
    game = start(random_base_board(rng), players, rng)

    positions = 0
    while not is_over(game) and game.turns < 40:
        step_randomly(game, rng)
        _check_blocks(game, players)
        positions += players
    assert positions > 100

    # Random play this short reaches no cities, so agreeing everywhere would
    # only say the two paths agree on settlements and empty vertices. Every
    # building and owner combination is planted here instead of hoped for.
    state = game._state
    for owner in range(players):
        state.vertex_building[owner] = Building.CITY
        state.vertex_owner[owner] = owner
        state.vertex_building[players + owner] = Building.SETTLEMENT
        state.vertex_owner[players + owner] = owner
        state.edge_owner[owner] = owner
    _check_blocks(game, players)


@pytest.mark.parametrize("players", [2, 3, 4])
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
            # Seat-relative: index i is the seat i places after the perspective.
            for i in range(players):
                seat = (perspective + i) % players
                assert points[i] == building_points(state, seat)
                scored += points[i] > 0

    assert scored > 0


# --- the live trade offer (trading design part 1) ---


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
    """Fix a hand slot for a test fixture, keeping `game.ledger` in sync so
    it reads as certain rather than as drift (`hexset.ledger.PublicLedger
    .spend`'s invariant check) the next time this game's own play spends it."""
    game._state.hands[player][resource] = n
    game.ledger.seats[player].known[resource] = n


def _ledger_width(players: int = 4) -> int:
    return (players - 1) * (NUM_RESOURCES + 1)


def _valuation_tail(obs, players: int = 4):
    """Every seat's published vector, seat-relative -- `players *
    NUM_RESOURCES` globals, sitting just before the ledger block."""
    width = players * NUM_RESOURCES
    ledger_width = _ledger_width(players)
    tail = obs.globals[-(width + ledger_width) : -ledger_width]
    return [tail[i * NUM_RESOURCES : (i + 1) * NUM_RESOURCES] for i in range(players)]


def test_the_valuation_block_is_zero_before_anybody_publishes():
    game = _main_phase_game()
    for perspective in range(4):
        for row in _valuation_tail(encode(game, perspective)):
            assert not row.any()


def test_every_seat_reads_every_published_vector_seat_relative():
    """The vectors are public -- what a table hears -- so nothing here is
    filtered by perspective; only the seat order rotates."""
    game = _main_phase_game()
    for seat in range(4):
        game.valuations[seat] = tuple(
            (seat + 1) / 10.0 if r == seat % NUM_RESOURCES else 0.0
            for r in range(NUM_RESOURCES)
        )
    for perspective in range(4):
        rows = _valuation_tail(encode(game, perspective))
        for seat in range(4):
            assert rows[_seat(seat, perspective, 4)] == pytest.approx(
                np.asarray(game.valuations[seat], dtype=np.float32)
            )


def test_a_negative_valuation_survives_unscaled():
    """Vectors are already in [-1, 1], so unlike hands they are not divided
    by anything on the way into the observation."""
    game = _main_phase_game()
    game.valuations[game.current_player] = (-1.0, 0.0, 0.0, 0.0, 1.0)
    rows = _valuation_tail(encode(game, game.current_player))
    assert rows[0][0] == pytest.approx(-1.0)
    assert rows[0][4] == pytest.approx(1.0)


def test_batched_valuation_encoding_matches_the_canonical_path():
    game = _main_phase_game()
    for seat in range(4):
        game.valuations[seat] = tuple(0.1 * (seat + 1) for _ in range(NUM_RESOURCES))
    games = [game] * 4
    perspectives = list(range(4))
    fast = encode_batch(games, perspectives)
    for got, perspective in zip(fast, perspectives, strict=True):
        want = encode(game, perspective)
        assert np.array_equal(got.globals, want.globals)
        assert got.globals.dtype == np.float32


# --- the public-knowledge ledger (trading-design §7.2) ---


def _ledger_tail(obs, players: int = 4):
    """Each opponent's known[5] + unknown, seat-relative, own seat excluded —
    always the very last `(players - 1) * (NUM_RESOURCES + 1)` globals."""
    tail = obs.globals[-_ledger_width(players):]
    known = [tail[i * 6 : i * 6 + 5] for i in range(players - 1)]
    unknown = [tail[i * 6 + 5] for i in range(players - 1)]
    return known, unknown


def test_global_features_counts_the_valuation_and_ledger_blocks():
    """87 at four players: the 18-float live-offer block is gone and
    `Phase` lost a member with it (`TRADE_RESPOND`), replaced by every
    seat's public valuation vector at `players * NUM_RESOURCES`."""
    assert global_features(4) == 87
    assert global_features(4) - global_features(3) == (
        # one more seat shows up in: opponent hand size, opponent dev count,
        # knights, public points, the two holder one-hots, the valuation
        # block, and the ledger block.
        1 + 1 + 1 + 1 + 2 + NUM_RESOURCES + (NUM_RESOURCES + 1)
    )


def test_the_ledger_block_is_zero_at_game_start():
    game = start(random_base_board(random.Random(1)), 4, random.Random(1))
    for perspective in range(4):
        known, unknown = _ledger_tail(encode(game, perspective))
        for k in known:
            assert not k.any()
        assert all(u == 0.0 for u in unknown)


def test_the_ledger_block_reads_each_opponent_seat_relative():
    from hexset.ledger import SeatLedger

    game = _main_phase_game()
    perspective = 0
    others = [s for s in range(4) if s != perspective]
    game.ledger.seats[others[0]] = SeatLedger(known=[2, 0, 0, 0, 0], unknown=1)
    game.ledger.seats[others[1]] = SeatLedger(known=[0, 1, 0, 3, 0], unknown=0)
    game.ledger.seats[others[2]] = SeatLedger(known=[0, 0, 0, 0, 4], unknown=2)

    known, unknown = _ledger_tail(encode(game, perspective))

    for seat in others:
        expected = game.ledger.seats[seat]
        slot = _seat(seat, perspective, 4) - 1
        assert known[slot] == pytest.approx(np.array(expected.known) / 10.0)
        assert unknown[slot] == pytest.approx(expected.unknown / 10.0)


def test_a_steal_shows_up_as_unknown_in_the_encoding():
    """A bystander's view (neither thief nor victim) is what the ledger's
    common-knowledge promise is actually about: the thief's block gains an
    `unknown`, and a victim certified for exactly one resource sees that
    entry floor to zero (`hexset.ledger.PublicLedger.steal` floors *every*
    `known[r]` by one, identity-independent by construction; with only one
    nonzero entry to begin with, that is the only one visibly touched) --
    picking a one-resource hand keeps the steal's outcome deterministic
    regardless of `robber.steal`'s random draw, without needing the ledger
    to know (or leak) which resource it actually drew."""
    from hexset.game import Phase, move_robber_to
    from hexset.ledger import SeatLedger

    game = _main_phase_game()
    game.phase = Phase.ROBBER
    thief, victim, bystander = 0, 1, 2
    game.current_player = thief
    game.ledger.seats[thief] = SeatLedger()
    game.ledger.seats[victim] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=0)
    game._state.hands[victim] = [1, 0, 0, 0, 0]

    # A target hex occupied by the victim, however this board happens to be
    # laid out -- the property under test is what the ledger does with a
    # steal, not which hex triggers one.
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


def test_batched_ledger_encoding_matches_the_canonical_path():
    from hexset.ledger import SeatLedger

    game = _main_phase_game()
    game.ledger.seats[1] = SeatLedger(known=[1, 0, 2, 0, 0], unknown=1)
    game.ledger.seats[2] = SeatLedger(known=[0, 0, 0, 1, 1], unknown=0)
    games = [game] * 4
    perspectives = list(range(4))
    fast = encode_batch(games, perspectives)
    for got, perspective in zip(fast, perspectives, strict=True):
        want = encode(game, perspective)
        assert np.array_equal(got.globals, want.globals)
        assert got.globals.dtype == np.float32
