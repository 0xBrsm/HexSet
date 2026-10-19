"""Contract 9's observation (`hexset.contract9`): one encoding for every seat."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from hexset import contract9
from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.contract9 import SEAT_COLUMNS as C, TABLE_COLUMNS as T, encode9, flatten, layout
from hexset.game import Phase, is_over, start, observe
from hexset.play import step_randomly
from hexset.view import View


def a_game(seed: int = 0, steps: int = 200):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(steps):
        if is_over(game):
            break
        step_randomly(game, rng)
    return game


def games(count: int = 12):
    for seed in range(count):
        yield a_game(seed, 40 + 23 * seed)


def test_shapes_and_the_flat_layout():
    game = a_game()
    o = encode9(game, 2)
    assert o.seats.shape == (4, contract9.SEAT_FEATURES) == (4, 39)
    assert o.table.shape == (contract9.TABLE_FEATURES,) == (20,)
    assert o.hexes.shape == (19, 7) and o.vertices.shape == (54, 7)
    assert o.vertex_owner.shape == (54,) and o.road_owner.shape == (72,) and o.reach.shape == (4, 54)
    flat = flatten(o)
    lay = layout(4, 19, 54, 72)
    assert flat.dtype == np.float32 and flat.shape == (lay.width,)
    for name, piece in lay.slices.items():
        np.testing.assert_array_equal(flat[piece].reshape(lay.shapes[name]), getattr(o, name))


def test_concrete_and_observed_states_encode_the_same_public_deck_size():
    for game in games():
        for seat in range(game.num_players):
            encoded = encode9(game, seat)
            assert encoded.table[T["deck"]] == np.float32(len(game._state.deck) / contract9.CARD_SCALE)
            np.testing.assert_array_equal(flatten(encoded), flatten(encode9(observe(game, seat), seat)))


def test_hidden_deck_composition_cannot_change_the_observation():
    game = a_game(0, 0)
    state = game._state
    held = state.deck.pop()
    state.dev_cards[1][held] += 1
    before = flatten(encode9(game, 0))
    index = next(i for i, card in enumerate(state.deck) if card != held)
    replacement = state.deck[index]
    state.deck[index] = held
    state.dev_cards[1][held] -= 1
    state.dev_cards[1][replacement] += 1
    np.testing.assert_array_equal(before, flatten(encode9(game, 0)))


def test_rows_are_seat_relative_and_pointers_name_rows():
    for game in games():
        state = game.state(0, hidden=False)
        for p in range(4):
            o = encode9(game, p)
            assert o.seats[0, C["me"]] == 1 and o.seats[1:, C["me"]].sum() == 0
            for row in range(4):
                seat = (p + row) % 4
                place = (seat - game.first) % 4
                assert o.seats[row, C["seat"]].tolist() == [float(i == place) for i in range(4)]
                assert o.seats[row, C["on_turn"]] == float(seat == game.current_player)
            for v, owner in enumerate(state.vertex_owner):
                assert o.vertex_owner[v] == (-1 if owner < 0 else (owner - p) % 4)
            for e, owner in enumerate(state.edge_owner):
                assert o.road_owner[e] == (-1 if owner < 0 else (owner - p) % 4)


def test_hands_are_the_views_bounds_and_mine_exact():
    for game in games():
        state = game.state(0, hidden=False)
        for p in range(4):
            o, view = encode9(game, p), View.from_game(game, p)
            np.testing.assert_allclose(o.seats[0, C["hand_least"]] * 10, state.hands[p])
            np.testing.assert_allclose(o.seats[0, C["hand_most"]] * 10, state.hands[p])
            for row in range(1, 4):
                seat = (p + row) % 4
                np.testing.assert_allclose(o.seats[row, C["hand_least"]] * 10, view.known[seat], atol=1e-5)
                most = np.asarray(view.known[seat]) + np.asarray(view.room[seat])
                np.testing.assert_allclose(o.seats[row, C["hand_most"]] * 10, most, atol=1e-5)
                assert (most <= np.asarray(view.known[seat]) + view.unknown[seat]).all()
                # The truth lies within the bounds.
                assert (np.asarray(state.hands[seat]) <= most).all()
                assert (np.asarray(state.hands[seat]) >= np.asarray(view.known[seat])).all()


def test_every_card_is_accounted_for_once():
    """Per resource, the bank, the pool and every seat's `hand.least` make the
    19 cards, wherever the view did not have to pad its pool."""
    exact = 0
    for game in games():
        state = game.state(0, hidden=False)
        for p in range(4):
            o, view = encode9(game, p), View.from_game(game, p)
            raw = [19 - state.bank[r] - sum(view.known[s][r] for s in range(4)) for r in range(5)]
            total = np.rint((o.seats[:, C["hand_least"]].sum(0) + o.table[T["bank"]] + o.table[T["pool"]]) * 10)
            if min(raw) >= 0 and sum(view.unknown) <= sum(raw):
                assert (total == 19).all()
                exact += 1
    assert exact > 0


def test_an_opponents_hidden_cards_never_reach_my_encoding():
    """Recompose an opponent's hand and development cards within what is
    public: my encoding cannot move."""
    moved = 0
    for game in games(8):
        state = game.state(0, hidden=False)
        for p in range(4):
            before = flatten(encode9(game, p))
            other = copy.deepcopy(game)
            s = other.state(0, hidden=False)
            seat = (p + 1) % 4
            hand = s.hands[seat]
            if sum(hand) >= 2:
                known = other.ledger.for_seat(p).seats[seat].known
                give = [r for r in range(5) if hand[r] > known[r]]
                take = [r for r in range(5) if r not in give]
                if give and take and s.bank[take[0]] > 0:
                    hand[give[0]] -= 1
                    hand[take[0]] += 1
                    s.bank[give[0]] += 1
                    s.bank[take[0]] -= 1
                    moved += 1
            for pile in (s.dev_cards[seat], s.new_dev_cards[seat]):
                if pile[DevCard.KNIGHT] and not pile[DevCard.MONOPOLY]:
                    pile[DevCard.KNIGHT] -= 1
                    pile[DevCard.MONOPOLY] += 1
                    moved += 1
            after = flatten(encode9(other, p))
            # The bank moved, which is public; nothing else may.
            lay = layout(4, 19, 54, 72)
            mask = np.ones(lay.width, dtype=bool)
            mask[lay.slices["table"]] = False
            np.testing.assert_array_equal(before[mask], after[mask])
    assert moved > 10


def test_pieces_points_and_the_board():
    for game in games():
        state = game.state(0, hidden=False)
        o = encode9(game, 0)
        for row in range(4):
            settled = sum(1 for v, w in enumerate(state.vertex_owner) if w == row and state.vertex_building[v] == 1)
            cities = sum(1 for v, w in enumerate(state.vertex_owner) if w == row and state.vertex_building[v] == 2)
            roads = sum(1 for w in state.edge_owner if w == row)
            np.testing.assert_allclose(o.seats[row, C["pieces"]] * 10, [5 - settled, 4 - cities, 15 - roads])
        np.testing.assert_allclose(o.vertices[:, -1] * 10, state.vertex_building)


def test_rolled_is_false_before_the_roll_even_in_a_knights_robber_phase():
    for seed in range(60):
        game = a_game(seed, 30 + seed * 7)
        if game.phase is not Phase.ROLL:
            continue
        assert encode9(game, game.current_player).table[T["rolled"]] == 0
        if Action(ActionType.PLAY_KNIGHT) in legal_actions(game):
            apply(game, Action(ActionType.PLAY_KNIGHT))
            if game.phase is Phase.ROBBER:
                o = encode9(game, game.current_player)
                assert o.table[T["rolled"]] == 0 and o.table[T["robber"]] == 1
                return
    pytest.skip("no position with a knight before the roll in these seeds")


def test_only_the_mover_loses_its_playable_cards_to_a_card_played():
    for seed in range(80):
        game = a_game(seed, 60 + seed * 5)
        if is_over(game) or game.phase is not Phase.MAIN or game.dev_card_played:
            continue
        mover = game.current_player
        state = game.state(0, hidden=False)
        other = next((s for s in range(4) if s != mover and state.dev_cards[s][DevCard.KNIGHT]), None)
        if other is None:
            continue
        game.dev_card_played = True
        mine = encode9(game, other).seats[0, C["dev_playable"]]
        theirs = encode9(game, mover).seats[0, C["dev_playable"]]
        assert mine[0] * 10 == state.dev_cards[other][DevCard.KNIGHT]
        assert not theirs.any()
        return
    pytest.skip("no seat off turn holding a matured knight in these seeds")


def _seat_tokens_one_at_a_time(game, perspective):
    """The seat tokens worked out a seat and a column at a time, each read
    through the engine's own function: what `encode9` fills in bulk."""
    from hexset.contract9 import CARD_SCALE, PIECE_SCALE, POINT_SCALE, SEAT_FEATURES, _ages, _closeness
    from hexset.devcards import dev_count
    from hexset.encoding import PLAYABLE_KINDS, knights_needed_all, to_frame
    from hexset.roads import roads_to_take_all
    from hexset.state import (MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, city_count, pile_size, road_count,
                              settlement_count)
    from hexset.victory import public_victory_points

    state = game._state
    players = state.num_players
    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()
    roads = roads_to_take_all(state)
    knights = knights_needed_all(state, perspective, unseen[DevCard.KNIGHT])
    mover = game.current_player
    seats = np.zeros((players, SEAT_FEATURES), dtype=np.float32)
    for i, s in enumerate(to_frame(range(players), perspective)):
        token = seats[i]
        token[C["seat"].start + (s - game.first) % players] = 1.0
        token[C["on_turn"]] = float(s == mover)
        token[C["me"]] = float(s == perspective)
        least = view.known[s]
        most = [k + r for k, r in zip(least, view.room[s])] if s != perspective else least
        token[C["hand_least"]] = np.asarray(least, dtype=np.float32) / CARD_SCALE
        token[C["hand_most"]] = np.asarray(most, dtype=np.float32) / CARD_SCALE
        token[C["hand_held"]] = view.sizes[s] / CARD_SCALE
        cards = dev_count(state, s)
        played = game.dev_card_played and s == mover
        if s == perspective:
            dev_most = [state.dev_cards[s][k] + state.new_dev_cards[s][k] for k in range(len(DevCard))]
            playable = [0 if played else state.dev_cards[s][kind] for kind in PLAYABLE_KINDS]
        else:
            dev_most = [min(cards, unseen[k]) for k in range(len(DevCard))]
            playable = [0 if played else min(pile_size(state.dev_cards[s]), unseen[kind]) for kind in PLAYABLE_KINDS]
        token[C["dev_most"]] = np.asarray(dev_most, dtype=np.float32) / CARD_SCALE
        token[C["dev_held"]] = cards / CARD_SCALE
        token[C["dev_oldest"]] = _ages(state, s)
        token[C["dev_playable"]] = np.asarray(playable, dtype=np.float32) / CARD_SCALE
        token[C["points"]] = public_victory_points(state, s) / POINT_SCALE
        token[C["to_longest_road"]] = _closeness(roads[s])
        token[C["to_largest_army"]] = _closeness(knights[s])
        token[C["must_discard"]] = (game.discard_quota[s] if game.discard_quota else 0) / CARD_SCALE
        token[C["pieces"]] = ((MAX_SETTLEMENTS - settlement_count(state, s)) / PIECE_SCALE,
                              (MAX_CITIES - city_count(state, s)) / PIECE_SCALE,
                              (MAX_ROADS - road_count(state, s)) / PIECE_SCALE)
    return seats


def test_the_seat_tokens_are_the_one_seat_at_a_time_reading_bit_for_bit():
    """`encode9` fills the tokens in bulk and reads reach once per position;
    every bit is what the seat-by-seat, column-by-column reading gives, on
    every perspective of a few hundred positions at four seats and at two,
    a second encoding of the same position (the cached reach) included."""
    from hexset.roads import reach_all

    checked = 0
    for seed in range(30):
        rng = random.Random(seed)
        players = 2 if seed % 5 == 0 else 4
        game = start(random_base_board(rng), players, rng)
        for step in range(400):
            if is_over(game):
                break
            if step % 9 == 0:
                for seat in range(players):
                    o = encode9(game, seat)
                    assert o.seats.tobytes() == _seat_tokens_one_at_a_time(game, seat).tobytes()
                    rows = [(seat + i) % players for i in range(players)]
                    reach = np.asarray(reach_all(game._state), dtype=np.float32)[rows]
                    reach = np.where(reach < 0, np.float32(0.0), 1.0 / (1.0 + np.maximum(reach, 0.0))).astype(np.float32)
                    assert o.reach.tobytes() == reach.tobytes() and o.reach.flags.writeable
                    assert flatten(encode9(game, seat)).tobytes() == flatten(o).tobytes()
                    checked += 1
            step_randomly(game, rng)
    assert checked > 500


def test_incidences_are_one_tuple_per_topology_and_equal_across_equal_ones():
    first, second = (random_base_board(random.Random(seed)).topology for seed in (1, 2))
    assert first is not second and first == second
    assert contract9.incidences(first) is contract9.incidences(first)
    assert contract9.incidences(second) == contract9.incidences(first)
    assert len(contract9.incidences(first)) == 144
