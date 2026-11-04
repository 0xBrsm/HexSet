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


# --- contract 9d, the two-seat observation -----------------------------------

from hexset.board.board import pips  # noqa: E402
from hexset.contract9 import (  # noqa: E402
    HEX_COLUMNS_9D as H9D, SEAT_COLUMNS_9D as C9D, TABLE_COLUMNS_9D as T9D, hex_features, next_roll_odds, rolled,
)
from hexset.rules import DUEL_VARIANT_GAME  # noqa: E402


def duel_positions(count: int = 6, steps: int = 400, every: int = 5):
    """Two-seat duel-variant games played at random: the game at the deal and
    then at every `every`th move."""
    for seed in range(count):
        rng = random.Random(100 + seed)
        game = start(random_base_board(rng), 2, rng, game_type=DUEL_VARIANT_GAME)
        yield game
        for step in range(steps):
            if is_over(game):
                break
            step_randomly(game, rng)
            if step % every == 0:
                yield game


def test_9d_is_the_two_seat_observation():
    assert contract9.features("9d") == (22, 20) and hex_features("9d") == 8 and hex_features("9") == 7
    assert layout(2, 19, 54, 72, "9d").width == 2 * 22 + 20 + 19 * 8 + 54 * 7 + 54 + 72 + 2 * 54 == 828
    assert "hand_most" not in C9D and "pool" not in T9D and "live" not in C9D
    # Nothing another column fixes: no bank, no deck, no opponent bounds by kind in a seat token.
    assert "bank" not in T9D and "deck" not in T9D and "dev_most" not in C9D and "dev_playable" not in C9D
    with pytest.raises(ValueError, match="two-seat"):
        encode9(a_game(), 0, "9d")
    for game in duel_positions(count=2, steps=60):
        for p in range(2):
            o = encode9(game, p, "9d")
            assert o.seats.shape == (2, 22) and o.table.shape == (20,) and o.hexes.shape == (19, 8)
            assert flatten(o).shape == (828,)
            assert o.seats[0, C9D["me"]] == 1.0 and o.seats[1, C9D["me"]] == 0.0
            # One seat opened the setup snake: `game.first`, whichever row it sits in.
            assert [float(s == game.first) for s in ((p + i) % 2 for i in range(2))] == list(o.seats[:, C9D["opened"]])


def test_9d_hands_are_exact_and_the_deck_is_the_one_unknown():
    """Across a table of two nothing in a hand is hidden: each row's hand is
    the seat's true hand, from either perspective. The perspective's own
    development cards are the table's, exact; each seat token carries the
    public count held and bought this turn."""
    checked = 0
    for game in duel_positions():
        state = game.state(0, hidden=False)   # true state: the exact hands the contract claims to read
        for p in range(2):
            o = encode9(game, p, "9d")
            np.testing.assert_allclose(o.seats[0, C9D["hand"]] * 10, state.hands[p], atol=1e-5)
            np.testing.assert_allclose(o.seats[1, C9D["hand"]] * 10, state.hands[1 - p], atol=1e-5)
            mine = [state.dev_cards[p][k] + state.new_dev_cards[p][k] for k in range(5)]
            np.testing.assert_allclose(o.table[T9D["dev_mine"]] * 10, mine, atol=1e-5)
            for row, seat in ((0, p), (1, 1 - p)):
                new = sum(state.new_dev_cards[seat])
                np.testing.assert_allclose(o.seats[row, C9D["dev_held"]] * 10, sum(state.dev_cards[seat]) + new,
                                           atol=1e-5)
                np.testing.assert_allclose(o.seats[row, C9D["dev_new"]] * 10, new, atol=1e-5)
            checked += 1
    assert checked > 200


def test_9d_odds_start_as_pips_and_follow_the_deck():
    """A fresh dice deck rolls as independent dice do, so each hex's odds
    column opens at its pips over 36 and the seven's at 6/36; after that it
    is the deck's chance for the seat rolling next: the seat on turn until
    it has rolled, then the other."""
    rng = random.Random(0)
    game = start(random_base_board(rng), 2, rng, game_type=DUEL_VARIANT_GAME)
    o = encode9(game, 0, "9d")
    tokens = game._state.board.tokens
    np.testing.assert_allclose(o.hexes[:, H9D["odds"]], [pips(t) / 36 for t in tokens], atol=1e-6)
    np.testing.assert_allclose(o.hexes[:, H9D["number"]], [t / 12 for t in tokens], atol=1e-6)
    assert abs(o.table[T9D["seven"]][0] - 6 / 36) < 1e-6
    before = after = 0
    for game in duel_positions(count=4):
        state = game.state(0, hidden=False)   # true state: the deck is public, read here directly
        roller = game.current_player if not rolled(game) else 1 - game.current_player
        odds = dict(state.dice.odds(roller))
        assert next_roll_odds(game) == odds
        for p in range(2):
            o = encode9(game, p, "9d")
            np.testing.assert_allclose(o.hexes[:, H9D["odds"]], [odds.get(t, 0.0) for t in state.board.tokens], atol=1e-6)
            np.testing.assert_allclose(o.table[T9D["seven"]], [odds.get(7, 0.0)], atol=1e-6)
            desert = [h for h, t in enumerate(state.board.tokens) if t == 0]
            assert (o.hexes[desert][:, [H9D["number"], H9D["odds"]]] == 0).all()
        if rolled(game):
            after += 1
        else:
            before += 1
    assert before > 20 and after > 20


def test_9d_shares_the_board_and_the_public_seat_reads_with_contract_9():
    """At two seats contract 9 and 9d read the same position: the board part
    bit for bit, every seat column both carry alike, and 9d's exact hand is
    9's lower bound plus the pool."""
    same_seat = ("on_turn", "me", "dev_held", "dev_oldest", "to_longest_road", "to_largest_army", "must_discard",
                 "pieces")
    same_table = ("dev_pool", "turn", "setup", "rolled", "robber")
    checked = 0
    for game in duel_positions(count=4, every=7):
        for p in range(2):
            nine, duel = encode9(game, p, "9"), encode9(game, p, "9d")
            for name in ("vertices", "vertex_owner", "road_owner", "reach"):
                assert getattr(nine, name).tobytes() == getattr(duel, name).tobytes(), name
            for name in same_seat:
                np.testing.assert_array_equal(nine.seats[:, C[name]], duel.seats[:, C9D[name]], err_msg=name)
            for name in same_table:
                np.testing.assert_array_equal(nine.table[T[name]], duel.table[T9D[name]], err_msg=name)
            # The perspective's own development cards: 9's first seat token, 9d's table.
            np.testing.assert_array_equal(nine.seats[0, C["dev_most"]], duel.table[T9D["dev_mine"]])
            np.testing.assert_array_equal(nine.seats[0, C["dev_playable"]], duel.table[T9D["dev_playable"]])
            np.testing.assert_allclose(duel.seats[0, C9D["hand"]], nine.seats[0, C["hand_least"]], atol=1e-6)
            np.testing.assert_allclose(duel.seats[1, C9D["hand"]], nine.seats[1, C["hand_least"]] + nine.table[T["pool"]],
                                       atol=1e-6)
            # Points over the game's 15, not contract 9's 10.
            np.testing.assert_allclose(duel.seats[:, C9D["points"]] * 15, nine.seats[:, C["points"]] * 10, atol=1e-5)
            np.testing.assert_array_equal(nine.hexes[:, :5], duel.hexes[:, H9D["terrain"]])
            np.testing.assert_array_equal(nine.hexes[:, 6], duel.hexes[:, H9D["robber"]])
            checked += 1
    assert checked > 100


def test_9d_drops_only_what_its_other_columns_fix():
    """What 9d no longer reads is a function of what it does: contract 9's
    bank is 19 of each less the two exact hands, its deck the unseen cards
    less the opponent's held, and its opponent bounds by kind the opponent's
    held (or, to play now, its held less bought this turn, none if the mover
    has played one) against the unseen of each kind."""
    from hexset.encoding import PLAYABLE_KINDS

    checked = 0
    for game in duel_positions(count=4, every=7):
        for p in range(2):
            nine, duel = encode9(game, p, "9"), encode9(game, p, "9d")
            hands = duel.seats[0, C9D["hand"]] + duel.seats[1, C9D["hand"]]
            np.testing.assert_allclose(nine.table[T["bank"]], 1.9 - hands, atol=1e-5)
            pool = duel.table[T9D["dev_pool"]]
            np.testing.assert_allclose(nine.table[T["deck"]], pool.sum() - duel.seats[1, C9D["dev_held"]], atol=1e-5)
            held = np.rint(duel.seats[1, C9D["dev_held"]][0] * 10)
            matured = held - np.rint(duel.seats[1, C9D["dev_new"]][0] * 10)
            unseen = np.rint(pool * 10)
            np.testing.assert_allclose(nine.seats[1, C["dev_most"]] * 10, np.minimum(held, unseen), atol=1e-4)
            played = duel.table[T9D["dev_played"]][0] == 1 and duel.seats[1, C9D["on_turn"]][0] == 1
            playable = [0 if played else min(matured, unseen[int(kind)]) for kind in PLAYABLE_KINDS]
            np.testing.assert_allclose(nine.seats[1, C["dev_playable"]] * 10, playable, atol=1e-4)
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("players", [3, 4])
def test_duel_layout_refuses_other_seat_counts(players):
    with pytest.raises(ValueError, match="two-seat"):
        contract9.layout(players, 19, 54, 72, "9d")


def test_9d_reads_no_constant_column():
    """Every column of the two-seat observation takes more than one value
    across random duels: a column a duel never moves is not read. `me`
    marks the perspective's token and differs between the two rows."""
    seats, tables, hexes = [], [], []
    for game in duel_positions(count=10, steps=700, every=3):
        for p in range(2):
            o = encode9(game, p, "9d")
            seats.append(o.seats)
            tables.append(o.table)
            hexes.append(o.hexes)
    s, t, h = np.concatenate(seats), np.stack(tables), np.concatenate(hexes)
    assert not [name for name, cols in C9D.items() if name != "me" and (np.ptp(s[:, cols], axis=0) == 0).any()]
    assert (s[0::2, C9D["me"]] == 1).all() and (s[1::2, C9D["me"]] == 0).all()
    assert not [name for name, cols in T9D.items() if (np.ptp(t[:, cols], axis=0) == 0).any()]
    assert not [name for name, col in H9D.items() if (np.ptp(h[:, col], axis=0) == 0).any()]


# --- contract 9d at a table dealt four with two seats retired ------------------

from hexset.actions import build_space  # noqa: E402
from hexset import decisions as D  # noqa: E402
from hexset.encoding import seat_frame  # noqa: E402


def retired_duel_positions(count: int = 4, steps: int = 400, every: int = 3, locked=(1, 3)):
    """Duel-variant games dealt four seats with `locked` retired, played at
    random: the game at the deal and then at every `every`th move."""
    for seed in range(count):
        rng = random.Random(200 + seed)
        game = start(random_base_board(rng), 4, rng, game_type=DUEL_VARIANT_GAME, locked=locked)
        yield game
        for step in range(steps):
            if is_over(game):
                break
            step_randomly(game, rng)
            if step % every == 0:
                yield game


def test_the_seat_frame_counts_every_dealt_seat_or_the_live_ones():
    full = a_game()
    assert seat_frame(full, 2, 4).seats == (2, 3, 0, 1) and seat_frame(full, 2, 4).dealt == 4
    game = next(retired_duel_positions(count=1))
    assert seat_frame(game, 0, 2).seats == (0, 2) and seat_frame(game, 2, 2).seats == (2, 0)
    frame = seat_frame(game, 2, 2)
    assert (frame.relative(0), frame.relative(2), frame.relative(4)) == (1, 0, 2)
    assert (frame.absolute(0), frame.absolute(1), frame.absolute(2)) == (2, 0, 4)
    assert frame.to_board([0.75, 0.25]) == (0.25, 0.0, 0.75, 0.0)
    with pytest.raises(ValueError, match="live seats"):
        seat_frame(game, 1, 2)          # a retired seat has no frame
    with pytest.raises(ValueError, match="live seats"):
        seat_frame(full, 0, 2)          # four live seats are not two


def test_9d_reads_a_table_dealt_four_by_its_two_live_seats():
    """At a table dealt four with two retired, 9d reads the two live seats as
    a table of two: the same shapes, the perspective first, each hand exact,
    every pointer naming a live seat's row, and nothing of the retired seats."""
    seen = 0
    for game in retired_duel_positions():
        state = game._state
        for p in (0, 2):
            other = 2 - p
            o = encode9(game, p, "9d")
            assert o.seats.shape == (2, 22) and flatten(o).shape == (828,)
            assert o.seats[0, C9D["me"]] == 1.0 and o.seats[1, C9D["me"]] == 0.0
            assert o.seats[0, C9D["on_turn"]] == float(game.current_player == p)
            assert o.seats[1, C9D["on_turn"]] == float(game.current_player == other)
            np.testing.assert_allclose(o.seats[1, C9D["hand"]] * contract9.CARD_SCALE, state.hands[other], atol=1e-5)
            np.testing.assert_allclose(o.seats[0, C9D["hand"]] * contract9.CARD_SCALE, state.hands[p], atol=1e-5)
            owner = np.asarray(state.vertex_owner)
            assert set(np.unique(o.vertex_owner)) <= {-1, 0, 1}
            assert np.array_equal(o.vertex_owner == 0, owner == p)
            assert np.array_equal(o.vertex_owner == 1, owner == other)
            assert np.array_equal(o.road_owner == 1, np.asarray(state.edge_owner) == other)
            seen += 1
    assert seen > 200


def test_robber_victims_count_the_live_seats_both_ways():
    """A two-seat space at a table dealt four: every legal flat move's slot
    names the same move again, robber victims and "no victim" included, and
    the decision's mask holds exactly the legal moves."""
    robber = 0
    for game in retired_duel_positions(count=6, steps=600, every=1):
        seat = game.current_player
        if game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.DISCARD) or D.pending_free_roads(game):
            continue
        topology = game._state.board.topology
        space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, 2)
        decision = D.tensors(game, seat, space)
        frame = D.frame_for(game, seat, space)
        pairs = contract9.incidences(topology)
        legal = [a for a in legal_actions(game, seat) if a.type is not ActionType.PLAY_ROAD_BUILDING]
        indices = {D.flat_index(space, a, seat, 4, frame) for a in legal}
        assert indices == set(np.flatnonzero(decision.flat))
        for action in legal:
            kind, index = D.locate(decision, action, game, seat, space)
            assert D.option_of(decision, kind, index, pairs, seat, space, 2, frame) == action
            robber += action.type is ActionType.MOVE_ROBBER
    assert robber > 50


# --- contract 9d.1: 9d with each seat's economy ---------------------------------


def test_9d1_appends_each_seats_production_and_bank_rates_to_9d():
    """9d.1 is 9d with ten columns appended to each seat token: production by
    resource at the long-run odds (the expectation of `state.production` over
    every roll but the seven) and the bank rate by resource over 4; every 9d
    column is unchanged, at a table of two and at one dealt four."""
    from hexset.contract9 import SEAT_COLUMNS_9D1 as C1, SEAT_FEATURES_9D1
    from hexset.economy import trade_ratios
    from hexset.state import production

    assert contract9.features("9d.1") == (SEAT_FEATURES_9D1, 20) == (32, 20)
    assert layout(2, 19, 54, 72, "9d.1").width == 848
    with pytest.raises(ValueError, match="two-seat"):
        layout(4, 19, 54, 72, "9d.1")
    seen, ported, producing = 0, 0, 0
    for game in [*duel_positions(count=3, steps=300), *retired_duel_positions(count=2, steps=300)]:
        st = game._state
        live = [s for s in range(st.num_players) if s not in game.locked]
        for p in live:
            a, b = encode9(game, p, "9d"), encode9(game, p, "9d.1")
            assert b.seats.shape == (2, 32)
            assert np.array_equal(a.seats, b.seats[:, :22]) and np.array_equal(a.table, b.table)
            assert np.array_equal(a.hexes, b.hexes) and np.array_equal(a.vertex_owner, b.vertex_owner)
            for row, seat in enumerate((p, next(s for s in live if s != p))):
                expected = [sum(pips(roll) / 36 * production(st, roll)[seat][r] for roll in range(2, 13) if roll != 7)
                            for r in range(5)]
                np.testing.assert_allclose(b.seats[row, C1["production"]], expected, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(b.seats[row, C1["rates"]], np.asarray(trade_ratios(st, seat)) / 4)
                ported += bool((b.seats[row, C1["rates"]] < 1).any())
                producing += bool(b.seats[row, C1["production"]].sum() > 0)
            seen += 1
    assert seen > 100 and ported > 10 and producing > 50
