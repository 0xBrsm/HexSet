# SPDX-License-Identifier: GPL-3.0-only
"""Joint hidden-hand constraints, including the bounded-deal fallback."""

from __future__ import annotations

import random
from itertools import combinations_with_replacement, product

import pytest

import hexset.view as belief
from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.game import start
from hexset.ledger import SeatLedger
from hexset.mcts import Search
from hexset.state import BANK_PER_RESOURCE
from hexset.view import View


def _scarce_pool_game():
    """Seat 1 must leave the only hidden wood for seat 2.

    The perspective's private rows restrict seat 1 to wood/brick and seat 2
    to wood/sheep. Seat 1 has one hidden card, seat 2 two, and the shared
    pool has one of each. Looking only at its own bounds, seat 1 could take
    wood and strand seat 2. Each opponent also has a certified ore.
    """
    rng = random.Random(0)
    game = start(random_base_board(rng), 3, rng)
    state = game._state
    state.hands = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 1]]
    state.bank = [BANK_PER_RESOURCE - sum(h[r] for h in state.hands) for r in range(5)]
    game.ledger.seats = [
        SeatLedger(known=state.hands[0][:]),
        SeatLedger(known=[0, 0, 0, 0, 1], unknown=1),
        SeatLedger(known=[0, 0, 0, 0, 1], unknown=2),
    ]
    private = game.ledger.knowledge(0)
    private.seats[1].most = [1, 1, 0, 0, 1]
    private.seats[2].most = [2, 0, 2, 0, 1]
    game.ledger.private = [private, None, None]
    for card, row in ((DevCard.VICTORY_POINT, state.dev_cards[0]),
                      (DevCard.KNIGHT, state.new_dev_cards[0])):
        state.deck.remove(int(card))
        row[card] += 1
    state.dev_ages[0] = [3, 0]
    return game


def _assert_consistent(view, state):
    for seat, hand in enumerate(state.hands):
        assert sum(hand) == view.sizes[seat]
        assert all(k <= h <= k + room for k, h, room in
                   zip(view.known[seat], hand, view.room[seat]))
    for r in range(5):
        assert state.bank[r] + sum(h[r] for h in state.hands) == BANK_PER_RESOURCE
    own = view.perspective
    assert state.hands[own] == view.state.hands[own]
    assert state.dev_cards[own] == view.state.dev_cards[own]
    assert state.new_dev_cards[own] == view.state.new_dev_cards[own]
    assert state.dev_ages[own] == view.state.dev_ages[own]
    assert state.bank == view.state.bank
    for card in DevCard:
        assert (state.deck.count(card)
                + sum(h[card] for h in state.dev_cards)
                + sum(h[card] for h in state.new_dev_cards)
                == view.state.deck.count(card)
                + sum(h[card] for h in view.state.dev_cards)
                + sum(h[card] for h in view.state.new_dev_cards))


@pytest.mark.parametrize("through_search", [False, True])
@pytest.mark.parametrize("tries", [0, 1, 16])
def test_fallback_reserves_scarce_cards_for_other_seats(monkeypatch, through_search, tries):
    monkeypatch.setattr(belief, "DEAL_TRIES", tries)
    game = _scarce_pool_game()
    view = game.state(0)
    assert view.pool == [1, 1, 1, 0, 0]
    assert view.unknown == [0, 1, 2]
    assert view.room[1] == [1, 1, 0, 0, 0]
    assert view.room[2] == [2, 0, 2, 0, 0]
    signature = view.signature()
    for seed in range(64):
        rng = random.Random(seed)
        if through_search:
            worlds = [world.state(0, hidden=False)
                      for _, world in Search(None, k=4, rng=rng).worlds(game)]
        else:
            worlds = [view.sample(rng)]
        for state in worlds:
            _assert_consistent(view, state)
            assert state.hands[1] == [0, 1, 0, 0, 1]
            assert state.hands[2] == [1, 0, 1, 0, 1]
    assert view.signature() == signature
    assert view.pool == [1, 1, 1, 0, 0]
    assert game.state(0) == view


def test_fallback_is_repeatable_and_reuses_residual_feasibility(monkeypatch):
    monkeypatch.setattr(belief, "DEAL_TRIES", 0)
    view = _scarce_pool_game().state(0)
    belief._feasible_deal.cache_clear()
    first_rng, second_rng = random.Random(4), random.Random(4)
    first = view.sample(first_rng)
    before = belief._feasible_deal.cache_info()
    second = view.sample(second_rng)
    after = belief._feasible_deal.cache_info()
    assert first.hands == second.hands
    assert first.dev_cards == second.dev_cards
    assert first.new_dev_cards == second.new_dev_cards
    assert first.deck == second.deck
    assert first_rng.getstate() == second_rng.getstate()
    assert after.hits > before.hits
    assert after.misses == before.misses
    assert after.maxsize is not None


@pytest.mark.parametrize("bounded", [False, True])
def test_successful_shuffle_keeps_the_original_fast_path_and_rng(monkeypatch, bounded):
    game = _scarce_pool_game()
    if not bounded:
        game.ledger.private = None
    view = game.state(0)

    def unexpected_fallback(*args):
        pytest.fail("a valid shuffle must not consult fallback feasibility")

    monkeypatch.setattr(belief, "_feasible_deal", unexpected_fallback)
    rng, expected_rng = random.Random(1), random.Random(1)
    cards = view._pool_cards()
    expected_rng.shuffle(cards)
    assert cards == [1, 2, 0]
    assert view._deal(rng) == [[], [1], [2, 0]]
    assert rng.getstate() == expected_rng.getstate()


def test_jointly_impossible_bounds_raise_without_changing_the_view():
    game = _scarce_pool_game()
    private = game.ledger.private[0]
    private.seats[1].most = [1, 0, 0, 0, 1]
    private.seats[2].most = [1, 0, 1, 0, 1]
    view = game.state(0)
    # Either seat fits alone, but both require the only hidden wood.
    assert all(sum(min(p, cap) for p, cap in zip(view.pool, view.room[s]))
               >= view.unknown[s] for s in (1, 2))
    signature = view.signature()
    with pytest.raises(ValueError, match="inconsistent resource bounds: no complete hidden deal"):
        view.sample(random.Random(0))
    assert view.signature() == signature
    assert game.state(0) == view


@pytest.mark.parametrize("pool,room", [([1, 1], [1, 0]), ([0, 0], [2, 2])])
def test_single_hand_fallback_never_widens_bounds_or_invents_cards(monkeypatch, pool, room):
    monkeypatch.setattr(belief, "DEAL_TRIES", 0)
    original = pool[:]
    with pytest.raises(ValueError, match="inconsistent resource bounds"):
        belief._capped_draw(random.Random(0), None, 2, room, pool)
    assert pool == original


def _enumerated_deals(pool, needs, rooms):
    """Every count allocation, independently of the flow-cut check."""
    hands = [combinations_with_replacement(range(len(pool)), n) for n in needs]
    for cards in product(*hands):
        counts = tuple(tuple(hand.count(r) for r in range(len(pool))) for hand in cards)
        if (all(n <= cap for row, room in zip(counts, rooms) for n, cap in zip(row, room))
                and all(sum(row[r] for row in counts) <= n for r, n in enumerate(pool))):
            yield counts


@pytest.mark.parametrize("seats,cap_values", [(1, range(3)), (2, range(3)), (3, range(2))])
def test_joint_feasibility_agrees_with_exhaustive_small_allocations(seats, cap_values):
    for pool in product(range(3), repeat=2):
        for needs in product(range(3), repeat=seats):
            for flat in product(cap_values, repeat=2 * seats):
                rooms = tuple(flat[2 * s:2 * s + 2] for s in range(seats))
                expected = next(_enumerated_deals(pool, needs, rooms), None) is not None
                assert belief._feasible_deal(pool, needs, rooms) == expected, (pool, needs, rooms)


def test_fallback_samples_only_enumerated_deals_with_contested_resources(monkeypatch):
    monkeypatch.setattr(belief, "DEAL_TRIES", 0)
    needs = (1, 1, 1)
    for pool in product(range(3), repeat=3):
        if sum(pool) != sum(needs):
            continue
        for flat in product(range(2), repeat=9):
            rooms = tuple(flat[3 * s:3 * s + 3] for s in range(3))
            possible = set(_enumerated_deals(pool, needs, rooms))
            view = object.__new__(View)
            view.num_players, view.perspective = 4, 3
            view.pool = list(pool) + [0, 0]
            view.unknown = list(needs) + [0]
            view.room = [list(room) + [0, 0] for room in rooms] + [[0] * 5]
            view.bounded = [True, True, True, False]
            if not possible:
                with pytest.raises(ValueError, match="inconsistent resource bounds"):
                    view._deal(random.Random(0))
                continue
            sampled = set()
            for seed in range(4):
                cards = view._deal(random.Random(seed))[:3]
                counts = tuple(tuple(hand.count(r) for r in range(3)) for hand in cards)
                assert counts in possible, (pool, rooms, counts)
                sampled.add(counts)
            if all(all(row) for row in rooms) and len(possible) > 1:
                assert len(sampled) > 1
