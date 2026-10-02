# SPDX-License-Identifier: GPL-3.0-only
"""Information-set access, hidden-state boundaries and imagined-game isolation."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

import pytest

from hexset.actions import apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.cards import DECK_COMPOSITION, DevCard
from hexset.game import (
    Phase,
    imagine,
    play_monopoly_card,
    play_road_building_card,
    play_year_of_plenty_card,
    start,
)
from hexset.ledger import SeatLedger
from hexset.view import View

SRC = Path(__file__).resolve().parent.parent

# Only engine-internal modules may read `Game`'s private field directly; these
# four directories are everything "outside the engine".
_OUTSIDE_ENGINE_DIRS = ("hexset/bots", "hexset/bench", "hexset/server", "hexset/clients")

# `x._state` for any `x` other than `self`: a module may legitimately own a
# `_state` member and reach it as `self._state`. Only the receiver being `self`
# is excluded, so `self.game._state` is still caught.
_PRIVATE_STATE_RE = re.compile(r"(?<!\bself)\._state\b")


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def after_setup(seed: int = 0, players: int = 4):
    game = a_game(players, seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def _set_known_hand(game, player: int, counts: list[int]) -> None:
    """Keeps `game.ledger` in sync; mirrors `test_ledger_engine`'s own."""
    state = game._state
    for r, n in enumerate(state.hands[player]):
        if n:
            state.bank[r] += n
            state.hands[player][r] = 0
    game.ledger.seats[player] = SeatLedger()
    for r, n in enumerate(counts):
        if n:
            state.bank[r] -= n
            state.hands[player][r] += n
            game.ledger.receive(player, r, n)


def test_the_views_known_and_unknown_match_the_ledger():
    game = after_setup()
    perspective = 0
    other = 1
    _set_known_hand(game, other, [2, 0, 1, 0, 0])
    view = game.state(perspective)
    seat_ledger = game.ledger.seats[other]
    assert view.known[other] == seat_ledger.known
    assert view.unknown[other] == seat_ledger.unknown


def test_the_views_hand_sizes_match_the_truth_for_every_seat():
    game = after_setup()
    perspective = 0
    view = game.state(perspective)
    for seat in range(game._state.num_players):
        true_size = sum(game._state.hands[seat])
        view_size = sum(view.known[seat]) + view.unknown[seat]
        assert view_size == true_size, f"seat {seat}: {view_size} != {true_size}"


def test_the_perspective_seats_own_hand_is_exact():
    game = after_setup()
    perspective = 2
    view = game.state(perspective)
    assert view.known[perspective] == game._state.hands[perspective]
    assert view.unknown[perspective] == 0


def test_unseen_dev_cards_is_exact_once_the_non_knight_cards_resolve():
    """Both Monopolies are played, so zero comes from the arithmetic, not the guard."""
    game = after_setup()
    game.phase = Phase.MAIN
    player = game.current_player
    perspective = (player + 1) % game._state.num_players

    game._state.dev_cards[player][DevCard.ROAD_BUILDING] = 1
    play_road_building_card(game)
    game.dev_card_played = False
    game.free_roads = 0             # its roads are not what this reads

    game._state.dev_cards[player][DevCard.YEAR_OF_PLENTY] = 1
    play_year_of_plenty_card(game, [Resource.WOOD, Resource.BRICK])
    game.dev_card_played = False

    game._state.dev_cards[player][DevCard.MONOPOLY] = 2
    play_monopoly_card(game, Resource.ORE)
    game.dev_card_played = False
    play_monopoly_card(game, Resource.ORE)

    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()
    assert unseen[DevCard.ROAD_BUILDING] == DECK_COMPOSITION[DevCard.ROAD_BUILDING] - 1
    assert unseen[DevCard.YEAR_OF_PLENTY] == DECK_COMPOSITION[DevCard.YEAR_OF_PLENTY] - 1
    assert unseen[DevCard.MONOPOLY] == 0
    assert DECK_COMPOSITION[DevCard.MONOPOLY] == 2, "both copies must be played above"


def test_p_holds_is_exact_where_it_can_be_and_sampled_where_it_cannot():
    game = after_setup()
    _set_known_hand(game, 1, [1, 1, 1, 0, 0])
    _set_known_hand(game, 2, [0, 0, 0, 2, 2])
    # Only seat 1's wood is certified: its other two cards are drawn from a
    # pool of six, its own two and seat 2's four.
    game.ledger.seats[1] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=2)
    game.ledger.seats[2] = SeatLedger(unknown=4)
    view = View.from_game(game, 0)
    assert view.p_holds(1, [1, 0, 0, 0, 0]) == 1.0
    assert view.p_holds(1, [0, 0, 0, 3, 0]) == 0.0, "three short, two hidden"
    pool, size = view.pool, view.pool_size
    assert size == 6
    one_short = 1.0 - math.comb(size - pool[1], 2) / math.comb(size, 2)
    assert view.p_holds(1, [0, 1, 0, 0, 0]) == pytest.approx(one_short)
    two_short = view.p_holds(1, [1, 1, 1, 0, 0], draws=200)
    assert 0.0 < two_short < 1.0
    assert two_short == view.p_holds(1, [1, 1, 1, 0, 0], draws=200), "a fixed stream by default"

    own = game.state(0, hidden=False).hands[0]
    short = [n + 1 if r == 0 else 0 for r, n in enumerate(own)]
    assert view.p_holds(0, short) == 0.0, "the perspective's own hand is exact"
    assert view.p_holds(0, own) == 1.0


def test_a_view_built_on_an_imagined_copy_diverges_after_a_mutation():
    game = after_setup()
    perspective = 0
    other = 1
    _set_known_hand(game, other, [1, 0, 0, 0, 0])

    child = imagine(game, random.Random(0))
    child._state.bank[1] += child._state.hands[other][1]
    child._state.hands[other][1] = 0
    child._state.bank[2] -= 1
    child._state.hands[other][2] += 1
    child.ledger.seats[other] = SeatLedger(known=[0, 0, 1, 0, 0], unknown=0)

    parent_view = game.state(perspective)
    child_view = child.state(perspective)

    assert parent_view.known[other] == [1, 0, 0, 0, 0]
    assert child_view.known[other] == [0, 0, 1, 0, 0]
    assert game._state.hands[other] != child._state.hands[other]


def test_no_bot_bench_server_or_client_module_reaches_into_game_state():
    offenders = []
    for rel in _OUTSIDE_ENGINE_DIRS:
        directory = SRC / rel
        assert directory.is_dir(), f"expected a directory at {directory}"
        for path in directory.rglob("*.py"):
            text = path.read_text()
            if _PRIVATE_STATE_RE.search(text):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "these files read Game's private state directly instead of going "
        f"through game.state(seat, hidden=...): {offenders}"
    )


def _hold(state, seat: int, card: DevCard, age: int) -> None:
    """`seat` holds `card`, kept `age` of its turns, taken out of the deck so
    the position stays one a game could reach."""
    state.deck.remove(int(card))
    state.dev_cards[seat][card] += 1
    state.dev_ages[seat].append(age)


class _OldCardsAreVictoryPoints:
    """A caller's reading: a card held past two turns is ten times likelier a
    victory point card."""

    def weight(self, card: int, age: int) -> float:
        return 10.0 if card == DevCard.VICTORY_POINT and age > 2 else 1.0


def _victory_points_dealt(view: View, rng: random.Random, hold=None) -> dict[int, int]:
    vp = {1: 0, 2: 0}
    unseen = view.unseen_dev_cards()
    for _ in range(400):
        world = view.sample(rng) if hold is None else view.sample(rng, hold)
        for seat in vp:
            vp[seat] += world.dev_cards[seat][DevCard.VICTORY_POINT]
        # Every unseen card is dealt once: the deck and the holdings between them.
        dealt = [world.deck.count(card) + sum(world.dev_cards[s][card] for s in (1, 2, 3))
                 for card in range(len(DevCard))]
        assert dealt == unseen
    return vp


def test_a_world_deals_every_unseen_development_card_alike_whatever_its_age():
    game = after_setup()
    state = game._state
    _hold(state, 1, DevCard.KNIGHT, 1)
    _hold(state, 2, DevCard.KNIGHT, 8)
    vp = _victory_points_dealt(View.from_game(game, 0), random.Random(0))
    assert 0.5 < vp[2] / vp[1] < 2


def test_a_world_deals_held_cards_by_the_callers_reading_of_their_age():
    game = after_setup()
    state = game._state
    _hold(state, 1, DevCard.KNIGHT, 1)
    _hold(state, 2, DevCard.KNIGHT, 8)
    vp = _victory_points_dealt(View.from_game(game, 0), random.Random(0), _OldCardsAreVictoryPoints())
    assert vp[2] > 3 * vp[1]
