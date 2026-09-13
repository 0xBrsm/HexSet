# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.clients.reachable` in isolation: pure vector arithmetic, so every
test here hand-crafts a hand (and, where it matters, a room/legal/deck
budget) rather than standing up a board or a game. `tests/clients/test_netbot.py`
covers the same gate wired into `NetworkBot`, budgets and all.
"""

from __future__ import annotations

from hexset.board.terrain import Resource
from hexset.clients.reachable import (
    Recipe,
    bank_conversions,
    branch_hands,
    maximal,
    purchase_multisets,
    reachable_recipes,
)

# No development cards, none playable, full piece room, a settlement's
# worth of legal placements everywhere unless a test says otherwise --
# the "nothing special about cards or the board" default most tests want.
NO_CARDS = (0, 0, 0, 0, 0)
FULL_ROOM = (15, 5, 4)
ONE_EACH_LEGAL = (1, 1, 1)
BASE_RATIOS = (4, 4, 4, 4, 4)
SETTLEMENT_COST = (1, 1, 1, 1, 0)  # wood, brick, sheep, wheat, ore


def _ctx(**overrides):
    ctx = dict(
        dev_cards_held=NO_CARDS,
        dev_card_played=False,
        bank=(19, 19, 19, 19, 19),
        known_others=[],
        ratios=BASE_RATIOS,
        room=FULL_ROOM,
        legal=ONE_EACH_LEGAL,
        deck_size=0,
    )
    ctx.update(overrides)
    return ctx


def test_a_4_to_1_conversion_reaches_a_settlement():
    """5 wood, 1 sheep, 1 wheat, no brick at all: a settlement needs one of
    each, so 4 of the 5 wood convert 4:1 into the missing brick, leaving
    exactly one wood behind for the settlement itself."""
    hand = (5, 0, 1, 1, 0)
    recipes = reachable_recipes(hand, **_ctx())
    assert (0, 1, 0, 0) in recipes
    recipe = recipes[(0, 1, 0, 0)]
    assert recipe.branch is None
    # The hand a settlement is actually paid from: four of the five wood
    # converted down to exactly the missing brick.
    assert recipe.hand == (1, 1, 1, 1, 0)


def test_a_2_to_1_port_reaches_it_from_fewer():
    """The same settlement, from only 3 wood -- a specific port on wood
    halves the rate `bank_conversions` needs to close the brick gap, so
    fewer wood are spent getting there than the 4:1 case above."""
    hand = (3, 0, 1, 1, 0)
    ratios = (2, 4, 4, 4, 4)  # a wood port, nothing else improved
    recipes = reachable_recipes(hand, **_ctx(ratios=ratios))
    assert (0, 1, 0, 0) in recipes
    assert recipes[(0, 1, 0, 0)].hand == (1, 1, 1, 1, 0)

    # The same 3 wood at the base 4:1 rate cannot close the gap at all.
    recipes_no_port = reachable_recipes(hand, **_ctx())
    assert (0, 1, 0, 0) not in recipes_no_port


def test_bank_conversions_is_closed_and_terminates():
    """Every ratio is at least 2:1, so the search shrinks the hand every
    step; from a hand with only wood, the reachable set is finite and
    contains hands built by converting one or more times."""
    reached = bank_conversions((8, 0, 0, 0, 0), (2, 4, 4, 4, 4))
    assert (8, 0, 0, 0, 0) in reached  # converting nothing is reachable too
    assert (6, 1, 0, 0, 0) in reached  # one 2:1 wood-for-brick trade
    assert (0, 4, 0, 0, 0) in reached  # converting every last would-be wood
    assert all(sum(h) <= 8 for h in reached)  # a trade never grows the hand


def test_a_monopoly_branch_adds_only_the_known_cards():
    """Monopoly sums `known_others` alone -- the certified lower bound this
    seat can prove, never a true hand it cannot read. A seat that could
    actually see three wheat from another player would still only branch on
    what its own ledger names."""
    branches = branch_hands(
        (0, 0, 0, 0, 0),
        dev_cards_held=(0, 0, 0, 0, 1),  # one Monopoly, matured
        dev_card_played=False,
        bank=(19, 19, 19, 19, 19),
        known_others=[(0, 0, 0, 2, 0), (0, 0, 0, 1, 0)],  # 3 wheat known, combined
    )
    wheat_branch = next(b for b in branches if b[0] == "monopoly" and b[1] == Resource.WHEAT)
    assert wheat_branch[2] == (0, 0, 0, 3, 0)  # 2 + 1 known, nothing more
    # No branch at all for a resource nobody is known to hold.
    assert not any(b[0] == "monopoly" and b[1] == Resource.WOOD for b in branches)


def test_a_year_of_plenty_branch_adds_a_pair():
    branches = branch_hands(
        (0, 0, 0, 0, 0),
        dev_cards_held=(0, 0, 0, 1, 0),  # one Year of Plenty, matured
        dev_card_played=False,
        bank=(19, 19, 19, 19, 19),
        known_others=[],
    )
    wood_wood = next(
        b for b in branches if b[0] == "year_of_plenty" and b[1] == (Resource.WOOD, Resource.WOOD)
    )
    assert wood_wood[2] == (2, 0, 0, 0, 0)
    # All 15 unordered pairs are offered when the bank can supply every one.
    assert sum(1 for b in branches if b[0] == "year_of_plenty") == 15


def test_a_card_already_played_this_turn_has_no_branch():
    branches = branch_hands(
        (0, 0, 0, 0, 0),
        dev_cards_held=(0, 0, 0, 1, 1),  # both Year of Plenty and Monopoly held
        dev_card_played=True,
        bank=(19, 19, 19, 19, 19),
        known_others=[(0, 0, 0, 5, 0)],
    )
    assert branches == ((None, None, (0, 0, 0, 0, 0)),)


def test_a_dev_card_bought_this_turn_is_not_a_branch():
    """`dev_cards_held` is `state.dev_cards`, matured cards only -- a card
    bought this turn (`state.new_dev_cards`) is never passed in here, and a
    seat that holds none matured gets no branch even if the bank could
    otherwise supply one."""
    branches = branch_hands(
        (0, 0, 0, 0, 0),
        dev_cards_held=NO_CARDS,  # nothing matured, whatever was bought this turn
        dev_card_played=False,
        bank=(19, 19, 19, 19, 19),
        known_others=[(0, 0, 0, 5, 0)],
    )
    assert branches == ((None, None, (0, 0, 0, 0, 0)),)


def test_a_multiset_needing_two_settlements_requires_two_legal_vertices():
    hand = (2, 2, 2, 2, 0)  # exactly two settlements' worth
    one_spot = purchase_multisets(hand, room=FULL_ROOM, legal=(0, 1, 0), deck_size=0)
    assert (0, 1, 0, 0) in one_spot
    assert (0, 2, 0, 0) not in one_spot  # affordable, but nowhere for the second

    two_spots = purchase_multisets(hand, room=FULL_ROOM, legal=(0, 2, 0), deck_size=0)
    assert (0, 2, 0, 0) in two_spots


def test_a_piece_cap_bounds_a_multiset_the_same_way():
    hand = (2, 2, 2, 2, 0)
    capped = purchase_multisets(hand, room=(15, 1, 4), legal=(0, 2, 0), deck_size=0)
    assert (0, 2, 0, 0) not in capped  # room for only one more settlement
    assert (0, 1, 0, 0) in capped


def test_a_dev_card_multiset_is_bounded_by_room_and_deck_together():
    hand = (0, 0, 3, 3, 3)  # three dev cards' worth of resources
    assert (0, 0, 0, 2) in purchase_multisets(hand, room=FULL_ROOM, legal=ONE_EACH_LEGAL, deck_size=2)
    assert (0, 0, 0, 3) not in purchase_multisets(
        hand, room=FULL_ROOM, legal=ONE_EACH_LEGAL, deck_size=2
    )


def test_maximal_drops_a_multiset_a_bigger_one_covers():
    reachable = frozenset({(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)})
    assert maximal(reachable) == [(1, 1, 0, 0)]


def test_maximal_keeps_two_incomparable_multisets():
    reachable = frozenset({(0, 0, 0, 0), (2, 0, 0, 0), (0, 1, 0, 0)})
    assert sorted(maximal(reachable)) == [(0, 1, 0, 0), (2, 0, 0, 0)]


def test_the_filter_is_exact_a_count_only_change_is_not_ignored():
    """The known simplification the old kind-based filter carried -- "two
    roads bought instead of one" does not survive -- is gone: `R(hand)`
    changing from one road to two is a real difference here."""
    one_road_hand = (1, 1, 0, 0, 0)
    two_road_hand = (2, 2, 0, 0, 0)
    ctx = _ctx(legal=(2, 1, 1), room=(15, 5, 4))
    one = frozenset(reachable_recipes(one_road_hand, **ctx))
    two = frozenset(reachable_recipes(two_road_hand, **ctx))
    assert (1, 0, 0, 0) in one and (2, 0, 0, 0) not in one
    assert (2, 0, 0, 0) in two
    assert one != two


def test_the_filter_is_exact_an_unrelated_change_is_ignored():
    """Adding a card no reachable multiset depends on at all leaves `R`
    exactly equal -- the same hand plus an ore nobody can spend on
    anything reachable."""
    hand = (1, 1, 0, 0, 0)
    richer = (1, 1, 0, 0, 5)  # +5 ore: nothing here affords a city or a dev card
    ctx = _ctx(legal=(1, 0, 0), room=(15, 0, 0), deck_size=0)
    before = frozenset(reachable_recipes(hand, **ctx))
    after = frozenset(reachable_recipes(richer, **ctx))
    assert before == after


def test_a_recipe_is_a_buildable_path_not_just_a_label():
    """Every multiset in `reachable_recipes`' result comes with a `Recipe`
    -- a branch (or none), and the exact hand a position would be built
    from -- not merely membership in a set."""
    recipes = reachable_recipes(
        (0, 0, 0, 0, 0),
        dev_cards_held=(0, 0, 0, 1, 0),
        dev_card_played=False,
        bank=(19, 19, 19, 19, 19),
        known_others=[],
        ratios=BASE_RATIOS,
        room=FULL_ROOM,
        legal=(0, 1, 0),
        deck_size=0,
    )
    # Year of Plenty on wood+brick, converted nowhere further, buys nothing
    # by itself (still short a settlement's sheep and wheat) but is at
    # least a reachable, buildable hand.
    recipe = recipes[(0, 0, 0, 0)]
    assert isinstance(recipe, Recipe)
    assert recipe.purchases == (0, 0, 0, 0)
