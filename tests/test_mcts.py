from __future__ import annotations

import random

import numpy as np
import pytest

from helpers import clear_hand, give

from catan.actions import Action, ActionType, apply, legal_actions, victim_of
from catan.board.board import random_base_board
from catan.board.terrain import Resource
from catan.cards import DevCard
from catan.game import Phase, imagine, start
from catan.bots import STANCES
from catan.mcts import (
    HIDDEN_DRAW,
    STANCE_ROWS,
    Leaf,
    Node,
    Search,
    _Chance,
    _drawn,
    visit_policy,
)


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


class Stub:
    """Uniform prior and a fixed value, remembering every wave it was handed.

    Torch is not involved anywhere in this file. That is the point of the
    module under test being separate from the network: the search's own
    bookkeeping is checkable without a forward.
    """

    def __init__(self, value=(0.0, 0.0, 0.0, 0.0), favour: int | None = None) -> None:
        self.value = value
        self.favour = favour
        self.waves: list[list[Leaf]] = []

    @property
    def leaves(self) -> int:
        return sum(len(wave) for wave in self.waves)

    def evaluate(self, leaves):
        self.waves.append(list(leaves))
        out = []
        for leaf in leaves:
            n = len(leaf.options)
            prior = np.full(n, 1.0 / n)
            if self.favour is not None and n > 1:
                prior = np.full(n, 0.01 / (n - 1))
                prior[self.favour % n] = 0.99
            out.append((prior, self.value))
        return out


def a_root(search: Search, game, stub: Stub) -> Node:
    """A root expanded by hand, so a test can drive `_descend` directly."""
    root = search._node(imagine(game, search.rng))
    (prior, value), = stub.evaluate([Leaf(root.game, root.mover, root.options)])
    root.prior = np.asarray(prior)
    root.value = tuple(value)
    return root


def test_every_simulation_lands_in_the_root_visit_counts():
    stub = Stub()
    search = Search(stub, simulations=32, wave=8, rng=random.Random(1))
    _, options, visits = search.run(a_game())
    assert len(options) == len(visits)
    assert visits.sum() == 32


def test_independent_roots_share_evaluator_calls_not_tree_statistics():
    stub = Stub()
    search = Search(stub, simulations=24, wave=4, rng=random.Random(1))
    results = search.run_many([a_game(seed) for seed in range(3)])

    # All roots are the first network batch; serial `run` would make three.
    assert len(stub.waves[0]) == 3
    assert len(results) == 3
    assert all(visits.sum() == 24 for _, _, visits in results)
    assert len({id(root) for root, _, _ in results}) == 3


def test_search_randomizes_the_hidden_deck_only_when_buying_a_card():
    class CountingRandom(random.Random):
        def __init__(self):
            super().__init__(1)
            self.shuffles = 0

        def shuffle(self, values):
            self.shuffles += 1
            super().shuffle(values)

    rng = CountingRandom()
    search = Search(Stub(), simulations=4, wave=2, rng=rng)
    game = a_game()
    search.run(game)
    assert rng.shuffles == 0

    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [3] * len(game.state.hands[0])
    node = search._node(imagine(game, random.Random(2), randomize_deck=False))
    node.options = (Action(ActionType.BUY_DEV_CARD),)
    before = len(node.game.state.deck)
    child = search._step(node, 0, None)

    assert rng.shuffles == 1
    assert len(child.game.state.deck) == before - 1


def after_setup(seed: int = 0):
    """The opening placements played out, so hexes have occupants and a robber
    move has somebody to steal from."""
    game = a_game(seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def a_steal(seed: int = 0):
    """A robber decision and the edge index of one that names a victim.

    The victim holds two kinds of card and not one, because the defect being
    pinned is an edge frozen on the first card it drew: a victim holding a
    single resource draws the same card every time and cannot show it.
    """
    game = after_setup(seed)
    game.phase = Phase.ROBBER
    game.current_player = 0
    for player in range(game.state.num_players):
        clear_hand(game.state, player)
    give(game.state, 1, Resource.WOOD, 6)
    give(game.state, 1, Resource.WHEAT, 6)
    index = next(
        i
        for i, action in enumerate(legal_actions(game))
        if action.type is ActionType.MOVE_ROBBER and victim_of(game, action.b) == 1
    )
    return game, index


def a_purchase(seed: int = 0):
    """A main-phase decision that can afford a development card, and its edge."""
    game = after_setup(seed)
    game.phase = Phase.MAIN
    game.current_player = 0
    clear_hand(game.state, 0)
    for resource in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        give(game.state, 0, resource, 3)
    index = next(
        i
        for i, action in enumerate(legal_actions(game))
        if action.type is ActionType.BUY_DEV_CARD
    )
    return game, index


def test_a_steal_edge_averages_over_the_cards_it_draws():
    """One frozen steal for the life of the tree was the defect. The edge now
    keeps one child per card actually stolen, the way a roll edge already kept
    one child per outcome actually rolled."""
    game, index = a_steal()
    search = Search(Stub(favour=index), simulations=96, wave=4, rng=random.Random(3))
    root, _, visits = search.run(game)

    slot = root.children[index]
    assert visits[index] == 96
    assert isinstance(slot, _Chance)
    assert sorted(slot.outcomes) == [Resource.WOOD, Resource.WHEAT]
    # Two children holding two different hands, which is what "an expectation
    # over the steal" means concretely.
    hands = {tuple(child.game.state.hands[0]) for child in slot.outcomes.values()}
    assert len(hands) == 2


def test_a_bought_card_edge_averages_over_the_deck():
    """Same defect on the other draw: the card off the deck. Ninety-six visits
    of a twenty-five card deck reach every kind in it."""
    game, index = a_purchase()
    search = Search(Stub(favour=index), simulations=96, wave=4, rng=random.Random(3))
    root, _, visits = search.run(game)

    slot = root.children[index]
    assert visits[index] == 96
    assert isinstance(slot, _Chance)
    assert sorted(slot.outcomes) == sorted(DevCard)


def test_a_robber_move_that_names_nobody_keeps_a_single_child():
    """An edge that draws nothing is not a chance edge and must not become one:
    it has one outcome, so a slot would rebuild an identical position on every
    visit and buy nothing for the engine time."""
    game, _ = a_steal()
    index = next(
        i
        for i, action in enumerate(legal_actions(game))
        if action.type is ActionType.MOVE_ROBBER and victim_of(game, action.b) is None
    )
    search = Search(Stub(favour=index), simulations=64, wave=4, rng=random.Random(3))
    root, _, visits = search.run(game)

    assert visits[index] == 64
    assert isinstance(root.children[index], Node)


def test_the_drawn_card_names_the_slot_it_is_cached_under():
    """`apply` returns nothing, so the outcome is read back off the state. A key
    that did not identify the draw would give every visit its own child, which
    is the frozen edge again with extra steps."""
    game, index = a_steal()
    search = Search(Stub(), simulations=4, wave=2, rng=random.Random(3))
    root = search._node(imagine(game, search.rng, randomize_deck=False))
    child = search._advance(root, index, None)
    stolen = _drawn(root.game, child, root.options[index])
    assert stolen in (Resource.WOOD, Resource.WHEAT)
    assert child.state.hands[0][stolen] == 1

    game, index = a_purchase()
    root = search._node(imagine(game, search.rng, randomize_deck=False))
    child = search._advance(root, index, None)
    bought = _drawn(root.game, child, root.options[index])
    assert bought in set(DevCard)
    assert child.state.new_dev_cards[0][bought] == 1


def test_two_visits_that_steal_the_same_card_share_one_child():
    """Keying on the outcome is what makes the edge an average: repeats land in
    the same subtree and accumulate, rather than each visit getting a private
    tree whose statistics nothing ever pools."""
    game, index = a_steal()
    search = Search(Stub(), simulations=4, wave=2, rng=random.Random(3))
    root = search._node(imagine(game, search.rng, randomize_deck=False))
    slot = _Chance()

    drawn = [search._sample(root, index, slot) for _ in range(40)]

    assert sorted(slot.outcomes) == [Resource.WOOD, Resource.WHEAT]
    assert {id(node) for node in drawn} == {id(node) for node in slot.outcomes.values()}


class Anchor:
    """A prior with a favourite and a value that varies with the position.

    The uniform `Stub` backs every edge up with the same number and leaves the
    visit counts flat, which would pin nothing. This one discriminates, so the
    counts below are a real fingerprint of the descent.
    """

    def evaluate(self, leaves):
        out = []
        for leaf in leaves:
            n = len(leaf.options)
            prior = np.full(n, 0.4 / max(n - 1, 1))
            prior[leaf.seat % n] = 0.6
            prior = prior / prior.sum()
            own = ((leaf.game.turns * 7 + n * 3) % 11) / 10.0 - 0.5
            value = tuple(own if s == leaf.seat else -own / 3.0 for s in range(4))
            out.append((prior, value))
        return out


class NoHiddenDraw(Search):
    """Every edge but the three that resolve a hidden card."""

    def _options(self, game):
        return tuple(a for a in super()._options(game) if a.type not in HIDDEN_DRAW)


def test_a_tree_that_draws_no_hidden_card_searches_exactly_as_it_did_before():
    """The off-path anchor, pinned to `33c6032` — the commit before chance slots
    reached the steal and the purchase.

    Rolls still resolve through `_Chance` here, so this covers the mechanism that
    was already correct alongside the deterministic edges. Both halves are
    pinned: the visit counts, and where the rng stream ended up, because a change
    that consumed a draw in a different order could reproduce one and not the
    other.
    """
    game = after_setup()
    while game.phase is not Phase.MAIN:
        apply(game, legal_actions(game)[0])

    rng = random.Random(5)
    _, _, visits = NoHiddenDraw(Anchor(), simulations=96, wave=8, rng=rng).run(game)

    assert [int(v) for v in visits] == [39, 3, 3, 2, 3, 3, 5, 9, 29]
    assert rng.random() == 0.2094563824951179


def test_a_setup_tree_searches_exactly_as_it_did_before():
    """The same anchor on the production `Search` itself rather than a subclass:
    the opening position's whole tree is deterministic placement, so `33c6032`'s
    counts stand unchanged and no rng is consumed at all."""
    rng = random.Random(5)
    _, _, visits = Search(Anchor(), simulations=96, wave=8, rng=rng).run(a_game())

    assert [int(v) for v in visits[:12]] == [67, 2, 2, 4, 3, 4, 1, 3, 1, 3, 3, 3]
    assert not visits[12:].any()
    assert rng.random() == 0.6229016948897019


def test_a_wave_is_never_larger_than_the_budget_left():
    stub = Stub()
    search = Search(stub, simulations=20, wave=8, rng=random.Random(1))
    search.run(a_game())
    assert all(len(wave) <= 8 for wave in stub.waves)
    # The budget, plus the root, which no simulation can descend without.
    assert stub.leaves <= 21


def test_the_root_is_expanded_before_anything_descends_through_it():
    """Otherwise the whole first wave lands on the unexpanded root, backs up an
    empty path, and spends that many simulations on nothing."""
    stub = Stub()
    search = Search(stub, simulations=16, wave=16, rng=random.Random(1))
    _, _, visits = search.run(a_game())
    assert len(stub.waves[0]) == 1
    assert visits.sum() == 16


def test_two_descents_that_collide_on_a_leaf_share_one_evaluation():
    """A wave wider than the branching factor must reuse edges, and virtual loss
    cannot prevent that. Both simulations still count."""

    class OneChoice(Search):
        def _options(self, game):
            return super()._options(game)[:2]

    stub = Stub()
    search = OneChoice(stub, simulations=8, wave=8, rng=random.Random(1))
    _, _, visits = search.run(a_game())
    assert visits.sum() == 8
    positions = [[id(leaf.game) for leaf in wave] for wave in stub.waves]
    assert all(len(set(wave)) == len(wave) for wave in positions)


def test_a_forced_move_is_not_searched():
    class OneWay(Search):
        def _options(self, game):
            return super()._options(game)[:1]

    stub = Stub()
    search = OneWay(stub, simulations=64, rng=random.Random(1))
    _, options, visits = search.run(a_game())
    assert len(options) == 1
    assert visits.tolist() == [1.0]
    assert stub.waves == []


def test_a_finished_game_has_nothing_to_search():
    game = a_game()
    game.phase = Phase.GAME_OVER
    stub = Stub()
    search = Search(stub, simulations=16, rng=random.Random(1))
    root, options, visits = search.run(game)
    assert options == ()
    assert visits.size == 0
    assert root.terminal and root.expanded
    assert stub.waves == []


def test_the_prior_decides_what_gets_tried_first():
    game = a_game()
    wanted = len(legal_actions(game)) - 1
    stub = Stub(favour=wanted)
    search = Search(stub, simulations=48, wave=4, rng=random.Random(1))
    _, options, visits = search.run(game)
    assert int(np.argmax(visits)) == wanted % len(options)


def test_root_noise_is_off_unless_asked_for():
    game = a_game()
    stub = Stub(favour=0)
    search = Search(stub, simulations=8, wave=4, rng=random.Random(1))
    root, _, _ = search.run(game)
    assert root.prior is not None
    assert root.prior.max() == pytest.approx(0.99)


def test_root_noise_moves_mass_off_the_priors_favourite():
    game = a_game()
    stub = Stub(favour=0)
    search = Search(
        stub,
        simulations=8,
        wave=4,
        root_noise=0.3,
        noise_fraction=0.25,
        rng=random.Random(1),
    )
    root, _, _ = search.run(game)
    assert root.prior is not None
    assert root.prior.sum() == pytest.approx(1.0)
    assert root.prior[0] < 0.99
    # Only the root is perturbed; a child expanded mid-search keeps its prior.
    child = next(c for c in root.children if isinstance(c, Node) and c.expanded)
    assert child.prior.max() == pytest.approx(0.99)


def test_virtual_loss_spreads_one_wave_over_several_edges():
    game = a_game()
    stub = Stub()
    search = Search(stub, simulations=8, wave=8, rng=random.Random(1))
    root = a_root(search, game, stub)
    picks = {search._descend(root)[0][0][1] for _ in range(4)}
    assert len(picks) == 4
    assert root.virtual.sum() == 4


def test_a_mover_reads_the_value_vector_with_its_own_stance():
    game = a_game()
    search = Search(Stub(), rng=random.Random(1))
    node = search._node(imagine(game, search.rng))
    node.prior = np.zeros(len(node.options))
    node.value = (0.0, 0.0, 0.0, 0.0)
    node.mover = 0
    node.virtual[:2] = 1.0
    search._backup([(node, 0)], [1.0, -1.0, 0.0, 0.0])
    search._backup([(node, 1)], [-1.0, 1.0, 0.0, 0.0])
    assert search._select(node) == 0

    node.visits[:2] = 0.0
    node.totals[:2] = 0.0
    node.ranked[:2] = 0.0
    node.mover = 1
    node.virtual[:2] = 1.0
    search._backup([(node, 0)], [1.0, -1.0, 0.0, 0.0])
    search._backup([(node, 1)], [-1.0, 1.0, 0.0, 0.0])
    assert search._select(node) == 1


@pytest.mark.parametrize("stance", ["own", "relative"])
def test_cached_linear_edge_scores_match_the_canonical_stance(stance):
    game = a_game()
    search = Search(Stub(), stance=stance, rng=random.Random(1))
    node = search._node(imagine(game, search.rng))
    values = ([0.4, -0.2, 0.1, -0.3], [-0.1, 0.3, -0.4, 0.2])
    for value in values:
        node.virtual[0] += 1
        search._backup([(node, 0)], value)

    expected = sum(STANCES[stance](value, node.mover) for value in values)
    assert node.ranked[0] == pytest.approx(expected)


def test_a_prior_of_the_wrong_width_is_refused():
    class Narrow(Stub):
        def evaluate(self, leaves):
            return [(np.ones(1), self.value) for _ in leaves]

    search = Search(Narrow(), simulations=4, rng=random.Random(1))
    with pytest.raises(ValueError, match="options"):
        search.run(a_game())


def test_an_evaluator_that_drops_a_leaf_is_refused():
    class Forgetful(Stub):
        def evaluate(self, leaves):
            return []

    search = Search(Forgetful(), simulations=4, rng=random.Random(1))
    with pytest.raises(ValueError, match="answered"):
        search.run(a_game())


def test_choose_returns_one_of_the_positions_legal_actions():
    game = a_game()
    search = Search(Stub(), simulations=16, wave=4, rng=random.Random(1))
    assert search.choose(game) in set(legal_actions(game))


def test_the_same_seed_searches_the_same_tree():
    left = Search(Stub(), simulations=24, wave=4, rng=random.Random(7))
    right = Search(Stub(), simulations=24, wave=4, rng=random.Random(7))
    assert left.run(a_game())[2].tolist() == right.run(a_game())[2].tolist()


def test_a_search_needs_a_known_stance_and_a_real_budget():
    with pytest.raises(ValueError, match="stance"):
        Search(Stub(), stance="hopeful")
    with pytest.raises(ValueError, match="at least one"):
        Search(Stub(), simulations=0)
    with pytest.raises(ValueError, match="at least one"):
        Search(Stub(), wave=0)


def test_visit_counts_become_a_distribution():
    assert visit_policy(np.array([3.0, 1.0])).tolist() == [0.75, 0.25]


def test_zero_temperature_is_argmax_with_ties_split():
    assert visit_policy(np.array([2.0, 2.0, 1.0]), temperature=0.0).tolist() == [
        0.5,
        0.5,
        0.0,
    ]


def test_temperature_sharpens_towards_the_most_visited():
    hot = visit_policy(np.array([3.0, 1.0]), temperature=1.0)
    cold = visit_policy(np.array([3.0, 1.0]), temperature=0.25)
    assert cold[0] > hot[0]


def test_an_unvisited_root_is_a_uniform_target_rather_than_a_division_by_zero():
    assert visit_policy(np.zeros(4)).tolist() == [0.25] * 4
    assert visit_policy(np.zeros(0)).size == 0


@pytest.mark.parametrize("stance", sorted(STANCE_ROWS))
def test_the_row_stances_agree_with_the_canonical_scalar_ones(stance):
    # `_select` reads a whole `totals` matrix rather than looping the scalar
    # stance over its rows. `relative` reassociates to get there, so this is a
    # tolerance and not an equality — see the note beside `STANCE_ROWS`.
    rng = np.random.default_rng(4)
    for seats in (2, 3, 4, 6):
        vectors = rng.normal(size=(9, seats)) * 3.0
        for seat in range(seats):
            fast = STANCE_ROWS[stance](vectors, seat)
            slow = [STANCES[stance](row, seat) for row in vectors]
            assert fast == pytest.approx(slow)


def test_a_row_stance_leaves_the_matrix_it_was_handed_alone():
    vectors = np.arange(12.0).reshape(4, 3)
    before = vectors.copy()
    for rows in STANCE_ROWS.values():
        rows(vectors, 1)
    assert (vectors == before).all()
