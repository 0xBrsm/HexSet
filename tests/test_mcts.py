from __future__ import annotations

import random

import numpy as np
import pytest

from catan.actions import Action, ActionType, legal_actions
from catan.board.board import random_base_board
from catan.game import Phase, imagine, start
from catan.bots import STANCES
from catan.mcts import STANCE_ROWS, Leaf, Node, Search, visit_policy


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
