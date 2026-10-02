# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

from helpers import clear_hand, give

from hexset.actions import Action, ActionType, apply, legal_actions, victim_of
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.cards import DevCard
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.bots import STANCES
from hexset.mcts import (
    HIDDEN_DRAW,
    STANCE_ROWS,
    Leaf,
    Node,
    Search,
    _Chance,
    _drawn,
    draws_hidden,
    lost_value,
    sampled_children,
    visit_policy,
)
from hexset.victory import relative_points, victory_points


def terminal_points(game) -> tuple[float, ...]:
    """A finished game's `relative_points`, board-seat order."""
    state = game.state(0, hidden=False)
    return relative_points(
        tuple(victory_points(state, seat) for seat in range(state.num_players)),
        winning_points=state.rules.winning_points,
    )


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


class Stub:

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

    def terminal(self, game):
        return self.value


def a_root(search: Search, game, stub: Stub) -> Node:
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


def test_search_randomizes_the_hidden_deck_only_when_buying_a_card():
    class CountingRandom(random.Random):
        def __init__(self):
            super().__init__(1)
            self.shuffles = 0

        def shuffle(self, values):
            self.shuffles += 1
            super().shuffle(values)

    rng = CountingRandom()
    # `hidden=False`: this pins the tree's own deck handling, and a determinized
    # root shuffles the belief pool `View.sample` deals from, which this counter
    # cannot tell from a deck shuffle.
    search = Search(Stub(), simulations=4, wave=2, hidden=False, rng=rng)
    game = a_game()
    search.run(game)
    assert rng.shuffles == 0

    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0] = [3] * len(game._state.hands[0])
    node = search._node(imagine(game, random.Random(2), randomize_deck=False))
    node.options = (Action(ActionType.BUY_DEV_CARD),)
    before = len(node.game._state.deck)
    child = search._step(node, 0, None)

    assert rng.shuffles == 1
    assert len(child.game._state.deck) == before - 1


def after_setup(seed: int = 0):
    game = a_game(seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def a_steal(seed: int = 0):
    """The victim holds two kinds of card, so a frozen first draw would show."""
    game = after_setup(seed)
    game.phase = Phase.ROBBER
    game.current_player = 0
    for player in range(game._state.num_players):
        clear_hand(game._state, player)
    give(game._state, 1, Resource.WOOD, 6)
    give(game._state, 1, Resource.WHEAT, 6)
    index = next(
        i
        for i, action in enumerate(legal_actions(game))
        if action.type is ActionType.MOVE_ROBBER and victim_of(game, action.b) == 1
    )
    return game, index


def a_purchase(seed: int = 0):
    game = after_setup(seed)
    game.phase = Phase.MAIN
    game.current_player = 0
    clear_hand(game._state, 0)
    for resource in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        give(game._state, 0, resource, 3)
    index = next(
        i
        for i, action in enumerate(legal_actions(game))
        if action.type is ActionType.BUY_DEV_CARD
    )
    return game, index


def test_a_steal_edge_averages_over_the_cards_it_draws():
    game, index = a_steal()
    # `hidden=False`: the victim's hand is the point of this test, and a
    # determinized root would redeal it from the mover's belief.
    # `wave=1`: no descent is in flight when the root selects, so virtual
    # loss sends none of the 96 elsewhere.
    search = Search(
        Stub(favour=index), simulations=96, wave=1, hidden=False, rng=random.Random(3)
    )
    root, _, visits = search.run(game)

    slot = root.children[index]
    assert visits[index] == 96
    assert isinstance(slot, _Chance)
    assert sorted(slot.outcomes) == [Resource.WOOD, Resource.WHEAT]
    hands = {tuple(child.game._state.hands[0]) for child in slot.outcomes.values()}
    assert len(hands) == 2


def test_a_bought_card_edge_averages_over_the_deck():
    """Ninety-six visits of a twenty-five card deck reach every kind in it."""
    game, index = a_purchase()
    search = Search(Stub(favour=index), simulations=96, wave=1, rng=random.Random(3))
    root, _, visits = search.run(game)

    slot = root.children[index]
    assert visits[index] == 96
    assert isinstance(slot, _Chance)
    assert sorted(slot.outcomes) == sorted(DevCard)


class Anchor:
    """Discriminating, so the pinned visit counts fingerprint the descent."""

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

    def terminal(self, game):
        return terminal_points(game)


class NoHiddenDraw(Search):

    def _options(self, game):
        return tuple(a for a in super()._options(game) if a.type not in HIDDEN_DRAW)


def test_a_tree_that_draws_no_hidden_card_searches_exactly_as_it_did_before():
    """A byte-identity anchor: visit counts and rng stream position both pinned.
    Only a change to the rules or to the search's own selection may re-pin
    them.
    """
    game = after_setup()
    while game.phase is not Phase.MAIN or len(legal_actions(game)) < 6:
        apply(game, legal_actions(game)[0])

    rng = random.Random(5)
    _, _, visits = NoHiddenDraw(
        Anchor(), simulations=96, wave=8, hidden=False, rng=rng
    ).run(game)

    assert [int(v) for v in visits] == [5, 4, 3, 72, 3, 9]
    assert rng.random() == 0.8262955117986266


def test_two_descents_that_collide_on_a_leaf_share_one_evaluation():

    class OneChoice(Search):
        def _options(self, game):
            return super()._options(game)[:2]

    stub = Stub()
    search = OneChoice(stub, simulations=8, wave=8, rng=random.Random(1))
    _, _, visits = search.run(a_game())
    assert visits.sum() == 8
    positions = [[id(leaf.game) for leaf in wave] for wave in stub.waves]
    assert all(len(set(wave)) == len(wave) for wave in positions)


class OneWay(Search):
    def _options(self, game):
        return super()._options(game)[:1]


def test_a_forced_move_is_evaluated_but_not_searched():
    """One evaluation, of the root: the value is the evaluator's and the
    visits are the whole budget, as any other root's are."""
    stub = Stub(value=(0.5, -0.5, 0.25, -0.25))
    search = OneWay(stub, simulations=64, rng=random.Random(1))
    root, options, visits = search.run(a_game())
    assert len(options) == 1
    assert visits.tolist() == [64.0]
    assert root.value == (0.5, -0.5, 0.25, -0.25)
    assert [len(wave) for wave in stub.waves] == [1]
    assert root.totals[0].tolist() == [32.0, -32.0, 16.0, -16.0]


def test_a_forced_move_carries_its_value_through_several_worlds():
    stub = Stub(value=(0.5, -0.5, 0.25, -0.25))
    game = after_setup()
    while game.phase is not Phase.MAIN:
        apply(game, legal_actions(game)[0])
    # Nobody else's cards are certified, so the mover's view holds several worlds.
    hands = game.state(0, hidden=False).hands
    for seat in range(4):
        if seat != to_move(game):
            game.ledger.seats[seat].known = [0] * 5
            game.ledger.seats[seat].unknown = sum(hands[seat])
    root, _, visits = OneWay(stub, simulations=16, k=4, rng=random.Random(3)).run(game)
    assert len(root.worlds) > 1
    assert visits.sum() == pytest.approx(16.0)
    assert root.value == pytest.approx((0.5, -0.5, 0.25, -0.25))


def test_a_finished_game_is_scored_by_the_evaluators_terminal():
    """The marked vector `evaluate` would never produce proves which supplied it."""

    class MarkedTerminal(Stub):
        def terminal(self, game):
            return (9.0, 8.0, 7.0, 6.0)

    game = a_game()
    game.phase = Phase.GAME_OVER
    stub = MarkedTerminal()
    search = Search(stub, simulations=16, rng=random.Random(1))
    root, _, _ = search.run(game)
    assert root.value == (9.0, 8.0, 7.0, 6.0)
    assert stub.waves == []


def test_the_policy_evaluator_scores_a_terminal_leaf_on_its_own_scale():
    from hexset.clients.netbot import LeafEvaluator

    evaluator = LeafEvaluator(policy=None)
    with pytest.raises(ValueError, match="has not finished"):
        evaluator.terminal(a_game())

    rng = random.Random(2)
    game = a_game(seed=2)
    moves = 0
    while not is_over(game) and moves < 20000:
        apply(game, rng.choice(legal_actions(game)))
        moves += 1
    assert is_over(game)

    scores = evaluator.terminal(game)
    assert sorted(scores) == [0.0, 0.0, 0.0, 1.0]
    assert scores[game.won_by] == 1.0
    assert tuple(scores) != tuple(terminal_points(game))


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


def test_a_lost_game_is_the_bottom_of_each_stance():
    assert lost_value("own", 4) == 0.0
    assert lost_value("relative", 4) == pytest.approx(-1 / 3)
    assert lost_value("relative", 2) == pytest.approx(-1.0)
    assert lost_value("paranoid", 4) == -1.0


@pytest.mark.parametrize("stance", sorted(STANCE_ROWS))
def test_a_descent_in_flight_discourages_another_down_the_same_edge(stance):
    """Both edges hold low win chances, so `relative` and `paranoid` read
    negative means. Counting the in-flight descent as a visit worth nothing
    would raise the first edge's mean towards zero and send the next descent
    after it. `exploration=0` leaves the means alone to decide."""
    game = a_game()
    search = Search(Stub(), stance=stance, exploration=0.0, rng=random.Random(1))
    node = search._node(imagine(game, search.rng))
    options = node.options[:2]
    node = Node(
        game=node.game, mover=0, options=options, value=(0.25,) * 4,
        prior=np.array([0.5, 0.5]), visits=np.zeros(2), virtual=np.zeros(2),
        totals=np.zeros((2, 4)), ranked=np.zeros(2), children=[None, None],
    )
    for index, chance in ((0, 0.10), (1, 0.09)):
        rest = (1.0 - chance) / 3
        search._credit(node, index, np.array([chance, rest, rest, rest]), 4)
    assert search._select(node) == 0
    node.virtual[0] = 1.0
    assert search._select(node) == 1


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


def test_the_same_seed_searches_the_same_tree():
    left = Search(Stub(), simulations=24, wave=4, rng=random.Random(7))
    right = Search(Stub(), simulations=24, wave=4, rng=random.Random(7))
    assert left.run(a_game())[2].tolist() == right.run(a_game())[2].tolist()


def test_an_unvisited_root_is_a_uniform_target_rather_than_a_division_by_zero():
    assert visit_policy(np.zeros(4)).tolist() == [0.25] * 4
    assert visit_policy(np.zeros(0)).size == 0


@pytest.mark.parametrize("stance", ["relative"])
def test_the_row_stances_agree_with_the_canonical_scalar_ones(stance):
    # `_select` reads a whole `totals` matrix rather than looping the scalar
    # stance over rows; `relative` reassociates, hence a tolerance not equality.
    rng = np.random.default_rng(4)
    for seats in (2, 3, 4, 6):
        vectors = rng.normal(size=(9, seats)) * 3.0
        for seat in range(seats):
            fast = STANCE_ROWS[stance](vectors, seat)
            slow = [STANCES[stance](row, seat) for row in vectors]
            assert fast == pytest.approx(slow)


def test_the_probes_and_the_tree_ask_one_question_about_a_hidden_draw():
    """Or the metric stops measuring what the search experiences."""
    game, _ = a_steal()
    search = Search(Stub(), simulations=1, rng=random.Random(0))
    for action in legal_actions(game):
        assert search._draws_hidden(game, action) == draws_hidden(game, action)


def test_a_chance_action_is_drawn_as_many_times_as_it_was_asked():
    game, index = a_steal()
    action = legal_actions(game)[index]
    assert draws_hidden(game, action)

    children = sampled_children(
        game, action, draws=8, rng=random.Random(5), extra=random.Random(11)
    )

    assert len(children) == 8
    assert len({_drawn(game, child, action) for child in children}) > 1


def test_a_purchase_draws_and_a_victimless_robber_move_does_not():
    game, index = a_purchase()
    action = legal_actions(game)[index]
    assert draws_hidden(game, action)
    assert len(
        sampled_children(
            game, action, draws=4, rng=random.Random(1), extra=random.Random(2)
        )
    ) == 4

    robber, _ = a_steal()
    victimless = next(
        action
        for action in legal_actions(robber)
        if action.type is ActionType.MOVE_ROBBER and victim_of(robber, action.b) is None
    )
    assert not draws_hidden(robber, victimless)


def test_the_search_resolves_a_discard_round_one_owing_seat_at_a_time():
    """The round is order-invariant, so these are positions the game can reach."""
    game = a_game(seed=7)
    game.phase = Phase.DISCARD
    game.current_player = 1
    state = game.state(0, hidden=False)
    for seat in range(4):
        clear_hand(state, seat)
    give(state, 0, Resource.WOOD, 4)
    give(state, 3, Resource.ORE, 4)
    game.discard_quota = [2, 0, 0, 2]

    search = Search(Stub(), simulations=4, wave=2, rng=random.Random(3))
    root = search._node(game)
    assert root.mover == 0
    assert {a.a for a in root.options} == {Resource.WOOD}

    after_one = search._advance(root, 0, None)
    child = search._node(after_one)
    assert child.mover == 0
    assert after_one._state.hands[3][Resource.ORE] == 4

    after_two = search._advance(child, 0, None)
    last = search._node(after_two)
    assert last.mover == 3
    assert {a.a for a in last.options} == {Resource.ORE}


def a_midgame(seed: int = 5, players: int = 4):
    """A main-phase position whose mover has several options and cannot see
    several of every opponent's cards."""
    from hexset.actions import options_for
    from hexset.play import step_randomly
    from hexset.view import View

    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(3000):
        if is_over(game):
            break
        seat = to_move(game)
        if (
            game.phase is Phase.MAIN
            and len(options_for(game)) > 3
            and sum(View.from_game(game, seat).unknown) >= 4
        ):
            return game, seat
        step_randomly(game, rng)
    raise AssertionError("no usable position")


def test_the_root_is_a_sampled_world_and_not_the_true_state():
    from hexset.view import View

    game, seat = a_midgame()
    truth = [list(hand) for hand in game.state(0, hidden=False).hands]
    drawn = []
    for seed in range(5):
        search = Search(Stub(), simulations=16, wave=4, rng=random.Random(seed))
        root, _, _ = search.run(game)
        sampled = [list(hand) for hand in root.game.state(0, hidden=False).hands]
        assert sampled[seat] == truth[seat], "the mover's own hand is not resampled"
        # Consistent with everything the mover can see, whatever was dealt.
        assert View.from_game(root.game, seat) == View.from_game(game, seat)
        assert root.worlds == ((1.0, root),)
        drawn.append(sampled)
    assert any(sampled != truth for sampled in drawn), (
        "five draws all reproduced the true hands: the root is the truth"
    )


def test_the_omniscient_root_is_reachable_only_by_asking_for_it():
    game, _ = a_midgame()
    truth = [list(hand) for hand in game.state(0, hidden=False).hands]
    search = Search(Stub(), simulations=16, wave=4, hidden=False, rng=random.Random(2))
    root, _, _ = search.run(game)
    assert [list(hand) for hand in root.game.state(0, hidden=False).hands] == truth


def test_several_worlds_each_get_the_whole_budget_and_combine_by_share():
    game, _ = a_midgame()
    search = Search(Stub(), simulations=16, wave=4, k=8, rng=random.Random(2))
    root, options, visits = search.run(game)

    assert len(root.worlds) > 1, "eight draws of this belief found one world"
    assert sum(share for share, _ in root.worlds) == pytest.approx(1.0)
    assert all(world.visits.sum() == 16 for _, world in root.worlds)
    assert visits.sum() == pytest.approx(16.0)
    assert len(options) == len(visits)
    assert root.prior.sum() == pytest.approx(1.0)
    assert not any(root.children), "a combined root holds no subtree of its own"
    assert root.game is root.worlds[0][1].game
