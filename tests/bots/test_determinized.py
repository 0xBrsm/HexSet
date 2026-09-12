# SPDX-License-Identifier: GPL-3.0-only
"""Capture-free world voting in the native engine; no model runtime required."""
import pickle
import random

import pytest

from hexset.actions import legal_actions, options_for
from hexset.arena import deal_game
from hexset.bots.determinized import determinized, holdings_signature, world_signature
from hexset.game import to_move
from hexset.state import copy_state
from hexset.view import View


@pytest.fixture
def game():
    return deal_game(96, 0, 4)


def script(monkeypatch, states):
    draws = iter(states)
    monkeypatch.setattr(View, "sample", lambda self, rng: copy_state(next(draws)))


def test_disabled_is_original_callable_and_does_not_draw(game):
    rng = random.Random(7)
    before = rng.getstate()
    choose = lambda g: legal_actions(g)[0]
    assert determinized(choose, 0, rng) is choose
    assert rng.getstate() == before


def test_duplicate_draws_vote_with_their_full_frequency(game, monkeypatch):
    state = game.state(0, hidden=False)
    a, b, c = [copy_state(state) for _ in range(3)]
    a.hands[1][0], b.hands[1][0], c.hands[1][0] = 1, 2, 3
    first, second = sorted(legal_actions(game))[:2]
    # Two distinct worlds favour first; the repeated world has more mass.
    script(monkeypatch, [a, b, c, c, c])
    calls = []

    def choose(world):
        n = world.state(0, hidden=False).hands[1][0]
        calls.append(n)
        return second if n == 3 else first

    assert determinized(choose, 5)(game) == second
    assert calls == [1, 2, 3]


def test_cache_is_per_decision_and_hypotheticals_are_safe_to_mutate(game):
    before = pickle.dumps(game)
    calls = []

    def choose(world):
        calls.append(world)
        action = legal_actions(world)[0]
        world.state(0, hidden=False).hands[0][0] += 3
        world.ledger.seats[0].known[0] += 3
        world.rng.random()
        world.locked = frozenset({1})
        return action

    pick = determinized(choose, 40, world_key=holdings_signature)
    pick(game)
    pick(game)
    pick(deal_game(97, 0, 4))
    assert len(calls) == 3  # all holdings known at setup, despite changing deck
    assert len({id(world) for world in calls}) == 3
    assert all(world is not game for world in calls)
    assert pickle.dumps(game) == before


@pytest.mark.parametrize("field", ["hands", "dev_cards", "new_dev_cards"])
def test_holdings_key_distinguishes_each_opponent_field(game, field):
    original = game.state(0, hidden=False)
    changed = copy_state(original)
    getattr(changed, field)[1][0] += 1
    assert holdings_signature(original, 0) != holdings_signature(changed, 0)


@pytest.mark.parametrize("key, expected_calls", [(world_signature, 2), (holdings_signature, 1)])
def test_deck_order_is_included_unless_explicitly_ignored(game, monkeypatch, key, expected_calls):
    a = game.state(0, hidden=False)
    b = copy_state(a)
    b.deck.reverse()
    assert a.deck != b.deck
    script(monkeypatch, [a, b, a, b])
    calls = []

    def choose(world):
        calls.append(world)
        return legal_actions(world)[0]

    determinized(choose, 4, world_key=key)(game)
    assert len(calls) == expected_calls


def test_abstentions_are_cached_and_all_abstentions_return_none(game):
    calls = []
    def choose(world):
        calls.append(world)
        return None
    assert determinized(choose, 40, world_key=holdings_signature)(game) is None
    assert len(calls) == 1


def test_abstentions_do_not_erase_valid_votes(game, monkeypatch):
    a = game.state(0, hidden=False)
    b = copy_state(a)
    b.hands[1][0] = 1
    script(monkeypatch, [a, a, b])
    action = legal_actions(game)[0]
    def choose(world):
        return action if world.state(0, hidden=False).hands[1][0] else None
    assert determinized(choose, 3)(game) == action


@pytest.mark.parametrize("temperature, weights", [(0., [0., 1.]), (1., [1/3, 2/3]), (0.5, [.2, .8]), (1e-300, [0., 1.])])
def test_sample_uses_temperature_adjusted_frequency(game, monkeypatch, temperature, weights):
    a = game.state(0, hidden=False)
    b = copy_state(a)
    b.hands[1][0] = 1
    script(monkeypatch, [b, a, b])
    actions = sorted(legal_actions(game))[:2]
    class RecordingRandom(random.Random):
        def choices(self, population, weights):
            assert population == actions
            assert list(weights) == pytest.approx(expected)
            return [population[-1]]
    expected = weights
    choose = lambda world: actions[world.state(0, hidden=False).hands[1][0]]
    assert determinized(choose, 3, RecordingRandom(0), temperature=temperature, select="sample")(game) == actions[-1]


def test_argmax_ties_use_action_order_not_world_order(game, monkeypatch):
    a = game.state(0, hidden=False)
    b = copy_state(a)
    b.hands[1][0] = 1
    script(monkeypatch, [b, a])
    actions = sorted(legal_actions(game))[:2]
    choose = lambda world: actions[world.state(0, hidden=False).hands[1][0]]
    assert determinized(choose, 2)(game) == actions[0]


def test_callback_chance_draws_do_not_change_sampled_worlds(game):
    def run(draws):
        seen = []
        def choose(world):
            seen.append(world_signature(world.state(0, hidden=False), 0))
            for _ in range(draws):
                world.rng.random()
            return legal_actions(world)[0]
        action = determinized(choose, 12, random.Random(3), select="sample")(game)
        return seen, action
    assert run(0) == run(37) == run(0)


def test_native_samples_preserve_known_cards_public_counts_and_live_state(game):
    state = game.state(0, hidden=False)
    # A valid residual pool of two resources, uncertified in the opponents'
    # ledger, plus one development card drawn by an opponent and one by us.
    state.hands[1][0] = state.hands[2][1] = state.hands[0][2] = 1
    for resource in range(3):
        state.bank[resource] -= 1
    state.dev_cards[1][state.deck.pop()] += 1
    state.new_dev_cards[0][state.deck.pop()] += 1
    before = pickle.dumps(game)
    samples = []
    def choose(world):
        sampled = world.state(0, hidden=False)
        samples.append(sampled)
        assert [sum(h) for h in sampled.hands] == [sum(h) for h in state.hands]
        assert sampled.hands[0] == state.hands[0]
        assert sampled.dev_cards[0] == state.dev_cards[0]
        assert sampled.new_dev_cards[0] == state.new_dev_cards[0]
        assert [sum(a) + sum(b) for a, b in zip(sampled.dev_cards, sampled.new_dev_cards)] == [
            sum(a) + sum(b) for a, b in zip(state.dev_cards, state.new_dev_cards)]
        assert len(sampled.deck) == len(state.deck)
        assert sampled.bank == state.bank
        assert sampled.vertex_owner == state.vertex_owner
        assert sampled.edge_owner == state.edge_owner
        for resource in range(5):
            assert sum(h[resource] for h in sampled.hands) + sampled.bank[resource] == 19
        return legal_actions(world)[0]
    determinized(choose, 12, random.Random(3))(game)
    assert len(samples) > 1
    assert pickle.dumps(game) == before


def test_runtime_neutral_policy_callback_needs_only_one_forward_for_known_holdings(game):
    class Policy:
        calls = 0
        def act_rows(self, rows):
            self.calls += 1
            return [options[0] for _, _, options in rows]
    policy = Policy()
    def choose(world):
        return policy.act_rows([(world, to_move(world), tuple(options_for(world)))])[0]
    action = determinized(choose, 40, world_key=holdings_signature)(game)
    assert action in legal_actions(game)
    assert policy.calls == 1


@pytest.mark.parametrize("kwargs", [
    {"worlds": -1}, {"select": "greedy"}, {"temperature": -1},
    {"temperature": float("nan")}, {"temperature": float("inf")},
])
def test_invalid_configuration_is_rejected_even_if_disabled(kwargs):
    with pytest.raises(ValueError):
        determinized(lambda game: None, **({"worlds": 0} | kwargs))


def test_noninteger_draw_count_is_rejected():
    with pytest.raises(TypeError):
        determinized(lambda game: None, 1.5)



def test_distinct_worlds_folds_the_draws_and_weights_them():
    """The one sampler both `determinized` and `Heximax.worlds` read: a pinned
    belief is one world with weight 1.0, and shares always sum to one."""
    import random

    from hexset.board.board import random_base_board
    from hexset.bots.determinized import distinct_worlds, holdings_signature
    from hexset.game import is_over, start, to_move
    from hexset.play import step_randomly
    from hexset.view import View

    rng = random.Random(5)
    board = random_base_board(rng)
    game = start(board, 2, rng)
    for _ in range(60):
        if is_over(game):
            break
        step_randomly(game, rng)
    seat = to_move(game)
    assert sum(View.from_game(game, seat).unknown) == 0
    worlds = distinct_worlds(game.state(seat), random.Random(0), 100, holdings_signature)
    assert len(worlds) == 1 and worlds[0][0] == 1.0

    rng = random.Random(11)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    for step in range(600):
        if is_over(game):
            break
        seat = to_move(game)
        if step > 40 and sum(View.from_game(game, seat).unknown) >= 3:
            break
        step_randomly(game, rng)
    worlds = distinct_worlds(game.state(seat), random.Random(0), 50, holdings_signature)
    assert 1 <= len(worlds) <= 50 and abs(sum(w for w, _ in worlds) - 1.0) < 1e-9
    keys = [holdings_signature(state, seat) for _, state in worlds]
    assert len(set(keys)) == len(keys)
