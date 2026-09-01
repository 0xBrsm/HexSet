# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.heximax`: the honest handcrafted baseline (design: heximax.md §6).

The tests here are the gate the design says must be written first. The one
that matters most is the information-set regression: two positions that the
public record cannot tell apart must draw the same move from `heximax`, and
the omniscient `search2` is shown to be able to tell them apart on at least
one such pair, which is the leak the regression guards against.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions, victim_of
from hexset.arena import PRESETS, Entrant, spawn
from hexset.board.board import make_board, random_base_board
from hexset.board.coords import Hex
from hexset.board.maps import islands
from hexset.board.terrain import NUM_RESOURCES, Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.bots import SearchBot, greedy
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase
from hexset.evaluate import Evaluator, Weights
from hexset.game import Phase, imagine, is_over, propose_trade, roll_dice, start, to_move
from hexset.heximax import (
    MODES,
    NO_TRADE_WEIGHTS,
    TRADING_WEIGHTS,
    Belief,
    Heximax,
    HonestEvaluator,
    heximax,
)
from hexset.ledger import SeatLedger
from hexset.mcts import draws_hidden
from hexset.play import step_randomly
from hexset.trading import bundle
from hexset.state import (
    MAX_CITIES,
    MAX_SETTLEMENTS,
    new_game,
    place_settlement,
    upgrade_to_city,
)
from hexset.victory import victory_points
from helpers import clear_hand, give, independent_vertices, mini_board

# The seed at which `search2` is pinned to read the two indistinguishable
# worlds of `test_heximax_cannot_tell_ledger_consistent_worlds_apart`
# differently. Found by `_first_seed_where_search2_differs`; pinned so the
# test documents a specific leak rather than hunting for one every run.
SEARCH2_LEAK_SEED = 3


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    board = random_base_board(rng)
    return start(board, players, rng)


def snapshot(game):
    state = game.state
    return (
        game.phase,
        game.current_player,
        game.turns,
        state.vertex_owner[:],
        state.vertex_building[:],
        state.edge_owner[:],
        state.robber,
        [hand[:] for hand in state.hands],
        state.bank[:],
        state.deck[:],
        [held[:] for held in state.dev_cards],
        game.rng.getstate(),
    )


def play_out(game, bots, cap: int = 60000) -> int:
    """Play to the end with one bot per seat (or one bot for every seat)."""
    moves = 0
    while not is_over(game):
        bot = bots[to_move(game)] if isinstance(bots, list) else bots
        apply(game, bot.choose(game))
        moves += 1
        if moves > cap:
            raise AssertionError("bots did not finish a game")
    return moves


def after_setup(seed: int = 0, players: int = 4):
    game = a_game(seed, players)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def set_known_hand(game, player: int, counts: list[int]) -> None:
    """Exactly `counts` for `player`, from the bank, with the ledger in sync."""
    state = game.state
    for r, n in enumerate(state.hands[player]):
        state.bank[r] += n
        state.hands[player][r] = 0
    game.ledger.seats[player] = SeatLedger()
    for r, n in enumerate(counts):
        if n:
            state.bank[r] -= n
            state.hands[player][r] += n
            game.ledger.receive(player, r, n)


def give_unknown(game, player: int, resource: int, n: int = 1) -> None:
    """`n` cards of `resource` from the bank, which the public record cannot type."""
    state = game.state
    state.bank[resource] -= n
    state.hands[player][resource] += n
    game.ledger.gain_unknown(player, n)


def steal_action(game, thief: int, victim: int, kind=ActionType.MOVE_ROBBER):
    game.current_player = thief
    for action in legal_actions(game):
        if action.type is kind and victim_of(game, action.b) == victim:
            return action
    raise AssertionError(f"no {kind.name} pairs {thief} -> {victim}")


def a_bot(game, seed: int = 0, **overrides) -> Heximax:
    return heximax(game.state.board, random.Random(seed), **overrides)


# --- the information-set regression -----------------------------------------


def _hidden_swap(game):
    """Two opponents of the mover and a card each that the ledger has not typed.

    Moving `r1` from A to B and `r2` from B to A keeps every hand size, every
    bank count and every certified `known` entry exactly as it was, so the
    public record — and therefore the honest belief — is the same before and
    after. Only the true hidden compositions differ.
    """
    mover = to_move(game)
    seats = [s for s in range(game.state.num_players) if s != mover]
    for a in seats:
        for b in seats:
            if a == b:
                continue
            for r1 in range(NUM_RESOURCES):
                if game.state.hands[a][r1] <= game.ledger.seats[a].known[r1]:
                    continue
                for r2 in range(NUM_RESOURCES):
                    if r2 == r1:
                        continue
                    if game.state.hands[b][r2] > game.ledger.seats[b].known[r2]:
                        return a, r1, b, r2
    return None


def two_worlds_the_record_cannot_tell_apart(seed: int, players: int = 4, cap: int = 600):
    """A mid-game main-phase position and its ledger-consistent perturbation."""
    game = a_game(seed, players)
    rng = random.Random(seed)
    for _ in range(cap):
        if game.phase is Phase.MAIN:
            swap = _hidden_swap(game)
            if swap is not None:
                a, r1, b, r2 = swap
                other = imagine(game, random.Random(seed), randomize_deck=False)
                other.state.hands[a][r1] -= 1
                other.state.hands[b][r1] += 1
                other.state.hands[b][r2] -= 1
                other.state.hands[a][r2] += 1
                return game, other
        step_randomly(game, rng)
    return None


def _record_says_the_same(one, two) -> bool:
    return (
        [(s.known, s.unknown) for s in one.ledger.seats]
        == [(s.known, s.unknown) for s in two.ledger.seats]
        and one.state.bank == two.state.bank
        and [sum(h) for h in one.state.hands] == [sum(h) for h in two.state.hands]
        and one.state.hands[to_move(one)] == two.state.hands[to_move(two)]
    )


def _search2_choice(game, seed: int) -> Action:
    bot = SearchBot(
        Evaluator(game.state.board), depth=2, width=6, rng=random.Random(seed)
    )
    return bot.choose(game)


def _first_seed_where_search2_differs(seeds=range(200)):
    for seed in seeds:
        worlds = two_worlds_the_record_cannot_tell_apart(seed)
        if worlds is None:
            continue
        one, two = worlds
        if _search2_choice(one, seed) != _search2_choice(two, seed):
            return seed
    return None


@pytest.mark.parametrize("seed", range(6))
def test_heximax_cannot_tell_ledger_consistent_worlds_apart(seed):
    """The handcrafted analogue of `test_opponent_hand_contents_do_not_leak`.

    Two positions whose public record is identical — same ledger, same bank,
    same hand sizes, same own hand — must draw the same action from an
    honest bot given the same seed, whatever the opponents actually hold.
    """
    worlds = two_worlds_the_record_cannot_tell_apart(seed)
    assert worlds is not None, f"seed {seed} never reached a swappable position"
    one, two = worlds
    assert _record_says_the_same(one, two)
    assert one.state.hands != two.state.hands

    for k in (1, 3):
        assert a_bot(one, seed, k=k).choose(one) == a_bot(two, seed, k=k).choose(two)


def test_search2_can_tell_the_same_two_worlds_apart():
    """The leak the regression above guards against, pinned to one seed.

    `search2` reads every seat's true hand, so on the pinned pair its move
    depends on which hidden cards the opponents hold.
    """
    worlds = two_worlds_the_record_cannot_tell_apart(SEARCH2_LEAK_SEED)
    assert worlds is not None
    one, two = worlds
    assert _record_says_the_same(one, two)
    assert _search2_choice(one, SEARCH2_LEAK_SEED) != _search2_choice(
        two, SEARCH2_LEAK_SEED
    )


# --- belief -------------------------------------------------------------------


def _positions(seed: int, count: int, players: int = 4, every: int = 25):
    """Up to `count` positions of one random game, `every` steps apart."""
    game = a_game(seed, players)
    rng = random.Random(seed)
    yielded = 0
    while yielded < count and not is_over(game):
        for _ in range(every):
            if is_over(game):
                break
            step_randomly(game, rng)
        if is_over(game):
            break
        yield game
        yielded += 1


@pytest.mark.parametrize("seed", range(4))
def test_a_sample_keeps_every_public_count(seed):
    rng = random.Random(seed)
    for game in _positions(seed, 30):
        seat = to_move(game)
        belief = Belief.from_game(game, seat)
        assert all(n >= 0 for n in belief.pool)
        for p in range(game.state.num_players):
            assert sum(belief.expected_hand(p)) == pytest.approx(sum(game.state.hands[p]))
        for _ in range(3):
            sampled = belief.sample(rng)
            for p in range(game.state.num_players):
                assert sum(sampled.hands[p]) == sum(game.state.hands[p])
                assert all(
                    sampled.hands[p][r] >= belief.known[p][r] for r in range(NUM_RESOURCES)
                )
                truth = game.state
                assert sum(sampled.dev_cards[p]) + sum(sampled.new_dev_cards[p]) == sum(
                    truth.dev_cards[p]
                ) + sum(truth.new_dev_cards[p])
            assert sampled.hands[seat] == game.state.hands[seat]
            assert sampled.dev_cards[seat] == game.state.dev_cards[seat]
            assert sampled.new_dev_cards[seat] == game.state.new_dev_cards[seat]
            assert len(sampled.deck) == len(game.state.deck)


def test_the_expected_hand_is_known_plus_the_pool_share():
    game = after_setup(1)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [2, 0, 0, 0, 0])
    set_known_hand(game, 1, [0, 1, 0, 0, 0])
    give_unknown(game, 1, Resource.WOOD, 1)
    give_unknown(game, 2, Resource.ORE, 1)

    belief = Belief.from_game(game, 0)
    # Two hidden cards, one wood and one ore, shared between seats 1 and 2.
    assert belief.pool == [1, 0, 0, 0, 1]
    assert belief.expected_hand(1) == pytest.approx([0.5, 1.0, 0.0, 0.0, 0.5])
    assert belief.expected_hand(2) == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.5])
    assert belief.expected_hand(0) == [2, 0, 0, 0, 0]
    assert belief.table_holding(Resource.WOOD) == pytest.approx(1.0)
    assert belief.table_holding(Resource.BRICK) == pytest.approx(1.0)


def test_p_holds_is_exact_for_one_card_and_certain_for_certified_cards():
    game = after_setup(1)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 1, [0, 1, 0, 0, 0])
    give_unknown(game, 1, Resource.WOOD, 1)
    give_unknown(game, 2, Resource.ORE, 1)

    belief = Belief.from_game(game, 0)
    assert belief.p_holds(1, (0, 1, 0, 0, 0)) == 1.0
    assert belief.p_holds(1, (1, 0, 0, 0, 0)) == pytest.approx(0.5)
    assert belief.p_holds(1, (0, 0, 1, 0, 0)) == 0.0
    assert belief.p_holds(1, (1, 1, 1, 0, 0)) == 0.0
    assert belief.p_holds(0, (0, 0, 0, 0, 0)) == 1.0
    estimate = belief.p_holds(1, (1, 1, 0, 0, 0), draws=200, rng=random.Random(0))
    assert 0.3 < estimate < 0.7


def test_an_open_offer_certifies_what_the_proposer_holds():
    """The engine only asks players who can cover the offer, and a proposer
    only offers what it holds. Both are public, so a sampled world in which
    either side could not complete the trade would be one the engine could
    never have reached."""
    game = after_setup(2)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, 1, Resource.ORE, 1)  # the proposer's hidden ore
    give_unknown(game, 2, Resource.WOOD, 1)  # the responder's hidden wood
    set_known_hand(game, 0, [1, 0, 0, 0, 0])
    game.current_player = 1
    game.phase = Phase.MAIN
    propose_trade(game, bundle(ore=1), bundle(wood=1))
    assert game.phase is Phase.TRADE_RESPOND
    responder = to_move(game)
    belief = Belief.from_game(game, responder)
    for _ in range(5):
        sampled = belief.sample(random.Random(_))
        assert sampled.hands[1][Resource.ORE] >= 1
        for pending in game.pending_responders:
            assert sampled.hands[pending][Resource.WOOD] >= 1


def test_a_desynced_fixture_does_not_break_the_belief():
    """Test fixtures poke `state.hands` behind the ledger's back. The belief
    has to shrug: clamp, pad, and carry on."""
    game = after_setup(3)
    clear_hand(game.state, 1)
    give(game.state, 1, Resource.ORE, 5)
    game.ledger.seats[1] = SeatLedger(known=[3, 3, 0, 0, 0], unknown=0)
    game.state.hands[2] = [6, 6, 6, 6, 6]  # conjured from nowhere
    belief = Belief.from_game(game, 0)
    assert all(n >= 0 for n in belief.pool)
    assert sum(belief.expected_hand(1)) == pytest.approx(5)
    assert sum(belief.expected_hand(2)) == pytest.approx(30)
    sampled = belief.sample(random.Random(0))
    assert sum(sampled.hands[1]) == 5
    assert sum(sampled.hands[2]) == 30


def test_omniscient_belief_is_the_truth():
    game = after_setup(4)
    belief = Belief.from_game(game, 0, omniscient=True)
    for p in range(4):
        assert belief.expected_hand(p) == game.state.hands[p]
    sampled = belief.sample(random.Random(0))
    assert sampled.hands == game.state.hands
    assert sampled.dev_cards == game.state.dev_cards


# --- evaluate -----------------------------------------------------------------


def test_a_seat_with_no_settlements_left_has_no_settlement_progress():
    board = mini_board()
    state = new_game(board, 2, random.Random(0))
    evaluator = HonestEvaluator(board)
    hand = list(COSTS[Purchase.SETTLEMENT])

    spots = independent_vertices(board, MAX_SETTLEMENTS)
    for vertex in spots[: MAX_SETTLEMENTS - 1]:
        place_settlement(state, 0, vertex, connected=False)
    assert evaluator.progress_toward(state, 0, hand, Purchase.SETTLEMENT) == 1.0
    assert evaluator.progress(state, 0, hand) == 1.0

    place_settlement(state, 0, spots[-1], connected=False)
    assert evaluator.progress_toward(state, 0, hand, Purchase.SETTLEMENT) == 0.0
    assert evaluator.progress(state, 0, hand) < 1.0


def test_a_seat_with_no_cities_left_has_no_city_progress():
    board = mini_board()
    state = new_game(board, 2, random.Random(0))
    evaluator = HonestEvaluator(board)
    hand = list(COSTS[Purchase.CITY])
    spots = independent_vertices(board, MAX_CITIES)
    for vertex in spots:
        place_settlement(state, 0, vertex, connected=False)
        upgrade_to_city(state, 0, vertex)
    assert evaluator.progress_toward(state, 0, hand, Purchase.CITY) == 0.0
    assert evaluator.progress_toward(state, 1, hand, Purchase.CITY) == 1.0


def test_the_honest_evaluator_agrees_with_the_evaluator_when_nothing_is_hidden():
    """With every hand certified the two must score alike, term for term: the
    honest evaluator is the same model read through the belief, not a new one."""
    game = after_setup(5)
    for p in range(4):
        set_known_hand(game, p, [p, 1, 0, 2, 1])
    honest = HonestEvaluator(game.state.board).evaluate_game(game, 0)
    plain = Evaluator(game.state.board).evaluate(game.state, 0)
    assert honest == pytest.approx(plain)


def test_opponent_terms_read_the_expected_hand_not_the_true_one():
    game = after_setup(6)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, 1, Resource.WHEAT, 2)
    give_unknown(game, 1, Resource.ORE, 3)  # a city in hand, in truth
    give_unknown(game, 2, Resource.WOOD, 5)

    evaluator = HonestEvaluator(game.state.board)
    honest = evaluator.evaluate_game(game, 0)
    truth = Evaluator(game.state.board).evaluate(game.state, 0)
    assert honest[1] != pytest.approx(truth[1])
    # ...but the perturbation the ledger cannot see does not move it.
    game.state.hands[1], game.state.hands[2] = game.state.hands[2], game.state.hands[1]
    assert evaluator.evaluate_game(game, 0) == pytest.approx(honest)


def test_the_two_weight_profiles_differ_where_trading_changed_the_fit():
    assert TRADING_WEIGHTS == Weights()
    assert NO_TRADE_WEIGHTS.production < TRADING_WEIGHTS.production
    assert NO_TRADE_WEIGHTS.progress > TRADING_WEIGHTS.progress
    assert NO_TRADE_WEIGHTS.victory_point == 1.0


# --- search -------------------------------------------------------------------


def nine_points_and_a_city_to_come():
    board = mini_board()
    game = start(board, 4, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0
    spots = independent_vertices(board, 4)
    for vertex in spots[:3]:
        place_settlement(game.state, 0, vertex, connected=False)
        upgrade_to_city(game.state, 0, vertex)
    place_settlement(game.state, 0, spots[3], connected=False)
    game.state.dev_cards[0][DevCard.VICTORY_POINT] += 2
    give(game.state, 0, Resource.WHEAT, 2)
    give(game.state, 0, Resource.ORE, 3)
    assert victory_points(game.state, 0) == 9
    return game, spots[3]


@pytest.mark.parametrize("mode", MODES)
def test_heximax_takes_a_winning_build(mode):
    game, vertex = nine_points_and_a_city_to_come()
    bot = a_bot(game, mode=mode)
    assert bot.choose(game) == Action(ActionType.BUILD_CITY, vertex)


def test_choosing_does_not_disturb_the_game_or_its_random_stream():
    game = a_game(seed=6)
    rng = random.Random(6)
    for _ in range(80):
        step_randomly(game, rng)
    before = snapshot(game)
    a_bot(game, 6, k=2).choose(game)
    assert snapshot(game) == before


def test_the_same_seed_plays_the_same_game():
    def run():
        game = a_game(seed=9)
        bot = a_bot(game, 9, k=2, max_nodes=200)
        chosen = []
        for _ in range(40):
            action = bot.choose(game)
            chosen.append(action)
            apply(game, action)
        return chosen, snapshot(game)

    assert run() == run()


def test_the_leaf_budget_is_never_exceeded():
    game = a_game(seed=11)
    rng = random.Random(11)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)
    for budget in (1, 8, 64, 300):
        bot = a_bot(game, 11, max_nodes=budget)
        seen = 0
        probe = imagine(game, random.Random(11))
        while seen < 50 and not is_over(probe):
            bot.choose(probe)
            assert bot.nodes <= budget
            seen += 1
            step_randomly(probe, rng)
        assert seen == 50


@pytest.mark.parametrize("budget", [1, 8, 64])
def test_a_game_finishes_under_any_budget(budget):
    game = a_game(seed=12, players=3)
    bots = [a_bot(game, seat, max_nodes=budget) for seat in range(3)]
    play_out(game, bots)
    assert is_over(game)


def test_deepening_stops_where_the_budget_says():
    """A budget that fits depth one but not depth two searches depth one, and
    a budget that fits both searches both — visible in the leaf count."""
    game = a_game(seed=13)
    rng = random.Random(13)
    while (
        game.phase is not Phase.MAIN
        or len(legal_actions(game)) < 4
        or any(draws_hidden(game, a) for a in legal_actions(game))
    ):
        step_randomly(game, rng)
    # Every option is a single deterministic child, so depth one costs exactly
    # one leaf per option.
    options = len(legal_actions(game))
    shallow = a_bot(game, 13, max_nodes=options + 1)
    shallow.choose(game)
    assert shallow.nodes <= options + 1
    assert shallow.depth_reached == 1
    deep = a_bot(game, 13, max_nodes=100000)
    deep.choose(game)
    assert deep.depth_reached == 2
    assert deep.nodes > shallow.nodes


def test_a_no_trade_bot_never_proposes_and_always_declines():
    game = a_game(seed=14)
    board = game.state.board
    quiet = heximax(board, random.Random(0), mode="notrade", max_nodes=64)
    assert quiet.max_offers == 0
    talkers = [greedy(Evaluator(board), random.Random(s), max_offers=3) for s in (1, 2, 3)]
    bots = [quiet, *talkers]
    asked = 0
    moves = 0
    while not is_over(game) and moves < 20000:
        seat = to_move(game)
        action = bots[seat].choose(game)
        if seat == 0:
            assert action.type is not ActionType.PROPOSE_TRADE
            assert action.type is not ActionType.ACCEPT_TRADE
            if game.phase is Phase.TRADE_RESPOND:
                asked += 1
                assert action.type is ActionType.DECLINE_TRADE
        apply(game, action)
        moves += 1
    assert is_over(game)
    assert asked > 0, "the table never asked seat 0 anything"


def test_a_trading_bot_does_propose():
    game = a_game(seed=15)
    bots = [a_bot(game, s, max_nodes=200) for s in range(4)]
    proposed = 0
    for _ in range(600):
        if is_over(game):
            break
        action = bots[to_move(game)].choose(game)
        proposed += action.type is ActionType.PROPOSE_TRADE
        apply(game, action)
    assert proposed > 0


def test_offers_stay_within_the_bots_own_budget():
    game = a_game(seed=16)
    bots = [a_bot(game, s, max_offers=2, max_nodes=200) for s in range(4)]
    peak = 0
    for _ in range(1500):
        if is_over(game):
            break
        apply(game, bots[to_move(game)].choose(game))
        peak = max(peak, game.offers_made)
    assert peak == 2


@pytest.mark.parametrize("players", [2, 3, 4])
def test_a_game_finishes_for_any_player_count(players):
    game = a_game(seed=17, players=players)
    bots = [a_bot(game, s, max_nodes=64) for s in range(players)]
    play_out(game, bots)
    assert is_over(game)


def two_islands():
    """A Seafarers-style layout: two radius-one islands sharing a coast."""
    topology = build_topology(islands(Hex(0, 0, 0), Hex(3, -3, 0), radius=1))
    producers = (
        Terrain.FOREST,
        Terrain.HILLS,
        Terrain.PASTURE,
        Terrain.FIELDS,
        Terrain.MOUNTAINS,
    )
    tokens_bag = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    terrain = []
    tokens = []
    for h in range(topology.num_hexes):
        if h == 0:
            terrain.append(Terrain.DESERT)
            tokens.append(0)
        else:
            terrain.append(producers[h % len(producers)])
            tokens.append(tokens_bag[h % len(tokens_bag)])
    return make_board(topology, tuple(terrain), tuple(tokens))


def test_a_game_finishes_on_a_two_island_board():
    board = two_islands()
    rng = random.Random(18)
    game = start(board, 3, rng)
    bots = [heximax(board, random.Random(s), max_nodes=64) for s in range(3)]
    play_out(game, bots)
    assert is_over(game)


def test_the_opening_settlement_comes_from_the_placement_prior():
    from hexset.placement import best

    game = a_game(seed=19)
    bot = a_bot(game, 19)
    options = [a.a for a in legal_actions(game)]
    assert bot.choose(game) == Action(
        ActionType.SETUP_SETTLEMENT, best(game.state, 0, options)
    )
    unprimed = a_bot(game, 19, placement=False)
    assert unprimed.choose(game).type is ActionType.SETUP_SETTLEMENT


def test_a_discard_gives_up_the_card_worth_least():
    game = after_setup(20)
    seat = 0
    set_known_hand(game, seat, [1, 1, 1, 1, 4])  # eight cards: a city and a settlement
    game.current_player = 1
    game.phase = Phase.ROLL
    roll_dice(game, 7)
    assert game.phase is Phase.DISCARD
    assert to_move(game) == seat
    bot = a_bot(game, 20)
    chosen = bot.choose(game)
    assert chosen.type is ActionType.DISCARD
    losses = {
        r: bot.marginal_loss(game, seat, r)
        for r in range(NUM_RESOURCES)
        if game.state.hands[seat][r]
    }
    assert losses[chosen.a] == min(losses.values())


def test_monopoly_names_the_resource_the_table_is_expected_to_hold_most_of():
    game = after_setup(21)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 1, [0, 0, 4, 0, 0])
    set_known_hand(game, 2, [0, 0, 3, 0, 0])
    set_known_hand(game, 3, [1, 0, 0, 0, 0])
    game.current_player = 0
    game.phase = Phase.MAIN
    game.state.dev_cards[0][DevCard.MONOPOLY] = 1
    bot = a_bot(game, 21)
    options = bot.root_options(game)
    monopolies = [a for a in options if a.type is ActionType.PLAY_MONOPOLY]
    assert monopolies == [Action(ActionType.PLAY_MONOPOLY, Resource.SHEEP)]


def test_a_steal_is_valued_as_the_expectation_over_the_victims_belief():
    """The robber's value is the probability-weighted mean over the cards the
    victim might hold, not one frozen draw."""
    game = after_setup(22)
    thief, victim, bystander = 0, 1, 2
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, victim, Resource.WOOD, 1)
    give_unknown(game, bystander, Resource.ORE, 1)
    game.phase = Phase.ROBBER
    action = steal_action(game, thief, victim)

    belief = Belief.from_game(game, thief)
    assert belief.expected_hand(victim) == pytest.approx([0.5, 0, 0, 0, 0.5])

    bot = a_bot(game, 22, depth=1)
    world = bot.worlds(game, thief)[0]
    children = bot.draw_children(world, action, thief)
    assert sorted(weight for weight, _ in children) == pytest.approx([0.5, 0.5])
    stolen = sorted(
        r for _, child in children for r in range(NUM_RESOURCES) if child.state.hands[thief][r]
    )
    assert stolen == [Resource.WOOD, Resource.ORE]

    expected = [0.0] * 4
    for weight, child in children:
        for p, v in enumerate(bot.evaluator.evaluate_game(child, thief)):
            expected[p] += weight * v
    value = bot._after(world, action, 1, thief)
    assert value == pytest.approx(expected, abs=1e-9)


def test_a_dev_card_purchase_is_valued_over_the_unseen_deck():
    game = after_setup(23)
    set_known_hand(game, 0, [0, 0, 1, 1, 1])
    game.current_player = 0
    game.phase = Phase.MAIN
    bot = a_bot(game, 23, depth=1)
    world = bot.worlds(game, 0)[0]
    children = bot.draw_children(world, Action(ActionType.BUY_DEV_CARD), 0)
    assert sum(weight for weight, _ in children) == pytest.approx(1.0)
    drawn = {
        next(c for c in range(len(DevCard)) if child.state.new_dev_cards[0][c])
        for _, child in children
    }
    assert drawn == set(range(len(DevCard)))


def test_an_omniscient_bot_ignores_k_and_searches_the_truth():
    game = after_setup(24)
    bot = a_bot(game, 24, mode="omniscient", k=4)
    worlds = bot.worlds(game, to_move(game))
    assert len(worlds) == 1
    assert worlds[0].state.hands == game.state.hands


def test_an_unknown_stance_or_mode_is_refused():
    game = a_game()
    with pytest.raises(ValueError, match="unknown stance"):
        Heximax(HonestEvaluator(game.state.board), stance="spiteful")
    with pytest.raises(ValueError, match="unknown heximax mode"):
        heximax(game.state.board, random.Random(0), mode="clairvoyant")


# --- trade valuation ------------------------------------------------------------


def test_marginal_values_read_the_hand_they_are_given():
    game = after_setup(25)
    set_known_hand(game, 0, [0, 0, 0, 2, 2])  # one ore short of a city
    bot = a_bot(game, 25)
    gains = [bot.marginal_gain(game, 0, r) for r in range(NUM_RESOURCES)]
    assert gains[Resource.ORE] == max(gains)
    assert bot.marginal_loss(game, 0, Resource.ORE) > 0
    # Nothing held, nothing to lose.
    assert bot.marginal_loss(game, 0, Resource.WOOD) == 0.0


def test_a_bundle_delta_depends_on_who_the_counterparty_is():
    game = after_setup(26)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [1, 0, 0, 0, 0])
    set_known_hand(game, 1, [0, 0, 0, 2, 3])  # a city in hand already
    set_known_hand(game, 2, [0, 0, 0, 0, 1])
    bot = a_bot(game, 26)
    give_bundle, want_bundle = (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)
    feeding_the_leader = bot.bundle_delta(game, 0, give_bundle, want_bundle, 1)
    feeding_a_trailer = bot.bundle_delta(game, 0, give_bundle, want_bundle, 2)
    assert feeding_the_leader != pytest.approx(feeding_a_trailer)


# --- presets ------------------------------------------------------------------


EXISTING_PRESETS = (
    "random",
    "greedy",
    "search2",
    "greedy-own",
    "search2-own",
    "search3",
    "greedy-tiered",
    "search2-tiered",
    "greedy-relative",
    "greedy-partner",
    "greedy-paranoid",
    "greedy-offers1",
    "greedy-offers2",
    "greedy-offers3",
    "search2-offers3",
    "search2-relative",
    "random-placement",
    "greedy-placement",
)


def test_the_heximax_presets_spawn_with_their_documented_modes():
    board = random_base_board(random.Random(0))
    honest = spawn(PRESETS["heximax"], board, random.Random(0))
    omni = spawn(PRESETS["heximax-omni"], board, random.Random(0))
    quiet = spawn(PRESETS["heximax-notrade"], board, random.Random(0))
    for bot in (honest, omni, quiet):
        assert isinstance(bot, Heximax)
        assert bot.placement
        assert bot.depth == 2 and bot.width == 6

    assert (honest.mode, honest.max_offers, honest.evaluator.omniscient) == ("honest", 3, False)
    assert honest.evaluator.weights == TRADING_WEIGHTS
    assert (omni.mode, omni.max_offers, omni.evaluator.omniscient) == ("omniscient", 3, True)
    assert (quiet.mode, quiet.max_offers, quiet.evaluator.omniscient) == ("notrade", 0, False)
    assert quiet.evaluator.weights == NO_TRADE_WEIGHTS


def test_existing_presets_are_untouched_and_still_spawn():
    assert tuple(PRESETS)[: len(EXISTING_PRESETS)] == EXISTING_PRESETS
    assert Entrant("x") == Entrant("x", mode="honest")
    board = random_base_board(random.Random(0))
    for name in EXISTING_PRESETS:
        assert spawn(PRESETS[name], board, random.Random(0)) is not None
