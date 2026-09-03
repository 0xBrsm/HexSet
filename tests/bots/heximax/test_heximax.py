# SPDX-License-Identifier: GPL-3.0-only
"""`heximax`: the honest handcrafted baseline (design: heximax.md §6).

The tests here are the gate the design says must be written first. The one
that matters most is the information-set regression: two positions that the
public record cannot tell apart must draw the same move from `heximax`, and
the omniscient `search2` is shown to be able to tell them apart on at least
one such pair, which is the leak the regression guards against.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions, victim_of
from hexset.arena import PRESETS, Entrant, spawn
from hexset.board.board import make_board, random_base_board
from hexset.board.coords import Hex
from hexset.board.maps import islands
from hexset.board.terrain import NUM_RESOURCES, Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.bots import SearchBot
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase
from hexset.bots.evaluate import Evaluator, Weights
from hexset.game import Phase, imagine, is_over, roll_dice, start, to_move
from hexset.bots.heximax import (
    MODES,
    NO_TRADE_WEIGHTS,
    TRADING_WEIGHTS,
    Heximax,
    HonestEvaluator,
    View,
    heximax,
)
from hexset.bots.heximax.search import MARGINAL_SCALE
from hexset.ledger import SeatLedger
from hexset.mcts import draws_hidden
from hexset.play import step_randomly
from hexset.trading import NO_VALUATION, bundle, one_for_one, publish_valuation
from hexset.state import (
    MAX_CITIES,
    MAX_SETTLEMENTS,
    copy_state,
    new_game,
    place_settlement,
    upgrade_to_city,
)
from hexset.victory import victory_points
from helpers import clear_hand, give, independent_vertices, mini_board

# The seed the pair of indistinguishable worlds below is built at. It used
# to be the seed at which `search2`'s *chosen action* differed between them
# (found by `_first_seed_where_search2_differs`); no seed in 0..1199 does any
# more, because the leak that reached the choice was the engine's offer
# sample -- `_offer_actions` read every opponent's hand to decide which
# `want` anyone could cover -- and trading is no longer an action. What
# `search2` still does is read those hands at its own leaves, which is what
# `test_search2_can_tell_the_same_two_worlds_apart` now pins.
SEARCH2_LEAK_SEED = 2


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    board = random_base_board(rng)
    return start(board, players, rng)


def snapshot(game):
    state = game._state
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
    state = game._state
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
    state = game._state
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
    return heximax(game._state.board, random.Random(seed), **overrides)


# --- the behaviour-preservation gate -----------------------------------------

# Re-baselined for the one-event trade mechanic (the trading design's §8):
# trading games change by construction, and the no-trade games changed too,
# which the registration did not predict. The reason, established by
# stubbing `actions._offer_actions` to `[]` on the old tree and reproducing
# every hash below exactly: `heximax-notrade` suppressed *proposing* and
# *accepting*, but `Heximax._options_in` handed the engine's offer sample
# to every node of the tree regardless, so the no-trade referent still
# searched hypothetical offers. The offer sample's removal is the whole of
# the difference; nothing else in the mechanic moves a no-trade game.
#
# Regenerate deliberately, never by hand, with `pytest
# tests/bots/heximax/test_heximax.py -k choices_are_byte_identical
# --write-census` -- see `conftest.py`.
CENSUS_FIXTURE = Path(__file__).parent / "fixtures" / "heximax_census.json"

# preset -> seeds. `heximax` gets the full 20-game census the design's
# robustness sweep never runs at four seats and all three presets; the other
# two presets get 5 games each -- enough to catch a regime a change touches
# only under `notrade` or `omniscient` without tripling the wall-clock.
CENSUS_SPECS: dict[str, range] = {
    "heximax": range(100, 120),
    "heximax-notrade": range(100, 105),
    "heximax-omni": range(100, 105),
}


def _census_game(preset: str, seed: int, players: int = 4) -> str:
    """Play one seeded game, every seat `PRESETS[preset]`, and hash the choices.

    Board and game start share one rng off `seed`, exactly as `a_game` builds
    it; each seat's bot gets its own rng, deterministic per seat
    (`f"{seed}:{seat}"`), so no seat's play depends on another seat's draws or
    on iteration order. The hash covers the full `(seat, action)` trace, not
    just the outcome, because two different move sequences can reach the same
    board -- this gate is about what heximax *chooses*, not merely where it
    ends up.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    bots = [
        spawn(PRESETS[preset], board, random.Random(f"{seed}:{seat}"))
        for seat in range(players)
    ]
    game.gates = tuple(bots)
    trace = []
    moves = 0
    while not is_over(game):
        seat = to_move(game)
        cleared = len(game.trades)
        # Once a turn, when the engine says it is due (`Game.publish_due`),
        # exactly as `arena.play` does it -- the PI amendment "publish
        # points and the event trigger" -- not after every action.
        if game.publish_due(seat):
            publish_valuation(game, seat, bots[seat])
        action = bots[seat].choose(game)
        apply(game, action)
        trace.append(
            (
                seat,
                int(action.type),
                action.a,
                action.b,
                [(t.a, t.b, t.received) for t in game.trades[cleared:]],
            )
        )
        moves += 1
        if moves > 60000:
            raise AssertionError(f"{preset} seed {seed} did not finish")
    return hashlib.sha256(repr(trace).encode()).hexdigest()


def test_choices_are_byte_identical_to_the_recorded_census(request):
    """The optimization pass's gate: every change must reproduce this exactly.

    Plays the 20 (`heximax`) + 5 (`heximax-notrade`) + 5 (`heximax-omni`)
    seeded games in `CENSUS_SPECS` and hashes each game's full `(seat,
    action)` sequence. A change to `heximax` that flips even one of
    these hashes has changed what the bot chooses somewhere in that game --
    not necessarily for the worse, but it is a behaviour change, and this
    pass is only allowed to make behaviour-*preserving* ones. `--write-census`
    (see `conftest.py`) regenerates the fixture when a change is deliberate.
    """
    computed = {
        preset: {str(seed): _census_game(preset, seed) for seed in seeds}
        for preset, seeds in CENSUS_SPECS.items()
    }
    if request.config.getoption("--write-census"):
        CENSUS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        CENSUS_FIXTURE.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote {CENSUS_FIXTURE}")
    recorded = json.loads(CENSUS_FIXTURE.read_text())
    assert computed == recorded


# --- the information-set regression -----------------------------------------


def _hidden_swap(game):
    """Two opponents of the mover and a card each that the ledger has not typed.

    Moving `r1` from A to B and `r2` from B to A keeps every hand size, every
    bank count and every certified `known` entry exactly as it was, so the
    public record — and therefore the honest belief — is the same before and
    after. Only the true hidden compositions differ.
    """
    mover = to_move(game)
    seats = [s for s in range(game._state.num_players) if s != mover]
    for a in seats:
        for b in seats:
            if a == b:
                continue
            for r1 in range(NUM_RESOURCES):
                if game._state.hands[a][r1] <= game.ledger.seats[a].known[r1]:
                    continue
                for r2 in range(NUM_RESOURCES):
                    if r2 == r1:
                        continue
                    if game._state.hands[b][r2] > game.ledger.seats[b].known[r2]:
                        return a, r1, b, r2
    return None


def two_worlds_the_record_cannot_tell_apart(
    seed: int, players: int = 4, cap: int = 900, min_turn: int = 12
):
    """A mid-game main-phase position and its ledger-consistent perturbation.

    `None` when `cap` random steps never reach one: not every seed sees a
    steal early enough, which is the only way a card goes untyped.
    """
    game = a_game(seed, players)
    rng = random.Random(seed)
    for _ in range(cap):
        if game.phase is Phase.MAIN and game.turns >= min_turn:
            swap = _hidden_swap(game)
            if swap is not None:
                a, r1, b, r2 = swap
                other = imagine(game, random.Random(seed), randomize_deck=False)
                other._state.hands[a][r1] -= 1
                other._state.hands[b][r1] += 1
                other._state.hands[b][r2] -= 1
                other._state.hands[a][r2] += 1
                return game, other
        step_randomly(game, rng)
    return None


def _record_says_the_same(one, two) -> bool:
    return (
        [(s.known, s.unknown) for s in one.ledger.seats]
        == [(s.known, s.unknown) for s in two.ledger.seats]
        and one._state.bank == two._state.bank
        and [sum(h) for h in one._state.hands] == [sum(h) for h in two._state.hands]
        and one._state.hands[to_move(one)] == two._state.hands[to_move(two)]
    )


def _search2_choice(game, seed: int) -> Action:
    bot = SearchBot(
        Evaluator(game._state.board), depth=2, width=6, rng=random.Random(seed)
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


# Seeds whose random prefix reaches a swappable mid-game position with more
# than a handful of options; `two_worlds_the_record_cannot_tell_apart` returns
# None for a seed that never sees a steal in time, which is not a failure of
# the bot and is kept out of the parametrization.
WORLD_SEEDS = (2, 3, 5, 6, 7, 10, 12, 16)


@pytest.mark.parametrize("seed", WORLD_SEEDS)
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
    assert one._state.hands != two._state.hands

    for k in (1, 3):
        assert a_bot(one, seed, k=k).choose(one) == a_bot(two, seed, k=k).choose(two)


def test_search2_can_tell_the_same_two_worlds_apart():
    """The leak the regression above guards against, pinned at the leaf.

    `search2` scores a position with `evaluate.Evaluator` on the true state
    (`SearchBot._from_state`, a sanctioned true-state read by design -- it is
    the project's perfect-information referent). So on a pair of worlds the
    public record cannot tell apart, its leaf values still differ, while
    `HonestEvaluator`'s -- read through the same seat's `View` -- cannot.

    Pinned at the leaf rather than at the chosen action: the leak that used
    to reach the *choice* was the engine's offer sample (see
    `SEARCH2_LEAK_SEED`), and with trading no longer an action no seed in
    0..1199 makes `search2` play differently between these two worlds.
    """
    worlds = two_worlds_the_record_cannot_tell_apart(SEARCH2_LEAK_SEED)
    assert worlds is not None
    one, two = worlds
    assert _record_says_the_same(one, two)

    seat = to_move(one)
    cheat = Evaluator(one._state.board)
    assert cheat.evaluate(one._state, seat) != cheat.evaluate(two._state, seat)

    honest = HonestEvaluator(one._state.board)
    assert honest.evaluate_game(one, seat) == pytest.approx(
        honest.evaluate_game(two, seat)
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
        belief = View.from_game(game, seat)
        assert all(n >= 0 for n in belief.pool)
        for p in range(game._state.num_players):
            assert sum(belief.expected_hand(p)) == pytest.approx(sum(game._state.hands[p]))
        for _ in range(3):
            sampled = belief.sample(rng)
            for p in range(game._state.num_players):
                assert sum(sampled.hands[p]) == sum(game._state.hands[p])
                assert all(
                    sampled.hands[p][r] >= belief.known[p][r] for r in range(NUM_RESOURCES)
                )
                truth = game._state
                assert sum(sampled.dev_cards[p]) + sum(sampled.new_dev_cards[p]) == sum(
                    truth.dev_cards[p]
                ) + sum(truth.new_dev_cards[p])
            assert sampled.hands[seat] == game._state.hands[seat]
            assert sampled.dev_cards[seat] == game._state.dev_cards[seat]
            assert sampled.new_dev_cards[seat] == game._state.new_dev_cards[seat]
            assert len(sampled.deck) == len(game._state.deck)


def test_the_expected_hand_is_known_plus_the_pool_share():
    game = after_setup(1)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [2, 0, 0, 0, 0])
    set_known_hand(game, 1, [0, 1, 0, 0, 0])
    give_unknown(game, 1, Resource.WOOD, 1)
    give_unknown(game, 2, Resource.ORE, 1)

    belief = View.from_game(game, 0)
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

    belief = View.from_game(game, 0)
    assert belief.p_holds(1, (0, 1, 0, 0, 0)) == 1.0
    assert belief.p_holds(1, (1, 0, 0, 0, 0)) == pytest.approx(0.5)
    assert belief.p_holds(1, (0, 0, 1, 0, 0)) == 0.0
    assert belief.p_holds(1, (1, 1, 1, 0, 0)) == 0.0
    assert belief.p_holds(0, (0, 0, 0, 0, 0)) == 1.0
    estimate = belief.p_holds(1, (1, 1, 0, 0, 0), draws=200, rng=random.Random(0))
    assert 0.3 < estimate < 0.7


def test_certify_adds_a_lower_bound_the_ledger_does_not_carry():
    """`View`'s `certify` hook survives the offer protocol that motivated it:
    a caller that knows a seat holds something says so, and nothing else
    about that seat moves."""
    game = after_setup(2)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, 1, Resource.ORE, 1)
    give_unknown(game, 2, Resource.WOOD, 1)

    plain = View(game._state, game.ledger, 0)
    told = View(game._state, game.ledger, 0, certify=[(1, bundle(ore=1))])
    assert plain.known[1] == [0, 0, 0, 0, 0]
    assert told.known[1] == [0, 0, 0, 0, 1]
    assert told.unknown[1] == 0
    assert told.known[2] == plain.known[2]
    assert told.unknown[2] == plain.unknown[2] == 1
    for seed in range(5):
        assert told.sample(random.Random(seed)).hands[1][Resource.ORE] >= 1


def test_a_desynced_fixture_does_not_break_the_belief():
    """Test fixtures poke `state.hands` behind the ledger's back. The belief
    has to shrug: clamp, pad, and carry on."""
    game = after_setup(3)
    clear_hand(game._state, 1)
    give(game._state, 1, Resource.ORE, 5)
    game.ledger.seats[1] = SeatLedger(known=[3, 3, 0, 0, 0], unknown=0)
    game._state.hands[2] = [6, 6, 6, 6, 6]  # conjured from nowhere
    belief = View.from_game(game, 0)
    assert all(n >= 0 for n in belief.pool)
    assert sum(belief.expected_hand(1)) == pytest.approx(5)
    assert sum(belief.expected_hand(2)) == pytest.approx(30)
    sampled = belief.sample(random.Random(0))
    assert sum(sampled.hands[1]) == 5
    assert sum(sampled.hands[2]) == 30


def test_omniscient_belief_is_the_truth():
    game = after_setup(4)
    belief = View.from_game(game, 0, omniscient=True)
    for p in range(4):
        assert belief.expected_hand(p) == game._state.hands[p]
    sampled = belief.sample(random.Random(0))
    assert sampled.hands == game._state.hands
    assert sampled.dev_cards == game._state.dev_cards


@pytest.mark.parametrize("omniscient", [False, True])
def test_the_belief_cache_is_exact_against_from_game_on_random_tree_nodes(omniscient):
    """The structural pass's step (a): `HonestEvaluator.belief_for`/
    `belief_from_game` memoize `Belief` construction within one decision. One
    evaluator is reused across 200 positions (several different games, each
    visited at many different seats), so the cache sees a realistic mix of
    misses and hits -- and every single lookup, hit or miss, must equal a
    fresh, uncached `Belief.from_game` field-for-field: that equality is the
    whole exactness argument for keying the cache the way `belief_for`'s
    docstring describes.
    """
    board = random_base_board(random.Random(0))
    evaluator = HonestEvaluator(board, omniscient=omniscient)
    checked = 0
    for seed in range(8):
        for game in _positions(seed, 25):
            for seat in range(game._state.num_players):
                cached = evaluator.belief_from_game(game, seat)
                fresh = View.from_game(game, seat, omniscient=omniscient)
                assert cached.known == fresh.known
                assert cached.unknown == fresh.unknown
                assert cached.pool == fresh.pool
                assert cached.pool_size == fresh.pool_size
                assert cached.sizes == fresh.sizes
                assert cached.perspective == fresh.perspective == seat
                assert cached.omniscient == fresh.omniscient == omniscient
                checked += 1
    assert checked >= 200


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
    honest = HonestEvaluator(game._state.board).evaluate_game(game, 0)
    plain = Evaluator(game._state.board).evaluate(game._state, 0)
    assert honest == pytest.approx(plain)


def test_opponent_terms_read_the_expected_hand_not_the_true_one():
    game = after_setup(6)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, 1, Resource.WHEAT, 2)
    give_unknown(game, 1, Resource.ORE, 3)  # a city in hand, in truth
    give_unknown(game, 2, Resource.WOOD, 5)

    evaluator = HonestEvaluator(game._state.board)
    honest = evaluator.evaluate_game(game, 0)
    truth = Evaluator(game._state.board).evaluate(game._state, 0)
    assert honest[1] != pytest.approx(truth[1])
    # ...but the perturbation the ledger cannot see does not move it.
    game._state.hands[1], game._state.hands[2] = game._state.hands[2], game._state.hands[1]
    assert evaluator.evaluate_game(game, 0) == pytest.approx(honest)


@pytest.mark.parametrize("omniscient", [False, True])
def test_the_evaluate_memo_is_exact_against_a_fresh_computation(omniscient):
    """The structural pass's step (b): `HonestEvaluator.evaluate` memoizes its
    per-seat vector within one decision. One evaluator is reused across 200
    positions so the memo sees real hits and misses, and every lookup must
    equal what a brand-new, uncached `HonestEvaluator` computes for the same
    `(state, knower)` -- i.e. the cache changes nothing about the answer.
    """
    board = random_base_board(random.Random(1))
    evaluator = HonestEvaluator(board, omniscient=omniscient)
    checked = 0
    for seed in range(8):
        for game in _positions(seed, 25):
            for seat in range(game._state.num_players):
                belief = evaluator.belief_from_game(game, seat)
                memoized = evaluator.evaluate(game._state, seat, belief)
                fresh_evaluator = HonestEvaluator(board, omniscient=omniscient)
                fresh_belief = View.from_game(game, seat, omniscient=omniscient)
                fresh = fresh_evaluator.evaluate(game._state, seat, fresh_belief)
                assert memoized == pytest.approx(fresh)
                checked += 1
    assert checked >= 200


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
        place_settlement(game._state, 0, vertex, connected=False)
        upgrade_to_city(game._state, 0, vertex)
    place_settlement(game._state, 0, spots[3], connected=False)
    game._state.dev_cards[0][DevCard.VICTORY_POINT] += 2
    give(game._state, 0, Resource.WHEAT, 2)
    give(game._state, 0, Resource.ORE, 3)
    assert victory_points(game._state, 0) == 9
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
    # one leaf per option. The root's own option count, not `legal_actions`':
    # a heximax proposal or two may still be among them.
    options = len(a_bot(game, 13, max_nodes=100000).root_options(game))
    shallow = a_bot(game, 13, max_nodes=options + 1)
    shallow.choose(game)
    assert shallow.nodes <= options + 1
    assert shallow.depth_reached == 1
    deep = a_bot(game, 13, max_nodes=100000)
    deep.choose(game)
    assert deep.depth_reached == 2
    assert deep.nodes > shallow.nodes


def test_a_no_trade_bot_publishes_nothing_and_refuses_everything():
    """`max_trades=0` is the whole of the no-trade referent: nothing is
    advertised, so no bundle it is party to can have positive public
    surplus, and the gate refuses anyway."""
    game = a_game(seed=14)
    board = game._state.board
    quiet = heximax(board, random.Random(0), mode="notrade", max_nodes=64)
    assert quiet.max_trades == 0
    view = game.state(0)
    assert quiet.valuation(view) == NO_VALUATION
    assert quiet.accepts(view, one_for_one(0, 4), 1) is False

    talkers = [a_bot(game, s, max_nodes=64) for s in (1, 2, 3)]
    bots = [quiet, *talkers]
    game.gates = tuple(bots)
    moves = 0
    while not is_over(game) and moves < 20000:
        seat = to_move(game)
        apply(game, bots[seat].choose(game))
        publish_valuation(game, seat, bots[seat])
        moves += 1
    assert is_over(game)
    assert all(t.a != 0 and t.b != 0 for t in game.trades)


def test_a_trading_bot_publishes_a_vector_and_trades():
    game = a_game(seed=15)
    bots = [a_bot(game, s, max_nodes=200) for s in range(4)]
    game.gates = tuple(bots)
    published = bots[0].valuation(game.state(0))
    assert len(published) == NUM_RESOURCES
    assert all(-1.0 <= v <= 1.0 for v in published)

    traded = 0
    moves = 0
    while not is_over(game) and moves < 20000:
        seat = to_move(game)
        cleared = len(game.trades)
        apply(game, bots[seat].choose(game))
        publish_valuation(game, seat, bots[seat])
        traded += len(game.trades[cleared:])
        moves += 1
    assert traded > 0


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
        ActionType.SETUP_SETTLEMENT, best(game._state, 0, options)
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
        r: bot._marginal_loss(game.state(seat), r)
        for r in range(NUM_RESOURCES)
        if game._state.hands[seat][r]
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
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
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

    belief = View.from_game(game, thief)
    assert belief.expected_hand(victim) == pytest.approx([0.5, 0, 0, 0, 0.5])

    bot = a_bot(game, 22, depth=1)
    world = bot.worlds(game, thief)[0]
    children = bot.draw_children(world, action, thief)
    assert sorted(weight for weight, _ in children) == pytest.approx([0.5, 0.5])
    stolen = sorted(
        r for _, child in children for r in range(NUM_RESOURCES) if child._state.hands[thief][r]
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
        next(c for c in range(len(DevCard)) if child._state.new_dev_cards[0][c])
        for _, child in children
    }
    assert drawn == set(range(len(DevCard)))


def test_an_omniscient_bot_ignores_k_and_searches_the_truth():
    game = after_setup(24)
    bot = a_bot(game, 24, mode="omniscient", k=4)
    worlds = bot.worlds(game, to_move(game))
    assert len(worlds) == 1
    assert worlds[0]._state.hands == game._state.hands


def test_an_unknown_stance_or_mode_is_refused():
    game = a_game()
    with pytest.raises(ValueError, match="unknown stance"):
        Heximax(HonestEvaluator(game._state.board), stance="spiteful")
    with pytest.raises(ValueError, match="unknown heximax mode"):
        heximax(game._state.board, random.Random(0), mode="clairvoyant")


# --- trading (`hexset.trading`) ------------------------------------------------


def test_marginal_values_read_the_hand_they_are_given():
    game = after_setup(25)
    set_known_hand(game, 0, [0, 0, 0, 2, 2])  # one ore short of a city
    bot = a_bot(game, 25)
    view = game.state(0)
    gains = [bot._marginal_gain(view, r) for r in range(NUM_RESOURCES)]
    assert gains[Resource.ORE] == max(gains)
    assert bot._marginal_loss(view, Resource.ORE) > 0
    # Nothing held, nothing to lose.
    assert bot._marginal_loss(view, Resource.WOOD) == 0.0


def test_the_published_vector_is_the_squashed_marginal():
    game = after_setup(25)
    set_known_hand(game, 0, [0, 0, 0, 2, 2])
    bot = a_bot(game, 25)
    view = game.state(0)
    published = bot.valuation(view)
    assert len(published) == NUM_RESOURCES
    assert all(-1.0 <= v <= 1.0 for v in published)
    for r in range(NUM_RESOURCES):
        assert published[r] == pytest.approx(
            math.tanh(bot._marginal_gain(view, r) / MARGINAL_SCALE)
        )
    # Ordering is what the clearing rule reads, and `tanh` preserves it.
    assert published[Resource.ORE] == max(published)


def test_every_seat_publishes_on_one_common_scale():
    """Not a per-decision rescaling: the clearing rule compares two seats'
    surpluses, so a seat with small marginals must publish small numbers
    rather than being stretched to fill the range."""
    game = after_setup(25)
    set_known_hand(game, 0, [0, 0, 0, 2, 2])
    set_known_hand(game, 1, [1, 1, 1, 1, 1])
    bot = a_bot(game, 25)
    keen = bot.valuation(game.state(0))
    calm = bot.valuation(game.state(1))
    assert max(keen) != pytest.approx(max(calm))


def test_a_gate_read_depends_on_who_the_counterparty_is():
    game = after_setup(26)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [1, 0, 0, 0, 0])
    set_known_hand(game, 1, [0, 0, 0, 2, 3])  # a city in hand already
    set_known_hand(game, 2, [0, 0, 0, 0, 1])
    bot = a_bot(game, 26)
    view = game.state(0)
    received = one_for_one(int(Resource.WOOD), int(Resource.ORE))
    feeding_the_leader = bot._delta(view, 0, 0, received, 1, bot._rank)
    feeding_a_trailer = bot._delta(view, 0, 0, received, 2, bot._rank)
    assert feeding_the_leader != pytest.approx(feeding_a_trailer)


def test_the_gate_is_strict():
    """Ties do not clear: the engine's termination argument rests on the
    acting seat's own valuation strictly increasing at every step."""
    game = after_setup(26)
    bot = a_bot(game, 26)
    view = game.state(0)
    nothing = tuple([0] * NUM_RESOURCES)
    assert bot.accepts(view, nothing, 1) is False


def _omniscient_truth_delta(bot, game, seat, received, counterparty):
    """`_delta`'s value computed the only way an omniscient seat could
    honestly compute it: evaluate the real position, move both hands exactly,
    evaluate again. Under omniscience every hand is read verbatim, so there is
    no belief to approximate through and this is not an estimate."""
    evaluator = bot.evaluator

    def read(state, row):
        belief = View(state, game.ledger, seat, omniscient=True)
        return bot._rank(evaluator.evaluate(state, seat, belief), row)

    after = copy_state(game._state)
    for r in range(NUM_RESOURCES):
        after.hands[seat][r] += received[r]
        after.hands[counterparty][r] -= received[r]
    return read(after, seat) - read(game._state, seat)


def test_an_omniscient_trade_read_moves_the_counterpartys_real_cards():
    """An omniscient seat reads every hand verbatim, so a trade valuation must
    move the counterparty's cards *by resource*, not fold its hand into one
    total the way the honest reading may (the honest evaluator never looks at
    a non-knower's composition, so folding is invisible there -- under
    omniscience it replaces the partner's real hand with an all-one-resource
    fiction, which is not the position being priced)."""
    game = after_setup(26)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [2, 1, 0, 0, 0])
    set_known_hand(game, 1, [0, 0, 2, 1, 1])
    set_known_hand(game, 2, [1, 1, 1, 0, 0])
    bot = a_bot(game, 26, mode="omniscient")
    received = one_for_one(int(Resource.WOOD), int(Resource.SHEEP))
    view = View.from_game(game, 0, omniscient=True)
    for counterparty in (1, 2):
        assert bot._delta(view, 0, 0, received, counterparty, bot._rank) == pytest.approx(
            _omniscient_truth_delta(bot, game, 0, received, counterparty)
        )


def test_an_honest_trade_read_is_unchanged_by_the_partners_real_cards():
    """Honesty, stated as an invariant: an honest seat's gate reads the
    counterparty through `expected_hand`, which depends only on the ledger's
    `known`/`unknown` and the shared pool, so it must be identical whatever
    the counterparty's real cards are."""
    received = one_for_one(int(Resource.WOOD), int(Resource.SHEEP))
    values = []
    for one, three in (([3, 0, 0, 0, 0], [0, 0, 1, 1, 1]), ([0, 0, 1, 1, 1], [3, 0, 0, 0, 0])):
        game = after_setup(26)
        for p in range(4):
            set_known_hand(game, p, [0] * NUM_RESOURCES)
        set_known_hand(game, 0, [2, 1, 0, 0, 0])
        for seat, secret in ((1, one), (3, three)):
            for r in range(NUM_RESOURCES):
                game._state.bank[r] -= secret[r]
                game._state.hands[seat][r] = secret[r]
            game.ledger.seats[seat] = SeatLedger()
            game.ledger.seats[seat].unknown = 3
        bot = a_bot(game, 26)
        values.append(bot._delta(game.state(0), 0, 0, received, 1, bot._rank))
    assert values[0] == pytest.approx(values[1])


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
    "greedy-paranoid",
    "greedy-notrade",
    "search2-notrade",
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

    assert (honest.mode, honest.max_trades, honest.evaluator.omniscient) == ("honest", None, False)
    assert honest.evaluator.weights == TRADING_WEIGHTS
    assert (omni.mode, omni.max_trades, omni.evaluator.omniscient) == ("omniscient", None, True)
    assert (quiet.mode, quiet.max_trades, quiet.evaluator.omniscient) == ("notrade", 0, False)
    assert quiet.evaluator.weights == NO_TRADE_WEIGHTS


def test_existing_presets_are_untouched_and_still_spawn():
    assert tuple(PRESETS)[: len(EXISTING_PRESETS)] == EXISTING_PRESETS
    assert Entrant("x") == Entrant("x", mode="honest")
    assert Entrant("x").k == 1
    assert PRESETS["search2"] == Entrant("search2", kind="search", depth=2, width=6)
    assert PRESETS["heximax"] == Entrant("heximax", kind="heximax", depth=2, width=6)
    board = random_base_board(random.Random(0))
    for name in EXISTING_PRESETS:
        assert spawn(PRESETS[name], board, random.Random(0)) is not None


# --- the published scale -------------------------------------------------------

MARGINAL_SCALE_SEEDS = range(100, 105)


def _recomputed_marginal_scale() -> float:
    """`MARGINAL_SCALE`, recomputed exactly as its comment describes it: the
    mean of `|Eval(hand + one r) - Eval(hand)|` under `TRADING_WEIGHTS` and
    the `relative` stance, over every resource at every position the mover
    reaches in the trade-free census games."""
    total = 0.0
    count = 0
    for seed in MARGINAL_SCALE_SEEDS:
        rng = random.Random(seed)
        board = random_base_board(rng)
        game = start(board, 4, rng)
        bots = [
            spawn(PRESETS["heximax-notrade"], board, random.Random(f"{seed}:{seat}"))
            for seat in range(4)
        ]
        game.gates = tuple(bots)
        meter = Heximax(HonestEvaluator(board, TRADING_WEIGHTS))
        while not is_over(game):
            seat = to_move(game)
            view = game.state(seat)
            for r in range(NUM_RESOURCES):
                total += abs(meter._marginal_gain(view, r))
                count += 1
            meter.evaluator._walk_cache.clear()
            meter.evaluator._belief_cache.clear()
            meter.evaluator._evaluate_cache.clear()
            apply(game, bots[seat].choose(game))
    return total / count


def test_marginal_scale_is_the_recorded_mean():
    """The constant every seat's published vector is squashed onto is pinned
    to the computation its own comment states, not to a number someone typed.

    The games it is measured over are `heximax-notrade`'s, which never trade,
    so they cannot depend on the constant they define -- that is what makes
    this reproducible rather than circular."""
    assert _recomputed_marginal_scale() == pytest.approx(MARGINAL_SCALE, abs=1e-9)


# --- honesty of the source ------------------------------------------------------


def test_heximax_reads_the_true_state_only_where_it_says_so():
    """Every `hidden=False` in heximax's own package carries a `# true state:`
    comment saying why, and there are no other routes to the raw state: the
    field is engine-private (`Game._state`), `game.state(seat)` is the honest
    view, and `View.from_game(..., omniscient=True)` is the one omniscient
    construction, reached only from `omniscient` mode.

    heximax is a package (`hexset.bots.heximax`, split by concern into
    `evaluate`/`search`/`presets`), so the source under test is those
    modules concatenated -- the invariant is about the package, not about
    which file a line lives in.
    """
    import inspect

    import hexset.bots.heximax as heximax_pkg
    from hexset.bots.heximax import evaluate, presets, search

    lines: list[str] = []
    for module in (heximax_pkg, evaluate, search, presets):
        lines.extend(inspect.getsource(module).splitlines())

    for i, line in enumerate(lines):
        if "hidden=False" not in line or line.lstrip().startswith("#"):
            continue
        window = "\n".join(lines[max(0, i - 14) : i + 1])
        assert "true state:" in window, f"unexplained true-state read: {line.strip()}"

    source = "\n".join(lines)
    assert "._state" not in source
    assert source.count("omniscient=self.omniscient") <= 2
