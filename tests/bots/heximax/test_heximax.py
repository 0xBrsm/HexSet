# SPDX-License-Identifier: GPL-3.0-only
"""`heximax`: the honest handcrafted baseline (design: heximax.md §6).

The tests here are the gate the design says must be written first. The one
that matters most is the information-set regression: two positions that the
public record cannot tell apart must draw the same move from `heximax`, and
the omniscient `search2` is shown to be able to tell them apart on at least
one such pair, which is the leak the regression guards against.
"""

from __future__ import annotations

import functools
import hashlib
import json
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
from hexset.bots import SearchBot, greedy
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase
from hexset.evaluate import Evaluator, Weights
from hexset.game import Phase, imagine, is_over, propose_trade, roll_dice, start, to_move
from heximax import (
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
from hexset.trading import Offer, bundle, can_propose, well_formed
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

# The seed at which `search2` is pinned to read the two indistinguishable
# worlds of `test_heximax_cannot_tell_ledger_consistent_worlds_apart`
# differently. Found by `_first_seed_where_search2_differs`; pinned so the
# test documents a specific leak rather than hunting for one every run.
SEARCH2_LEAK_SEED = 2


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


# --- the behaviour-preservation gate -----------------------------------------

# The recorded census is keyed to the tree at `ecb5252` (HEAD of
# `feat/heximax-p2` before this pass's optimizations). Regenerate it
# deliberately, never by hand, with `pytest tests/test_heximax.py -k
# choices_are_byte_identical --write-census` -- see `conftest.py`.
CENSUS_FIXTURE = Path(__file__).parent / "fixtures" / "heximax_census_ecb5252.json"

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
    trace = []
    moves = 0
    while not is_over(game):
        seat = to_move(game)
        action = bots[seat].choose(game)
        trace.append(
            (
                seat,
                int(action.type),
                action.a,
                action.b,
                list(action.give),
                list(action.want),
                list(action.ask),
            )
        )
        apply(game, action)
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


def a_response_with_another_seat_pending(seed: int, swapped: bool):
    """heximax (seat 0) is asked first; exactly one other seat is also pending.

    Seats 2 and 3 each hold one card the record cannot type -- a wood and a
    sheep between them. `swapped` says which holds which. The bank, every
    hand size and every `known` entry are identical either way, so the two
    positions are one information set for seat 0. The engine's
    `pending_responders`, built from the true hands, differ: only the seat
    that truly holds the wanted wood is pending.
    """
    game = after_setup(seed)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [1, 0, 0, 0, 0])
    set_known_hand(game, 1, [0, 0, 0, 0, 1])
    wood_holder, sheep_holder = (3, 2) if swapped else (2, 3)
    give_unknown(game, wood_holder, Resource.WOOD, 1)
    give_unknown(game, sheep_holder, Resource.SHEEP, 1)
    game.current_player = 1
    game.phase = Phase.MAIN
    propose_trade(game, bundle(ore=1), bundle(wood=1), ask=(0,))
    assert game.phase is Phase.TRADE_RESPOND
    assert to_move(game) == 0
    assert len(game.pending_responders) == 2
    return game


@pytest.mark.parametrize("seed", range(4))
def test_who_else_could_cover_an_offer_is_hidden_from_the_responder(seed):
    """`pending_responders` is the engine's true eligibility list. Under the
    rules a decline reveals nothing, so from the responder's seat the other
    pending seats' coverage is hidden, and the belief may not read it."""
    one = a_response_with_another_seat_pending(seed, swapped=False)
    two = a_response_with_another_seat_pending(seed, swapped=True)
    assert _record_says_the_same(one, two)
    assert one.pending_responders != two.pending_responders

    for seat in (2, 3):
        assert Belief.from_game(one, 0).expected_hand(seat) == pytest.approx(
            Belief.from_game(two, 0).expected_hand(seat)
        )
    for k in (1, 4):
        assert a_bot(one, seed, k=k).choose(one) == a_bot(two, seed, k=k).choose(two)


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


def test_an_open_offer_certifies_the_proposers_side_and_nothing_else():
    """A proposer only offers what it holds, and the offer is announced, so
    `give` is public. Who else can cover `want` is not: the engine's
    `pending_responders` is built from the true hands, and a decline reveals
    nothing, so the belief about the other pending seats must be the plain
    ledger reading."""
    game = after_setup(2)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    give_unknown(game, 1, Resource.ORE, 1)  # the proposer's hidden ore
    give_unknown(game, 2, Resource.WOOD, 1)  # another responder's hidden wood
    set_known_hand(game, 0, [1, 0, 0, 0, 0])
    game.current_player = 1
    game.phase = Phase.MAIN
    propose_trade(game, bundle(ore=1), bundle(wood=1), ask=(0,))
    assert game.phase is Phase.TRADE_RESPOND
    assert to_move(game) == 0
    assert len(game.pending_responders) == 2

    belief = Belief.from_game(game, 0)
    assert belief.known[1] == [0, 0, 0, 0, 1]  # `give` certified
    other = next(p for p in game.pending_responders if p != 0)
    assert belief.known[other] == game.ledger.seats[other].known  # `want` not
    assert belief.unknown[other] == game.ledger.seats[other].unknown == 1
    # ...so the reading of `other` is what the ledger plus the announced
    # `give` alone support: the same belief built with only that certified.
    plain = Belief(game.state, game.ledger, 0, certify=[(1, bundle(ore=1))])
    assert belief.known == plain.known and belief.unknown == plain.unknown
    assert belief.expected_hand(other) == pytest.approx(plain.expected_hand(other))
    assert belief.p_holds(other, bundle(wood=1)) == plain.p_holds(other, bundle(wood=1))
    for seed in range(5):
        sampled = belief.sample(random.Random(seed))
        assert sampled.hands[1][Resource.ORE] >= 1


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
            for seat in range(game.state.num_players):
                cached = evaluator.belief_from_game(game, seat)
                fresh = Belief.from_game(game, seat, omniscient=omniscient)
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
            for seat in range(game.state.num_players):
                belief = evaluator.belief_from_game(game, seat)
                memoized = evaluator.evaluate(game.state, seat, belief)
                fresh_evaluator = HonestEvaluator(board, omniscient=omniscient)
                fresh_belief = Belief.from_game(game, seat, omniscient=omniscient)
                fresh = fresh_evaluator.evaluate(game.state, seat, fresh_belief)
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
        # At least 4 non-trade options: P2's adapter replaces every
        # `PROPOSE_TRADE` in `legal_actions` with its own (possibly empty)
        # candidates, so a position that only clears 4 by counting the
        # engine's one-for-one trade sample can collapse to a single
        # option (END_TURN) once that sample is gone.
        or len([a for a in legal_actions(game) if a.type is not ActionType.PROPOSE_TRADE]) < 4
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


def _omniscient_truth_delta(bot, game, seat, give, want, counterparty):
    """`bundle_delta`'s value computed the only way an omniscient seat could
    honestly compute it: evaluate the real position, move both hands exactly,
    evaluate again. Under omniscience every hand is read verbatim, so there is
    no belief to approximate through and this is not an estimate."""
    evaluator = bot.evaluator

    def read(state):
        belief = Belief(state, game.ledger, seat, omniscient=True)
        return bot._rank(evaluator.evaluate(state, seat, belief), seat)

    after = copy_state(game.state)
    for r in range(NUM_RESOURCES):
        after.hands[seat][r] += want[r] - give[r]
        after.hands[counterparty][r] += give[r] - want[r]
    return read(after) - read(game.state)


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
    give, want = (1, 0, 0, 0, 0), (0, 0, 1, 0, 0)
    for counterparty in (1, 2):
        assert bot.bundle_delta(game, 0, give, want, counterparty) == pytest.approx(
            _omniscient_truth_delta(bot, game, 0, give, want, counterparty)
        )


def test_an_omniscient_partner_read_moves_the_partners_real_cards():
    """The same, read from a row that is not the knower's own -- what
    `score_proposal`'s `willing` gate and `rank_partners` ask: would this trade
    help *them*? Folding the partner's hand into one total makes every trade
    look ruinous for the partner, so an omniscient bot's `willing` gate never
    fires and it stops proposing."""
    game = after_setup(26)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    set_known_hand(game, 0, [2, 1, 0, 0, 0])
    set_known_hand(game, 1, [0, 0, 2, 1, 1])
    bot = a_bot(game, 26, mode="omniscient")
    give, want = (1, 0, 0, 0, 0), (0, 0, 1, 0, 0)
    # seat 1 gives `want` and receives `give` from seat 0; read from seat 1's
    # row, through omniscient seat 0's (perfect) information.
    theirs = bot._partner_delta(game, 0, 1, want, give, 0, bot._rank)
    evaluator = bot.evaluator

    def read(state):
        belief = Belief(state, game.ledger, 0, omniscient=True)
        return bot._rank(evaluator.evaluate(state, 0, belief), 1)

    after = copy_state(game.state)
    for r in range(NUM_RESOURCES):
        after.hands[1][r] += give[r] - want[r]
        after.hands[0][r] += want[r] - give[r]
    assert theirs == pytest.approx(read(after) - read(game.state))


def test_an_honest_trade_read_is_unchanged_by_the_exact_partner_move():
    """The counterpart guarantee: honesty is not weakened by the fix. An honest
    seat's `bundle_delta` reads the counterparty through `expected_hand`, which
    depends only on the ledger's `known`/`unknown` and the shared pool, so it
    must be identical whatever the counterparty's real cards are."""
    give, want = (1, 0, 0, 0, 0), (0, 0, 1, 0, 0)
    # Seats 1 and 3 each hold three untyped cards; swapping their real
    # compositions leaves the bank, both hand sizes and both ledger rows
    # identical, so the public record cannot tell the two worlds apart.
    values = []
    for one, three in (([3, 0, 0, 0, 0], [0, 0, 1, 1, 1]), ([0, 0, 1, 1, 1], [3, 0, 0, 0, 0])):
        game = after_setup(26)
        for p in range(4):
            set_known_hand(game, p, [0] * NUM_RESOURCES)
        set_known_hand(game, 0, [2, 1, 0, 0, 0])
        for seat, secret in ((1, one), (3, three)):
            for r in range(NUM_RESOURCES):
                game.state.bank[r] -= secret[r]
                game.state.hands[seat][r] = secret[r]
            game.ledger.seats[seat] = SeatLedger()
            game.ledger.seats[seat].unknown = 3
        bot = a_bot(game, 26)
        values.append(bot.bundle_delta(game, 0, give, want, 1))
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
    assert Entrant("x").k == 1
    assert PRESETS["search2"] == Entrant("search2", kind="search", depth=2, width=6)
    assert PRESETS["heximax"] == Entrant("heximax", kind="heximax", depth=2, width=6, max_offers=3)
    board = random_base_board(random.Random(0))
    for name in EXISTING_PRESETS:
        assert spawn(PRESETS[name], board, random.Random(0)) is not None


def test_entrant_margin_knobs_default_unchanged_and_reach_the_spawned_bot():
    """`Entrant.accept_margin`/`propose_margin` (P3's margin-grid harness gap)
    default to `heximax()`'s own defaults, so every existing preset spawns a
    bot byte-identical to before this field existed -- `PRESETS["heximax"] ==
    Entrant("heximax", ...)` above already pins that by equality. A
    non-default value on the `Entrant` reaches the built `Heximax`, which is
    all `_spawn`'s `kind == "heximax"` branch is responsible for; nothing
    reads these fields for any other `kind`.
    """
    assert Entrant("x", kind="heximax").accept_margin == 0.0
    assert Entrant("x", kind="heximax").propose_margin == 0.0
    board = random_base_board(random.Random(0))
    bot = spawn(replace(PRESETS["heximax"], accept_margin=0.2), board, random.Random(0))
    assert bot.accept_margin == 0.2
    assert bot.propose_margin == 0.0
    bot = spawn(replace(PRESETS["heximax"], propose_margin=0.4), board, random.Random(0))
    assert bot.propose_margin == 0.4
    assert bot.accept_margin == 0.0


# --- trade adapter (P2) --------------------------------------------------------

CENSUS_GAMES = 20
CENSUS_MAX_NODES = 64


def _play_and_census(seed: int):
    """Play one seeded four-`heximax`-seat game to completion, checking every
    `PROPOSE_TRADE` heximax emits, as it is chosen, for legality and shape.

    Returns `(multi, one_for_one, accepted)` for this one game.
    """
    game = a_game(seed)
    bots = [
        heximax(game.state.board, random.Random(seed * 97 + s), max_nodes=CENSUS_MAX_NODES)
        for s in range(4)
    ]
    multi = one_for_one = accepted = 0
    moves = 0
    while not is_over(game) and moves < 8000:
        seat = to_move(game)
        action = bots[seat].choose(game)
        if action.type is ActionType.PROPOSE_TRADE:
            offer = Offer(proposer=seat, give=action.give, want=action.want)
            assert well_formed(offer)
            assert can_propose(game.state, offer)
            assert not any(g and w for g, w in zip(action.give, action.want))
            assert sum(action.give) <= 2 and sum(action.want) <= 2
            if sum(action.give) > 1 or sum(action.want) > 1:
                multi += 1
            else:
                one_for_one += 1
        elif action.type is ActionType.ACCEPT_TRADE:
            accepted += 1
        apply(game, action)
        moves += 1
    assert is_over(game), f"seed {seed} did not finish in {moves} moves"
    return multi, one_for_one, accepted


@functools.lru_cache(maxsize=1)
def _trade_census():
    """The 20-game census, run once and cached: both gate tests below read
    the same run rather than paying for it twice."""
    per_game = tuple(_play_and_census(seed) for seed in range(CENSUS_GAMES))
    totals = tuple(sum(column) for column in zip(*per_game))
    return per_game, totals


def test_every_proposal_is_legal_and_shaped_right_over_twenty_games():
    """Every `PROPOSE_TRADE` heximax emits, over 20 seeded four-`heximax`-seat
    games (`CENSUS_MAX_NODES = 64`), is `well_formed`, passes `can_propose`
    against the state at the moment it was chosen, never repeats a resource
    on both sides, and never exceeds two cards a side -- checked inline, as
    each proposal is chosen, inside `_play_and_census`.
    """
    _trade_census()


def test_multi_card_and_one_for_one_proposals_both_occur_over_twenty_games():
    """Measured on this run (seeds 0-19, `CENSUS_MAX_NODES = 64`, four
    `heximax` seats, `functools.lru_cache`d with the legality test above so
    the games are only played once): 32 multi-card proposals against 112
    one-for-one ones -- a 22.2% multi-card share of 144 total -- and 65
    accepted trades across the 20 games, 3.25 a game. Both shapes occur in
    every run this was measured on; the exact counts drift with any change
    to the valuation, the search, or `CENSUS_MAX_NODES`, but the qualitative
    claim this test gates -- neither shape disappears -- does not. The
    volume itself is low relative to the engine's naive one-for-one sample
    it replaced: `score_proposal`'s crisp `willing` gate only credits an
    opponent when accepting is a genuine gain for them under `relative`,
    which a table that also improves the proposer's own position often
    fails on the same trade (see `heximax.py`'s module docstring, "Cost").

    Review F: those exact counts lived only in this docstring, not in an
    assertion, so a real regression in trade shape (say, the multi-card
    share dropping to near zero, or every proposal turning multi-card)
    would not have failed this test. Bounds below are deliberately wide --
    several times the measured values in either direction -- so a future,
    intentional tuning change (a different weight profile, a different
    `propose_margin`) does not turn this into a brittle pin, but a real
    collapse of one shape or a wildly skewed share does trip it.
    """
    _per_game, (total_multi, total_one_for_one, total_accepted) = _trade_census()
    assert total_multi > 0, "no multi-card proposal was ever emitted"
    assert total_one_for_one > 0, "one-for-one proposals stopped occurring"
    assert total_accepted > 0
    share = total_multi / (total_multi + total_one_for_one)
    assert 0.05 <= share <= 0.60, f"multi-card share {share:.3f} outside the expected band"
    accepted_per_game = total_accepted / CENSUS_GAMES
    assert 1.0 <= accepted_per_game <= 8.0, (
        f"{accepted_per_game:.2f} accepted trades/game outside the expected band"
    )


def test_accept_rule_takes_a_clearly_positive_offer():
    """A city-completing ore for a wood I have no use for: clearly worth it."""
    game = after_setup(30)
    for p in range(4):
        set_known_hand(game, p, list(bundle()))
    set_known_hand(game, 0, list(bundle(wood=1, ore=2, wheat=2)))  # 1 ore short of a city
    set_known_hand(game, 1, list(bundle(ore=3)))
    game.current_player = 1
    game.phase = Phase.MAIN
    bot = a_bot(game, 30)
    offer = Offer(proposer=1, give=bundle(ore=1), want=bundle(wood=1))
    assert bot.accept_rule(game, 0, offer, 0.0)


def test_accept_rule_declines_a_clearly_negative_offer():
    """Giving up the ore that completes a city outright, for a useless sheep."""
    game = after_setup(31)
    for p in range(4):
        set_known_hand(game, p, list(bundle()))
    set_known_hand(game, 0, list(bundle(ore=3, wheat=2)))  # a city ready to build
    set_known_hand(game, 1, list(bundle(sheep=1)))
    game.current_player = 1
    game.phase = Phase.MAIN
    bot = a_bot(game, 31)
    offer = Offer(proposer=1, give=bundle(sheep=1), want=bundle(ore=1))
    assert not bot.accept_rule(game, 0, offer, 0.0)


def test_accept_rule_declines_a_leader_denial_case():
    """Helps me a little, helps the runaway leader a lot -- declined under
    `relative`. Seat 1 already has three cities and is one ore short of a
    fourth; I hold that ore and would receive a wood I have some use for
    (there is no other seat mid-purchase to route it to). The same trade
    read with a trailing seat as the counterparty (no cities, nothing
    in progress) costs the leader-denial term nothing extra, and is read
    as less negative -- the counterparty-dependence `bundle_delta`'s own
    docstring promises under `relative`."""
    board = mini_board()
    spots = independent_vertices(board, 4)

    def a_table(proposer: int):
        game = start(board, 4, random.Random(0))
        game.phase = Phase.MAIN
        game.current_player = proposer
        for v in spots[:3]:
            place_settlement(game.state, 1, v, connected=False)
            upgrade_to_city(game.state, 1, v)
        for p in range(4):
            set_known_hand(game, p, list(bundle()))
        set_known_hand(game, 1, list(bundle(wood=1, ore=2, wheat=2)))  # 1 ore short of a 4th city
        set_known_hand(game, 2, list(bundle(wood=1)))
        set_known_hand(game, 0, list(bundle(ore=1)))
        return game

    leader_game = a_table(proposer=1)
    trailing_game = a_table(proposer=2)
    leader_bot = a_bot(leader_game, 32)
    trailing_bot = a_bot(trailing_game, 32)
    leader_offer = Offer(proposer=1, give=bundle(wood=1), want=bundle(ore=1))
    trailing_offer = Offer(proposer=2, give=bundle(wood=1), want=bundle(ore=1))

    leader_delta = leader_bot.bundle_delta(
        leader_game, 0, leader_offer.want, leader_offer.give, leader_offer.proposer
    )
    trailing_delta = trailing_bot.bundle_delta(
        trailing_game, 0, trailing_offer.want, trailing_offer.give, trailing_offer.proposer
    )
    assert leader_delta < trailing_delta
    assert not leader_bot.accept_rule(leader_game, 0, leader_offer, 0.0)


def test_rank_partners_asks_first_whoever_it_helps_least():
    """Seat 1 is one ore short of a city (giving up ore costs it dearly);
    seat 2 is stacked five ore deep (giving up one costs it almost
    nothing). Both truly hold ore, so both are eligible; seat 1, whom the
    trade helps least (hurts most), is asked first."""

    def a_table(swapped: bool):
        game = after_setup(40, players=3)
        for p in range(3):
            set_known_hand(game, p, list(bundle()))
        set_known_hand(game, 0, list(bundle(wood=2)))
        set_known_hand(game, 1, list(bundle(ore=2, wheat=2)))
        set_known_hand(game, 2, list(bundle(ore=5)))
        game.current_player = 0
        game.phase = Phase.MAIN
        # One untyped card each for seats 1 and 2 -- which true resource it
        # is differs, but the ledger (known/unknown) is identical either way.
        a, b = (Resource.SHEEP, Resource.BRICK) if not swapped else (Resource.BRICK, Resource.SHEEP)
        give_unknown(game, 1, a, 1)
        give_unknown(game, 2, b, 1)
        return game

    give, want = bundle(wood=1), bundle(ore=1)
    one = a_table(swapped=False)
    two = a_table(swapped=True)
    assert _record_says_the_same(one, two)
    bot_one = a_bot(one, 40)
    bot_two = a_bot(two, 40)
    order = bot_one.rank_partners(one, 0, give, want)
    assert order == (1, 2)
    assert bot_two.rank_partners(two, 0, give, want) == order


def test_counter_of_returns_a_valid_bundle_from_my_surplus_or_none():
    game = after_setup(33)
    for p in range(4):
        set_known_hand(game, p, list(bundle()))
    set_known_hand(game, 0, list(bundle(wood=3, brick=2)))
    game.current_player = 1
    game.phase = Phase.MAIN
    bot = a_bot(game, 33)
    standing = Offer(proposer=1, give=bundle(ore=1), want=bundle(wheat=2))

    result = bot.counter_of(game, 0, standing)
    assert result is not None
    give, want = result
    assert result in bot.candidate_bundles(game, 0)
    counter = Offer(proposer=0, give=give, want=want)
    assert well_formed(counter)
    assert can_propose(game.state, counter)

    clear_hand(game.state, 0)
    assert bot.counter_of(game, 0, standing) is None


def test_proposals_are_among_the_root_options_at_a_main_phase_position():
    """A real four-`heximax`-seat game (seed 0, `max_nodes=64` -- the census
    configuration) reaches a MAIN-phase decision where the search itself
    chooses a MULTI-CARD `PROPOSE_TRADE`: proof the adapter's candidates are
    actually among the root options during play, not merely present unused.
    `choose` can only return an action `root_options` offered -- most
    MAIN-phase positions score every candidate at or below `propose_margin`
    (the crisp `willing` gate is strict), so this walks a real game to the
    first one that does not, rather than asserting it of an arbitrary
    position.

    Review F: the earlier version of this test only checked for *any*
    `PROPOSE_TRADE`, which P1's engine-sample one-for-one offer (already
    present in `legal_actions`, `_root_options`'s "seen" union before P2's
    filter/replace) would also satisfy -- so it passed unchanged with the P2
    adapter reverted to P1's shape, proving nothing about the adapter
    specifically. A one-for-one offer is not proof of anything P2-specific;
    a *multi-card* one is, since P1's engine sample -- and the P1-shaped
    `_root_options` this test would see with the adapter disabled -- can
    never produce one (`trading-design.md` §1, "one-for-one offers only").
    Verified by temporarily reverting `_root_options` to the P1 shape
    (dropping the `propose_actions` extension) and re-running this test: it
    fails with "heximax never chose a multi-card trade in this game", exactly
    because no multi-card candidate ever reaches `root_options` without the
    adapter. Re-enabled before this commit.
    """
    game = a_game(0)
    bots = [heximax(game.state.board, random.Random(97 * s), max_nodes=64) for s in range(4)]
    found = False
    moves = 0
    while not is_over(game) and moves < 2000:
        seat = to_move(game)
        action = bots[seat].choose(game)
        if action.type is ActionType.PROPOSE_TRADE and (
            sum(action.give) > 1 or sum(action.want) > 1
        ):
            found = True
            break
        apply(game, action)
        moves += 1
    assert found, "heximax never chose a multi-card trade in this game"


def test_heximax_source_never_calls_responders_or_reads_pending_responders():
    """`game.pending_responders` is the engine's true-hand eligibility list
    and `trading.responders` is built from it (`bots.SearchBot._addressed`
    uses both) -- both are off limits to an honest bot. The one line that
    names `pending_responders` is the P1 comment explaining why.

    heximax is a package (`hexset.bots.heximax`, split by concern into
    `belief`/`evaluate`/`search`/`trade`/`presets`), not the single file this
    test was first written against, so the source under test is every one of
    those submodules concatenated -- the invariant is about the package as a
    whole, not about which file a line happens to live in now. The deprecated
    top-level `heximax` shim is one import line and is not itself part of
    the source this checks.
    """
    import importlib
    import inspect

    # `importlib.import_module` returns the actual `sys.modules` entry;
    # neither `import hexset.bots.heximax as x` nor
    # `from hexset.bots import heximax` would: both resolve through
    # attribute access on `hexset.bots`, whose own `from .heximax import
    # (..., heximax, ...)` rebinds the attribute named `heximax` to the
    # *factory function*, shadowing the submodule of the same name.
    heximax_pkg = importlib.import_module("hexset.bots.heximax")
    from hexset.bots.heximax import belief, evaluate, presets, search, trade

    source = "\n".join(
        inspect.getsource(module)
        for module in (heximax_pkg, belief, evaluate, search, trade, presets)
    )
    assert "responders(" not in source
    mentions = [line for line in source.splitlines() if "pending_responders" in line]
    assert len(mentions) == 1
    assert "is the engine's" in mentions[0]
