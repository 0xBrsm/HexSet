# SPDX-License-Identifier: GPL-3.0-only
"""`heximax`: the honest handcrafted baseline (design: heximax.md §6).

The tests here are the gate the design says must be written first. The one
that matters most is the information-set regression: two positions that the
public record cannot tell apart must draw the same move from `heximax`, and
the omniscient `search2` is shown to be able to tell them apart on at least
one such pair, which is the leak the regression guards against.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, apply, legal_actions
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.bots import SearchBot
from hexset.bots.evaluate import Evaluator
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.bots.heximax import (
    NO_TRADE_WEIGHTS,
    TRADING_WEIGHTS,
    Heximax,
    HonestEvaluator,
    View,
    heximax,
)
from hexset.ledger import SeatLedger
from hexset.play import step_randomly
from hexset.trading import one_for_one
from hexset.state import copy_state
from helpers import clear_hand, give

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


def a_bot(game, seed: int = 0, **overrides) -> Heximax:
    return heximax(game._state.board, random.Random(seed), **overrides)


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
WORLD_SEEDS = (2, 3)


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


# --- evaluate -----------------------------------------------------------------


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


# --- search -------------------------------------------------------------------


def test_a_no_trade_bot_refuses_everything():
    """`max_trades=0` is the whole of the no-trade referent: the gate
    prices every candidate below zero, so nothing it is party to ever
    clears."""
    game = a_game(seed=14)
    board = game._state.board
    quiet = heximax(board, random.Random(0), mode="notrade", max_nodes=64)
    assert quiet.max_trades == 0
    view = game.state(0)
    assert quiet.accepts(view, one_for_one(0, 4), 1) is False
    assert quiet.gains_many(view, [one_for_one(0, 4)], [1]) == [-1.0]

    talkers = [a_bot(game, s, max_nodes=64) for s in (1, 2, 3)]
    bots = [quiet, *talkers]
    game.gates = tuple(bots)
    moves = 0
    while not is_over(game) and moves < 20000:
        seat = to_move(game)
        apply(game, bots[seat].choose(game))
        moves += 1
    assert is_over(game)
    assert all(t.a != 0 and t.b != 0 for t in game.trades)


def test_a_trading_bot_trades():
    game = a_game(seed=15)
    bots = [a_bot(game, s, max_nodes=200) for s in range(4)]
    game.gates = tuple(bots)

    traded = 0
    moves = 0
    while not is_over(game) and moves < 20000:
        seat = to_move(game)
        cleared = len(game.trades)
        apply(game, bots[seat].choose(game))
        traded += len(game.trades[cleared:])
        moves += 1
    assert traded > 0


@pytest.mark.parametrize("players", [4])
def test_a_game_finishes_for_any_player_count(players):
    game = a_game(seed=17, players=players)
    bots = [a_bot(game, s, max_nodes=64) for s in range(players)]
    play_out(game, bots)
    assert is_over(game)


def test_an_unknown_stance_or_mode_is_refused():
    game = a_game()
    with pytest.raises(ValueError, match="unknown stance"):
        Heximax(HonestEvaluator(game._state.board), stance="spiteful")
    with pytest.raises(ValueError, match="unknown heximax mode"):
        heximax(game._state.board, random.Random(0), mode="clairvoyant")


# --- trading (`hexset.trading`) ------------------------------------------------


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
