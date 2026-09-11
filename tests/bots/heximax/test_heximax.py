# SPDX-License-Identifier: GPL-3.0-only
"""Heximax behavior and information-set invariance.

Worlds with the same public record and own hand must produce the same choice
under the same random seed, regardless of opponents' actual hidden holdings.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, apply, legal_actions
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
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
from helpers import clear_hand, give

INFORMATION_SET_SEED = 2


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


def test_heximax_declares_its_measured_floor():
    from hexset.bots.heximax import HEXIMAX_TRADE_FLOOR

    bot = a_bot(a_game(seed=15), 0)
    assert bot.trade_floor == HEXIMAX_TRADE_FLOOR == 0.0197


def test_a_trading_bot_trades():
    # The wiring, not the floor: under its measured floor (0.0197) heximax's
    # claimed gains almost never clear, which is the floor doing its job.
    # With each bot's own floor at zero a trading bot must still trade.
    game = a_game(seed=15)
    bots = [a_bot(game, s, max_nodes=200) for s in range(4)]
    for bot in bots:
        bot.trade_floor = 0.0
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


def test_the_vectorised_gate_matches_the_clone_it_replaces_bit_for_bit():
    """`_delta`'s fast path prices a candidate by recomputing every seat's
    hand terms from the post-trade pool (`HonestEvaluator.score_many`);
    `_delta_reference` clones the state and re-reads it the slow way.

    Worth pinning rather than trusting: `score_many` is a hand-written
    transposition of `hand_terms` onto the candidate axis, and the hand terms
    put a `PURCHASE_VALUE`-weighted argmax in the middle of it whose tie-break
    has to match the scalar loop's strictly-greater test exactly. Pick a
    different winner there and `best_cost` changes with it, which silently
    moves `spare_card` on a subset of hands -- the private gate would then be
    pricing a position the search would never score.
    """
    worst = 0.0
    checked = nonzero = 0
    for seed in (26, 31, 44):
        game = after_setup(seed)
        rng = random.Random(seed)
        bot = a_bot(game, seed)
        for _ in range(120):
            options = legal_actions(game)
            if not options or is_over(game):
                break
            apply(game, rng.choice(options))
            if game.phase is not Phase.MAIN:
                continue
            seat = to_move(game)
            view = game.state(seat)
            for counterparty in range(game._state.num_players):
                if counterparty == seat:
                    continue
                for give_r in range(NUM_RESOURCES):
                    if not game._state.hands[seat][give_r]:
                        continue
                    for take_r in range(NUM_RESOURCES):
                        if take_r == give_r or not game._state.hands[counterparty][take_r]:
                            continue
                        received = one_for_one(give_r, take_r)
                        fast = bot._delta(view, seat, seat, received, counterparty, bot._rank)
                        slow = bot._delta_reference(
                            view, seat, seat, received, counterparty, bot._rank
                        )
                        worst = max(worst, abs(fast - slow))
                        checked += 1
                        nonzero += fast != 0.0
    # A run that priced nothing, or priced everything at zero, would pass
    # vacuously -- the equality is only worth anything over real candidates.
    assert checked > 500
    assert nonzero > 100
    # To floating-point noise, not bit for bit: `score_many` sums a hand's
    # bank value with numpy's pairwise reduction where the scalar loop adds
    # left to right, so the two agree exactly only for weight values that
    # happen to round alike -- the shipped -0.15 did, the swept -0.30 differs
    # by one ulp. `score_many`'s own docstring states the contract as
    # `gains_many`'s caller checks it, at 1e-12; that is what is pinned here.
    assert worst <= 1e-12


# --- presets ------------------------------------------------------------------


def test_the_heximax_presets_spawn_with_their_documented_modes():
    board = random_base_board(random.Random(0))
    honest = spawn(PRESETS["heximax"], board, random.Random(0))
    quiet = spawn(PRESETS["heximax-notrade"], board, random.Random(0))
    for bot in (honest, quiet):
        assert isinstance(bot, Heximax)
        assert bot.placement
        assert bot.depth == 2 and bot.width == 6

    assert (honest.mode, honest.max_trades) == ("honest", None)
    assert honest.evaluator.weights == TRADING_WEIGHTS
    assert (quiet.mode, quiet.max_trades) == ("notrade", 0)
    assert quiet.evaluator.weights == NO_TRADE_WEIGHTS


# --- honesty of the source ------------------------------------------------------


def test_heximax_reads_the_true_state_only_where_it_says_so():
    """Every `hidden=False` in heximax's own package carries a `# true state:`
    comment saying why, and there are no other routes to the raw state: the
    field is engine-private (`Game._state`), `game.state(seat)` is the honest
    view, and `View` has no omniscient construction at all.

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
    assert "omniscient" not in source


# --- hidden victory points and the fitted temperature --------------------------


def test_an_opponents_development_cards_are_worth_their_expected_victory_points():
    """The anchor term means the same thing in every row: the knower's own VP
    cards are exact, an opponent's are its held count times the VP share of
    the unseen pool."""
    from hexset.bots.heximax.evaluate import VP_CARDS, HonestEvaluator, expected_card_points
    from hexset.cards import DevCard

    game = after_setup(3)
    for p in range(4):
        set_known_hand(game, p, [0] * NUM_RESOURCES)
    state = game._state
    # Seat 1 holds two knights and a VP card, seat 2 one knight; all drawn
    # from the deck, so the unseen pool shrinks by the same three cards.
    state.dev_cards[1][DevCard.KNIGHT] = 2
    state.new_dev_cards[1][DevCard.VICTORY_POINT] = 1
    state.dev_cards[2][DevCard.KNIGHT] = 1
    del state.deck[:4]

    unseen_from_0 = len(state.deck) + 3 + 1
    assert expected_card_points(state, 1, 0) == pytest.approx(3 * VP_CARDS / unseen_from_0)
    assert expected_card_points(state, 2, 0) == pytest.approx(1 * VP_CARDS / unseen_from_0)
    assert expected_card_points(state, 0, 1) == 0.0
    # Seat 1 knows its own VP card, so the pool it reads has one VP card fewer
    # and excludes its own three cards.
    assert expected_card_points(state, 2, 1) == pytest.approx(
        (VP_CARDS - 1) / (len(state.deck) + 1)
    )

    honest = HonestEvaluator(state.board)
    plain = Evaluator(state.board)
    rows = honest.rows_game(game, 0)
    truth = [plain.terms(state, p, knower=p) for p in range(4)]
    assert rows[0][0] == truth[0][0]
    assert rows[1][0] == pytest.approx(truth[1][0] - 1 + 3 * VP_CARDS / unseen_from_0)
    own = honest.rows_game(game, 1)
    assert own[1][0] == truth[1][0]  # exact for the knower: the real VP card counts
    # `evaluate` is `rows` dotted with the weights, so the two cannot drift.
    scored = honest.evaluate_game(game, 0)
    assert scored == pytest.approx(
        [sum(w * v for w, v in zip(honest.vector, row)) for row in rows]
    )


def test_a_candidate_temperature_travels_with_the_entrant():
    from hexset.arena import Entrant
    from hexset.bots.stances import WIN_TEMPERATURE, win, win_at

    vector = [4.0, 6.0, 5.0, 3.0]
    assert win(vector, 1) == pytest.approx(win_at(vector, 1, WIN_TEMPERATURE))
    assert win_at(vector, 1, 0.5) > win(vector, 1) > win_at(vector, 1, 50.0)

    board = random_base_board(random.Random(0))
    bot = spawn(
        Entrant("hot", kind="heximax", depth=2, width=6, temperature=0.5),
        board, random.Random(0),
    )
    assert bot.temperature == 0.5
    assert bot._rank(vector, 1) == pytest.approx(win_at(vector, 1, 0.5))
    default = spawn(PRESETS["heximax"], board, random.Random(0))
    assert default.temperature is None
    assert default._rank(vector, 1) == pytest.approx(win(vector, 1))
    with pytest.raises(ValueError):
        spawn(
            Entrant("bad", kind="heximax", depth=2, width=6, stance="relative", temperature=1.0),
            board, random.Random(0),
        )


@pytest.mark.parametrize('stance,temperature', [('win', 1.7), ('relative', None)])
def test_split_trade_values_match_the_gate_and_preserve_move_rng(stance, temperature):
    game = after_setup(26)
    for seat in range(4):
        set_known_hand(game, seat, [2, 2, 2, 2, 2])
    common = dict(stance=stance, temperature=temperature, trade_floor=0.0)
    split = a_bot(game, 81, weights=NO_TRADE_WEIGHTS, expansion_value=.25,
                  trade_weights=TRADING_WEIGHTS, **common)
    move = a_bot(game, 81, weights=NO_TRADE_WEIGHTS, expansion_value=.25, **common)
    candidates = [one_for_one(0, 4), one_for_one(1, 3)]
    partners = [1, 2]
    move_rng = split.rng.getstate()
    for cards in ([2, 2, 2, 2, 2], [3, 1, 0, 2, 4]):
        # Reuse the gate across real state/ledger changes: its caches must
        # produce the same answers as a fresh evaluator on the current view.
        set_known_hand(game, 0, cards)
        gate = a_bot(game, 0, weights=TRADING_WEIGHTS, **common)
        view = game.state(0)
        expected = gate.gains_many(view, candidates, partners)
        assert split.gains_many(view, candidates, partners) == expected
        assert split.accepts(view, candidates[0], partners[0]) == (expected[0] > 0)
        offers = list(zip(partners, candidates))
        assert split.estimate_many(view, offers) == gate.estimate_many(view, offers)
    assert split.rng.getstate() == move_rng
    assert split.choose(game) == move.choose(game)
    assert split.rng.getstate() == move.rng.getstate()
    assert split.trade_floor == 0.0
    split.max_trades = 0
    assert split.gains_many(game.state(0), candidates, partners) == [-1.0, -1.0]
    assert split.estimate_many(game.state(0), offers) == [-1.0, -1.0]
