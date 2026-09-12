from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.game import Phase, start, to_move  # noqa: E402
from hexset.clients.onnxbot import network_bot  # noqa: E402
from conftest import step_randomly  # noqa: E402

FIXTURE_V2 = Path(__file__).parent / "fixtures" / "stub-contract6.onnx"
FIXTURE_VALUED = Path(__file__).parent / "fixtures" / "stub-contract6-valued.onnx"


@pytest.fixture
def checkpoint_v2():
    """`stub-contract6.onnx`: a record-contract stub for a 4-player base
    board — uniform-over-legal prior, zero value, no learned weights (see
    `tests/fixtures/build_stub.py`). This is the only policy `onnxbot`
    serves: contract 1 (and the `encoding_v1`-based policy that read it) was
    dropped 2026-09-02, so every
    generic bot behaviour below — caching, staleness, offer budgets, player
    checks, determinism — is exercised against this fixture rather than a
    frozen contract-1 checkpoint.

    Contract *dispatch* — which graph shape a `contract` value routes to, and
    whether a genuine dev-HexNet export loads at all — is
    `test_contract_dispatch.py`'s job, against a real export rather than this
    stub.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE_V2), board
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def test_a_v2_checkpoint_plays_a_legal_action_from_every_phase(checkpoint_v2):
    from hexset.actions import apply

    path, board = checkpoint_v2
    bot = network_bot(path, board)
    rng = random.Random(3)
    game = start(board, 4, rng)

    seen = set()
    for _ in range(400):
        if game.won_by is not None:
            break
        action = bot.choose(game)
        assert action in options_for(game)
        seen.add(game.phase)
        apply(game, action)
    assert len(seen) > 3


def test_a_v2_checkpoints_value_head_is_already_board_seat_order(checkpoint_v2):
    from hexset.clients.onnxbot import load

    path, board = checkpoint_v2
    policy = load(path, board.topology).policy
    game = start(board, 4, random.Random(2))
    for _ in range(30):
        step_randomly(game, random.Random(2))

    for seat in range(4):
        vector = policy.value_rows([(game, seat)])[0]
        assert len(vector) == 4
        assert vector == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_a_v2_search_over_a_learned_prior_plays_a_legal_action(checkpoint_v2):
    from hexset.actions import apply
    from hexset.clients.onnxbot import searcher

    path, board = checkpoint_v2
    search = searcher(path, board, simulations=16, wave=4, rng=random.Random(0))
    game = start(board, 4, random.Random(2))
    for _ in range(20):
        action = search.choose(game)
        assert action in set(options_for(game))
        apply(game, action)


def test_a_searched_checkpoint_trades_through_its_own_value_head(checkpoint_v2):
    """`Search` decides moves and nothing else; a served checkpoint exported
    with `search: mcts` therefore had no trade gate at all and never traded
    (`linear24` at the table, 2026-09-08). The searcher now carries the plain
    bot's gate, seated where `choose` last was, and answers `accepts_many`
    with one verdict per candidate -- the same verdicts the plain bot gives."""
    from hexset.clients.onnxbot import network_bot, searcher
    from hexset.trading import _candidates, valued_many

    path, board = checkpoint_v2
    search = searcher(path, board, simulations=8, wave=4, rng=random.Random(0))
    plain = network_bot(path, board)
    game = start(board, 4, random.Random(2))
    for _ in range(40):
        step_randomly(game, random.Random(2))
    seat = to_move(game)
    search.choose(game)
    plain.choose(game)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))
    received = [b for _, b in candidates]
    counterparties = [c for c, _ in candidates]
    assert search.trade_floor == plain.trade_floor == 0.0  # a boolean gate has no resolution
    verdicts = search.accepts_many(game.state(seat), received, counterparties)
    assert len(verdicts) == len(candidates)
    assert verdicts == plain.accepts_many(game.state(seat), received, counterparties)
    # What the clearing house and the round read: the value head's own
    # deltas, the same numbers the plain bot gives, not a sign.
    gains = valued_many(search, game.state(seat), received, counterparties)
    assert gains == plain.gains_many(game.state(seat), received, counterparties)
    assert verdicts == [g > 0.0 for g in gains]
    assert search.estimate_many(game.state(seat), candidates) == plain.estimate_many(game.state(seat), candidates)


def test_the_network_gate_scores_both_sides_of_an_exchange(checkpoint_v2):
    """`gains_many` is this seat's own row of the value head, after minus
    before, on a position where *both* hands moved; `estimate_many` is the
    counterparty's row on the same two positions. One float per candidate,
    `-1.0` for one this seat cannot cover, and every candidate two cards or
    fewer a side is scored however many there are. The stub's value head is
    zero, so every scored delta is exactly zero here -- the shape and the
    cover rule are what this pins; the numbers come from a real checkpoint."""
    from hexset.clients.onnxbot import network_bot
    from hexset.trading import _candidates

    path, board = checkpoint_v2
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(2))
    for _ in range(40):
        step_randomly(game, random.Random(2))
    seat = to_move(game)
    bot.choose(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    gains = bot.gains_many(view, received, thems)
    estimates = bot.estimate_many(view, candidates)
    assert len(gains) == len(estimates) == len(candidates)
    scored = [g for g in gains if g != -1.0]
    assert scored and all(g == 0.0 for g in scored)
    assert all(e == 0.0 for e in estimates if e != -1.0)

    # A bundle this seat cannot cover is never scored.
    hand = list(view.known[seat])
    short = tuple(-(hand[r] + 1) if r == 0 else (1 if r == 1 else 0) for r in range(len(hand)))
    assert bot.gains_many(view, [short], [thems[0]]) == [-1.0]
    # The live game is left exactly as it was.
    assert game.state(seat, hidden=False).hands[seat] == hand


def test_a_terminal_leaf_is_scored_on_the_win_probability_scale(checkpoint_v2):
    """Contract-6 value heads are trained on `hexn.rewards.win_loss`, so every
    non-terminal leaf in a wave is a win probability. A terminal leaf has to be
    one too, or the search backs a points margin up the tree beside them."""
    from hexset.clients.onnxbot import LeafEvaluator, load
    from hexset.game import is_over

    path, board = checkpoint_v2
    loaded = load(path, board.topology)
    evaluator = LeafEvaluator(policy=loaded.policy)

    game = start(board, 4, random.Random(2))
    with pytest.raises(ValueError, match="has not finished"):
        evaluator.terminal(game)

    rng = random.Random(2)
    moves = 0
    while not is_over(game) and moves < 20000:
        step_randomly(game, rng)
        moves += 1
    assert is_over(game)

    scores = evaluator.terminal(game)
    assert len(scores) == 4
    assert sorted(scores) == [0.0, 0.0, 0.0, 1.0]
    assert scores[game.won_by] == 1.0


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(checkpoint_v2):
    path, _ = checkpoint_v2
    board3 = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="trained for 4 players"):
        network_bot(path, board3).choose(start(board3, 3, random.Random(0)))


def test_threads_caps_the_session_pools_and_keys_the_cache(checkpoint_v2):
    """A caller that has already sharded games across processes asks for one
    thread each, rather than every process sizing a pool from the whole core
    count and oversubscribing the box."""
    from hexset.clients.onnxbot import load

    path, board = checkpoint_v2
    capped = load(path, board.topology, threads=1)
    options = capped.policy.session.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1

    # `threads` is part of the cache key, so a differently-capped request is
    # not handed back the session built for the first one.
    assert load(path, board.topology, threads=1) is capped
    assert load(path, board.topology) is not capped


# --- Trading: `accepts` off the value head, mirroring
# `hexnet.policy.DerivedTrader` ---


@pytest.fixture
def checkpoint_valued():
    """`stub-contract6-valued.onnx`: same shape as `stub-contract6.onnx`, but
    `value` reads `own_hand` through five fixed, distinct, non-zero
    per-resource weights instead of being identically zero (see
    `fixtures/build_stub.py --valued`). Linear in the hand, so a one-card
    imagined successor's delta is exactly that resource's weight whatever the
    starting hand holds -- deterministic and non-degenerate enough to
    exercise `NetworkBot.accepts` for real, without pretending this is a
    trained network.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE_VALUED), board
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def _seated_at_main(path: str, board, seat: int = 0, hand=(1, 2, 0, 1, 3)):
    """A bot that has just chosen once at a `MAIN`-phase position with
    `seat`'s hand pinned to `hand` -- `accepts` only ever answers for the
    game `choose` last handed the bot (see `NetworkBot._seated`'s
    docstring), so every trading test needs one `choose` first, exactly as
    the server's own `Tables.act` does before it ever asks the gate.

    A settlement at vertex 0 gives `seat` a city to upgrade to -- the
    affordability filter (`hexset.clients.netbot._kinds_of`) needs at least
    one purchase kind reachable, or every candidate reads as changing
    nothing regardless of the hand.
    """
    from hexset.state import Building

    bot = network_bot(path, board)
    game = start(board, 4, random.Random(1))
    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    state.hands[seat] = list(hand)
    state.vertex_owner[0] = seat
    state.vertex_building[0] = Building.SETTLEMENT
    bot.choose(game)
    return bot, game


def test_accepts_refuses_a_trade_with_zero_delta(checkpoint_valued):
    """Strict, not `>=`: an exchange that leaves the hand (and so the value)
    exactly where it was must be refused, the same termination argument
    `hexnet.policy.DerivedTrader.accepts` and `hexset.trading.trade_event`
    both rest on."""
    path, board = checkpoint_valued
    bot, game = _seated_at_main(path, board)
    seat = to_move(game)
    view = game.state(seat)

    no_change = (0, 0, 0, 0, 0)
    assert bot.accepts(view, no_change, (seat + 1) % 4) is False


def test_accepts_takes_a_strictly_improving_exchange(checkpoint_valued):
    """`stub-contract6-valued.onnx`'s weights are `[0.006, -0.011, 0.004,
    0.013, -0.008]`: giving up resource 1 (the most negative weight) for
    resource 3 (the most positive) strictly increases the linear value, so
    the private gate must say yes -- the mirror image of the zero-delta
    refusal above, pinning that `accepts` is not vacuously `False`."""
    path, board = checkpoint_valued
    bot, game = _seated_at_main(path, board, hand=(1, 2, 0, 1, 3))
    seat = to_move(game)
    view = game.state(seat)

    improving = (0, -1, 0, 1, 0)
    assert bot.accepts(view, improving, (seat + 1) % 4) is True


def test_accepts_many_agrees_with_accepts_row_by_row(checkpoint_valued):
    """The batched gate (`agents/reference/trading-design.md`'s post-data
    note, "the collector cost gate fails at 2.9-3.6x") must answer exactly
    what looping `accepts` would, one graph call instead of many: a mix of
    a refused zero-delta trade, the strictly-improving trade above, its
    strict reverse (refused), and an uncoverable bundle (negative resulting
    count -- refused without ever reaching the graph)."""
    path, board = checkpoint_valued
    bot, game = _seated_at_main(path, board, hand=(1, 2, 0, 1, 3))
    seat = to_move(game)
    view = game.state(seat)

    no_change = (0, 0, 0, 0, 0)
    improving = (0, -1, 0, 1, 0)
    worsening = (0, 1, 0, -1, 0)
    uncoverable = (0, 0, -1, 0, 0)  # this hand holds zero sheep
    received = [no_change, improving, worsening, uncoverable]
    counterparties = [(seat + 1) % 4] * len(received)

    expected = [bot.accepts(view, r, c) for r, c in zip(received, counterparties)]
    many = bot.accepts_many(view, received, counterparties)

    assert many == expected
    assert many == [False, True, False, False]
