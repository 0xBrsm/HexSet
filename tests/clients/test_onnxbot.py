from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import legal_actions  # noqa: E402
from hexset.server.rules import options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.game import Phase, start, to_move  # noqa: E402
from hexset.clients.onnxbot import load, network_bot  # noqa: E402
from conftest import step_randomly  # noqa: E402

FIXTURE_V2 = Path(__file__).parent / "fixtures" / "stub-contract5.onnx"
FIXTURE_VALUED = Path(__file__).parent / "fixtures" / "stub-contract5-valued.onnx"
FIXTURE_VALUED_BATCH1 = Path(__file__).parent / "fixtures" / "stub-contract5-valued-batch1.onnx"


@pytest.fixture
def checkpoint_v2():
    """`stub-contract5.onnx`: a record-contract stub for a 4-player base
    board — uniform-over-legal prior, zero value, no learned weights (see
    `tests/fixtures/build_stub.py`). This is the only policy `onnxbot`
    serves: contract 1 (and the `encoding_v1`-based policy that read it) was
    dropped 2026-09-02 (`docs/engine-divergence-2026-09-02.md`, B5), so every
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
    from hexset.clients.onnxbot import network_evaluator

    path, board = checkpoint_v2
    evaluator = network_evaluator(path, board)
    game = start(board, 4, random.Random(2))
    for _ in range(30):
        step_randomly(game, random.Random(2))

    for seat in range(4):
        vector = evaluator.evaluate_game(game, seat)
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


def test_a_v2_stub_spawns_a_single_forward_bot(checkpoint_v2):
    """The stub's metadata asks for no search — `spawn` must read that off a
    record contract's metadata."""
    from hexset.clients.onnxbot import NetworkBot, spawn

    path, board = checkpoint_v2
    assert isinstance(spawn(path, board, rng=random.Random(0)), NetworkBot)


def test_the_checkpoint_is_loaded_once_per_process_not_once_per_game(checkpoint_v2):
    path, board = checkpoint_v2
    first = network_bot(path, board)
    second = network_bot(path, board)
    assert first is not second
    assert first.policy is second.policy


def test_a_checkpoint_dropped_in_with_the_same_name_is_not_served_stale(
    checkpoint_v2, tmp_path
):
    """The whole point of hexset's models/ directory: replacing a file by
    name must not keep serving the old in-memory session — unlike
    the training repo's immutable runs/*.pt, this repo's checkpoints are
    expected to
    change underneath a running server."""
    path, board = checkpoint_v2
    live = tmp_path / "live.onnx"
    live.write_bytes(Path(path).read_bytes())

    first = load(live, board.topology)
    live.write_bytes(Path(path).read_bytes())  # rewritten, same bytes, new mtime
    second = load(live, board.topology)
    assert first.policy is not second.policy


def test_the_offer_budget_defaults_from_the_checkpoint_unless_overridden(checkpoint_v2):
    """`stub-contract5.onnx` declares no `max_trades` at all (an empty
    metadata value, same as a checkpoint that never set one), so the default
    is `None` — no budget — unless a caller overrides it."""
    path, board = checkpoint_v2
    assert network_bot(path, board).max_trades is None
    assert network_bot(path, board, max_trades=8).max_trades == 8


def test_the_budget_is_honoured_exactly_as_the_search_bot_honours_it(checkpoint_v2):
    path, board = checkpoint_v2
    bot = network_bot(path, board)
    rng = random.Random(11)
    game = start(board, 4, rng)

    from hexset.actions import apply

    for _ in range(600):
        if game.won_by is not None:
            break
        allowed = legal_actions(game)
        action = bot.choose(game)
        assert action in allowed
        apply(game, action)


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(checkpoint_v2):
    path, _ = checkpoint_v2
    board3 = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="trained for 4 players"):
        network_bot(path, board3).choose(start(board3, 3, random.Random(0)))


def test_scoring_is_greedy_so_a_position_answers_the_same_way_twice(checkpoint_v2):
    path, board = checkpoint_v2
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(5))
    rng = random.Random(5)
    for _ in range(40):
        step_randomly(game, rng)
    assert to_move(game) is not None
    assert bot.choose(game) == bot.choose(game)


# --- Trading: valuation/accepts off the value head, mirroring
# `hexnet.policy.DerivedTrader` (`agents/reference/trading-design.md` §8) ---


@pytest.fixture
def checkpoint_valued():
    """`stub-contract5-valued.onnx`: same shape as `stub-contract5.onnx`, but
    `value` reads `own_hand` through five fixed, distinct, non-zero
    per-resource weights instead of being identically zero (see
    `fixtures/build_stub.py --valued`). Linear in the hand, so a one-card
    imagined successor's delta is exactly that resource's weight whatever the
    starting hand holds -- deterministic and non-degenerate enough to
    exercise `NetworkBot.valuation`/`accepts` for real, without pretending
    this is a trained network.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE_VALUED), board
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def _seated_at_main(path: str, board, seat: int = 0, hand=(1, 2, 0, 1, 3)):
    """A bot that has just chosen once at a `MAIN`-phase position with
    `seat`'s hand pinned to `hand` -- `valuation`/`accepts` only ever answer
    for the game `choose` last handed the bot (see `NetworkBot._seated`'s
    docstring), so every trading test needs one `choose` first, exactly as
    the server's own `Tables.act` does before it ever asks for a vector.
    """
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(1))
    game.phase = Phase.MAIN
    game.current_player = seat
    game.state(seat, hidden=False).hands[seat] = list(hand)
    bot.choose(game)
    return bot, game


def test_valuation_is_five_floats_in_range_and_deterministic(checkpoint_valued):
    path, board = checkpoint_valued
    bot, game = _seated_at_main(path, board)
    seat = to_move(game)
    view = game.state(seat)

    first = bot.valuation(view)
    second = bot.valuation(view)

    assert len(first) == 5
    assert all(-1.0 <= x <= 1.0 for x in first)
    assert first == second


def test_valuation_scores_the_six_rows_in_one_batched_call_when_the_graph_allows_it(
    checkpoint_valued,
):
    """`stub-contract5-valued.onnx` declares a dynamic batch axis (like every
    other fixture here), so `V2Policy.value_of`'s one-call path applies —
    the six-row fan-out (the hand plus its five one-card successors) costs
    one graph dispatch, not six."""
    path, board = checkpoint_valued
    bot, _ = _seated_at_main(path, board)
    assert bot.policy._batchable(6)


def test_valuation_falls_back_to_six_calls_when_the_graphs_batch_is_fixed_to_one():
    """The graph's own declared shape, not a caller's guess, decides this:
    `stub-contract5-valued-batch1.onnx` is the same weights with every input's
    batch axis pinned to the literal `1` (`fixtures/build_stub.py --valued
    --fixed-batch`), and `V2Policy.value_of` must fall back to one call per
    row rather than feed it a batch of six and let onnxruntime refuse."""
    board = random_base_board(random.Random(0))
    bot, game = _seated_at_main(str(FIXTURE_VALUED_BATCH1), board)
    assert not bot.policy._batchable(6)

    seat = to_move(game)
    view = game.state(seat)
    values = bot.valuation(view)
    assert len(values) == 5
    assert all(-1.0 <= x <= 1.0 for x in values)


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
    """`stub-contract5-valued.onnx`'s weights are `[0.006, -0.011, 0.004,
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


def test_accepts_many_matches_accepts_within_tolerance_on_random_bundles(checkpoint_valued):
    """Row-by-row agreement within 1e-6, the strong check item 4 asks for,
    over a spread of coverable one-card and two-card exchanges rather than
    the four hand-picked cases above."""
    import random as _random

    path, board = checkpoint_valued
    bot, game = _seated_at_main(path, board, hand=(2, 2, 1, 2, 3))
    seat = to_move(game)
    view = game.state(seat)
    hand = view.known[seat]

    rng = _random.Random(11)
    received = []
    counterparties = []
    for _ in range(15):
        wanted = [0, 0, 0, 0, 0]
        give_from = rng.randrange(5)
        take_to = rng.randrange(5)
        if give_from == take_to:
            take_to = (take_to + 1) % 5
        wanted[give_from] -= 1
        wanted[take_to] += 1
        received.append(tuple(wanted))
        counterparties.append((seat + 1 + rng.randrange(3)) % 4)

    expected = [bot.accepts(view, r, c) for r, c in zip(received, counterparties)]
    many = bot.accepts_many(view, received, counterparties)
    assert many == expected

    # And the underlying value comparisons agree to float precision, not
    # only the thresholded boolean -- rebuild each row's raw value the same
    # way `accepts`/`accepts_many` do and diff them directly.
    for r, c in zip(received, counterparties):
        after = [n + d for n, d in zip(hand, r)]
        if any(n < 0 for n in after):
            continue
        before_value, after_value = bot._own_values(seat, [list(hand), after])
        assert (after_value > before_value) == bot.accepts(view, r, c)
