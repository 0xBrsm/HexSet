from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.server.rules import options_for  # noqa: E402
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


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(checkpoint_v2):
    path, _ = checkpoint_v2
    board3 = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="trained for 4 players"):
        network_bot(path, board3).choose(start(board3, 3, random.Random(0)))


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
    """
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(1))
    game.phase = Phase.MAIN
    game.current_player = seat
    game.state(seat, hidden=False).hands[seat] = list(hand)
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
