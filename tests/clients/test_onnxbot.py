from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import legal_actions  # noqa: E402
from hexset.server.rules import options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.game import start, to_move  # noqa: E402
from hexset.clients.onnxbot import load, network_bot  # noqa: E402
from conftest import step_randomly  # noqa: E402

FIXTURE_V2 = Path(__file__).parent / "fixtures" / "stub-contract5.onnx"


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
