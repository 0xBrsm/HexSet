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
    """`stub-contract6.onnx`: uniform-over-legal prior, zero value, no learned
    weights. Contract *dispatch* is `test_contract_dispatch.py`'s job.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE_V2), board
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def test_a_v2_search_over_a_learned_prior_plays_a_legal_action(checkpoint_v2):
    from hexset.actions import apply
    from hexset.clients.onnxbot import searcher

    path, board = checkpoint_v2
    search = searcher(path, board, simulations=16, wave=4, rng=random.Random(0))
    game = start(board, 4, random.Random(2), turn_cap=3000)
    for _ in range(20):
        action = search.choose(game)
        assert action in set(options_for(game))
        apply(game, action)


def test_a_terminal_leaf_is_scored_on_the_win_probability_scale(checkpoint_v2):
    from hexset.clients.netbot import LeafEvaluator
    from hexset.clients.onnxbot import load
    from hexset.game import is_over

    path, board = checkpoint_v2
    loaded = load(path, board.topology)
    evaluator = LeafEvaluator(policy=loaded.policy)

    game = start(board, 4, random.Random(2), turn_cap=3000)
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
    from hexset.clients.onnxbot import load

    path, board = checkpoint_v2
    capped = load(path, board.topology, threads=1)
    options = capped.policy.session.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1

    assert load(path, board.topology, threads=1) is capped
    # The bare call *is* the capped one: one thread is the default, because
    # the sharded caller in this package never passed anything.
    assert load(path, board.topology) is capped
    # `None` is the opt-out -- onnxruntime's own choice -- and keys its own
    # entry, so asking for the box does not hand back a capped session.
    assert load(path, board.topology, threads=None) is not capped


# --- Trading: `accepts` off the value head ---


@pytest.fixture
def checkpoint_valued():
    """Same shape, but `value` reads `own_hand` through five fixed, distinct,
    non-zero weights: linear in the hand, so a one-card successor's delta is
    exactly that resource's weight.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE_VALUED), board
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def _seated_at_main(path: str, board, seat: int = 0, hand=(1, 2, 0, 1, 3)):
    """`accepts` only answers for the game `choose` last handed the bot, so every
    trading test needs one `choose` first. The settlement at vertex 0 gives
    `seat` a city to upgrade to.
    """
    from hexset.state import Building

    bot = network_bot(path, board)
    game = start(board, 4, random.Random(1))
    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    state.hands[seat] = list(hand)
    # The next seat holds cards, so an exchange with it is one it could make:
    # a gate prices a counterparty with no room for its side at `-1.0`.
    state.hands[(seat + 1) % 4] = [2, 2, 2, 2, 2]
    state.vertex_owner[0] = seat
    state.vertex_building[0] = Building.SETTLEMENT
    bot.choose(game)
    return bot, game


def test_accepts_many_agrees_with_accepts_row_by_row(checkpoint_valued):
    """The fixture's weights are `[0.006, -0.011, 0.004, 0.013, -0.008]`: only
    the brick-for-wheat exchange is strictly improving, and a zero delta is
    refused."""
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


# --- the thread cap ---------------------------------------------------------


def test_the_arena_spawns_capped_sessions(monkeypatch, checkpoint_v2):
    """onnxruntime's own default is one intra-op thread per visible core, and a
    duel is thirty worker processes: measured, that cost 4x the wall clock.
    `register_entrants` builds its loaders as `loader(path, topology)` -- two
    arguments -- so whatever `threads` defaults to is what every arena worker
    gets."""
    from hexset import arena
    from hexset.clients import netbot, onnxbot

    path, board = checkpoint_v2
    asked: list[object] = []
    real = onnxbot._load_cached

    def spy(spy_path, topology, device, mtime_ns, threads):
        asked.append(threads)
        return real(spy_path, topology, device, mtime_ns, threads)

    spy.cache_clear = real.cache_clear  # the fixture's teardown calls it
    monkeypatch.setattr(onnxbot, "_load_cached", spy)
    netbot.register_entrants(onnxbot.load)

    entrant = arena.Entrant("network", kind="network", weights=path)
    arena.spawn(entrant, board, random.Random(0))

    assert asked == [onnxbot.DEFAULT_THREADS], (
        f"the arena asked for {asked}, not a capped pool"
    )


def test_a_file_that_names_a_trader_is_seated_with_it(checkpoint_v2, tmp_path):
    """A `trader` key in the file's metadata seats the checkpoint's moves over
    that bot's trading (`hexset.arena.traded`): the web board and a lineup's
    `network:x.onnx` both reach it through `spawn`. A file without the key is
    seated as it always was."""
    onnx = pytest.importorskip("onnx")
    from hexset.bots import TradesBy
    from hexset.clients.netbot import NetworkBot
    from traders import ScarcityTrader
    from hexset.clients.onnxbot import load, spawn

    path, board = checkpoint_v2
    model = onnx.load(path)
    entry = model.metadata_props.add()
    entry.key, entry.value = "trader", "test-trader"
    named = tmp_path / "hybrid.onnx"
    onnx.save(model, str(named))

    assert load(str(named), board.topology).trader == "test-trader"
    seat = spawn(str(named), board, rng=random.Random(0))
    assert isinstance(seat, TradesBy)
    assert isinstance(seat.mover, NetworkBot) and isinstance(seat.trader, ScarcityTrader)
    assert load(path, board.topology).trader is None
    assert isinstance(spawn(path, board, rng=random.Random(0)), NetworkBot)


def test_a_file_naming_a_trader_nothing_registers_fails_at_spawn_naming_the_fix(
    checkpoint_v2, tmp_path
):
    """At spawn, not at the first trade: the error names the file, the
    metadata key and how to load the runtime that would register it."""
    onnx = pytest.importorskip("onnx")
    from hexset.clients.onnxbot import load, spawn

    path, board = checkpoint_v2
    model = onnx.load(path)
    entry = model.metadata_props.add()
    entry.key, entry.value = "trader", "no-such-bot"
    named = tmp_path / "unregistered.onnx"
    onnx.save(model, str(named))

    assert load(str(named), board.topology).trader == "no-such-bot"
    with pytest.raises(ValueError) as caught:
        spawn(str(named), board, rng=random.Random(0))
    message = str(caught.value)
    assert str(named) in message
    assert "`trader`" in message and "'no-such-bot'" in message
    assert "--runtime" in message


def test_spawn_refuses_a_checkpoint_trained_for_another_player_count(checkpoint_v2):
    from hexset.clients.onnxbot import spawn

    path, board = checkpoint_v2
    with pytest.raises(ValueError) as caught:
        spawn(path, board, rng=random.Random(0), players=3)
    assert path in str(caught.value) and "4 players" in str(caught.value)
    assert spawn(path, board, rng=random.Random(0), players=4) is not None
