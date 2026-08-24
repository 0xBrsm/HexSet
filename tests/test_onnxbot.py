from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="catan.onnxbot needs onnxruntime installed")

from catan.actions import legal_actions, within_offer_budget  # noqa: E402
from catan.arena import Entrant, compete, spawn  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.game import start, to_move  # noqa: E402
from catan.onnxbot import load, network_bot  # noqa: E402
from catan.play import step_randomly  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.onnx"


@pytest.fixture
def checkpoint():
    """`tiny.onnx`: a width=16/rounds=1 network for a 4-player base board,
    built and exported once by dev-catan's catan.export_onnx (see the
    commit that added this fixture) so this suite stays torch-free. Matches
    `test_netbot.py::checkpoint`'s role, minus building the checkpoint here
    — this repo has no torch to build one with.
    """
    board = random_base_board(random.Random(0))
    yield str(FIXTURE), board
    from catan.onnxbot import _load_cached

    _load_cached.cache_clear()


def test_a_checkpoint_plays_a_legal_action_from_every_phase(checkpoint):
    path, board = checkpoint
    bot = network_bot(path, board)
    rng = random.Random(3)
    game = start(board, 4, rng)

    seen = set()
    for _ in range(400):
        if game.won_by is not None:
            break
        action = bot.choose(game)
        assert action in legal_actions(game)
        seen.add(game.phase)
        from catan.actions import apply

        apply(game, action)
    assert len(seen) > 3


def test_the_checkpoint_is_loaded_once_per_process_not_once_per_game(checkpoint):
    path, board = checkpoint
    first = network_bot(path, board)
    second = network_bot(path, board)
    assert first is not second
    assert first.policy is second.policy


def test_a_checkpoint_dropped_in_with_the_same_name_is_not_served_stale(
    checkpoint, tmp_path
):
    """The whole point of catan-web's models/ directory: replacing a file by
    name must not keep serving the old in-memory session — unlike
    dev-catan's immutable runs/*.pt, this repo's checkpoints are expected to
    change underneath a running server."""
    path, board = checkpoint
    live = tmp_path / "live.onnx"
    live.write_bytes(Path(path).read_bytes())

    first = load(live, board.topology)
    live.write_bytes(Path(path).read_bytes())  # rewritten, same bytes, new mtime
    second = load(live, board.topology)
    assert first.policy is not second.policy


def test_the_offer_budget_comes_from_the_checkpoint_unless_overridden(checkpoint):
    path, board = checkpoint
    assert network_bot(path, board).max_offers == 3
    assert network_bot(path, board, max_offers=8).max_offers == 8


def test_the_budget_is_honoured_exactly_as_the_search_bot_honours_it(checkpoint):
    path, board = checkpoint
    bot = network_bot(path, board)
    rng = random.Random(11)
    game = start(board, 4, rng)

    from catan.actions import apply

    for _ in range(600):
        if game.won_by is not None:
            break
        allowed = within_offer_budget(game, legal_actions(game), 3)
        action = bot.choose(game)
        assert action in allowed
        apply(game, action)


def test_a_network_entrant_can_play_a_whole_tournament(checkpoint):
    path, board = checkpoint
    lineup = [
        Entrant("network", kind="network", weights=path),
        Entrant("network2", kind="network", weights=path),
        Entrant("random", kind="random"),
        Entrant("random2", kind="random"),
    ]
    result = compete(lineup, 4, seed=1, action_cap=2000)
    assert result.games == 4
    assert sum(s.wins for s in result.standings) + result.unfinished == 4


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(checkpoint):
    path, _ = checkpoint
    board3 = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="trained for 4 players"):
        network_bot(path, board3).choose(start(board3, 3, random.Random(0)))


def test_scoring_is_greedy_so_a_position_answers_the_same_way_twice(checkpoint):
    path, board = checkpoint
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(5))
    rng = random.Random(5)
    for _ in range(40):
        step_randomly(game, rng)
    assert to_move(game) is not None
    assert bot.choose(game) == bot.choose(game)


def test_the_search_reads_the_value_head_in_board_seat_order(checkpoint):
    from catan.encoding import encode
    from catan.onnxbot import network_evaluator

    path, board = checkpoint
    evaluator = network_evaluator(path, board)
    game = start(board, 4, random.Random(2))
    for _ in range(30):
        step_randomly(game, random.Random(2))

    for seat in range(4):
        rotated = evaluator.policy.values([encode(game, seat)])[0]
        vector = evaluator.evaluate_game(game, seat)
        assert vector[seat] == pytest.approx(rotated[0])
        for i, score in enumerate(rotated.tolist()):
            assert vector[(seat + i) % 4] == pytest.approx(score)


def test_a_search_over_a_learned_prior_plays_a_legal_action(checkpoint):
    from catan.onnxbot import searcher

    path, board = checkpoint
    search = searcher(path, board, simulations=16, wave=4, rng=random.Random(0))
    game = start(board, 4, random.Random(2))
    for _ in range(20):
        action = search.choose(game)
        assert action in set(legal_actions(game))
        from catan.actions import apply

        apply(game, action)


def test_the_prior_covers_every_offer_rather_than_one_arbitrary_one(checkpoint):
    from catan.actions import ActionType
    from catan.mcts import Leaf
    from catan.onnxbot import searcher

    path, board = checkpoint
    search = searcher(path, board, rng=random.Random(0))
    rng = random.Random(11)
    game = start(board, 4, rng)
    for _ in range(400):
        options = within_offer_budget(game, legal_actions(game), 3)
        offers = [o for o in options if o.type is ActionType.PROPOSE_TRADE]
        if len(offers) > 1:
            break
        step_randomly(game, rng)
    else:
        pytest.skip("no position offering more than one trade turned up")

    seat = to_move(game)
    (prior, _), = search.evaluator.evaluate([Leaf(game, seat, tuple(options))])
    weights = [prior[options.index(offer)] for offer in offers]
    assert all(w > 0 for w in weights)


def test_a_network_entrant_needs_a_path_rather_than_weights():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="checkpoint path"):
        spawn(Entrant("bogus", kind="network", weights=[1.0]), board, random.Random(0))
