from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import legal_actions, space_for, within_offer_budget  # noqa: E402
from catan.arena import Entrant, compete, spawn  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import static_graph  # noqa: E402
from catan.game import start, to_move  # noqa: E402
from catan.model import CatanNet, ModelConfig  # noqa: E402
from catan.netbot import load, network_bot  # noqa: E402
from catan.play import step_randomly  # noqa: E402


def a_checkpoint(path, *, players: int = 4, max_offers: int | None = 3, seed: int = 0):
    """A checkpoint in the shape `catan.train.save` writes, tiny enough to load."""
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = CatanNet(
        space_for(game), graph, players, ModelConfig(width=16, rounds=1)
    )
    torch.save(
        {
            "iteration": 7,
            "net": net.state_dict(),
            "args": {
                "players": players,
                "width": 16,
                "rounds": 1,
                "max_offers": max_offers,
            },
        },
        path,
    )
    return board


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    board = a_checkpoint(path)
    load.cache_clear()
    yield str(path), board
    load.cache_clear()


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
    """`arena.spawn` runs per game per worker; a `torch.load` there would be
    most of what the duel measured."""
    path, board = checkpoint
    first = network_bot(path, board)
    second = network_bot(path, board)
    assert first is not second
    assert first.policy is second.policy
    assert load.cache_info().hits == 1


def test_the_offer_budget_comes_from_the_checkpoint_unless_overridden(checkpoint):
    path, board = checkpoint
    assert network_bot(path, board).max_offers == 3
    assert network_bot(path, board, max_offers=8).max_offers == 8


def test_the_budget_is_honoured_exactly_as_the_search_bot_honours_it(checkpoint):
    """The policy never sees a slot the budget forbids, in scoring as in training."""
    path, board = checkpoint
    bot = network_bot(path, board)
    rng = random.Random(11)
    game = start(board, 4, rng)

    spent = 0
    for _ in range(600):
        if game.won_by is not None:
            break
        allowed = within_offer_budget(game, legal_actions(game), 3)
        action = bot.choose(game)
        assert action in allowed
        spent += game.offers_made >= 3
        from catan.actions import apply

        apply(game, action)
    # Vacuous unless the budget actually bound somewhere in the game.
    assert spent > 0


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


def test_a_network_entrant_needs_a_path_rather_than_weights():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="checkpoint path"):
        spawn(Entrant("bogus", kind="network", weights=[1.0]), board, random.Random(0))


def test_a_checkpoint_refuses_a_table_it_was_not_trained_for(tmp_path):
    path = tmp_path / "three.pt"
    board = a_checkpoint(path, players=3)
    load.cache_clear()
    bot = network_bot(str(path), board)
    game = start(board, 4, random.Random(0))
    with pytest.raises(ValueError, match="trained for 3 players"):
        bot.choose(game)
    load.cache_clear()


def test_scoring_is_greedy_so_a_position_answers_the_same_way_twice(checkpoint):
    """A sampled policy is the behaviour distribution PPO needed, not the
    policy worth scoring, and a duel of one is not a measurement."""
    path, board = checkpoint
    bot = network_bot(path, board)
    game = start(board, 4, random.Random(5))
    rng = random.Random(5)
    for _ in range(40):
        step_randomly(game, rng)
    assert to_move(game) is not None
    assert bot.choose(game) == bot.choose(game)
