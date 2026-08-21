from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import legal_actions, space_for, within_offer_budget  # noqa: E402
from catan.arena import Entrant, compete, spawn  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import encode, static_graph  # noqa: E402
from catan.game import start, to_move  # noqa: E402
from catan.model import CatanNet, ModelConfig  # noqa: E402
from catan.netbot import load, network_bot  # noqa: E402
from catan.play import step_randomly  # noqa: E402


def a_checkpoint(
    path,
    *,
    players: int = 4,
    max_offers: int | None = 3,
    seed: int = 0,
    shape: dict | None = None,
):
    """A checkpoint in the shape `catan.train.save` writes, tiny enough to load.

    `shape` is omitted by default, which is the case that matters: it is what a
    checkpoint written before the head-shape flags existed looks like, and the
    loader has to keep rebuilding those as `"linear"` on both.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = CatanNet(
        space_for(game), graph, players, ModelConfig(width=16, rounds=1, **(shape or {}))
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
                **(shape or {}),
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


def test_a_checkpoint_that_predates_the_head_flags_still_loads_as_the_old_shape(
    checkpoint,
):
    """Every checkpoint under `runs/` has no head shape in its `args`.

    The default the loader reads has to be the shape those runs trained with, or
    `load_state_dict` fails on the keys — which is a working checkpoint made
    unloadable by a flag nobody passed.
    """
    path, board = checkpoint
    config = load(path, board.topology, "cpu").policy.net.config

    assert (config.value_head, config.policy_head) == ("linear", "linear")


@pytest.mark.parametrize(
    "shape",
    [
        {"value_head": "mlp"},
        {"value_head": "pooled"},
        {"value_head": "attn"},
        {"policy_head": "mlp"},
        {"value_head": "mlp_pooled", "policy_head": "mlp"},
    ],
)
def test_a_head_shape_round_trips_through_the_checkpoints_args_dict(tmp_path, shape):
    """A run's shape lives in its `args`, so a duel rebuilds what it scored.

    Without this the shape is known only to the launching command line, and a
    checkpoint from a shaped run would be rebuilt as `"linear"` and fail to
    load — with a key error a week after the run, not at the point of the bug.
    """
    path = tmp_path / "shaped.pt"
    board = a_checkpoint(path, shape=shape)
    load.cache_clear()

    loaded = load(str(path), board.topology, "cpu")

    config = loaded.policy.net.config
    for field, value in shape.items():
        assert getattr(config, field) == value
    position = encode(start(board, 4, random.Random(5)), 0)
    assert loaded.policy.values([position])[0].shape == (4,)
    load.cache_clear()


def test_loading_places_the_network_on_the_requested_device(checkpoint):
    path, board = checkpoint
    loaded = load(path, board.topology, "cpu")
    assert {parameter.device.type for parameter in loaded.policy.net.parameters()} == {
        "cpu"
    }


def test_loading_can_compile_the_network(checkpoint, monkeypatch):
    path, board = checkpoint
    calls = []

    def compile_(net, *, mode):
        calls.append((net, mode))
        return net

    monkeypatch.setattr(torch, "compile", compile_)
    loaded = load(path, board.topology, "cpu", "default")

    assert calls == [(loaded.policy.net, "default")]


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

    from catan.actions import apply

    for _ in range(600):
        if game.won_by is not None:
            break
        allowed = within_offer_budget(game, legal_actions(game), 3)
        action = bot.choose(game)
        assert action in allowed
        apply(game, action)

    # The guard this once used — `game.offers_made >= 3` somewhere in the game —
    # is unreachable rather than merely unlucky. `choose` is argmax, and a fresh
    # checkpoint's policy heads are initialised at gain 0.01, so its logits are
    # near-equal and the argmax simply never lands on the trade slot: measured,
    # this bot proposes 0 times in 600 steps and `offers_made` never leaves 0.
    # A budget of zero binds at every trading-legal position instead, so the
    # binding case is exercised by construction and counted rather than hoped for.
    strict = network_bot(path, board, max_offers=0)
    game = start(board, 4, random.Random(11))
    bound = 0
    for _ in range(600):
        if game.won_by is not None:
            break
        actions = legal_actions(game)
        allowed = within_offer_budget(game, actions, 0)
        bound += len(allowed) != len(actions)
        action = strict.choose(game)
        assert action in allowed
        apply(game, action)
    assert bound > 0, "the budget never removed an action, so nothing was tested"


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


def test_the_search_reads_the_value_head_in_board_seat_order(checkpoint):
    """The encoder rotates the mover to slot 0; `SearchBot` indexes by board
    seat. Getting this backwards would search fine and play nonsense."""
    from catan.encoding import encode
    from catan.netbot import network_evaluator

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


def test_a_search_over_learned_leaves_plays_the_whole_game(checkpoint):
    from catan.arena import Entrant, compete

    path, _ = checkpoint
    lineup = [
        Entrant("netsearch", kind="search", depth=2, width=6,
                evaluator="network", weights=path),
        Entrant("random0", kind="random"),
        Entrant("random1", kind="random"),
        Entrant("random2", kind="random"),
    ]
    result = compete(lineup, 4, seed=3, action_cap=3000)
    assert result.games == 4
    # Not "every game finishes": these are the weights a fresh network is born
    # with, and a search over a meaningless evaluation can stall on the action
    # cap. What is being tested is that the leaves are read at all.
    assert len(result.decided()) >= 1


def test_learned_leaves_inherit_the_checkpoints_offer_budget(checkpoint):
    from catan.arena import Entrant, spawn

    path, board = checkpoint
    bot = spawn(
        Entrant("netsearch", kind="search", evaluator="network", weights=path),
        board,
        random.Random(0),
    )
    assert bot.max_offers == 3
    stated = spawn(
        Entrant(
            "netsearch8", kind="search", evaluator="network", weights=path, max_offers=8
        ),
        board,
        random.Random(0),
    )
    assert stated.max_offers == 8


def test_a_handcrafted_evaluation_is_still_asked_for_the_state_alone(checkpoint):
    """The `evaluate_game` hook must not cost the baselines anything."""
    from catan.arena import PRESETS, spawn
    from catan.board.board import random_base_board

    board = random_base_board(random.Random(0))
    bot = spawn(PRESETS["search2"], board, random.Random(0))
    assert not hasattr(bot.evaluator, "evaluate_game")
    game = start(board, 4, random.Random(0))
    assert len(bot._leaf(game, 0)) == 4


def test_the_leaf_evaluator_hands_a_whole_wave_to_one_forward(checkpoint):
    from catan.actions import legal_actions
    from catan.mcts import Leaf
    from catan.netbot import searcher

    path, board = checkpoint
    search = searcher(path, board, simulations=8, wave=8, rng=random.Random(0))
    game = start(board, 4, random.Random(5))
    leaves = [Leaf(game, to_move(game), tuple(legal_actions(game)))] * 3
    scored = search.evaluator.evaluate(leaves)
    assert len(scored) == 3
    for prior, value in scored:
        assert prior.shape == (len(leaves[0].options),)
        assert prior.sum() == pytest.approx(1.0)
        assert len(value) == 4


def test_leaf_evaluation_can_pad_to_one_compiled_shape(checkpoint):
    from catan.actions import legal_actions
    from catan.mcts import Leaf
    from catan.netbot import searcher

    path, board = checkpoint
    search = searcher(path, board, inference_batch=8, rng=random.Random(0))
    game = start(board, 4, random.Random(5))
    leaves = [Leaf(game, to_move(game), tuple(legal_actions(game)))] * 3
    policy = search.evaluator.policy
    score = policy.score
    seen = []

    def recording_score(observations, masks, pairs):
        seen.append(len(observations))
        return score(observations, masks, pairs)

    policy.score = recording_score
    scored = search.evaluator.evaluate(leaves)

    assert seen == [8]
    assert len(scored) == 3


def test_the_prior_covers_every_offer_rather_than_one_arbitrary_one(checkpoint):
    """`PROPOSE_TRADE` is one slot and many options. If the slot's mass were not
    split by the offer heads, one offer would carry the policy's whole appetite
    for trading and the rest would be unreachable."""
    from catan.actions import ActionType
    from catan.mcts import Leaf
    from catan.netbot import searcher

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
    assert len(set(weights)) > 1


def test_a_search_over_a_learned_prior_plays_a_legal_action(checkpoint):
    from catan.netbot import searcher

    path, board = checkpoint
    search = searcher(path, board, simulations=16, wave=4, rng=random.Random(0))
    game = start(board, 4, random.Random(2))
    for _ in range(20):
        action = search.choose(game)
        assert action in set(legal_actions(game))
        from catan.actions import apply

        apply(game, action)


def test_a_scored_run_records_the_networks_value_beside_someone_elses_play():
    """`benchmarks.value_head --behaviour` measures the head off-policy.

    The whole point is that the action and the prediction come from different
    places, so the two things worth pinning are that the recorded value is the
    network's and that row order survives — a transposed batch would produce a
    plausible, wrong headline number rather than an error.
    """
    import numpy as np

    from benchmarks.value_head import Scored
    from catan.selfplay import Request

    class Values:
        def values(self, observations):
            return np.arange(len(observations) * 4, dtype=np.float64).reshape(-1, 4)

    class Always:
        def __init__(self, index):
            self.index = index

        def choose(self, game):
            return legal_actions(game)[self.index]

    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    options = tuple(legal_actions(game))
    requests = [
        Request(lane=i, seat=0, observation=object(), mask=np.zeros(1, dtype=bool),
                options=options, game=game)
        for i in range(3)
    ]

    choices = Scored(Always(2), Values()).act(requests)
    assert [c.value for c in choices] == [
        (0.0, 1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0, 7.0),
        (8.0, 9.0, 10.0, 11.0),
    ]
    assert all(c.action == options[2] for c in choices)
    assert Scored(Always(0), Values()).act([]) == []
