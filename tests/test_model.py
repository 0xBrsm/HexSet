from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import encode, static_graph  # noqa: E402
from catan.game import start  # noqa: E402
from catan.model import (  # noqa: E402
    CatanNet,
    ModelConfig,
    collate,
    pack,
    packing,
    unpack,
)
from catan.play import step_randomly  # noqa: E402
from catan.readout import scatter_logits  # noqa: E402


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        step_randomly(game, rng)
    return game


def a_net(players: int = 4, seed: int = 0, **kwargs):
    torch.manual_seed(seed)
    game = a_game(players=players)
    space = space_for(game)
    net = CatanNet(space, static_graph(game.state.board.topology), players, ModelConfig(**kwargs))
    return game, space, net


def observations(count: int, players: int = 4):
    return [encode(a_game(players=players, seed=s, steps=60 + s)) for s in range(count)]


@pytest.mark.parametrize("players", [2, 3, 4])
def test_output_shapes(players):
    game, space, net = a_net(players=players)
    batch = collate(observations(3, players=players))

    out = net(*batch)

    assert out.logits.shape == (3, space.size)
    assert out.value.shape == (3, players)
    assert out.give.shape == (3, 5)
    assert out.want.shape == (3, 5)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_packing_round_trips_to_the_same_batch_as_collate(players):
    """`pack` writes through reshaped views, which numpy may silently copy.

    If a slice ever reshaped to a copy instead of a view, `pack` would stack
    into a temporary and hand back an uninitialised buffer — garbage inputs,
    no error. This compares it against `collate`, which builds the same batch
    the obvious way.
    """
    obs = observations(3, players=players)
    layout = packing(static_graph(a_game(players=players).state.board.topology), players)
    packed = pack(layout, obs)

    assert packed.shape == (3, layout.width)
    for got, want in zip(unpack(layout, packed), collate(obs), strict=True):
        assert torch.equal(got, want)


def test_a_packed_batch_feeds_the_net_unchanged():
    game, space, net = a_net()
    obs = observations(3)
    layout = packing(static_graph(game.state.board.topology), 4)

    assert torch.allclose(
        net(*unpack(layout, pack(layout, obs))).logits, net(*collate(obs)).logits
    )


def test_the_flat_permutation_agrees_with_the_numpy_scatter():
    """Ties the model's one-gather shortcut to the tested reference in readout.

    The model concatenates its heads and permutes once; `scatter_logits` writes
    each head into place. They must agree or the policy is trained on the wrong
    actions, which is the failure this whole mapping exists to prevent.
    """
    _, _, net = a_net()
    rng = np.random.default_rng(0)

    emitted = {
        head.source: rng.standard_normal((head.num_nodes, head.width)).astype(np.float32)
        for head in net.readout.heads
    }
    expected = scatter_logits(net.readout, emitted)

    stacked = torch.cat(
        [torch.from_numpy(emitted[head.source]).reshape(1, -1) for head in net.readout.heads],
        dim=1,
    )
    actual = stacked.index_select(1, net.flat_gather)

    assert np.allclose(actual.numpy()[0], expected)


def test_each_row_of_a_batch_is_independent():
    """Catches message passing that leaks across the batch dimension."""
    _, _, net = a_net()
    net.eval()
    obs = observations(4)

    with torch.no_grad():
        together = net(*collate(obs))
        apart = [net(*collate([one])) for one in obs]

    for i, single in enumerate(apart):
        assert torch.allclose(together.logits[i], single.logits[0], atol=1e-5)
        assert torch.allclose(together.value[i], single.value[0], atol=1e-5)


def test_a_backward_pass_reaches_every_parameter():
    _, _, net = a_net()
    out = net(*collate(observations(2)))

    (out.logits.sum() + out.value.sum() + out.give.sum() + out.want.sum()).backward()

    unused = [name for name, p in net.named_parameters() if p.grad is None or not p.grad.any()]
    assert unused == []


def test_outputs_are_finite():
    _, _, net = a_net()
    with torch.no_grad():
        out = net(*collate(observations(4)))
    for name in ("logits", "value", "give", "want"):
        assert torch.isfinite(getattr(out, name)).all(), name


def test_the_model_stays_small():
    """Parameter economy is a project goal, not an accident."""
    _, _, net = a_net()
    total = sum(p.numel() for p in net.parameters())
    assert total < 500_000, total


def test_a_deeper_model_has_more_parameters_but_the_same_interface():
    _, space, shallow = a_net(rounds=1)
    _, _, deep = a_net(rounds=3)

    assert sum(p.numel() for p in deep.parameters()) > sum(
        p.numel() for p in shallow.parameters()
    )
    for net in (shallow, deep):
        assert net(*collate(observations(1))).logits.shape == (1, space.size)


def test_collate_stacks_in_order():
    obs = observations(3)
    hexes, vertices, edges, globals_ = collate(obs)

    assert hexes.shape[0] == 3
    assert np.array_equal(vertices[1].numpy(), obs[1].vertices)
    assert np.array_equal(globals_[2].numpy(), obs[2].globals)
    assert edges.dtype == torch.float32
