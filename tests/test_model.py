from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import encode, encode_batch, static_graph  # noqa: E402
from catan.game import start  # noqa: E402
from catan.model import (  # noqa: E402
    POLICY_HEADS,
    VALUE_HEADS,
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


def test_pack_reuses_the_batch_encoders_contiguous_buffer():
    games = [a_game(seed=seed, steps=60 + seed) for seed in range(8)]
    observations = encode_batch(games, [game.current_player for game in games])
    layout = packing(static_graph(games[0].state.board.topology), players=4)

    packed = pack(layout, observations)

    assert observations[0]._packed is not None
    assert packed.data_ptr() == observations[0]._packed.ctypes.data
    canonical = [
        encode(game, game.current_player)
        for game in games
    ]
    for got, want in zip(unpack(layout, packed), collate(canonical), strict=True):
        assert torch.equal(got, want)


def test_a_packed_batch_feeds_the_net_unchanged():
    """One float32 ULP apart, not bit-identical, and that is expected.

    The round-trip test above already proves the inputs match bit for bit, so
    any difference here is the matmul taking a different path over a strided
    view. `allclose`'s default `atol` of 1e-8 cannot absorb a last-bit
    difference on a logit that happens to sit near zero.
    """
    game, space, net = a_net()
    obs = observations(3)
    layout = packing(static_graph(game.state.board.topology), 4)

    assert torch.allclose(
        net(*unpack(layout, pack(layout, obs))).logits,
        net(*collate(obs)).logits,
        atol=1e-6,
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


# Written out rather than derived from a freshly built net, because a net is
# exactly the thing under test: generating the expectation from the code would
# make this pass no matter what the code did. Every checkpoint under `runs/`
# holds these names and no others, so this list is what makes a shape flag safe
# to add.
DEFAULT_KEYS = frozenset(
    {
        "embed_hex.weight",
        "embed_hex.bias",
        "embed_vertex.weight",
        "embed_vertex.bias",
        "embed_edge.weight",
        "embed_edge.bias",
        "embed_global.weight",
        "embed_global.bias",
        "trade_give.weight",
        "trade_give.bias",
        "trade_want.weight",
        "trade_want.bias",
        "value.weight",
        "value.bias",
        # Buffers are in the `state_dict` too, and a checkpoint carries them.
        "hv_hex",
        "hv_vertex",
        "ve_vertex",
        "ve_edge",
        "n_vertex_from_hex",
        "n_hex_from_vertex",
        "n_edge_from_vertex",
        "n_vertex_from_edge",
        "flat_gather",
    }
    | {
        f"heads.{source}.{parameter}"
        for source in ("hexes", "vertices", "edges", "globals")
        for parameter in ("weight", "bias")
    }
    | {
        f"rounds.{round_}.{block}.{layer}.{parameter}"
        for round_ in range(2)
        for block in ("hex", "vertex", "edge", "globe")
        for layer in (0, 2)
        for parameter in ("weight", "bias")
    }
)


def test_the_default_config_writes_exactly_the_state_dict_keys_it_always_has():
    """The one test that protects every checkpoint already on disk.

    `ModelConfig` gained `value_head` and `policy_head`, and the obvious way to
    implement either — wrap the head in a module and dispatch on its type —
    renames `value.weight` to `value.out.weight` and makes every run recorded so
    far unloadable. The default has to *be* the old network, not a new network
    that happens to compute the same function.
    """
    _, _, net = a_net()

    assert set(net.state_dict()) == DEFAULT_KEYS


def test_the_default_config_is_the_shape_every_run_on_record_used():
    assert ModelConfig().value_head == "linear"
    assert ModelConfig().policy_head == "linear"


@pytest.mark.parametrize("shape", VALUE_HEADS)
def test_every_value_head_shape_predicts_one_number_per_seat(shape):
    _, space, net = a_net(value_head=shape)

    out = net(*collate(observations(3)))

    assert out.value.shape == (3, 4)
    assert out.logits.shape == (3, space.size)
    assert torch.isfinite(out.value).all()


@pytest.mark.parametrize("shape", POLICY_HEADS)
def test_every_policy_head_shape_fills_the_same_flat_action_space(shape):
    """The scatter is the contract: a deeper head may not move a single slot."""
    _, space, net = a_net(policy_head=shape)

    out = net(*collate(observations(3)))

    assert out.logits.shape == (3, space.size)
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("shape", VALUE_HEADS)
def test_a_backward_pass_reaches_every_parameter_of_every_value_head(shape):
    """Catches an attention query, or a pooled input, wired in but never read."""
    _, _, net = a_net(value_head=shape)

    out = net(*collate(observations(2)))
    (out.logits.sum() + out.value.sum() + out.give.sum() + out.want.sum()).backward()

    unused = [
        name for name, p in net.named_parameters() if p.grad is None or not p.grad.any()
    ]
    assert unused == []


@pytest.mark.parametrize("shape", ["pooled", "mlp_pooled", "attn"])
def test_detaching_the_value_cuts_every_tensor_the_head_reads_not_only_the_global(
    shape,
):
    """`detach_value` cuts the head off the trunk; it must not cut off the head.

    The flag detaches every tensor the head reads, not only `g`, or a pooled
    head would keep the exact gradient path the flag exists to remove. The head's
    own parameters still have to train, which is the whole point of detaching
    rather than freezing.
    """
    _, _, net = a_net(value_head=shape)
    net.detach_value = True

    net(*collate(observations(2))).value.sum().backward()

    head = [p for name, p in net.named_parameters() if name.startswith("value")]
    assert any(p.grad is not None and p.grad.any() for p in head)
    trunk = net.embed_vertex.weight.grad
    assert trunk is None or not trunk.any()


def test_a_deeper_head_only_adds_keys_under_its_own_name():
    """A shape change may not disturb the trunk, or an ablation is confounded."""
    _, _, deep = a_net(value_head="mlp", policy_head="mlp")
    keys = set(deep.state_dict())

    trunk = {name for name in DEFAULT_KEYS if not name.startswith(("value", "heads."))}
    assert trunk <= keys
    assert "value.0.weight" in keys and "value.2.weight" in keys
    assert "value.weight" not in keys
    assert "heads.vertices.0.weight" in keys and "heads.vertices.2.weight" in keys


def test_the_attention_head_carries_its_query_and_the_pooled_head_does_not():
    _, _, attn = a_net(value_head="attn")
    _, _, pooled = a_net(value_head="pooled")

    assert "value_query.weight" in attn.state_dict()
    assert "value_query.weight" not in pooled.state_dict()
    # `g` plus a pooled vertex vector, so twice the width in.
    assert attn.value.weight.shape == (4, 2 * 64)
    # `g` plus a max-pool of each of the three node types.
    assert pooled.value.weight.shape == (4, 4 * 64)


def test_the_output_layer_of_a_deep_head_is_the_one_that_gets_the_small_gain():
    """`_initialise`'s reasoning, restated as a property.

    A hidden layer inside a head feeds a SiLU and is trunk by every property
    that method cares about, so it keeps gain √2. Only the layer emitting a
    logit is held near zero — giving the whole `Sequential` gain 0.01 would
    start a two-layer head weaker than the one-layer head it is being compared
    against, which would decide the ablation by initialisation.
    """
    _, _, net = a_net(value_head="mlp", policy_head="mlp")

    hidden = net.heads["vertices"][0].weight.std().item()
    output = net.heads["vertices"][2].weight.std().item()
    assert output < hidden / 10

    # The value head predicts at its target's scale, so its output stays at 1.0.
    assert net.value[2].weight.std().item() > net.value[0].weight.std().item() / 10


@pytest.mark.parametrize(
    "field,value",
    [("value_head", "attention"), ("policy_head", "pooled"), ("value_head", "")],
)
def test_an_unknown_head_shape_is_refused_before_a_run_starts(field, value):
    """A typo that silently fell back to the default would be discovered as an
    ablation that read flat, which is the most expensive way to find it."""
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_a_shaped_head_costs_parameters_but_keeps_the_model_small():
    _, _, plain = a_net()
    _, _, shaped = a_net(value_head="mlp_pooled", policy_head="mlp")

    plain_total = sum(p.numel() for p in plain.parameters())
    shaped_total = sum(p.numel() for p in shaped.parameters())
    assert shaped_total > plain_total
    assert shaped_total < 500_000, shaped_total


def test_collate_stacks_in_order():
    obs = observations(3)
    hexes, vertices, edges, globals_ = collate(obs)

    assert hexes.shape[0] == 3
    assert np.array_equal(vertices[1].numpy(), obs[1].vertices)
    assert np.array_equal(globals_[2].numpy(), obs[2].globals)
    assert edges.dtype == torch.float32
