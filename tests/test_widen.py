"""`catan.widen`: a wider net that computes the same function, until asked not to."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import encode, static_graph  # noqa: E402
from catan.game import is_over, start  # noqa: E402
from catan.model import CatanNet, ModelConfig, collate  # noqa: E402
from catan.play import step_randomly  # noqa: E402
from catan.widen import compare, widen_checkpoint, widen_state_dict  # noqa: E402


def a_net(width: int, seed: int = 0, **kwargs) -> CatanNet:
    torch.manual_seed(seed)
    rng = random.Random(0)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    net = CatanNet(space_for(game), static_graph(board.topology), 4, ModelConfig(width=width, **kwargs))
    # Orthogonal init leaves the heads near zero; scale everything up so a
    # wiring mistake shows as a large delta rather than a small one.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return net.eval()


def observations(count: int = 12) -> list:
    out = []
    for i in range(count):
        rng = random.Random(100 + i)
        board = random_base_board(rng)
        game = start(board, 4, rng)
        for _ in range(50 + 10 * i):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.append(encode(game))
    return out


def widen(narrow: CatanNet, width: int, **kwargs) -> CatanNet:
    wide = a_net(width, seed=99, **{k: v for k, v in kwargs.items() if k not in ("noise", "seed")})
    wide.load_state_dict(
        widen_state_dict(
            narrow.state_dict(), wide.state_dict(), narrow.config.width, width,
            noise=kwargs.get("noise", 0.0), seed=kwargs.get("seed", 0),
        ),
        strict=True,
    )
    return wide.eval()


@pytest.mark.parametrize("old,new", [(16, 32), (16, 24), (8, 40)])
def test_widening_preserves_the_function_on_both_forward_paths(old, new):
    narrow = a_net(old)
    wide = widen(narrow, new)
    report = compare(narrow, wide, observations())
    assert report["max_rel_logit_delta_reference"] < 1e-5
    assert report["max_rel_value_delta_reference"] < 1e-5
    assert report["max_rel_logit_delta_fused"] < 1e-5
    assert report["max_rel_value_delta_fused"] < 1e-5


@pytest.mark.parametrize("shape", ["mlp", "pooled", "attn", "quantile"])
def test_every_head_shape_widens_exactly(shape):
    kwargs = {"value_head": shape} if shape != "mlp" else {"value_head": "mlp", "policy_head": "mlp"}
    narrow = a_net(16, **kwargs)
    wide = widen(narrow, 32, **kwargs)
    report = compare(narrow, wide, observations(6))
    assert report["max_rel_logit_delta_reference"] < 1e-5
    assert report["max_rel_value_delta_reference"] < 1e-5


def test_the_wide_net_has_exactly_a_fresh_wide_nets_parameters():
    narrow = a_net(16)
    wide = widen(narrow, 32)
    fresh = a_net(32)
    assert sum(p.numel() for p in wide.parameters()) == sum(p.numel() for p in fresh.parameters())
    assert set(wide.state_dict()) == set(fresh.state_dict())


def test_the_first_old_units_are_the_narrow_net():
    narrow = a_net(16)
    wide = widen(narrow, 32)
    assert torch.equal(wide.embed_hex.weight[:16], narrow.embed_hex.weight)
    assert torch.equal(wide.embed_hex.weight[16:], narrow.embed_hex.weight)
    # A consumer reads each copy at half weight.
    head = narrow.heads["vertices"].weight
    assert torch.allclose(wide.heads["vertices"].weight[:, :16], head / 2)
    assert torch.allclose(wide.heads["vertices"].weight[:, 16:], head / 2)


def test_noise_separates_the_copies_and_moves_the_function_only_slightly():
    narrow = a_net(16)
    wide = widen(narrow, 32, noise=0.01, seed=3)
    assert not torch.equal(wide.embed_hex.weight[16:], wide.embed_hex.weight[:16])
    # Sources are untouched; only the copies carry noise.
    assert torch.equal(wide.embed_hex.weight[:16], narrow.embed_hex.weight)
    report = compare(narrow, wide, observations())
    assert 0.0 < report["mean_policy_kl_reference"] < 1e-2
    assert report["max_abs_logit_delta_reference"] > 0.0


def test_the_noise_is_seeded():
    narrow = a_net(16)
    a = widen(narrow, 32, noise=0.01, seed=7)
    b = widen(narrow, 32, noise=0.01, seed=7)
    c = widen(narrow, 32, noise=0.01, seed=8)
    assert torch.equal(a.embed_hex.weight, b.embed_hex.weight)
    assert not torch.equal(a.embed_hex.weight, c.embed_hex.weight)


def test_narrowing_is_refused():
    narrow = a_net(16)
    wide = a_net(8)
    with pytest.raises(ValueError):
        widen_state_dict(narrow.state_dict(), wide.state_dict(), 16, 8)


def test_a_checkpoint_round_trips_with_the_resume_contract(tmp_path):
    narrow = a_net(16)
    optimiser = torch.optim.Adam(narrow.parameters(), lr=3e-4, eps=1e-5)
    source = tmp_path / "latest.pt"
    torch.save(
        {
            "iteration": 805,
            "games_started": 114850,
            "net": narrow.state_dict(),
            "optimiser": optimiser.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "args": {"width": 16, "rounds": 2, "learning_rate": 3e-4, "adam_eps": 1e-5, "seed": 0},
            "config": {},
        },
        source,
    )
    out, report = widen_checkpoint(source, 32, noise=0.01, seed=0, observations=8)
    assert out["iteration"] == 805 and out["games_started"] == 114850
    assert out["args"]["width"] == 32 and out["config"]["width"] == 32
    assert out["widen"]["width_from"] == 16 and out["widen"]["noise"] == 0.01
    assert out["widen"]["source_sha256"]
    assert report["parameters_after"] > report["parameters_before"]
    # The optimiser state is fresh and loads onto a width-32 net the way
    # `catan.train --resume` loads it.
    wide = a_net(32)
    wide.load_state_dict(out["net"], strict=True)
    fresh = torch.optim.Adam(wide.parameters(), lr=1.0)
    fresh.load_state_dict(out["optimiser"])
    assert fresh.param_groups[0]["lr"] == 3e-4 and fresh.param_groups[0]["eps"] == 1e-5
    assert fresh.state == {}
    # And it is the narrow function, up to the noise asked for.
    assert report["mean_policy_kl_reference"] < 1e-2
