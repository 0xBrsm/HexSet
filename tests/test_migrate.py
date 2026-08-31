# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.migrate`: a pre-offer-observation checkpoint onto the wider
globals, function preserved exactly.

A real pre-change checkpoint cannot exist under this code (`global_features`
is derived, not stored), so the tests manufacture one the way the migration
undoes: slice a fresh net's `embed_global` down by the 18 offer columns.
Migrating that slice back must reproduce the sliced net's function exactly
on positions with and without a standing offer.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from hexset.actions import space_for
from hexset.board.board import random_base_board
from hexset.encoding import encode, global_features, static_graph
from hexset.game import is_over, start
from hexset.migrate import compare, migrate_state_dict
from hexset.model import HexNet, ModelConfig, collate
from hexset.play import step_randomly

OFFER_FEATURES = 18  # 2 * NUM_RESOURCES + 2 * players at four players


def _net(seed: int = 0) -> HexNet:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    return HexNet(space_for(game), static_graph(board.topology), 4, ModelConfig())


def _observations(count: int = 32, seed: int = 1000) -> list:
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        game = start(random_base_board(rng), 4, rng)
        for _ in range(40 + (i % 80)):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.append(encode(game))
    return out


def _pre_change_state(net: HexNet) -> dict:
    state = {k: v.clone() for k, v in net.state_dict().items()}
    state["embed_global.weight"] = state["embed_global.weight"][:, :-OFFER_FEATURES].clone()
    return state


def test_migration_restores_the_shapes_and_zeroes_the_new_columns():
    net = _net()
    template = net.state_dict()
    old = _pre_change_state(net)

    migrated = migrate_state_dict(old, template)
    weight = migrated["embed_global.weight"]
    assert weight.shape == template["embed_global.weight"].shape
    assert torch.equal(weight[:, :-OFFER_FEATURES], old["embed_global.weight"])
    assert weight[:, -OFFER_FEATURES:].abs().max() == 0.0
    for key, tensor in migrated.items():
        if key != "embed_global.weight":
            assert torch.equal(tensor, old[key])


def test_migration_preserves_the_function_on_offer_carrying_positions():
    net = _net()
    old_in = global_features(4) - OFFER_FEATURES
    migrated = migrate_state_dict(_pre_change_state(net), net.state_dict())
    net.load_state_dict(migrated, strict=True)
    net.eval()

    report = compare(net, old_in, _observations())
    assert report["max_abs_embed_delta"] == 0.0
    for tag in ("reference", "fused"):
        assert report[f"max_abs_logit_delta_{tag}"] == 0.0
        assert report[f"max_abs_value_delta_{tag}"] == 0.0


def test_only_embed_global_may_change_shape():
    net = _net()
    template = net.state_dict()
    bad = {k: v.clone() for k, v in template.items()}
    bad["embed_hex.weight"] = bad["embed_hex.weight"][:, :-1].clone()
    with pytest.raises(ValueError, match="embed_global"):
        migrate_state_dict(bad, template)


def test_a_matching_checkpoint_passes_through_untouched():
    net = _net()
    template = net.state_dict()
    out = migrate_state_dict({k: v.clone() for k, v in template.items()}, template)
    for key in template:
        assert torch.equal(out[key], template[key])
