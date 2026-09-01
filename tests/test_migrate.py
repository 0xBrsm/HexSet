# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.migrate`: a pre-widening checkpoint onto the current globals,
function preserved exactly.

A real pre-change checkpoint cannot exist under this code (`global_features`
is derived, not stored), so the tests manufacture one the way the migration
undoes: slice a fresh net's `embed_global` down by however many trailing
columns a later change appended. Migrating that slice back must reproduce
the sliced net's function exactly on real positions.

Two widenings have landed so far, both globals-tail appends handled by the
same generic `migrate_state_dict` (it does not know or care how many
columns changed, only that `embed_global.weight` is the one tensor allowed
to grow): the live trade offer (68 -> 86 covers the ledger step) and,
composed, the pre-offer checkpoint straight onto today's width (50 -> 86,
naming a checkpoint the offer-observation change itself never wrote, since
this repo went straight from 50 to 68 to 86 without one sitting at 68 for
long, but which validates that `migrate_state_dict`'s zero-padding does not
assume it is only ever bridging one widening at a time).
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
LEDGER_FEATURES = 18  # (players - 1) * (NUM_RESOURCES + 1) at four players


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


def _pre_change_state(net: HexNet, cut: int = OFFER_FEATURES) -> dict:
    state = {k: v.clone() for k, v in net.state_dict().items()}
    state["embed_global.weight"] = state["embed_global.weight"][:, :-cut].clone()
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
    """68 -> 86: a checkpoint from just after the offer observation landed,
    straight onto the ledger's wider globals."""
    net = _net()
    old_in = global_features(4) - LEDGER_FEATURES
    migrated = migrate_state_dict(_pre_change_state(net, LEDGER_FEATURES), net.state_dict())
    net.load_state_dict(migrated, strict=True)
    net.eval()

    assert old_in == 68

    report = compare(net, old_in, _observations())
    assert report["max_abs_embed_delta"] == 0.0
    for tag in ("reference", "fused"):
        assert report[f"max_abs_logit_delta_{tag}"] == 0.0
        assert report[f"max_abs_value_delta_{tag}"] == 0.0


def test_migration_from_before_the_offer_observation_preserves_the_function():
    """50 -> 86: a checkpoint from before *either* widening, straight onto
    today's globals in one step -- `migrate_state_dict` zero-pads whatever
    the gap is, not just the 18-column case the other test exercises."""
    net = _net()
    cut = OFFER_FEATURES + LEDGER_FEATURES
    old_in = global_features(4) - cut
    migrated = migrate_state_dict(_pre_change_state(net, cut), net.state_dict())
    net.load_state_dict(migrated, strict=True)
    net.eval()

    assert old_in == 50

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
