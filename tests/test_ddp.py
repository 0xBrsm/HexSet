# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.ddp import UpdateCrew, UpdateSpec  # noqa: E402
from hexset.ppo import PPOConfig, assemble, update  # noqa: E402
from hexset.selfplay import Collector  # noqa: E402
from hexset.train import build  # noqa: E402


def _fixture():
    args = SimpleNamespace(
        seed=5, players=4, width=8, rounds=1, device="cpu", learning_rate=3e-4
    )
    policy, optimiser, _ = build(args)
    config = PPOConfig(minibatch=256, epochs=2)
    collector = Collector(
        policy, lanes=4, players=4, seed=5, action_cap=500, max_offers=3, deal=4
    )
    batch = assemble(collector.drain(), policy.layout, config)
    return args, policy, optimiser, config, batch


def test_the_sharded_update_matches_the_single_device_update():
    """The whole module's claim: same schedule, same math, same result.

    Both paths start from identical weights and optimizer state and walk the
    same seeded minibatch order. Only float summation order differs — the crew
    sums shard gradients weighted by row count — so the bar is allclose, not
    equality. The tolerance is set by Adam, not by the gradients: dividing by
    sqrt(v) makes a near-zero-gradient parameter's step sign-sensitive, so a
    1e-8 reduction-order wobble can become a full lr-sized (3e-4) step on a
    few parameters within a couple of epochs. Measured divergence ~1e-4; a
    genuinely different update moves parameters by ~5e-3, an order above the
    bar, which is what keeps the test non-vacuous.
    """
    args, policy, optimiser, config, batch = _fixture()
    weights = copy.deepcopy(policy.net.state_dict())
    opt_state = copy.deepcopy(optimiser.state_dict())
    initial = (
        torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach().clone()
    )

    single = update(
        policy, optimiser, batch, config, generator=torch.Generator().manual_seed(3)
    )
    reference = torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach()

    policy.net.load_state_dict(weights)
    optimiser.load_state_dict(opt_state)
    crew = UpdateCrew([UpdateSpec(seed=5, players=4, width=8, rounds=1)] * 2)
    try:
        sharded = crew.update(
            policy,
            optimiser,
            batch,
            config,
            generator=torch.Generator().manual_seed(3),
        )
    finally:
        crew.close()
    result = torch.nn.utils.parameters_to_vector(policy.net.parameters()).detach()

    assert torch.allclose(reference, result, atol=3e-4, rtol=1e-3), (
        f"largest divergence {float((reference - result).abs().max())}"
    )
    assert sharded.positions == single.positions
    assert abs(sharded.policy_loss - single.policy_loss) < 1e-4
    assert abs(sharded.value_loss - single.value_loss) < 1e-4
    assert abs(sharded.entropy - single.entropy) < 1e-4
    assert abs(sharded.explained_variance - single.explained_variance) < 1e-3
    # Anti-vacuity: the update actually moved the weights, by far more than
    # the tolerance above, so "same update" and "no update" stay distinguishable.
    assert float((result - initial).abs().max()) > 1e-3


def test_a_worker_with_no_rows_in_a_minibatch_is_harmless():
    # Eight workers over a small batch: some minibatches miss some shards
    # entirely, which must contribute nothing rather than hang or skew.
    args, policy, optimiser, config, batch = _fixture()
    crew = UpdateCrew([UpdateSpec(seed=5, players=4, width=8, rounds=1)] * 8)
    try:
        stats = crew.update(
            policy,
            optimiser,
            batch,
            config,
            generator=torch.Generator().manual_seed(4),
        )
    finally:
        crew.close()
    assert stats.positions == len(batch)
    assert stats.value_loss > 0.0
