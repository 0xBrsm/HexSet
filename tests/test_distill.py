from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import ActionType, space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.distill import (  # noqa: E402
    DistillConfig,
    assemble,
    losses,
    project,
    update,
)
from catan.encoding import static_graph  # noqa: E402
from catan.expert import SearchPolicy  # noqa: E402
from catan.game import start  # noqa: E402
from catan.mcts import Search, visit_policy  # noqa: E402
from catan.model import CatanNet, ModelConfig, packing  # noqa: E402
from catan.netbot import LeafEvaluator  # noqa: E402
from catan.policy import NetworkPolicy, pair_index  # noqa: E402
from catan.ppo import rotate  # noqa: E402
from catan.rewards import reward  # noqa: E402
from catan.selfplay import Collector  # noqa: E402


def a_setup(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    graph = static_graph(board.topology)
    space = space_for(game)
    layout = packing(graph, players)
    torch.manual_seed(seed)
    net = CatanNet(space, graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space, layout), space, layout


def searched_episodes(policy, space, games: int = 2, seed: int = 0):
    """Games played by a search over the network, so every transition has a target."""
    search = Search(
        LeafEvaluator(policy=policy, space=space),
        simulations=8,
        wave=4,
        rng=random.Random(seed),
    )
    expert = SearchPolicy(search, rng=random.Random(seed))
    return Collector(expert, lanes=2, seed=seed, action_cap=3000).collect(games)


def a_batch(policy, space, layout, config=None, games: int = 2):
    episodes = searched_episodes(policy, space, games=games)
    return assemble(episodes, space, layout, config or DistillConfig())


# --- the projection -------------------------------------------------------


def test_the_projection_is_a_distribution_over_slots():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout)
    totals = batch.slot_target.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_offers_sharing_the_trade_slot_sum_into_it_and_split_within_it():
    # Two different one-for-one offers land on one flat slot, so the slot must
    # carry both their visits and the offer row must carry their ratio. This is
    # the whole reason `Target` keeps its options.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    trade_slot = space.offsets[ActionType.PROPOSE_TRADE]
    exercised = 0
    for episode in episodes:
        for trajectory in episode.trajectories:
            for transition in trajectory:
                target = transition.aux
                proposals = [
                    (option, visit)
                    for option, visit in zip(target.options, target.visits)
                    if option.type is ActionType.PROPOSE_TRADE
                ]
                if len(proposals) < 2:
                    continue
                slots, offers, mass = project(target, space, 1.0)
                assert slots[trade_slot] == pytest.approx(mass, abs=1e-6)

                if mass == 0:
                    # Proposals were legal and the search visited none of them.
                    # The offer row stays empty and the mass gates it off, which
                    # is the documented reading rather than a uniform fallback.
                    assert not offers.any()
                    continue

                exercised += 1
                assert offers.sum() == pytest.approx(1.0, abs=1e-6)
                for option, visit in proposals:
                    if visit > 0:
                        assert offers[pair_index(option.give, option.want)] > 0
    assert exercised, "no position split visits across two proposals; untested"


def test_a_position_that_cannot_propose_carries_no_offer_mass():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout)
    trade_slot = space.offsets[ActionType.PROPOSE_TRADE]
    quiet = batch.slot_target[:, trade_slot] == 0
    assert quiet.any()
    assert torch.all(batch.trade_mass[quiet] == 0)
    assert torch.all(batch.offer_target[quiet] == 0)


def test_cooling_drains_the_trade_slot_because_its_mass_is_a_sum():
    """Temperature acts on options, so aggregate slots lose out as it falls.

    This is the consequence of the ordering the module argues for, and it is
    not a wart. The trade slot's mass is a sum over dozens of individually
    unpopular offers, while a settlement slot is one option. Sharpening before
    aggregation therefore drains the slot that was only ever winning on
    aggregate — which is the honest reading of what the search preferred, and
    the opposite of what sharpening the slot row afterwards would have said.
    """
    policy, space, layout = a_setup()
    warm = a_batch(policy, space, layout, DistillConfig(temperature=1.0))
    cold = a_batch(policy, space, layout, DistillConfig(temperature=0.25))
    trade_slot = space.offsets[ActionType.PROPOSE_TRADE]

    assert cold.slot_target.max(-1).values.mean() > warm.slot_target.max(-1).values.mean()

    spread = warm.slot_target[:, trade_slot] > 0
    assert spread.any()
    assert (
        cold.slot_target[spread, trade_slot].mean()
        < warm.slot_target[spread, trade_slot].mean()
    )


# --- the loss -------------------------------------------------------------


def test_the_factored_loss_equals_the_cross_entropy_over_options():
    """The identity the module rests on, checked against a direct sum.

    The policy never scores an option directly — it scores a slot and, for a
    proposal, a pair. So the target has to be projected onto that factorisation,
    and this is the check that the projection loses nothing: the two weighted
    cross-entropies must equal the plain cross-entropy over concrete options.
    """
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    config = DistillConfig()
    batch = assemble(episodes, space, layout, config)

    targets = [
        transition.aux
        for episode in episodes
        for trajectory in episode.trajectories
        for transition in trajectory
    ]

    rows = torch.arange(len(batch))
    slots, offers, _ = policy.distributions(batch.buffer, batch.mask, batch.pair)
    slot_loss, offer_loss = losses(slots, offers, batch, rows)
    factored = float((slot_loss + offer_loss).detach())
    slots, offers = slots.detach(), offers.detach()

    trade_slot = space.offsets[ActionType.PROPOSE_TRADE]
    direct = 0.0
    for row, target in enumerate(targets):
        weights = visit_policy(target.visits, config.temperature)
        for option, share in zip(target.options, weights):
            index = space.index(option)
            log_q = float(slots[row, index])
            if index == trade_slot:
                log_q += float(offers[row, pair_index(option.give, option.want)])
            direct -= share * log_q
    direct /= len(targets)

    assert factored == pytest.approx(direct, rel=1e-4)


def test_the_offer_term_is_weighted_by_the_mass_on_the_trade_slot():
    # Unweighted, the offer row would attract as much gradient at a position
    # that never proposes as at one that always does.
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout)
    rows = torch.arange(len(batch))
    slots, offers, _ = policy.distributions(batch.buffer, batch.mask, batch.pair)

    _, offer_loss = losses(slots, offers, batch, rows)
    silenced = batch.trade_mass.clone()
    batch.trade_mass = torch.zeros_like(silenced)
    _, muted = losses(slots, offers, batch, rows)

    assert float(muted.detach()) == pytest.approx(0.0, abs=1e-9)
    assert float(offer_loss.detach()) > 0


# --- the update -----------------------------------------------------------


def test_a_batch_refuses_transitions_that_carry_no_search_target():
    policy, space, layout = a_setup()
    episodes = Collector(policy, lanes=2, seed=0, action_cap=3000).collect(1)
    with pytest.raises(ValueError, match="search targets"):
        assemble(episodes, space, layout, DistillConfig())


def test_distilling_a_fixed_batch_moves_the_policy_toward_the_search():
    """The end-to-end property: repeated updates raise agreement and cut loss."""
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, games=3)
    optimiser = torch.optim.Adam(policy.net.parameters(), lr=1e-3)
    config = DistillConfig(epochs=1, minibatch=256)

    first = update(policy, optimiser, batch, config)
    for _ in range(20):
        last = update(policy, optimiser, batch, config)

    assert last.policy_loss < first.policy_loss
    assert last.agreement > first.agreement


def test_the_value_head_is_trained_on_the_terminal_outcome():
    # Not on the search's backed-up root value. Deliberate, and argued in the
    # module docstring: bootstrapping is a separate experiment.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    batch = assemble(episodes, space, layout, DistillConfig())

    row = 0
    for episode in episodes:
        payoffs = reward(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            expected = torch.tensor(rotate(payoffs, seat), dtype=torch.float32)
            for _ in trajectory:
                assert torch.allclose(batch.value_target[row], expected)
                row += 1
    assert row == len(batch)
