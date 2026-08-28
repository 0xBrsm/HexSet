from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import ActionType, space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.distill import (  # noqa: E402
    DistillConfig,
    assemble,
    contested,
    losses,
    project,
    stake,
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
    for _ in range(100):
        last = update(policy, optimiser, batch, config)

    assert last.policy_loss < first.policy_loss
    # 100 updates rather than 20, because argmax agreement on a three-game batch
    # wobbles by about +/-0.02 while it climbs and 20 landed in a trough. The
    # measured curve from the near-uniform init (policy heads at gain 0.01, so
    # the initial argmax is arbitrary and its agreement is an artifact rather
    # than a baseline): 0.687 at step 1, 0.678 at 10, 0.669 at 21, 0.689 at 50,
    # 0.700 at 100, 0.715 at 200, against a policy_loss that falls monotonically
    # 0.972 -> 0.766 and a value_loss that falls 0.609 -> 0.018 throughout. The
    # trend is the property; 20 updates was measuring the wobble.
    assert last.agreement > first.agreement


def test_a_bootstrapped_target_reads_the_estimate_that_many_decisions_later():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    horizon = 3
    batch = assemble(
        episodes, space, layout, DistillConfig(value_horizon=horizon)
    )

    row = 0
    bootstrapped = 0
    for episode in episodes:
        payoffs = reward(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            terminal = torch.tensor(rotate(payoffs, seat), dtype=torch.float32)
            for index in range(len(trajectory)):
                ahead = index + horizon
                estimate = (
                    trajectory[ahead].value if ahead < len(trajectory) else ()
                )
                if estimate:
                    # Rotated, because the search stores board order and the
                    # target is in the seat's own frame.
                    expected = torch.tensor(
                        rotate(estimate, seat), dtype=torch.float32
                    )
                    bootstrapped += 1
                else:
                    expected = terminal
                assert torch.allclose(batch.value_target[row], expected)
                row += 1
    assert row == len(batch)
    assert bootstrapped > 0, "no transition actually bootstrapped"


def test_a_bootstrapped_target_is_not_the_terminal_one():
    """Otherwise the test above would pass on a horizon that did nothing."""
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    terminal = assemble(episodes, space, layout, DistillConfig())
    near = assemble(episodes, space, layout, DistillConfig(value_horizon=3))
    assert not torch.allclose(terminal.value_target, near.value_target)


def test_the_value_head_is_trained_on_the_terminal_outcome():
    # The default, and what AlphaZero does. `--value-horizon` bootstraps
    # instead; `benchmarks.floor` is the argument for doing so here.
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


def a_target(visits, prior=None, values=None):
    """A `Target` whose options are placeholders: `contested` reads only their
    count, so a real action space is not needed to pin the filter."""
    import numpy as np

    from catan.expert import Target

    return Target(
        options=tuple(range(len(visits))),
        visits=np.asarray(visits, dtype=np.float64),
        prior=None if prior is None else np.asarray(prior, dtype=np.float64),
        values=None if values is None else np.asarray(values, dtype=np.float64),
    )


def test_a_search_that_agrees_with_the_prior_is_not_contested():
    assert not contested(a_target([70.0, 20.0, 10.0], [0.7, 0.2, 0.1]))


def test_a_search_that_overrules_the_prior_is_contested():
    assert contested(a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1]))


def test_a_row_the_search_never_expanded_is_not_contested():
    # 24 of 72 rows in the first screen were exactly this: all-zero visits,
    # where `argmax` on ties invents a disagreement out of index order.
    assert not contested(a_target([0.0, 0.0, 0.0], [0.1, 0.7, 0.2]))


def test_a_corpus_without_recorded_priors_is_never_contested():
    # Keeps an old corpus readable: the filter is a no-op rather than an
    # empty batch.
    assert not contested(a_target([20.0, 70.0, 10.0], prior=None))


def test_the_stake_is_the_gap_between_what_the_search_picked_and_what_it_overruled():
    target = a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1], [0.10, 0.16, -0.30])
    assert stake(target) == pytest.approx(0.06)


def test_a_search_whose_own_value_prefers_the_option_it_overruled_stakes_nothing():
    # The exploration bonus can carry visits somewhere the means never followed.
    # The counts call this contested; the gap says the correction is worth less
    # than nothing, and weighting by it is what drops the row.
    target = a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1], [0.16, 0.10, -0.30])
    assert stake(target) < 0.0


def test_a_corpus_collected_before_values_were_recorded_stakes_nothing():
    assert stake(a_target([20.0, 70.0, 10.0], [0.7, 0.2, 0.1])) == 0.0


def test_the_stake_scale_weights_a_contested_row_by_what_it_is_worth():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    filtered = assemble(
        episodes, space, layout, DistillConfig(contested_only=True)
    )
    staked = assemble(
        episodes,
        space,
        layout,
        DistillConfig(contested_only=True, stake_scale=0.05),
    )
    assert (staked.policy_weight <= filtered.policy_weight + 1e-6).all()
    assert (staked.policy_weight >= 0.0).all()
    # The anchor holds whatever the policy term let go, and a part-staked row is
    # part-held: the two weights have to keep summing to one where a prior was
    # recorded, or the row is trained on less than all of itself.
    both = staked.policy_weight + staked.anchor_weight
    recorded = filtered.anchor_weight + filtered.policy_weight > 0.0
    assert torch.allclose(both[recorded], torch.ones_like(both[recorded]), atol=1e-6)


def test_the_hard_target_puts_all_of_a_row_on_one_option():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, config=DistillConfig(hard_target=True))
    totals = batch.slot_target.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    # A one-hot over *options*, so the only way a slot holds less than 1 is the
    # trade slot standing for an offer the row split across -- which cannot
    # happen here, because one option took the whole weight.
    assert torch.allclose(
        batch.slot_target.max(-1).values,
        torch.ones_like(totals),
        atol=1e-5,
    )


def test_the_searched_corpus_records_a_prior_beside_its_visits():
    # Without this the filter has nothing to compare against, and it cannot be
    # recovered later: by training time the policy has moved.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    targets = [
        t.aux
        for e in episodes
        for traj in e.trajectories
        for t in traj
        if t.aux is not None
    ]
    assert targets
    # A forced position is never expanded, so it has no prior -- and it cannot
    # be contested either, which is why `contested` treats a missing prior as
    # agreement rather than as an error.
    searched = [t for t in targets if len(t.options) > 1]
    assert searched
    assert all(t.prior is not None for t in searched)
    assert all(len(t.prior) == len(t.options) for t in searched)
    assert all(t.prior is None for t in targets if len(t.options) == 1)


def test_filtering_zeroes_the_policy_weight_where_the_search_agreed():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    plain = assemble(episodes, space, layout, DistillConfig())
    filtered = assemble(
        episodes, space, layout, DistillConfig(contested_only=True)
    )
    assert torch.all(plain.policy_weight == 1.0)
    # The measured disagreement rate is ~5%, so most rows must be switched off
    # and the batch must keep its length -- the value head still sees them all.
    assert len(filtered) == len(plain)
    assert filtered.policy_weight.sum() < plain.policy_weight.sum()


# --- the anchor -----------------------------------------------------------
#
# `contested_only` filters the policy loss but cannot filter its effect: the
# trunk is shared, the value loss is unweighted, and a network has no
# per-position parameters. `runs/filtered` measured the drift that follows --
# agreement with the search fell 0.941 to 0.788 over ten iterations, and the
# checkpoint lost to its own parent by 0.555 VP. These pin the restoring force.


def test_the_anchor_weight_is_the_complement_of_the_policy_weight():
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=3)
    filtered = assemble(episodes, space, layout, DistillConfig(contested_only=True))
    # Only where a prior was recorded. A forced position has none, and both
    # weights are zero there -- it is neither trained nor anchored.
    anchorable = filtered.anchor_slots.sum(-1) > 0
    assert bool(anchorable.any())
    assert torch.allclose(
        filtered.anchor_weight[anchorable],
        1.0 - filtered.policy_weight[anchorable],
    )


def test_a_row_without_a_recorded_prior_cannot_be_anchored():
    # The alternative -- a uniform target -- would pull the policy toward
    # uniform on exactly the rows it knows least about.
    policy, space, layout = a_setup()
    episodes = searched_episodes(policy, space, games=2)
    batch = assemble(episodes, space, layout, DistillConfig(contested_only=True))
    unanchorable = batch.anchor_slots.sum(-1) == 0
    assert torch.all(batch.anchor_weight[unanchorable] == 0.0)


def test_the_anchor_target_is_a_distribution_over_the_same_slots():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, DistillConfig(contested_only=True))
    anchored = batch.anchor_weight > 0
    assert bool(anchored.any())
    totals = batch.anchor_slots[anchored].sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_the_anchor_is_off_by_default_so_recorded_runs_are_untouched():
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, DistillConfig(contested_only=True))
    assert DistillConfig().anchor == 0.0
    stats = update(
        policy,
        torch.optim.Adam(policy.net.parameters(), lr=1e-3),
        batch,
        DistillConfig(contested_only=True, epochs=1),
        generator=torch.Generator().manual_seed(0),
    )
    assert stats.anchor_loss == 0.0


def test_the_anchor_holds_the_settled_rows_the_filter_let_go():
    """The whole point: rows the policy loss zeroed must not drift freely."""
    import copy

    policy, space, layout = a_setup(seed=3)
    batch = a_batch(policy, space, layout, DistillConfig(contested_only=True), games=4)
    settled = batch.anchor_weight > 0
    assert bool(settled.any())

    def drift(anchor: float) -> float:
        student = NetworkPolicy(copy.deepcopy(policy.net), space, layout)
        config = DistillConfig(contested_only=True, anchor=anchor, epochs=2)
        update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=3e-3),
            batch,
            config,
            generator=torch.Generator().manual_seed(0),
        )
        with torch.no_grad():
            slots, _, _ = student.distributions(batch.buffer, batch.mask, batch.pair)
            # How far the settled rows ended up from the prior they were
            # collected under, which is exactly what the anchor penalises.
            per_row = -(batch.anchor_slots * slots).sum(-1)
            return float(per_row[settled].mean())

    assert drift(anchor=1.0) < drift(anchor=0.0)


# --- the buffer and the packing -------------------------------------------
#
# Collection is 92% of wall clock -- 707 s against 66 s of update on the live
# run -- and with `--contested-only` about 3% of what it buys carries policy
# gradient. Both of these attack that: reuse the corpus, and spend the policy
# term's steps on rows that are actually in it.


def test_concatenating_batches_keeps_every_row():
    policy, space, layout = a_setup()
    from catan.distill import Batch

    first = a_batch(policy, space, layout, games=2)
    second = a_batch(policy, space, layout, games=2)
    joined = Batch.concat([first, second])
    assert len(joined) == len(first) + len(second)
    for name in Batch.FIELDS:
        assert torch.equal(
            getattr(joined, name)[: len(first)], getattr(first, name)
        )
        assert torch.equal(
            getattr(joined, name)[len(first) :], getattr(second, name)
        )


def test_concatenating_one_batch_is_that_batch():
    policy, space, layout = a_setup()
    from catan.distill import Batch

    only = a_batch(policy, space, layout, games=2)
    assert Batch.concat([only]) is only


def test_refreshing_against_the_collecting_policy_reproduces_the_filter():
    """The recorded prior came from this policy, so the two must agree.

    Asserted on the rows where the two definitions provably coincide, which
    takes two restrictions rather than one.

    Where the search put no mass on the trade slot, no aggregation can separate
    an option-space argmax from a slot-space one, and `--max-offers` cannot have
    truncated the option tuple in a way that matters. A slot-space comparison
    scored 0.659 on the whole batch, which is what motivated `_best_option` --
    the trade slot is a sum over every offer on the row, so a search that spread
    its visits can top that slot while its argmax option is not a trade at all.

    And where the max is *strict*. Tied options have no argmax, only a
    tie-break, and the two sides break ties in unrelated orders: `contested`
    takes `np.argmax` over the option tuple, `_best_option` takes `Tensor.max`
    over slots. Swept against the fixture's simulation count the mismatch is the
    tie rate and nothing else -- 33.0% of rows tied gives 1.54% of plain rows
    disagreeing, 14.2% gives 0.27%, 3.3% gives 0.02%, and at 512 simulations
    0.66% tie and none disagree. The invariant is real; asserting past the ties
    would only assert that two arbitrary orderings coincide on one seed.
    """
    from catan.distill import refresh

    policy, space, layout = a_setup(seed=5)
    config = DistillConfig(contested_only=True)
    batch = a_batch(policy, space, layout, config, games=3)
    refreshed = refresh(policy, batch, config)

    plain = batch.trade_mass == 0.0
    # On a plain row the option -> slot map is injective, so a tie among the
    # target's top options shows up as a tie among the top slots.
    top = batch.slot_target.max(-1, keepdim=True).values
    decided = plain & ((batch.slot_target >= top).sum(-1) == 1)

    assert int(plain.sum()) > 0
    assert int(decided.sum()) > 0
    assert torch.equal(batch.policy_weight[decided], refreshed.policy_weight[decided])


def test_refreshing_tracks_the_policy_rather_than_the_recorded_prior():
    """The whole point of a buffer: the filter must not age with the corpus."""
    import copy

    from catan.distill import refresh

    policy, space, layout = a_setup(seed=7)
    config = DistillConfig(contested_only=True, hard_target=True)
    batch = a_batch(policy, space, layout, config, games=3)

    moved = NetworkPolicy(copy.deepcopy(policy.net), space, layout)
    # Train it hard on the search's own targets, so it now agrees with the
    # search where it used to be overruled.
    update(
        moved,
        torch.optim.Adam(moved.net.parameters(), lr=1e-2),
        batch,
        DistillConfig(contested_only=True, hard_target=True, epochs=6),
        generator=torch.Generator().manual_seed(0),
    )
    before = float(batch.policy_weight.sum())
    after = float(refresh(moved, batch, config).policy_weight.sum())
    assert before > 0
    assert after != before


def test_refreshing_never_contests_a_row_the_search_did_not_expand():
    # The projected target of an all-zero row is a one-hot on whichever option
    # `argmax` reached first, so slot space cannot see what `contested` reads
    # off the raw visits. `searched` carries it.
    from catan.distill import refresh

    policy, space, layout = a_setup()
    config = DistillConfig(contested_only=True)
    batch = a_batch(policy, space, layout, config, games=2)
    refreshed = refresh(policy, batch, config)
    unexpanded = batch.searched == 0.0
    assert torch.all(refreshed.policy_weight[unexpanded] == 0.0)


def test_the_refreshed_anchor_is_the_live_policys_own_distribution():
    from catan.distill import refresh

    policy, space, layout = a_setup()
    config = DistillConfig(contested_only=True)
    batch = a_batch(policy, space, layout, config, games=2)
    refreshed = refresh(policy, batch, config)
    with torch.no_grad():
        slots, _, _ = policy.distributions(batch.buffer, batch.mask, batch.pair)
    assert torch.allclose(refreshed.anchor_slots, slots.exp(), atol=1e-5)
    totals = refreshed.anchor_slots.sum(-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


def test_packing_is_off_by_default():
    assert DistillConfig().pack_contested is False
    assert DistillConfig().refresh_prior is False
    assert DistillConfig().buffer_iterations == 1


def test_the_reported_density_matches_the_filter():
    policy, space, layout = a_setup()
    config = DistillConfig(contested_only=True, epochs=1)
    batch = a_batch(policy, space, layout, config, games=3)
    stats = update(
        policy,
        torch.optim.Adam(policy.net.parameters(), lr=1e-4),
        batch,
        config,
        generator=torch.Generator().manual_seed(0),
    )
    assert stats.contested_positions == int(batch.policy_weight.sum())


def test_packing_reports_the_policy_loss_over_contested_rows_only():
    """Unpacked, a minibatch holding no contested row reports a zero loss.

    Those zeros are averaged into `policy_loss`, so the reported number is
    diluted by the filter's selectivity rather than describing the rows the
    loss actually trained on. Packed, every reported step has contested rows in
    it.
    """
    import copy

    policy, space, layout = a_setup(seed=11)
    batch = a_batch(
        policy, space, layout, DistillConfig(contested_only=True), games=3
    )
    assert float(batch.policy_weight.sum()) > 0

    # The dilution needs a minibatch with nothing contested in it, and this
    # corpus does not supply one: `losses` divides by the weight rather than by
    # the row count, so a minibatch holding any contested row already reports
    # the mean over those rows, and at this batch's natural density every one of
    # its seven minibatches holds hundreds. Thinned to three contested rows, six
    # of the seven report the zero this test is about — and the assertion stops
    # being a comparison of two numbers that agree to a tenth of a percent.
    keep = torch.nonzero(batch.policy_weight > 0.0).squeeze(-1)[:3]
    thinned = torch.zeros_like(batch.policy_weight)
    thinned[keep] = batch.policy_weight[keep]
    batch = batch.__class__(
        **{
            name: thinned if name == "policy_weight" else getattr(batch, name)
            for name in batch.FIELDS
        }
    )

    def run(pack: bool) -> float:
        student = NetworkPolicy(copy.deepcopy(policy.net), space, layout)
        stats = update(
            student,
            torch.optim.Adam(student.net.parameters(), lr=1e-4),
            batch,
            DistillConfig(
                contested_only=True, hard_target=True, epochs=1, pack_contested=pack
            ),
            generator=torch.Generator().manual_seed(0),
        )
        return stats.policy_loss

    assert run(pack=True) > run(pack=False)


def test_packing_with_nothing_contested_still_runs():
    # `contested_only` on a corpus with no recorded priors switches every row
    # off, so the policy pass has no rows at all.
    policy, space, layout = a_setup()
    batch = a_batch(policy, space, layout, DistillConfig(), games=2)
    batch = batch.__class__(
        **{
            name: (
                torch.zeros_like(batch.policy_weight)
                if name == "policy_weight"
                else getattr(batch, name)
            )
            for name in batch.FIELDS
        }
    )
    stats = update(
        policy,
        torch.optim.Adam(policy.net.parameters(), lr=1e-4),
        batch,
        DistillConfig(contested_only=True, epochs=1, pack_contested=True),
        generator=torch.Generator().manual_seed(0),
    )
    assert stats.contested_positions == 0
    assert stats.policy_loss == 0.0
