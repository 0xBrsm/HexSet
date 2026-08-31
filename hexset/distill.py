# SPDX-License-Identifier: GPL-3.0-only
"""Distillation: move the policy toward what the search decided.

`hexset.expert.SearchPolicy` plays games with a tree and files what the tree
concluded on `Transition.aux`. This module is the other half — the training step
that consumes those targets. The case for it is measured rather than assumed:
`mcts@256` beats the same checkpoint's raw policy 56.8%, so the visit counts
carry something the policy does not already contain.

## The loss is a cross-entropy, and it factors exactly

The search's target is a distribution over *concrete options*. The policy's
distribution is not: it emits one row over flat action slots plus a second row
over one-for-one offer pairs, because `PROPOSE_TRADE` is a single slot standing
for many offers. So the two are not the same shape and the target has to be
projected onto the policy's factorisation.

That projection is exact, not an approximation. The policy scores an option as
`log q(slot) + [slot is trade] log q(pair)` — that is `_chosen_log_prob`, the
same decomposition acting and PPO already use. Substituting it into the
cross-entropy and splitting the sum by slot gives

    -sum_o p(o) log q(o)  =  -sum_s p(s) log q(s)  -  p(trade) * sum_k p(k|trade) log q(k)

where `p(s)` sums the target's mass over every option landing on slot `s` and
`p(k|trade)` is the target restricted to the offers and renormalised. Two
ordinary cross-entropies, the second weighted by the target's own mass on the
trade slot. `test_the_factored_loss_equals_the_cross_entropy_over_options` pins
the identity against a direct sum over options, because getting it wrong is not
a crash — it is a policy that trains and trades badly.

The weighting is the part worth not losing. An unweighted offer term would
spend as much of the gradient on the offer row at the great majority of
positions where the search never proposes at all, which is the same mistake the
entropy bonus in `hexset.policy.evaluate` avoids for the same reason.

## Temperature is applied over options, before the projection

`visit_policy` sharpens the distribution the search actually chose among, so it
belongs at option level. Applying it after aggregating to slots would sharpen
the trade slot against the rest using a mass that is already a sum over dozens
of offers, which is a different and less defensible quantity.

## The value head is still trained on terminal outcomes

Not on the search's backed-up root value, which is the tempting alternative and
is deliberately not taken here. `benchmarks.floor` found 68.4% of the head's
error is irreducible outcome variance, and bootstrapping off the search is one
of the two ideas that could beat that ceiling — but it is an untested change to
the target, and folding it in alongside a new policy loss would leave a
disappointing run with two suspects. So the value half is exactly what PPO
already does, `rotate` and `reward` reused unchanged, and bootstrapping is a
separate experiment with its own control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import Action, ActionSpace, ActionType
from .expert import Target
from .mcts import visit_policy
from .model import Packing, pack
from .policy import NUM_PAIRS, NetworkPolicy, _entropy, pair_index, pair_mask
from .rewards import reward
from .selfplay import Episode, Transition

# Borrowed from `hexset.ppo` rather than lifted into a third module: minibatch
# shuffling, explained variance and the reward rotation are four short functions
# that mean the same thing in both trainers, and a `hexset.training` holding only
# those would be more indirection than it saves.
from .ppo import _explained_variance, _minibatches, rotate


@dataclass(frozen=True)
class DistillConfig:
    temperature: float = 1.0
    # Train the policy only where the search overruled it, and on a hard argmax
    # rather than the visit distribution. Both default off, so every recorded
    # run's arithmetic is untouched.
    #
    # Why, measured. The search moves the policy's argmax on ~4.8% of
    # decisions, and the signed value of those moves is +0.040 VP each --
    # which times ~10-15 contested decisions a seat-game reproduces the duel's
    # +0.536 VP. So the teacher's whole content is in that 5%. The other 95% is
    # the policy's own answer handed back, and handing back the visit
    # *distribution* is worse than handing back nothing: its entropy is set by
    # the search's own exploration settings, not by the position, so every
    # distillation arm on record converged to the tree's stationary entropy
    # (~0.43-0.46 against the parent's 0.339) whatever else changed. Play reads
    # the argmax; training read the distribution. These two flags close that gap.
    contested_only: bool = False
    hard_target: bool = False
    # Visit share the search's pick must lead the policy's by before the row
    # counts as contested. 0.0 is every recorded arm. See `contested` for the
    # measurement: at 0.0 the filter is 64% trade-bundle near-ties.
    contested_margin: float = 0.0
    # The Q-gap at which a contested row earns full weight, in reward units --
    # 0.01 is 0.1 victory points. 0.0 keeps the 0/1 filter every arm on record
    # used. Above it the weight is `min(gap / stake_scale, 1)`, and rows whose
    # gap is negative -- the search's own value says its pick is the worse of
    # the two -- fall out entirely.
    #
    # Why, given `contested_margin` was refuted. The margin gated on the *visit*
    # lead, which is a measure of how sure the search is; it made the arm worse
    # on both columns, so certainty is not what separates a real correction from
    # a coin flip. The gap gates on what the correction is *worth*, which is a
    # different quantity and the one `_select` actually ranks: five votes
    # between two options the search values identically is noise however sure
    # the count looks, and five votes across half a victory point is not.
    stake_scale: float = 0.0
    # Weight on the anchor: a cross-entropy toward the *recorded prior* on the
    # rows `contested_only` zeroed. 0.0 keeps every recorded run's arithmetic.
    #
    # Why it is needed. `contested_only` filters the policy loss but cannot
    # filter its effect. The trunk is shared and `self.value` is one linear
    # layer on it, the value loss is an unweighted mean over every row, and a
    # network has no per-position parameters -- so fitting the contested 5%
    # moves the policy on the other 95% with nothing holding it there. Measured
    # on `runs/filtered`, ten iterations off `ppo4-585`: top-1 agreement with
    # the search fell 0.941 -> 0.788 (disagreement 5.9% -> 21.2%, 3.6x) while
    # trade acceptance went 3.3% -> 23.7% against the 15.9% cited in
    # `benchmarks.behaviour`.
    #
    # Why the prior and not the visits. Anchoring on the visit distribution is
    # exactly what flattened every earlier arm: its entropy belongs to the
    # search operator -- `exploration`, `simulations`, branching -- not to the
    # position, which is why all three arms converged to ~0.43-0.46 whatever
    # the value target was. The prior's entropy is the policy's own 0.339, so
    # this is a restoring force toward where the policy already was and carries
    # no flattening pressure of its own. It is PPO's trust region, spelled as a
    # cross-entropy against the collection-time policy.
    anchor: float = 0.0
    # How many iterations of collected rows the update trains on. 1 discards
    # each corpus after its four epochs, which is what every run on record did.
    #
    # Why more is safe here and is not safe for PPO. `ppo5` confirmed that a
    # batch four generations off-policy costs strength and caps the learning
    # rate -- but that is a *policy gradient*, an estimator that needs an
    # importance correction to stay valid off-policy. Distillation is a
    # supervised cross-entropy against a fixed label: no ratio, no correction,
    # nothing to break. And the label is local -- a visit count is what 256
    # simulations measured at that position, so it inherits none of the 2-3 VP
    # terminal noise that makes the *game* the independent unit for PPO.
    buffer_iterations: int = 1
    # Recompute the filter and the anchor against the live policy instead of the
    # prior recorded at collection time. This is what makes a buffer worth
    # having: the search costs ~700 s a corpus and its counts stay a valid label
    # for their position, while the prior they are compared against is one
    # forward pass and goes stale the moment the policy moves. Recomputing it
    # also turns the 97% that carry no policy gradient from waste into
    # not-yet-contested -- rows cross into the contested set as the student
    # drifts, so one corpus keeps yielding fresh targets.
    refresh_prior: bool = False
    # Give the policy term its own dense minibatches instead of letting it ride
    # the value term's. At ~3% contested density a 1024-row minibatch carries
    # ~31 contested rows, so the policy loss is a 31-sample estimate taken ~419
    # times an iteration. Packed, it is ~3 steps of 1024 an epoch -- the same
    # rows, the same number of visits each, at a sixteenth of the per-step
    # variance. The anchor rides the value pass on purpose: that is the pass
    # that moves the trunk, and holding the policy while it does is the whole
    # point of the anchor. Same shape PPG uses, for the same reason.
    pack_contested: bool = False
    value_coefficient: float = 0.5
    epochs: int = 4
    minibatch: int = 1024
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    # 0 keeps the terminal Monte Carlo target. Anything higher bootstraps: see
    # `_value_targets`, and `benchmarks.floor` for why the terminal one cannot
    # be fitted well enough for a search to use.
    value_horizon: int = 0


@dataclass(frozen=True)
class Stats:
    """What one update did, for the run log rather than for the maths."""

    positions: int
    policy_loss: float
    offer_loss: float
    value_loss: float
    agreement: float
    explained_variance: float
    anchor_loss: float = 0.0
    # The quantity that diagnosed the flattening failure, which the run built to
    # fix it did not log. Mean policy entropy over the batch: the parent reads
    # 0.339 and every distribution-distilled arm left for 0.43-0.46.
    entropy: float = 0.0
    # `agreement` pools two movements that point opposite ways -- the contested
    # rows the loss pulls toward the search, and the settled rows nothing holds.
    # One number cannot say which moved, so it is also reported split. Both are
    # 0.0 when the corresponding subset is empty, which `contested_only=False`
    # makes true of `agreement_settled` by construction.
    agreement_contested: float = 0.0
    agreement_settled: float = 0.0
    # How many rows actually carried policy gradient. Derived from the split
    # gauges before, which needed algebra and an assumption; logged now.
    contested_positions: int = 0


@dataclass
class Batch:
    """One update's worth of searched positions, flattened out of episodes."""

    buffer: Tensor
    mask: Tensor
    pair: Tensor
    slot_target: Tensor
    offer_target: Tensor
    trade_mass: Tensor
    value_target: Tensor
    # 1 where the policy should learn from this row, 0 where it should not.
    # A weight rather than a filter so the *value* head still sees every
    # position: restricting it to contested rows would train it on a biased
    # slice of the state distribution, which is a different experiment.
    policy_weight: Tensor
    # The anchor's target and its weight: the recorded prior on the same two
    # rows, and `1 - policy_weight` wherever a prior was actually recorded. Zero
    # weight on a corpus collected before `Target.prior` existed, so an old
    # corpus stays readable and simply cannot be anchored.
    anchor_slots: Tensor
    anchor_offers: Tensor
    anchor_mass: Tensor
    anchor_weight: Tensor
    # 1 where the search actually expanded this row. `contested` reads it off
    # the raw visits, but a projected slot target cannot: an all-zero row
    # becomes a one-hot on whichever option `argmax` reaches first, and
    # `refresh` would call that a disagreement. 24 of 72 rows in the first
    # screen were exactly this.
    searched: Tensor

    FIELDS = (
        "buffer",
        "mask",
        "pair",
        "slot_target",
        "offer_target",
        "trade_mass",
        "value_target",
        "policy_weight",
        "anchor_slots",
        "anchor_offers",
        "anchor_mass",
        "anchor_weight",
        "searched",
    )

    @staticmethod
    def concat(batches: Sequence["Batch"]) -> "Batch":
        """One batch out of several iterations' worth, for the replay buffer.

        Assembled batches rather than raw episodes: the projection is already
        paid, the tensors are compact, and nothing in a `Batch` depends on which
        iteration produced it once `refresh` recomputes the prior.
        """
        if not batches:
            raise ValueError("nothing to concatenate")
        if len(batches) == 1:
            return batches[0]
        return Batch(
            **{
                name: torch.cat([getattr(b, name) for b in batches])
                for name in Batch.FIELDS
            }
        )

    def nbytes(self) -> int:
        return sum(
            getattr(self, name).element_size() * getattr(self, name).nelement()
            for name in self.FIELDS
        )

    def __len__(self) -> int:
        return self.buffer.shape[0]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            **{name: getattr(self, name).to(device) for name in self.FIELDS}
        )


def contested(target: Target, margin: float = 0.0) -> bool:
    """Did the search overrule the policy at this position, and mean it?

    False when the prior was not recorded, which keeps an old corpus readable
    and makes `contested_only` a no-op on it rather than an empty batch.
    A row of all-zero visits is not contested however the argmaxes fall: it
    means the search never expanded here, and `argmax` on ties would invent a
    disagreement -- 24 of 72 rows in the first screen were exactly that.

    `margin` is the visit share the search's pick must lead the policy's by
    before the disagreement counts, and it defaults to 0 so every arm on record
    keeps its meaning. It exists because the unguarded definition is dominated
    by decisions the search cannot actually call: on a fixed 12,625-row corpus
    258 of 405 disagreements are two trade *bundles* sharing one flat slot,
    their median lead is 0.020 -- five votes in 256 -- and 51.2% lead by under
    2%, against 24.5% for every other kind of correction. A `--hard-target`
    one-hot on a five-vote lead, carrying ~28x the per-row gradient from the
    subset-mean normalisation, is the largest single input to a programme in
    which every arm so far lost to its own parent.
    """
    if target.prior is None or not len(target.options):
        return False
    if float(np.max(target.visits)) <= 0.0:
        return False
    best = int(np.argmax(target.visits))
    theirs = int(np.argmax(target.prior))
    if best == theirs:
        return False
    if margin <= 0.0:
        return True
    total = float(np.sum(target.visits))
    lead = float(target.visits[best] - target.visits[theirs]) / max(total, 1.0)
    return lead >= margin


def stake(target: Target) -> float:
    """What the search says its correction is worth, in reward units.

    The backed-up mean of the search's pick less that of the policy's, both read
    from the mover's seat exactly as `_select` ranks them. Positive means the
    search's own value agrees with its own counts; negative means it does not,
    which happens when the exploration bonus carried the visits somewhere the
    means never followed.

    0.0 when there is nothing to compare -- a corpus collected before `Target`
    recorded values, no prior, an unexpanded row -- so a caller weighting by this
    drops those rows rather than inventing a stake for them.
    """
    if target.values is None or target.prior is None or not len(target.options):
        return 0.0
    if float(np.max(target.visits, initial=0.0)) <= 0.0:
        return 0.0
    values = np.asarray(target.values, dtype=np.float64)
    if values.shape != target.visits.shape:
        return 0.0
    best = int(np.argmax(target.visits))
    theirs = int(np.argmax(target.prior))
    return float(values[best] - values[theirs])


def project(
    target: Target, space: ActionSpace, temperature: float, hard: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """A visit distribution over options, as the policy's two rows see it.

    Returns the slot distribution, the offer distribution *conditional on
    proposing*, and the mass that landed on the trade slot. The offer row is
    zero when nothing was proposed, and the caller must not read it as uniform:
    it is gated by the mass, which is zero there too.
    """
    if hard:
        # The object play actually uses. `visit_policy` with a temperature
        # approaching zero is the same limit, but an exact one-hot avoids
        # asking what 0 ** (1/eps) should be on a row of zero-visit edges.
        weights = np.zeros(len(target.options), dtype=np.float64)
        weights[int(np.argmax(target.visits))] = 1.0
    else:
        weights = visit_policy(target.visits, temperature)
    return _spread(target.options, weights, space)


def _spread(
    options: Sequence[Action], weights: np.ndarray, space: ActionSpace
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lay a distribution over options onto the policy's two rows.

    Shared by the visit target and the anchor's prior so the two land on the
    slot space identically -- several options can share one slot, and the trade
    slot is a sum over offers, so this cannot be done twice by hand.
    """
    slots = np.zeros(space.size, dtype=np.float32)
    offers = np.zeros(NUM_PAIRS, dtype=np.float32)

    for option, share in zip(options, weights):
        slots[space.index(option)] += share
        if option.type is ActionType.PROPOSE_TRADE:
            offers[pair_index(option.give, option.want)] += share

    mass = float(offers.sum())
    if mass > 0:
        offers /= mass
    return slots, offers, mass


def project_prior(
    target: Target, space: ActionSpace
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The root prior over the same options, on the same two rows.

    The anchor's target. `None` when there is no prior to anchor to -- an old
    corpus that never recorded one, or a degenerate row -- and the caller pairs
    that with zero anchor weight rather than inventing a uniform target, which
    would pull the policy toward uniform on exactly the rows it knows least
    about.
    """
    if target.prior is None or not len(target.options):
        return None
    prior = np.asarray(target.prior, dtype=np.float64)
    total = float(prior.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return _spread(target.options, prior / total, space)


def assemble(
    episodes: Sequence[Episode],
    space: ActionSpace,
    layout: Packing,
    config: DistillConfig,
) -> Batch:
    """Flatten searched episodes into one update's tensors.

    Every transition must carry a `Target`. A batch that silently skipped the
    ones that did not would train on whichever positions happened to be searched
    and quietly weight the corpus by that, so a missing target is an error.
    """
    observations = []
    masks: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    slot_targets: list[np.ndarray] = []
    offer_targets: list[np.ndarray] = []
    trade_masses: list[float] = []
    weights: list[float] = []
    values: list[np.ndarray] = []
    anchor_slots: list[np.ndarray] = []
    anchor_offers: list[np.ndarray] = []
    anchor_masses: list[float] = []
    anchor_weights: list[float] = []
    searched_flags: list[float] = []

    for episode in episodes:
        payoffs = reward(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            seen = np.asarray(rotate(payoffs, seat), dtype=np.float32)
            wanted = _value_targets(trajectory, seen, seat, config.value_horizon)
            for transition, want in zip(trajectory, wanted):
                target = _target(transition)
                slots, offers, mass = project(
                    target, space, config.temperature, config.hard_target
                )
                weight = (
                    1.0
                    if not config.contested_only
                    or contested(target, config.contested_margin)
                    else 0.0
                )
                if weight > 0.0 and config.stake_scale > 0.0:
                    weight = min(
                        max(stake(target), 0.0) / config.stake_scale, 1.0
                    )
                observations.append(transition.observation)
                masks.append(transition.mask)
                # Rebuilt from the options rather than read off `aux`, which is
                # where PPO finds it: a searched transition spends `aux` on the
                # target, and `Target` keeps its options for exactly this.
                pairs.append(pair_mask(target.options))
                slot_targets.append(slots)
                offer_targets.append(offers)
                trade_masses.append(mass)
                weights.append(weight)
                values.append(want)
                # The anchor holds the rows the policy loss let go of, so its
                # weight is the complement -- and zero where there is no prior
                # to hold them to.
                anchored = project_prior(target, space)
                if anchored is None:
                    anchor_slots.append(np.zeros(space.size, dtype=np.float32))
                    anchor_offers.append(np.zeros(NUM_PAIRS, dtype=np.float32))
                    anchor_masses.append(0.0)
                    anchor_weights.append(0.0)
                else:
                    prior_slots, prior_offers, prior_mass = anchored
                    anchor_slots.append(prior_slots)
                    anchor_offers.append(prior_offers)
                    anchor_masses.append(prior_mass)
                    anchor_weights.append(1.0 - weight)
                searched_flags.append(
                    1.0 if float(np.max(target.visits, initial=0.0)) > 0.0 else 0.0
                )

    if not observations:
        raise ValueError("no searched transitions to learn from")

    return Batch(
        buffer=pack(layout, observations),
        mask=torch.from_numpy(np.stack(masks)),
        pair=torch.from_numpy(np.stack(pairs)),
        slot_target=torch.from_numpy(np.stack(slot_targets)),
        offer_target=torch.from_numpy(np.stack(offer_targets)),
        trade_mass=torch.tensor(trade_masses, dtype=torch.float32),
        value_target=torch.from_numpy(np.stack(values)),
        policy_weight=torch.tensor(weights, dtype=torch.float32),
        anchor_slots=torch.from_numpy(np.stack(anchor_slots)),
        anchor_offers=torch.from_numpy(np.stack(anchor_offers)),
        anchor_mass=torch.tensor(anchor_masses, dtype=torch.float32),
        anchor_weight=torch.tensor(anchor_weights, dtype=torch.float32),
        searched=torch.tensor(searched_flags, dtype=torch.float32),
    )


def _value_targets(
    trajectory: Sequence[Transition],
    terminal: np.ndarray,
    seat: int,
    horizon: int,
) -> list[np.ndarray]:
    """What each of one seat's decisions is told the position was worth.

    `horizon` 0 gives every decision the terminal outcome, which is what
    AlphaZero does and what this trainer did until now. It is the wrong target
    for a dice game: `benchmarks.floor` attributes 80% of the value head's
    squared error to the conditional variance of that outcome, so no fit
    removes it, and `benchmarks.sibling` measures the surviving error at 19x
    the spread the head puts across a position's children. A search ranking
    children on that is ranking them on noise.

    A bootstrapped target refuses to look that far. Counting `horizon` of the
    seat's *own* decisions ahead — not game steps, since three other seats move
    in between and the gap in dice rolled is what the variance is about — the
    target becomes the estimate recorded there, and the dice between the two
    stop entering it. The floor is a property of the question, so asking a
    nearer one lowers it.

    The estimate used is the search's backed-up root mean rather than the raw
    head, because `SearchPolicy` already stores that in `Transition.value` and
    an average over a tree is the better of the two for free.

    **`Transition.value` is in board order here, and the target is in the
    seat's frame.** `SearchPolicy._value` returns `Node.totals`, whose columns
    are board seats, while `rotate` puts the seat itself first to match what the
    encoder fed the network. The two policies that fill this field disagree on
    frame — `NetworkPolicy` emits the mover's frame — which nothing caught
    because until now nothing read it. Rotating is not optional; skipping it
    trains without complaint and plays nonsense.

    Falls back to the terminal outcome at the end of a trajectory, and for a
    forced move, where the search returns no estimate because there was nothing
    to search.
    """
    if horizon <= 0:
        return [terminal] * len(trajectory)
    targets = []
    for index in range(len(trajectory)):
        ahead = index + horizon
        estimate = trajectory[ahead].value if ahead < len(trajectory) else ()
        if estimate:
            targets.append(np.asarray(rotate(estimate, seat), dtype=np.float32))
        else:
            targets.append(terminal)
    return targets


def _target(transition: Transition) -> Target:
    if not isinstance(transition.aux, Target):
        raise ValueError(
            "a distillation batch needs search targets; this transition came "
            "from a policy that does not produce them"
        )
    return transition.aux


def losses(
    slot_log_probs: Tensor,
    offer_log_probs: Tensor,
    batch: Batch,
    rows: Tensor,
) -> tuple[Tensor, Tensor]:
    """The two cross-entropy terms, the offer one already weighted.

    Split out of `update` because the identity in the module docstring is worth
    testing on its own, without an optimiser in the way.
    """
    weight = batch.policy_weight[rows]
    # Normalise by the weight, not the row count: a minibatch where one row in
    # twenty is contested must produce the same gradient scale as a full one,
    # or the effective learning rate falls with the filter's selectivity.
    total = weight.sum().clamp(min=1e-8)
    slot_loss = (weight * -(batch.slot_target[rows] * slot_log_probs).sum(-1)).sum() / total
    per_row = -(batch.offer_target[rows] * offer_log_probs).sum(-1)
    offer_loss = (weight * batch.trade_mass[rows] * per_row).sum() / total
    return slot_loss, offer_loss


def _best_option(
    slot_probs: Tensor, offer_probs: Tensor, trade_slot: int
) -> tuple[Tensor, Tensor]:
    """The argmax *option* of a factored distribution, as (slot, pair).

    `pair` is -1 on a non-trade slot. An option's probability is `p(slot)` for a
    plain action and `p(trade) * p(pair)` for an offer, so the two candidates are
    the best non-trade slot and the best offer, compared directly.
    """
    plain = slot_probs.clone()
    plain[:, trade_slot] = -1.0
    plain_value, plain_slot = plain.max(-1)

    pair_value, pair_index = offer_probs.max(-1)
    trade_value = slot_probs[:, trade_slot] * pair_value

    trades = trade_value > plain_value
    slot = torch.where(
        trades, torch.full_like(plain_slot, trade_slot), plain_slot
    )
    pair = torch.where(trades, pair_index, torch.full_like(pair_index, -1))
    return slot, pair


def refresh(
    policy: NetworkPolicy,
    batch: Batch,
    config: DistillConfig,
    *,
    chunk: int = 4096,
) -> Batch:
    """Recompute the filter and the anchor against the live policy.

    What a buffer needs, and it is cheap for a specific reason: the two halves
    of the target age at completely different rates. The visit counts cost ~700 s
    of search a corpus and stay a valid label for their position; the prior they
    are compared against is one forward pass and is stale as soon as the policy
    moves -- which is exactly why `Target.prior` had to be recorded in the first
    place. Recomputing it costs ~25 us a position batched against ~7 ms a
    position to search, so a cached corpus is reusable for about a 200th of what
    it cost to collect.

    The anchor becomes the *current* policy's own distribution, which is the
    correct trust region rather than an approximation of one: it holds the
    settled rows where they are at the start of this update, the way PPO anchors
    to `pi_old`.

    Nothing here has a gradient. It runs once per iteration, not per epoch.

    **The comparison is over options, not slots, and it has to be.** Comparing
    slot argmaxes looks equivalent and is not: the trade slot is a *sum* over
    every offer on the row, so a search that spread its visits over five offers
    can put more mass on that one slot than on any single non-trade action while
    its argmax *option* is not a trade at all. Measured on a searched corpus, the
    two definitions disagree on about a third of rows -- far too many to swap one
    for the other silently, since play reads the option and so does `contested`.
    `_best_option` reconstructs the option-space argmax from the factored rows,
    which is exact: `slot_target[trade] * offer_target[k]` is the target's mass
    on offer `k`, and `q(trade) * q(k)` is the policy's, which is
    `_chosen_log_prob`'s own decomposition.
    """
    rows = len(batch)
    device = policy.device
    slot_rows: list[Tensor] = []
    offer_rows: list[Tensor] = []

    with torch.no_grad():
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            slots, offers, _ = policy.distributions(
                batch.buffer[start:stop].to(device),
                batch.mask[start:stop].to(device),
                batch.pair[start:stop].to(device),
            )
            slot_rows.append(slots.exp().cpu())
            offer_rows.append(offers.exp().cpu())

    current = torch.cat(slot_rows)
    current_offers = torch.cat(offer_rows)

    if config.contested_only:
        trade = policy.trade_slot
        want_slot, want_pair = _best_option(
            batch.slot_target, batch.offer_target, trade
        )
        has_slot, has_pair = _best_option(current, current_offers, trade)
        overruled = ((want_slot != has_slot) | (want_pair != has_pair)).to(
            torch.float32
        )
        # An unexpanded row is not a disagreement however the argmaxes fall.
        weight = overruled * batch.searched
    else:
        weight = torch.ones(rows, dtype=torch.float32)

    replaced = {name: getattr(batch, name) for name in Batch.FIELDS}
    replaced["policy_weight"] = weight
    replaced["anchor_slots"] = current
    replaced["anchor_offers"] = current_offers
    replaced["anchor_mass"] = current[:, policy.trade_slot].contiguous()
    replaced["anchor_weight"] = 1.0 - weight
    return Batch(**replaced)


def anchor_losses(
    slot_log_probs: Tensor,
    offer_log_probs: Tensor,
    batch: Batch,
    rows: Tensor,
) -> tuple[Tensor, Tensor]:
    """The trust-region terms: the same two cross-entropies, toward the prior.

    Structurally identical to `losses` on the complementary weight, which is the
    point -- the anchor has to be the same object as the target so the two can
    be traded off by one scalar. Minimising H(prior, pi) minimises
    KL(prior || pi): they differ by H(prior), a constant of the corpus and not
    of the parameters, so the gradient is the same one a KL penalty would give.
    """
    weight = batch.anchor_weight[rows]
    total = weight.sum().clamp(min=1e-8)
    slot_loss = (
        weight * -(batch.anchor_slots[rows] * slot_log_probs).sum(-1)
    ).sum() / total
    per_row = -(batch.anchor_offers[rows] * offer_log_probs).sum(-1)
    offer_loss = (weight * batch.anchor_mass[rows] * per_row).sum() / total
    return slot_loss, offer_loss


def update(
    policy: NetworkPolicy,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    config: DistillConfig,
    *,
    generator: torch.Generator | None = None,
) -> Stats:
    """One distillation update: `config.epochs` passes over `batch`.

    Two shapes, chosen by `pack_contested`.

    Unpacked is the original: one pass, every term on the same minibatch. The
    policy term then rides the value term's sampling, and at ~3% contested
    density a 1024-row minibatch carries ~31 contested rows -- a 31-sample
    gradient estimate taken ~419 times an iteration.

    Packed gives the two terms their own cadence, which is the shape PPG uses
    and for the same reason: their noise scales
    differ by orders of magnitude, so one minibatch cannot serve both. The value
    pass keeps every row at `minibatch`, and **carries the anchor**, because it
    is the pass that moves the trunk and holding the policy while it does is
    what the anchor is for. The policy pass then walks the contested rows
    densely -- the same rows, seen the same `epochs` times each, at a fraction
    of the per-step variance.
    """
    batch = batch.to(policy.device)
    size = len(batch)

    slot_losses, offer_losses, value_losses, agreements = [], [], [], []
    anchor_reports: list[float] = []
    entropies: list[float] = []
    contested_agreements: list[float] = []
    settled_agreements: list[float] = []
    predicted_for_variance = None

    contested_rows = torch.nonzero(batch.policy_weight > 0.0).squeeze(-1)
    packed = config.pack_contested and len(contested_rows) > 0

    def step(loss: Tensor) -> None:
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.net.parameters(), config.max_grad_norm)
        optimiser.step()

    for _ in range(config.epochs):
        # The value pass. Every row, unbiased -- restricting it to contested rows
        # would train the value head on a biased slice of the state distribution,
        # which is a different experiment.
        for rows in _minibatches(size, config.minibatch, generator):
            rows = rows.to(policy.device)
            slots, offers, value = policy.distributions(
                batch.buffer[rows], batch.mask[rows], batch.pair[rows]
            )

            value_loss = (value - batch.value_target[rows]).pow(2).mean()
            loss = config.value_coefficient * value_loss

            slot_loss = offer_loss = None
            if not packed:
                slot_loss, offer_loss = losses(slots, offers, batch, rows)
                loss = loss + slot_loss + offer_loss

            anchored = 0.0
            if config.anchor > 0.0:
                anchor_slot, anchor_offer = anchor_losses(slots, offers, batch, rows)
                penalty = anchor_slot + anchor_offer
                loss = loss + config.anchor * penalty
                anchored = float(penalty.detach())

            step(loss)

            with torch.no_grad():
                if slot_loss is not None:
                    slot_losses.append(float(slot_loss))
                    offer_losses.append(float(offer_loss))
                value_losses.append(float(value_loss))
                anchor_reports.append(anchored)
                entropies.append(float(_entropy(slots).mean()))
                # Top-1 agreement with the search, which is the number to watch:
                # the cross-entropy falls whenever the policy sharpens anywhere,
                # and only this says it sharpened toward the right option. Read
                # off the value pass in both shapes, because only that pass sees
                # every row.
                #
                # This is a *slot*-space comparison and the contested filter is
                # an *option*-space one -- see `_best_option`, where the trade
                # slot's aggregation makes the two disagree on a third of rows.
                # Deliberately not changed: every run on record logs this
                # definition, and a gauge that shifts meaning mid-campaign is
                # worse than one that is merely approximate. Read it as a trend
                # against other runs, not as the filter's own density, which
                # `contested_positions` now reports directly.
                hit = (slots.argmax(-1) == batch.slot_target[rows].argmax(-1)).float()
                agreements.append(float(hit.mean()))
                # Split by which side of the filter the row fell on. Pooled, a
                # fall cannot be read: the contested rows are being pulled
                # toward the search while the settled ones are held by nothing,
                # and those move opposite ways.
                trained = batch.policy_weight[rows]
                settled = 1.0 - trained
                if float(trained.sum()) > 0.0:
                    contested_agreements.append(
                        float((hit * trained).sum() / trained.sum())
                    )
                if float(settled.sum()) > 0.0:
                    settled_agreements.append(
                        float((hit * settled).sum() / settled.sum())
                    )
                predicted_for_variance = (value.detach(), rows)

        if not packed:
            continue

        # The policy pass. Dense contested rows, one visit each per epoch, so a
        # row is trained on exactly as often as it would be unpacked.
        for chosen in _minibatches(len(contested_rows), config.minibatch, generator):
            rows = contested_rows[chosen.to(contested_rows.device)]
            slots, offers, _ = policy.distributions(
                batch.buffer[rows], batch.mask[rows], batch.pair[rows]
            )
            slot_loss, offer_loss = losses(slots, offers, batch, rows)
            step(slot_loss + offer_loss)
            with torch.no_grad():
                slot_losses.append(float(slot_loss))
                offer_losses.append(float(offer_loss))

    with torch.no_grad():
        if predicted_for_variance is None:
            variance = 0.0
        else:
            predicted, rows = predicted_for_variance
            variance = _explained_variance(
                predicted[:, 0], batch.value_target[rows][:, 0]
            )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return Stats(
        positions=size,
        policy_loss=mean(slot_losses),
        offer_loss=mean(offer_losses),
        value_loss=mean(value_losses),
        agreement=mean(agreements),
        explained_variance=variance,
        anchor_loss=mean(anchor_reports),
        entropy=mean(entropies),
        agreement_contested=mean(contested_agreements),
        agreement_settled=mean(settled_agreements),
        contested_positions=int(len(contested_rows)),
    )
