# SPDX-License-Identifier: GPL-3.0-only
"""PPO over self-play episodes: GAE, the clipped surrogate, and the value loss.

`hexset.selfplay` hands back episodes whose transitions are already demultiplexed
by seat, which is the property this module rests on. A seat's next state is not
the position that follows its action — three other seats act in between — so an
advantage computed over the interleaved stream would be nonsense. Per seat, the
trajectory is an ordinary MDP again and the usual arithmetic applies.

## gamma is 1, and it is not a parameter

`hexset.rewards` explains why at length; the short form is that the reward is
zero-sum, so about half of all terminal values are negative, and discounting
makes a negative terminal cheaper the later it arrives. That pays a losing
policy to stall, and trading in circles is that move available for free. So
there is no `gamma` field on `PPOConfig`, no flag on the trainer, and
`test_the_discount_is_one_and_is_not_configurable` pins it.

`lam` is the knob instead, and it is the safe one: it trades bias against
variance in the *advantage estimator* without changing which policy is optimal.

## The value head's target, and the one knob on it

By default every transition's value target is the game's terminal reward vector,
rotated into that seat's frame. With gamma 1 and a reward that is zero everywhere
except the end, the Monte Carlo return *is* the terminal reward, so the default
is not an approximation of a bootstrapped target — it is the return, exactly,
with no bootstrap anywhere in it.

Two things follow. The estimator is unbiased and high-variance, which is the
trade `lam` exists to manage on the policy side and which the value head then
eats. And the head is trained on all `players` outputs from every position, not
just the acting seat's, so one position teaches four numbers. That is what makes
it usable by `hexset.bots.SearchBot`'s max^n backup later: a head trained only on
the mover's own component would have nothing to say about the other three seats
and could not be backed up at all.

`value_lam` below 1 replaces the *mover's own* component with a lambda-return —
the standard PPO target this project had until now declined. The case for it is
measured: `benchmarks.horizon` scored fixed horizons and lambda-mixtures against
the same rollouts, and at every matched noise level the mixture is the better
teacher, with the margin widening as more variance is spent (teacher_ratio 0.898
to 0.783 at sigma 0.05, 0.749 to 0.470 at sigma 0.10). A single horizon sits on a
worse trade-off curve than the mixture at every point, so `--value-horizon 8`
failed for its shape rather than for its number.

Two properties of this implementation are deliberate.

**All four components are bootstrapped, off this seat's own trajectory.** The
recorded estimate at a decision is the whole vector — what the mover thinks each
seat will score — so the sequence of estimates of *seat j* along *this seat's*
decisions is well defined, and mixing it is the same operation for every j. That
is exactly the alignment the terminal target already assumes: it hands seat j's
outcome to this seat's decision times too.

**The mixed target is projected back onto the zero-sum plane.**
`relative_points` is zero-sum by construction, so the terminal target sums to
zero on every row and `hexset.mcts`'s `relative` stance reads that sum. A
bootstrapped target does not inherit the property, because it is built from the
head's own outputs and nothing constrains those to sum to zero — measured at up
to 0.89 off on an untrained head, which is the regime a resumed run passes
through if anything goes wrong. Subtracting the row mean restores it exactly.

That projection is not cosmetic. The truth being predicted lies in the zero-sum
plane, so whatever component of the head's error points out of that plane is
error and nothing else. Removing it can only shrink the distance to the target,
and it is the same relativisation `relative_points` applies to the outcome.

## The value loss under a quantile head

`ModelConfig.value_head = "quantile"` changes the value *loss* and nothing
else. The head emits `players x Q` numbers whose mean is `V`, so `advantages`,
`lambda_returns`, the zero-sum projection and the whole policy-gradient wire
read exactly what they read before -- the mean, one scalar per seat. What
`minibatch_terms` swaps is the term it differentiates: the per-seat quantile
Huber loss (`hexset.model.quantile_huber_loss`) against the same
`value_target` vector, in place of that vector's squared error.

That is the whole of the treatment, and it is deliberate. Gate A2 of the
variance screen measured the mean-of-quantiles pricing this target no better
than a squared-error head on frozen trunk features (bias^2 down 1.0% against a
20% line), so the surviving mechanism is the richer loss shaping the shared
trunk -- which only a real training run can express.

`Stats.value_mse` exists for the reading. The two arms' `value_loss` columns
are on different scales (a pinball loss is roughly `E|u|/2`, a squared error is
`E u^2`), so a curve comparison across arms needs one number both arms compute
the same way: the plain squared error of the mean, logged alongside and never
differentiated. `explained_variance` is already that kind of number, since it
reads the mean too.

**`value_lam = 1.0` keeps the old code path rather than reproducing it.** The
identity `G^lam = A^lam + V` makes lam=1 the terminal return exactly in real
arithmetic, but the telescoping sum is not bit-identical to the assignment, and a
control arm has to be the thing it controls for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import ActionType
from .model import Packing, pack
from .policy import NUM_PAIRS, NetworkPolicy, pair_index
from .rewards import reward
from .selfplay import Episode, Transition

# Not a configurable. See the module docstring and `hexset.rewards`.
GAMMA = 1.0

# torch's Adam defaults to 1e-8; 1e-5 is the standard PPO value. It matters more
# here than usual: `masked_log_softmax` zeroes the gradient at illegal positions
# and a position offers ~6 legal actions out of 553, so a given logit's row sees
# gradient from a small minority of a minibatch. At 1e-8 Adam normalises those
# few tiny, noisy gradients up to a near-full +/-lr step, so rarely-legal logits
# random-walk at the full learning rate. 1e-5 damps exactly that.
ADAM_EPS = 1e-5


@dataclass(frozen=True)
class PPOConfig:
    lam: float = 0.95
    # 1.0 keeps the terminal outcome as the value target, which is what every
    # run on record trained under. Below 1 the mover's own component becomes a
    # lambda-return. See the module docstring.
    value_lam: float = 1.0
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch: int = 1024
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    # Every run on record shares this across seats; carried on the config
    # (rather than passed straight to the optimiser, as `hexset.train` does)
    # so `hexset.league` can vary it per learner like every other knob.
    adam_eps: float = ADAM_EPS
    # Which wires connect the value head to learning. The head touches training
    # through exactly two: it prices each decision (GAE turns its estimates
    # into advantages) and its loss gradient shapes the shared trunk. "gae" is
    # every run on record — both wires. "none" cuts both: the advantage is the
    # seat's terminal return and no value term enters the loss, which is
    # REINFORCE with the zero-sum reward as an opponent baseline. "aux" keeps
    # the trunk-shaping wire and cuts the pricing one: the head trains exactly
    # as under "gae" while the policy gradient never reads it. The head module
    # is built in every mode so checkpoints stay loadable and the duel tooling
    # is untouched; the modes ablate the wires, not the module.
    critic: str = "gae"
    # Stop taking epochs once a finished epoch's mean `approx_kl` exceeds this;
    # 0 disables, which is every run on record. A ceiling, not a target — it
    # can only remove epochs, never raise a rate. The ppo7 collapse was a
    # controller steering *toward* 0.02 from below, a failure mode a one-sided
    # break cannot reproduce. The number itself is measured, not inherited:
    # healthy last-epoch KL runs ~0.004-0.008 on record, the highest reading
    # with no damage is 0.0182, the lowest with confirmed damage is 0.045, so
    # 0.02 sits at the conservative edge of the (0.018, 0.045) boundary band.
    kl_break: float = 0.0
    # The board-paired advantage baseline (variance screen, candidate 1). Each
    # seat's terminal payoff on the policy-gradient wire becomes r - (r+r')/2,
    # where r' is the same seat's reward in the mate game `index ^ 1` of a
    # board pair. Given (board, seat, policy) — which is what paired dealing
    # plus the paired caster hold fixed — r' is independent of this game's
    # actions, so the subtraction is a valid control variate and the gradient
    # stays unbiased. The value target keeps the RAW terminal: the baseline
    # changes what the policy is paid, never what the head is trained toward.
    # Off by default, so every recorded batch assembles bit-identically.
    pair_baseline: bool = False

    def __post_init__(self) -> None:
        if self.critic not in ("gae", "none", "aux"):
            raise ValueError(f"unknown critic mode {self.critic!r}")
        if self.critic != "gae" and self.value_lam < 1.0:
            raise ValueError(
                "value_lam below 1 mixes the head's estimates into the value "
                "target; run it only with critic='gae' so one flag is one wire"
            )


@dataclass(frozen=True)
class Stats:
    """What a single update did, for the run log rather than for the maths."""

    positions: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    # Everything below defaults, so `hexset.ddp` and `hexset.distill` keep
    # constructing this positionally without knowing about the new gauges.
    #
    # `approx_kl` above is the mean over every minibatch of every epoch, which
    # is not a step size and is not comparable to the conventional 0.01-0.02
    # band that band is quoted for the *end* of an update. Measured on a fixed
    # batch at lr 3e-4: 0.0007 in epoch 1 rising to 0.0044 by epoch 3, so the
    # average understates the finished update's divergence by about 1.7x. The
    # three fields below are the readable versions.
    #
    # `approx_kl_first_minibatch` is the diagnostic that matters most: on a
    # genuinely on-policy batch it must be ~0, because nothing has stepped yet.
    # Measured at 0.0007 on a freshly drained single-policy batch. If a
    # production run logs a materially larger number, the batch is off-policy
    # before the update starts — which is what `lanes / games_per_iteration`
    # predicts, that ratio being how many iterations a game survives in flight.
    approx_kl_first_minibatch: float = 0.0
    approx_kl_last_epoch: float = 0.0
    clip_fraction_last_epoch: float = 0.0
    # The pre-clip global norm, median over the update's steps.
    # `clip_grad_norm_` has always returned it and all three call sites threw it
    # away, which is why "is max_grad_norm binding?" was unanswerable from the
    # logs for the whole campaign. (Measured: median 0.256 against a 0.5 clip,
    # so it is not. Note that under Adam a *constant* rescale of the gradient
    # cannot change the step anyway, since m/sqrt(v) is invariant to it.)
    grad_norm: float = 0.0
    # Logged beside `explained_variance` so a shrinking denominator can never
    # again be misread as a improving head: EV against a terminal return is
    # bounded near ~0.6 for this target, and the repo has measured that twice.
    value_target_variance: float = 0.0
    # The rate Adam actually stepped with. Absent this, `--learning-rate 6e-4`
    # was silently discarded by `--resume` for an entire 150-iteration block and
    # nothing in the log could show it.
    lr: float = 0.0
    # Epochs the update actually took. Equal to `config.epochs` unless
    # `kl_break` stopped it early; under a break regime this column is the
    # knob's own telemetry, and the threshold gets re-derived from it.
    epochs_taken: int = 0
    # The squared error of the *mean* the rest of the system reads as `V`,
    # whatever the value head's shape and whatever loss was differentiated.
    # Equal to `value_loss` under every scalar head; under the quantile head
    # `value_loss` is the pinball loss and this is the only column on which the
    # two arms of a heat can be read against each other. Never differentiated.
    value_mse: float = 0.0


def rotate(rewards: Sequence[float], seat: int) -> tuple[float, ...]:
    """A board-order reward vector as `seat` sees it, with itself first.

    The same rotation `hexset.encoding` applies — slot `i` is board seat
    `(perspective + i) % players` — so the value head's outputs line up with its
    inputs. Getting this backwards trains fine and plays nonsense, which is why
    it is one function pinned by one test rather than an inline expression in
    three places.
    """
    players = len(rewards)
    return tuple(rewards[(seat + i) % players] for i in range(players))


def advantages(
    values: np.ndarray, terminal: float, lam: float
) -> np.ndarray:
    """GAE over one seat's trajectory, with gamma fixed at 1.

    `values` is the seat's own value estimate at each of its decisions, in
    order. `terminal` is the reward the game ended on, from this seat's point of
    view. The reward is zero at every step but the last, so the residual is just
    the change in the seat's own estimate, and only the final step sees a payoff.
    """
    steps = len(values)
    out = np.zeros(steps, dtype=np.float32)
    running = 0.0
    for t in reversed(range(steps)):
        # The bootstrap is the *next* estimate, except at the end of the game
        # where there is no next state and the payoff arrives instead.
        nxt = values[t + 1] if t + 1 < steps else 0.0
        payoff = terminal if t + 1 == steps else 0.0
        delta = payoff + GAMMA * nxt - values[t]
        running = delta + GAMMA * lam * running
        out[t] = running
    return out


def lambda_returns(values: np.ndarray, terminal: np.ndarray, lam: float) -> np.ndarray:
    """One seat's trajectory of value *vectors*, mixed over every horizon.

    The same recursion as `advantages`, run over every component at once and
    with `V` added back: `G^lam = A^lam + V`, because the advantage and the
    return are one geometric mixture read two ways. At lam=0 a row is
    `V(s_next)`, the one-step bootstrap; at lam=1 it is the terminal vector
    exactly, by the same telescoping the GAE relies on.

    `values` is (steps, players) in the seat's frame, `terminal` is that seat's
    rotated outcome. Rows come back projected onto the zero-sum plane — see the
    module docstring for why that is a correction rather than a tidy-up. With
    zero-sum inputs the projection is a no-op, which is what keeps the identity
    above exactly testable.
    """
    values = np.asarray(values, dtype=np.float32)
    terminal = np.asarray(terminal, dtype=np.float32)
    steps = len(values)
    out = np.zeros_like(values)
    running = np.zeros(values.shape[1], dtype=np.float32)
    zero = np.zeros_like(running)
    for t in reversed(range(steps)):
        nxt = values[t + 1] if t + 1 < steps else zero
        payoff = terminal if t + 1 == steps else zero
        delta = payoff + GAMMA * nxt - values[t]
        running = delta + GAMMA * lam * running
        out[t] = running + values[t]
    return out - out.mean(axis=1, keepdims=True)


@dataclass
class Batch:
    """One update's worth of positions, flattened out of many episodes."""

    buffer: Tensor
    mask: Tensor
    pair: Tensor
    chosen: Tensor
    offer: Tensor
    log_prob: Tensor
    advantage: Tensor
    value_target: Tensor

    def __len__(self) -> int:
        return self.buffer.shape[0]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "buffer",
                    "mask",
                    "pair",
                    "chosen",
                    "offer",
                    "log_prob",
                    "advantage",
                    "value_target",
                )
            }
        )


def _offer_slot(transition: Transition) -> int:
    if transition.action.type is ActionType.PROPOSE_TRADE:
        return pair_index(transition.action.give, transition.action.want)
    return 0


def _pair_mask(transition: Transition) -> np.ndarray:
    # `aux` is the mask the policy applied when it sampled. It is absent on
    # every non-trading position and on anything a scripted policy produced,
    # and an all-False row is exactly right there: `masked_log_softmax` falls
    # back to uniform and the offer term is gated off by the slot anyway.
    if isinstance(transition.aux, np.ndarray):
        return transition.aux
    return np.zeros(NUM_PAIRS, dtype=bool)


def assemble(
    episodes: Sequence[Episode], layout: Packing, config: PPOConfig
) -> Batch:
    """Flatten finished episodes into one update's tensors.

    Only finished episodes: a game still in flight has no terminal reward, and
    the value head is never bootstrapped, so there is nothing to learn from a
    partial trajectory. `Collector.collect` returns exactly the finished ones.
    """
    observations = []
    masks: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    chosen: list[int] = []
    offers: list[int] = []
    log_probs: list[float] = []
    advantage_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    # The pair baseline needs every game's mate in the same batch: a game is a
    # pure function of (seed, index), so the pairing key is the same pair.
    # `owned` keeps every episode (it empties seats, never drops games), so a
    # learner's slice of a paired cohort still carries both halves.
    dealt = (
        {(e.seed, e.index): reward(e.outcome) for e in episodes}
        if config.pair_baseline
        else {}
    )

    for episode in episodes:
        payoffs = reward(episode.outcome)
        mate = None
        if config.pair_baseline:
            mate = dealt.get((episode.seed, episode.index ^ 1))
            if mate is None:
                raise ValueError(
                    f"game {episode.index} (seed {episode.seed}) has no mate "
                    f"{episode.index ^ 1} in this batch; pair_baseline needs "
                    "paired dealing (pair_boards) and an even cohort of "
                    "complete pairs"
                )
        for seat, trajectory in enumerate(episode.trajectories):
            if not trajectory:
                continue
            seen = rotate(payoffs, seat)
            # Component 0 is this seat's own payoff, because `rotate` puts it
            # first — the same convention the encoder and the value head use.
            own = np.float32(seen[0])
            if mate is not None:
                # (r - r')/2 is the register's r - (r + r')/2 in exact
                # arithmetic, computed in the form whose float rounding makes
                # the two halves' adjusted payoffs bit-exact negatives —
                # negating a difference is exact, subtracting a rounded
                # midpoint is not. The raw `seen` still feeds `target` below:
                # only the policy-gradient terminal moves.
                own = np.float32((seen[0] - mate[seat]) / 2)
            if config.critic == "gae":
                # Raise rather than substitute 0.0 for a missing estimate. A single
                # zero mid-chain corrupts two GAE residuals directly and ~20 more
                # through `running`, silently — and it is reachable: `expert`'s
                # search policy returns `()` on forced moves and its values are in
                # *board* order rather than the mover's frame, so putting a
                # search-collected episode through this loop would read seat 0's
                # estimate for every seat and never complain. Today only
                # `NetworkPolicy` fills recorded seats, which always emits all
                # `players` components in the mover's frame.
                for transition in trajectory:
                    if not transition.value:
                        raise ValueError(
                            "a recorded transition has no value estimate; GAE cannot "
                            "substitute one without silently corrupting the chain"
                        )
                estimates = np.array(
                    [t.value[0] for t in trajectory],
                    dtype=np.float32,
                )
                advantage_blocks.append(advantages(estimates, own, config.lam))
            else:
                # REINFORCE: every decision in the game is credited with the
                # seat's terminal return, whole. Not GAE-with-zeros — with V≡0
                # the recursion yields lam**(T-t) * G, a horizon discount no
                # one asked for — the return itself, at every step. The
                # zero-sum reward already subtracts the opponents' mean and the
                # per-minibatch normalisation in `update` is unchanged, so this
                # differs from "gae" in exactly one thing: what the head
                # contributes to the gradient.
                advantage_blocks.append(
                    np.full(len(trajectory), own, dtype=np.float32)
                )

            target = np.asarray(seen, dtype=np.float32)
            mixed = (
                lambda_returns(
                    np.array([t.value for t in trajectory], dtype=np.float32),
                    target,
                    config.value_lam,
                )
                if config.value_lam < 1.0
                else None
            )
            for step, transition in enumerate(trajectory):
                observations.append(transition.observation)
                masks.append(transition.mask)
                pairs.append(_pair_mask(transition))
                chosen.append(transition.index)
                offers.append(_offer_slot(transition))
                log_probs.append(transition.log_prob)
                targets.append(target if mixed is None else mixed[step])

    if not observations:
        raise ValueError("no transitions to learn from")

    return Batch(
        buffer=pack(layout, observations),
        mask=torch.from_numpy(np.stack(masks)),
        pair=torch.from_numpy(np.stack(pairs)),
        chosen=torch.tensor(chosen, dtype=torch.int64),
        offer=torch.tensor(offers, dtype=torch.int64),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        advantage=torch.from_numpy(np.concatenate(advantage_blocks)),
        value_target=torch.from_numpy(np.stack(targets)),
    )


def _minibatches(
    size: int, minibatch: int, generator: torch.Generator | None
) -> Iterator[Tensor]:
    """A shuffled partition of `size` rows into chunks of at most `minibatch`.

    **A trailing chunk of exactly one row is folded into its predecessor**, and
    that is not tidiness. One row cannot be advantage-normalised: `Tensor.std()`
    defaults to `correction=1`, so a single row divides by zero and returns nan,
    the `+ 1e-8` in the caller does not rescue a nan, `clip_grad_norm_`
    propagates it, and `optimiser.step()` then writes nan into every parameter
    while the run carries on logging. Since `positions` varies per iteration
    that was a ~1-in-`minibatch` dice roll every iteration — about a 40% chance
    of silently destroying a 500-iteration run. The smallest remainder actually
    observed over the 592 logged iterations in `runs/` is 3, so it never fired.
    That was luck, not design.

    All three update paths — this module's, `hexset.ddp`'s and `hexset.distill`'s
    — draw their minibatches here, so the guard lands once for all of them.
    """
    order = torch.randperm(size, generator=generator)
    bounds = [(start, min(start + minibatch, size)) for start in range(0, size, minibatch)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    for start, stop in bounds:
        yield order[start:stop]


@dataclass(frozen=True)
class Terms:
    """One minibatch's PPO arithmetic: the loss to differentiate plus the
    numbers the log wants. One function produces it (`minibatch_terms`) and
    both the single-device update below and the sharded one in `hexset.ddp`
    consume it, so the two cannot drift apart."""

    loss: Tensor
    value: Tensor
    # The three summands of `loss`, still attached to the graph. `loss` is what
    # the update differentiates; these exist so a caller can differentiate one
    # term at a time — which is the only way to ask whether the policy or the
    # value head dominates the shared trunk, an open question four separate
    # audits answered four different ways from the loss magnitudes alone.
    policy_term: Tensor
    value_term: Tensor
    entropy_term: Tensor
    policy_loss: float
    value_loss: float
    # The mean's plain squared error, on one scale across head shapes. Equal to
    # `value_loss` under a scalar head, by the same expression.
    value_mse: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def minibatch_terms(
    policy: NetworkPolicy,
    buffer: Tensor,
    mask: Tensor,
    pair: Tensor,
    chosen: Tensor,
    offer: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    value_target: Tensor,
    config: PPOConfig,
) -> Terms:
    """The clipped surrogate, value loss and entropy for one minibatch.

    `advantage` arrives already normalised — the caller owns that, because a
    sharded worker only holds a slice of the minibatch and cannot compute the
    minibatch's own mean and std.
    """
    evaluation = policy.evaluate(buffer, mask, pair, chosen, offer)
    ratio = (evaluation.log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clamped = ratio.clamp(1 - config.clip, 1 + config.clip) * advantage
    policy_loss = -torch.min(unclipped, clamped).mean()
    # A scalar value head takes the branch it always took, character for
    # character. `test_a_linear_head_s_loss_is_bit_identical_to_the_pre_quantile_
    # arithmetic` pins that this is not "the same in exact arithmetic" but the
    # same graph, gradients included.
    if evaluation.quantiles is None:
        value_loss = (evaluation.value - value_target).pow(2).mean()
        value_mse = value_loss
    else:
        # `players x Q` predictions against the same `value_target` vector.
        # The head owns its levels and its Huber width, so the loss cannot
        # drift from the shape that produced the numbers.
        value_loss = policy.net.value.loss(evaluation.quantiles, value_target)
        with torch.no_grad():
            # Logged, never differentiated: the comparable column across arms.
            value_mse = (evaluation.value - value_target).pow(2).mean()
    entropy = evaluation.entropy.mean()
    # Under critic="none" the value term is absent from the loss, so no
    # gradient reaches the head or the trunk through it; `value_loss` is still
    # computed and logged, and reads as the untrained head's error.
    value_term = (
        config.value_coefficient * value_loss
        if config.critic != "none"
        else torch.zeros_like(policy_loss)
    )
    loss = policy_loss + value_term - config.entropy_coefficient * entropy
    with torch.no_grad():
        # Schulman's low-variance estimator, which unlike the plain log-ratio
        # mean is non-negative and so cannot hide a diverging update behind
        # cancellation.
        log_ratio = evaluation.log_prob - old_log_prob
        # `expm1`, not `ratio - 1`. The estimator is non-negative in exact
        # arithmetic, but in float32 `exp(x)` rounds to 1.0 for any |x| below
        # the ~1.2e-7 epsilon, so the difference collapses to `-log_ratio` and
        # the gauge reports a negative KL. That is exactly the regime cohort
        # collection put it in: an on-policy batch has log-ratios around 1e-9
        # from reduction order alone, and the reading came back -5.8e-10 rather
        # than the true ~1.7e-19. `expm1` is accurate for small arguments, so
        # the floor reads 0 instead of a small impossible number. Nothing above
        # a log-ratio of ~1e-3 changes, which is every reading on record.
        kl = float((torch.expm1(log_ratio) - log_ratio).mean())
        clipped = float((ratio - 1).abs().gt(config.clip).float().mean())
    return Terms(
        loss=loss,
        value=evaluation.value.detach(),
        policy_term=policy_loss,
        value_term=value_term,
        entropy_term=-config.entropy_coefficient * entropy,
        # Detached before conversion: these still carry grad here, unlike the
        # old inline code where the floats were taken inside no_grad.
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        value_mse=float(value_mse.detach()),
        entropy=float(entropy.detach()),
        approx_kl=kl,
        clip_fraction=clipped,
    )


def _explained_variance(predicted: Tensor, actual: Tensor) -> float:
    """1 - Var(residual)/Var(actual), the standard read on a value head.

    Zero means the head is no better than predicting the mean, which is the
    number to watch early: a value head stuck at zero explained variance is the
    single clearest sign a run is not learning.
    """
    variance = actual.var()
    if variance < 1e-8:
        return 0.0
    return float(1 - (actual - predicted).var() / variance)


def update(
    policy: NetworkPolicy,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    config: PPOConfig,
    *,
    generator: torch.Generator | None = None,
) -> Stats:
    """One PPO update: `config.epochs` passes over `batch` in minibatches."""
    batch = batch.to(policy.device)
    size = len(batch)

    policy_losses, value_losses, entropies, kls, clipped = [], [], [], [], []
    value_mses: list[float] = []
    grad_norms: list[float] = []
    epoch_kls: list[list[float]] = []
    epoch_clips: list[list[float]] = []
    predicted_for_variance = None

    epochs_taken = 0
    for _ in range(config.epochs):
        epoch_kls.append([])
        epoch_clips.append([])
        for rows in _minibatches(size, config.minibatch, generator):
            rows = rows.to(policy.device)
            advantage = batch.advantage[rows]
            # Normalised per minibatch, which is what makes one clip range work
            # across a run whose reward scale is fixed but whose advantage
            # spread collapses as the value head improves. `_minibatches`
            # guarantees at least two rows, without which `std()` returns nan.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            terms = minibatch_terms(
                policy,
                batch.buffer[rows],
                batch.mask[rows],
                batch.pair[rows],
                batch.chosen[rows],
                batch.offer[rows],
                batch.log_prob[rows],
                advantage,
                batch.value_target[rows],
                config,
            )

            optimiser.zero_grad(set_to_none=True)
            terms.loss.backward()
            # The return value is the norm *before* clipping, which is the only
            # way to know whether `max_grad_norm` is binding. Kept, not dropped.
            grad_norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        policy.net.parameters(), config.max_grad_norm
                    )
                )
            )
            optimiser.step()

            policy_losses.append(terms.policy_loss)
            value_losses.append(terms.value_loss)
            value_mses.append(terms.value_mse)
            entropies.append(terms.entropy)
            kls.append(terms.approx_kl)
            clipped.append(terms.clip_fraction)
            epoch_kls[-1].append(terms.approx_kl)
            epoch_clips[-1].append(terms.clip_fraction)
            predicted_for_variance = (terms.value, rows)

        epochs_taken += 1
        # The break reads the epoch that just finished, so the damage an over-
        # long update does is bounded at one epoch past the ceiling rather
        # than three — the two recorded blowouts were fine at epoch 1 and
        # diverging by epoch 4.
        if config.kl_break > 0 and float(np.mean(epoch_kls[-1])) > config.kl_break:
            break

    with torch.no_grad():
        if predicted_for_variance is None:
            variance = 0.0
        else:
            predicted, rows = predicted_for_variance
            variance = _explained_variance(
                predicted[:, 0], batch.value_target[rows][:, 0]
            )

    return Stats(
        positions=size,
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approx_kl=float(np.mean(kls)),
        clip_fraction=float(np.mean(clipped)),
        explained_variance=variance,
        approx_kl_first_minibatch=float(epoch_kls[0][0]) if epoch_kls[0] else 0.0,
        approx_kl_last_epoch=float(np.mean(epoch_kls[-1])) if epoch_kls[-1] else 0.0,
        clip_fraction_last_epoch=(
            float(np.mean(epoch_clips[-1])) if epoch_clips[-1] else 0.0
        ),
        grad_norm=float(np.median(grad_norms)) if grad_norms else 0.0,
        value_target_variance=float(batch.value_target[:, 0].var()),
        lr=float(optimiser.param_groups[0]["lr"]),
        epochs_taken=epochs_taken,
        value_mse=float(np.mean(value_mses)) if value_mses else 0.0,
    )
