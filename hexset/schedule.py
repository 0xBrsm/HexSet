# SPDX-License-Identifier: GPL-3.0-only
"""The learning rate as a controller driven by a live gauge, not a constant.

Every run through ppo4 set `--learning-rate` once and never touched it again:
`torch.optim.Adam(net.parameters(), lr=args.learning_rate)` and that was the
whole schedule. That is the supervised-training habit, and on-policy RL breaks
it. In supervised training the loss surface is fixed and the schedule can be
written in advance; in PPO the data distribution is the policy, so how far an
update should travel is a *measured* quantity — and PPO already measures it, as
`approx_kl` per update.

**The rule implemented here is the one the robotics PPO implementations settled
on** (`rsl_rl`'s and `rl_games`' adaptive schedulers, standard in the Isaac
Gym/Lab locomotion work): keep the per-update KL inside a band around a target
by moving the learning rate multiplicatively, and clamp it so a deaf gauge
cannot walk the rate off to infinity. Multiplicative because the quantity being
steered is roughly quadratic in step size, so additive nudges are the wrong
shape; a dead zone because a controller with no dead zone oscillates on
sampling noise alone.

The clamp is not a formality here. ppo4 measured **approx_kl 0.0061 at lr 6e-4
against 0.0062 at 3e-4** — doubling the rate moved the gauge by one part in six
hundred. A band controller reading 0.006 against a 0.02 target would call that
"far too cold" and multiply by 1.5 every iteration until something else gave.
Whatever is actually setting the step size in that regime, `max_lr` is what
stops the controller from running away while looking for it, and
`AdaptiveLR.deaf` is how a run notices it is happening rather than discovering
it in the wreckage.

`linear_anneal` is kept alongside because it is the other standard answer and
the two compose: anneal the *ceiling* over a run while the controller works
underneath it. What this module deliberately does **not** do is touch the
entropy coefficient. Exploration here is steered by the accept canary
(`accepts_per_seat_game`), a gauge on a different timescale with a different
failure mode, and two controllers moving at once is how ppo2 became
unattributable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HasParamGroups(Protocol):
    """Just the surface of an optimiser this module needs.

    A Protocol rather than `torch.optim.Optimizer` so the controller and its
    tests stay importable where torch is not — the dev container has no torch,
    and a scheduling rule is exactly the kind of thing that should be testable
    without a GPU image.
    """

    param_groups: list[dict]


def current_lr(optimiser: HasParamGroups) -> float:
    """The first group's rate, which is the rate: nothing here builds groups."""
    return float(optimiser.param_groups[0]["lr"])


def set_lr(optimiser: HasParamGroups, lr: float) -> None:
    """Mutate every group in place — the same thing a torch LRScheduler does.

    Adam's moment estimates are deliberately left alone. They are statistics of
    the gradient, not of the step size, so a rate change should ride the warm
    state rather than restart it; resetting them would make every controller
    move also a momentum reset, and the two effects would be inseparable.
    """
    for group in optimiser.param_groups:
        group["lr"] = float(lr)


@dataclass(frozen=True)
class AdaptiveLR:
    """Hold `approx_kl` in a band around `target_kl` by scaling the rate.

    `band` is the half-width as a *ratio*, not a difference: the dead zone is
    `[target_kl / band, target_kl * band]`, so at the defaults the controller
    sits still anywhere between 0.01 and 0.04 and only acts outside that. The
    conventional PPO KL band is 0.01-0.02, which is where `target_kl` comes
    from; `factor` and `band` are `rsl_rl`'s values.
    """

    target_kl: float = 0.02
    band: float = 2.0
    factor: float = 1.5
    min_lr: float = 1e-5
    max_lr: float = 1e-2

    def __post_init__(self) -> None:
        if self.band < 1.0:
            raise ValueError("band is a ratio >= 1; below 1 the dead zone inverts")
        if self.factor <= 1.0:
            raise ValueError("factor is a ratio > 1; at 1 the controller is a constant")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr above max_lr leaves no rate to choose")

    def next_lr(self, lr: float, kl: float) -> float:
        """The rate for the next update, given the KL this one measured."""
        if kl > self.target_kl * self.band:
            return max(self.min_lr, lr / self.factor)
        if kl < self.target_kl / self.band:
            return min(self.max_lr, lr * self.factor)
        return lr

    def deaf(self, lr: float, kl: float) -> bool:
        """True when the controller is pinned at a clamp and still out of band.

        The failure ppo4's numbers predict: the gauge does not answer the knob,
        so the controller saturates and every further iteration is a no-op it
        cannot report. Worth logging as its own flag rather than inferring from
        a flat `lr` column, because a *correctly* converged controller also
        holds the rate still.
        """
        if kl < self.target_kl / self.band:
            return lr >= self.max_lr
        if kl > self.target_kl * self.band:
            return lr <= self.min_lr
        return False


def linear_anneal(initial: float, iteration: int, total: int, floor: float = 0.0) -> float:
    """`initial` decaying linearly to `initial * floor` over `total` iterations.

    The other standard answer, and the one to reach for when a run's job is to
    consolidate rather than to explore — a terminal cool-down block. `floor` is
    a fraction of `initial` rather than an absolute rate so that the same
    schedule can be pointed at any starting rate.
    """
    if total <= 0:
        return initial
    progress = min(max(iteration / total, 0.0), 1.0)
    return initial * (1.0 - progress * (1.0 - floor))
