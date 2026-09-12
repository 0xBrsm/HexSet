# SPDX-License-Identifier: GPL-3.0-only
"""Runtime-neutral interfaces for batched neural policies and checkpoints.

Rows contain live games, explicit perspective seats and legal options. The
runtime owns encoding, action masking and inference. These games contain
private state: use a perspective-aware encoder to enforce the experiment's
information assumptions, and never mutate input games.

Every returned value vector uses board-seat order, regardless of the
runtime's internal seat rotation. Action priors align with the supplied
options. `hexset.clients.netbot` adapts these interfaces into playable bots,
trade gates and policy/value-guided search without importing a model runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hexset.actions import Action, ActionSpace
from hexset.game import Game


class Policy(Protocol):
    """Batched policy and value inference over live positions.

    Every method returns one result per input row, in input order. Runtime
    integrations own stochastic action selection and its random seed.
    """

    space: ActionSpace

    def act_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[Action]:
        """Choose one action from each row's supplied legal options."""
        ...

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """Return one value vector per row, with one entry per board seat.

        Checkpoint adapters interpret these entries as win probabilities.
        Search terminal values and trade gains use that same scale.
        """
        ...

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[Sequence[float], tuple[float, ...]]]:
        """Return `(priors, values)` for each search leaf.

        Priors are nonnegative, sum to one, and align with the supplied
        options. Values follow `value_rows`' board-seat order and scale.
        """
        ...


class Checkpoint(Protocol):
    """A policy and the metadata required to construct its bot adapters.

    Loaders may attach additional metadata such as architecture, training
    iteration and search settings.

    One of those is read by name when present, structurally rather than by
    inheritance, the same way `hexset.trading` reads a gate's own surface:
    `trade_floor` (this checkpoint's measured clearing floor, in the win
    probability its value head answers in). It describes the exported value
    head rather than the adapter, so a loader that has one should say so; a
    checkpoint that declares none is read at
    `hexset.clients.modelmeta`'s unmeasured default and trades exactly as
    it did before the key existed.
    """

    policy: Policy
    #: The same action space used by `policy`.
    space: ActionSpace
    #: Supported seat count; adapters reject games with a different count.
    players: int
    #: Training trade setting: zero disables trading; None leaves it unbounded.
    max_trades: int | None
