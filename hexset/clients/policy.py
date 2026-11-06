# SPDX-License-Identifier: GPL-3.0-only
"""Runtime-neutral interfaces for batched neural policies and checkpoints.

Rows carry live games, explicit perspective seats and legal options; the
runtime owns encoding, masking and inference. These games carry private state,
so encode perspective-aware and never mutate an input game. Value vectors are
always in board-seat order; priors align with the supplied options. A policy
may also keep plans for decisions it chooses whole, and list a decision's
whole options (`Policy`'s optional names)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hexset.actions import Action, ActionSpace
from hexset.game import Game

__all__ = [
    "Policy",
    "Checkpoint",
]


class Policy(Protocol):
    """Batched policy and value inference over live positions: one result per
    input row, in input order. Runtime integrations own stochastic action
    selection and its random seed."""

    space: ActionSpace

    def act_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[Action]:
        """Choose one action from each row's supplied legal options."""
        ...

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """One value vector per row, one entry per board seat. Adapters read
        the entries as win probabilities; search terminals and trade gains use
        that same scale."""
        ...

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[Sequence[float], tuple[float, ...]]]:
        """`(priors, values)` per search leaf. Priors are nonnegative, sum to
        one, and align with the supplied options; values follow `value_rows`'
        board-seat order and scale."""
        ...

    # Optional, read by name where present (`hexset.rollout.PairedRollouts`):
    #
    # `plans`, a `hexset.decisions.Plans`: a policy that chooses a decision of
    # several moves whole and plays it a move at a time keeps the rest of it
    # there. Its `act_rows` is then the move it plays alone, which need not be
    # the argmax of its `score_rows` prior; a caller that plays another move
    # where a plan is held drops the game's plan (`Plans.drop`).
    #
    # `option_rows(rows: Sequence[tuple[Game, int]]) -> list[tuple[tuple[Option,
    # ...], np.ndarray]]`: every whole option of each row's decision
    # (`hexset.decisions.Option`, as `hexset.decisions.options` states them,
    # a position inside a decision as its legal moves) with its probability,
    # nonnegative and summing to one, in block order (flat actions by index,
    # the setup turn or the double turn, Road Building, the discard), and
    # its `act_rows` decision the most probable of them. A policy that has the
    # method but cannot answer it sets `has_options` false.


class Checkpoint(Protocol):
    """A policy and the metadata required to construct its bot adapters.

    How the checkpoint bargains is read by name when present, structurally
    rather than by inheritance: `trade_params` (a whole
    `hexset.trading.TradeParams` -- card caps and direction, clearing floor,
    responder price, offer budget, fragment plan and continuation budget)
    or, for a loader that carries only the two loose attributes, `trade_floor`
    and `gate_plies`. Declaring none of them reads `hexset.trading.UNLIMITED`,
    which is no limits of its own: no card caps -- the table's is then the only
    bound -- a floor at zero, nothing charged for a response, no broadcast
    budget asked for, no planning.

    A checkpoint may also name a `trader`: a bot, as a lineup names it, that
    answers every trade for it while the checkpoint plays every move
    (`hexset.arena.traded`). Declaring none trades through its own gate.
    """

    policy: Policy
    #: The same action space used by `policy`.
    space: ActionSpace
    #: Supported seat count; adapters reject games with a different count.
    players: int

