# SPDX-License-Identifier: GPL-3.0-only
"""What a development card another seat holds is, read from how long it has
been kept: the form of a `hexset.view.HoldReading` a checkpoint can declare.

A victory point card is never played. Every other kind is played sooner or
later, each at its own pace: a card of kind ``k`` is still unplayed after
``c`` of its owner's turns with probability ``scale[k] ** (c ** shape)``
(`HoldModel.kept`). A held card is then each kind in proportion to the unseen
cards of that kind times that chance (`HoldModel.odds`): a card kept long is
most likely a victory point card or a slow kind.

The engine has no parameters of its own for it. A checkpoint declares them
in its metadata (`hold_shape`, `hold_scale`; `HoldModel.meta` writes them,
`hexset.clients._modelmeta.hold_config` reads them), and the network seats
deal their search worlds by what it declares (`hexset.clients.netbot`). A
caller may also hand any `HoldReading` to `View.sample(rng, hold)`,
`hexset.mcts.Search(hold=...)` or `hexset.rollout.PairedRollouts(hold=...)`.
Without one every unseen card is dealt alike.

Everything a reading reads is public: each held card's age
(`GameState.dev_ages`) and, through `View.sample`, the cards the perspective
has not seen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from hexset.cards import NUM_DEV_CARDS


@dataclass(frozen=True)
class HoldModel:
    """How long each kind of development card is kept before it is played:
    `shape` shared by every kind, `scale[k]` the chance a card of kind `k`
    (by `DevCard`) is kept through its first playable turn. A `scale` of 1.0
    is never played."""

    shape: float
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.shape) and self.shape > 0):
            raise ValueError(f"a hold shape is a positive number, not {self.shape!r}")
        if len(self.scale) != NUM_DEV_CARDS:
            raise ValueError(f"a hold scale gives {NUM_DEV_CARDS} kinds, not {len(self.scale)}")
        if not all(math.isfinite(s) and 0.0 < s <= 1.0 for s in self.scale):
            raise ValueError(f"each hold scale is in (0, 1], not {self.scale!r}")

    def kept(self, card: int, chances: int) -> float:
        """The chance a card of kind `card` is still unplayed after `chances`
        of its owner's turns."""
        if chances <= 0 or self.scale[card] >= 1.0:
            return 1.0
        return self.scale[card] ** (chances ** self.shape)

    def weight(self, card: int, age: int) -> float:
        """`kept` at a card's age (`GameState.dev_ages`): the reading
        `hexset.view.View.sample` deals a held card by."""
        return self.kept(card, chances(age))

    def odds(self, unseen: Sequence[int], chances: int) -> tuple[float, ...]:
        """A card kept through `chances` turns, as each kind: the unseen cards
        of the kind times the chance one is kept that long."""
        weights = [count * self.kept(card, chances) for card, count in enumerate(unseen)]
        total = sum(weights)
        return tuple(w / total for w in weights) if total else (0.0,) * len(unseen)

    def meta(self) -> dict[str, str]:
        """The checkpoint metadata that declares this model, as
        `hexset.clients._modelmeta.hold_config` reads it back exactly."""
        return {
            "hold_shape": repr(float(self.shape)),
            "hold_scale": ",".join(repr(float(s)) for s in self.scale),
        }


def chances(age: int) -> int:
    """The turns a card of `age` (`GameState.dev_ages`) has been playable
    through: the turn it was bought in is not one."""
    return max(age - 1, 0)


__all__ = ["HoldModel", "chances"]
