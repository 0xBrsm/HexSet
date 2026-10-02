# SPDX-License-Identifier: GPL-3.0-only
"""Casting laws: which policy id holds each seat, as a pure function of the
game index -- so a worker that knows only `k` can reconstruct who sat where
in game `k` with no shared state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .arena import seat_of

__all__ = [
    "Cast",
    "Caster",
    "alternating",
    "paired",
    "league_rotation",
    "rotating",
    "swapped",
]


Cast = tuple[int, ...]
Caster = Callable[[int], Cast]


def alternating(players: int, flip: bool = False) -> Caster:
    """The duel caster: id `1` on every other seat, the pair it holds swapping
    by game parity. A *fixed* seat-parity assignment would cancel the seat
    effect only in the mean, the pairing being correlated with the game index;
    `flip` gives the complementary assignment to difference that term out."""
    offset = 1 if flip else 0

    def caster(index: int) -> Cast:
        cast = [0] * players
        for seat in range(1 - (index + offset) % 2, players, 2):
            cast[seat] = 1
        return tuple(cast)

    return caster


def paired(caster: Caster) -> Caster:
    """Games `2k` and `2k+1` share `caster(k)`, so a mate game is a same-seat,
    same-policy baseline for the game beside it, at twice the cadence."""

    def caster_paired(index: int) -> Cast:
        return caster(index // 2)

    return caster_paired


def league_rotation(
    learners: int, players: int, order: Sequence[int] | None = None
) -> Caster:
    """Rotate `learners` ids over `players` seats by game index, balancing every
    share of every seat over any `players` consecutive indices. `order` permutes
    the ids first, changing *who sits next to whom*."""
    seq = tuple(order) if order is not None else tuple(range(learners))
    if sorted(seq) != list(range(learners)):
        raise ValueError(
            f"learner order must be a permutation of 0..{learners - 1}: {seq}"
        )

    def caster(index: int) -> Cast:
        return tuple(seq[(seat + index) % learners] for seat in range(players))

    return caster


def rotating(lineup: Sequence[int]) -> Caster:
    """`arena.compete`'s own seating as a caster: entrant `e` sits at
    `arena.seat_of(e, index, seats)`, each holding each seat once over any
    `seats` consecutive indices. `lineup` is that lineup as policy ids."""
    seats = len(lineup)
    if seats < 2:
        raise ValueError("a cast needs at least two seats")
    order = tuple(lineup)
    if any(pid < 0 for pid in order):
        raise ValueError(f"lineup {order} names a policy that cannot exist")

    def caster(index: int) -> Cast:
        cast = [0] * seats
        for entrant, pid in enumerate(order):
            cast[seat_of(entrant, index, seats)] = pid
        return tuple(cast)

    return caster


def swapped(caster: Caster, a: int = 0, b: int = 1) -> Caster:
    """The complementary cast: ids `a` and `b` exchange seats on every index --
    the other half of an antithetic pair, same board and dice with seat sets
    traded. Exchanging ids complements *any* cast, which a seat shift does not."""

    def caster_swapped(index: int) -> Cast:
        return tuple(
            b if pid == a else a if pid == b else pid for pid in caster(index)
        )

    return caster_swapped
