# SPDX-License-Identifier: GPL-3.0-only
"""Casting laws: which policy id holds each seat, as a pure function of the
game index.

`arena.seat_of` answers "who sits where" when the entrants are a fixed
lineup and the question is only rotating that lineup through every seat for
one `compete` tournament. A training loop asks a different question: given
one or more *ids* naming a policy rather than a fixed `Entrant` -- a
reference, a learner, a pool of learners -- which id holds which seat in
game `k`, so every worker can replay the same assignment from the index
alone, with no shared state and nothing to keep in sync across processes.

Every function here is a pure function of an integer game index -- same
input, same output, forever -- because that purity is what makes a caster
replayable: a worker that only knows `k` can reconstruct exactly who sat
where in game `k`, whether it is training an evaluation the first time or
replaying it the tenth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .arena import seat_of

Cast = tuple[int, ...]
Caster = Callable[[int], Cast]


def alternating(players: int, flip: bool = False) -> Caster:
    """The duel caster: id `1` on every other seat, the pair of seats it
    holds swapping by game parity; id `0` holds the rest.

    A caster that put the same id on the same seat-parity in every game
    would look, on average, like it cancels the seat effect over an
    even-sized cohort -- but that is only true of the *mean*. The seat
    pairing is correlated with the game index (the board itself is a
    function of `(seed, index)`), so a single fixed pairing folds a real,
    parity-correlated seat term into what reads as ordinary noise. Measured
    on one checkpoint duelled against itself: -0.77 VP over 48 games and
    -0.08 over 800, with 55% of a single-order duel's variance coming from
    that residual alone rather than from the board or the dice.

    `flip` names the complementary assignment: `alternating(players, True)`
    holds the opposite seat-parity from `alternating(players, False)` on
    every index, which is what lets a caller play one board both ways and
    difference the seat term back out.
    """
    offset = 1 if flip else 0

    def caster(index: int) -> Cast:
        cast = [0] * players
        for seat in range(1 - (index + offset) % 2, players, 2):
            cast[seat] = 1
        return tuple(cast)

    return caster


def paired(caster: Caster) -> Caster:
    """Games `2k` and `2k+1` share `caster(k)`: within a pair, the same id
    holds the same seat on the same board, so a mate game's outcome can serve
    as a same-seat, same-policy baseline for the game beside it rather than a
    comparison between two different players. Still a pure function of the
    index, the law every caster here obeys.

    The balance consequence: whatever share of every seat `caster` gives
    each id over any window of indices, `paired(caster)` gives that same
    exact share over a window twice as wide -- nothing is lost, the cadence
    is just doubled.
    """

    def caster_paired(index: int) -> Cast:
        return caster(index // 2)

    return caster_paired


def league_rotation(
    learners: int, players: int, order: Sequence[int] | None = None
) -> Caster:
    """Rotate `learners` ids over `players` seats by game index, so no
    learner owns a seat and every learner's share of every seat balances
    over any window of `players` consecutive indices. Pure in the index,
    like every caster here.

    `order` permutes the learner ids before the rotation, which changes
    *who sits next to whom* while leaving every learner's per-seat share
    exactly as balanced. Rotation alone cannot vary that: the cyclic seating
    order is always `0, 1, 2, ...`, so id `0`'s turn-order successor is
    always id `1` in every game. That fixed adjacency is worth being able to
    break when adjacency itself is the thing under test --
    `order=(0, 2, 1, 3)` reseats the cycle as `0, 2, 1, 3`, pairing
    `{0, 2}`/`{1, 3}` instead of `{0, 1}`/`{2, 3}` for whatever a caller
    measures against it.
    """
    seq = tuple(order) if order is not None else tuple(range(learners))
    if sorted(seq) != list(range(learners)):
        raise ValueError(
            f"learner order must be a permutation of 0..{learners - 1}: {seq}"
        )

    def caster(index: int) -> Cast:
        return tuple(seq[(seat + index) % learners] for seat in range(players))

    return caster


def rotating(lineup: Sequence[int]) -> Caster:
    """`arena.compete`'s own seating, read as a caster.

    A tournament rotates a fixed lineup through every seat -- entrant `e`
    sits at `arena.seat_of(e, index, seats)` -- so that over any `seats`
    consecutive indices each entrant holds each seat exactly once. `lineup`
    is that same lineup written as policy ids: `(0, 0, 1, 1)` is the duel
    `compete` runs, the learner on two adjacent seats and the reference on
    the other two, cycling by index.

    It is here rather than left inside `compete` because a batched
    evaluation deals the same boards under the same law without going
    through the tournament's per-position job at all, and a seating law
    written twice is how two evaluations of the same pair come to disagree
    about what they measured.
    """
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
    """The complementary cast: ids `a` and `b` exchange seats on every index.

    This is the other half of an antithetic pair -- same index, so the same
    board and the same dice, with the two sides' seat sets traded, which is
    what lets a caller average a board's two readings and difference the
    seat term out exactly rather than hope it cancels in the mean.

    Exchanging the two ids is the complement for *any* cast, which a seat
    shift is not: `arena.compete` writes its own complement as a shift of
    `seats // 2`, and that is only the complement of its adjacent lineup --
    the same shift leaves `alternating`'s interleaved cast untouched.
    `swapped(alternating(n))` is `alternating(n, flip=True)`, and
    `swapped(rotating((0, 0, 1, 1)))` is `compete`'s antithetic partner.
    """

    def caster_swapped(index: int) -> Cast:
        return tuple(
            b if pid == a else a if pid == b else pid for pid in caster(index)
        )

    return caster_swapped
