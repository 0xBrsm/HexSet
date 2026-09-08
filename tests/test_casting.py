# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

from hexset.arena import seat_of
from hexset.casting import (
    alternating,
    league_rotation,
    paired,
    rotating,
    swapped,
)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_alternating_is_a_pure_function_of_the_index(players):
    caster = alternating(players)
    other = alternating(players)  # a fresh caster, not the same closure
    for index in range(20):
        assert caster(index) == caster(index) == other(index)


def test_paired_is_a_pure_function_of_the_index():
    caster = paired(alternating(4))
    other = paired(alternating(4))
    for index in range(20):
        assert caster(index) == caster(index) == other(index)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_league_rotation_is_a_pure_function_of_the_index(players):
    caster = league_rotation(players, players)
    other = league_rotation(players, players)
    for index in range(20):
        assert caster(index) == caster(index) == other(index)


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_alternating_flip_is_the_seat_complement(players):
    plain = alternating(players)
    flipped = alternating(players, flip=True)
    for index in range(20):
        assert flipped(index) == tuple(1 - seat for seat in plain(index))


def test_paired_gives_identical_casts_to_2k_and_2k_plus_1():
    caster = alternating(4)
    doubled = paired(caster)
    for k in range(10):
        assert doubled(2 * k) == doubled(2 * k + 1) == caster(k)


@pytest.mark.parametrize("players", [3, 4, 5])
def test_league_rotation_gives_every_learner_every_seat_equally_often(players):
    """Over any window of `players` consecutive indices, each seat is held by
    every learner exactly once -- balanced whichever window a caller starts
    counting from, not just from index 0."""
    learners = players
    caster = league_rotation(learners, players)
    for start in range(-5, 10):
        window = [caster(i) for i in range(start, start + players)]
        for seat in range(players):
            held_by = sorted(cast[seat] for cast in window)
            assert held_by == list(range(learners))


def test_league_rotation_order_permutes_who_sits_next_to_whom():
    default = league_rotation(4, 4)
    reseated = league_rotation(4, 4, order=(0, 2, 1, 3))
    casts = [default(i) for i in range(4)]
    reseated_casts = [reseated(i) for i in range(4)]
    assert casts != reseated_casts
    # still balanced: every learner still holds every seat once per window.
    for seat in range(4):
        assert sorted(cast[seat] for cast in reseated_casts) == [0, 1, 2, 3]


def test_league_rotation_rejects_an_order_that_is_not_a_permutation():
    with pytest.raises(ValueError):
        league_rotation(4, 4, order=(0, 1, 1, 3))


@pytest.mark.parametrize("players", [2, 4, 6])
def test_rotating_is_arenas_own_seating(players):
    """`compete`'s lineup rotation, read as a caster over the game index."""
    lineup = tuple(0 if entrant < players // 2 else 1 for entrant in range(players))
    caster = rotating(lineup)
    for index in range(12):
        cast = caster(index)
        assert cast == caster(index)  # pure in the index, like every caster here
        for entrant, pid in enumerate(lineup):
            assert cast[seat_of(entrant, index, players)] == pid


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_swapped_alternating_is_alternatings_own_flip(players):
    """One complement law: exchanging the ids is what `flip` already did."""
    plain = alternating(players)
    for index in range(20):
        assert swapped(plain)(index) == alternating(players, flip=True)(index)


def test_swapped_rotating_is_competes_antithetic_partner():
    """`compete` shifts the seats by `seats // 2`; the ids exchange instead.

    The two are the same permutation of its adjacent lineup, which is what
    lets a batched duel deal `compete`'s second half without repeating the
    tournament's inline `divmod`.
    """
    lineup = (0, 0, 1, 1)
    caster = rotating(lineup)
    for pair in range(12):
        shifted = [0] * 4
        for entrant, pid in enumerate(lineup):
            shifted[seat_of(entrant, pair + 4 // 2, 4)] = pid
        assert swapped(caster)(pair) == tuple(shifted)
