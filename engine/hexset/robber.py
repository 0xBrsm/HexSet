# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from .board.terrain import Resource
from .state import NO_OWNER, GameState

DISCARD_THRESHOLD = 7


def occupants(state: GameState, hex_index: int) -> tuple[int, ...]:
    """Distinct players with a building on the given hex."""
    topology = state.board.topology
    found = {
        state.vertex_owner[v]
        for v in topology.hex_vertices[hex_index]
        if state.vertex_owner[v] != NO_OWNER
    }
    return tuple(sorted(found))


def victims(state: GameState, hex_index: int, thief: int) -> tuple[int, ...]:
    """Players the thief may steal from: present on the hex, and holding cards."""
    return tuple(
        p
        for p in occupants(state, hex_index)
        if p != thief and sum(state.hands[p]) > 0
    )


def move_robber(state: GameState, target: int) -> None:
    if not 0 <= target < state.board.num_hexes:
        raise ValueError(f"no such hex: {target}")
    if target == state.robber:
        raise ValueError("the robber must move to a different hex")
    state.robber = target


def steal(
    state: GameState, thief: int, victim: int, rng: random.Random
) -> Resource | None:
    """Take one card at random, so the chance of each resource follows the hand."""
    hand = state.hands[victim]
    total = sum(hand)
    if total == 0:
        return None

    pick = rng.randrange(total)
    for resource, count in enumerate(hand):
        if pick < count:
            hand[resource] -= 1
            state.hands[thief][resource] += 1
            return Resource(resource)
        pick -= count
    raise AssertionError("unreachable")


def discard_count(state: GameState, player: int) -> int:
    """How many cards a player must discard when a seven is rolled."""
    held = sum(state.hands[player])
    return held // 2 if held > DISCARD_THRESHOLD else 0


def discard(
    state: GameState, player: int, cards: list[int], required: int | None = None
) -> None:
    if required is None:
        required = discard_count(state, player)
    if sum(cards) != required:
        raise ValueError(f"player {player} must discard exactly {required}")
    hand = state.hands[player]
    for resource, count in enumerate(cards):
        if count > hand[resource]:
            raise ValueError(f"player {player} lacks {count} of resource {resource}")
    for resource, count in enumerate(cards):
        hand[resource] -= count
        state.bank[resource] += count


def random_discard(state: GameState, player: int, rng: random.Random) -> list[int]:
    cards = [0] * len(state.hands[player])
    for _ in range(discard_count(state, player)):
        pool = [r for r, n in enumerate(state.hands[player]) if n > cards[r]]
        cards[rng.choice(pool)] += 1
    discard(state, player, cards)
    return cards
