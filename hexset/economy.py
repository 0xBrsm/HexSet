# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from enum import IntEnum

from .board.ports import BASE_TRADE_RATIO
from .board.terrain import NUM_RESOURCES, Resource, check_resource
from .state import BANK_PER_RESOURCE, GameState, is_hidden, pile_size, production

__all__ = [
    "Purchase",
    "COSTS",
    "hand_size",
    "can_afford",
    "check_afford",
    "pay",
    "trade_ratios",
    "bank_trade",
    "distribute",
    "total_in_play",
    "expected_total",
]


class Purchase(IntEnum):
    """What a seat can buy; `COSTS` prices each."""

    ROAD = 0
    SETTLEMENT = 1
    CITY = 2
    DEV_CARD = 3


def _cost(**amounts: int) -> tuple[int, ...]:
    cost = [0] * NUM_RESOURCES
    for name, count in amounts.items():
        cost[Resource[name.upper()]] = count
    return tuple(cost)


COSTS: dict[Purchase, tuple[int, ...]] = {
    Purchase.ROAD: _cost(wood=1, brick=1),
    Purchase.SETTLEMENT: _cost(wood=1, brick=1, sheep=1, wheat=1),
    Purchase.CITY: _cost(wheat=2, ore=3),
    Purchase.DEV_CARD: _cost(sheep=1, wheat=1, ore=1),
}


def hand_size(state: GameState, player: int) -> int:
    """How many resource cards a seat holds. Works on a referee's state and
    on an observed one alike, so every size-only reader calls this rather than
    summing `state.hands[player]`."""
    return pile_size(state.hands[player])


def can_afford(state: GameState, player: int, purchase: Purchase) -> bool:
    """Whether `player`'s hand covers `COSTS[purchase]`. An identity read: a
    hidden hand raises `HiddenRead`; `check_afford` checks one by count."""
    hand = state.hands[player]
    return all(hand[r] >= n for r, n in enumerate(COSTS[purchase]))


def check_afford(state: GameState, player: int, purchase: Purchase) -> None:
    """Raise `ValueError` unless `pay` would succeed; writes nothing. Of a
    hidden hand only the count can be checked, as `pay` checks it."""
    hand = state.hands[player]
    cost = COSTS[purchase]
    if is_hidden(hand):
        if len(hand) < sum(cost):
            raise ValueError(
                f"player {player} holds {len(hand)} cards, too few for {purchase.name}")
    elif not can_afford(state, player, purchase):
        raise ValueError(f"player {player} cannot afford {purchase.name}")


def pay(state: GameState, player: int, purchase: Purchase) -> None:
    """Move `COSTS[purchase]` from `player`'s hand to the bank. Raises
    `ValueError` as `check_afford` does."""
    check_afford(state, player, purchase)
    hand = state.hands[player]
    if is_hidden(hand):
        # An observed state: the host already ruled the purchase legal, and
        # what is paid is public, so the cards go back to the bank by name.
        _pay_unseen(state, hand, COSTS[purchase], player, purchase.name)
        return
    for r, n in enumerate(COSTS[purchase]):
        hand[r] -= n
        state.bank[r] += n


def _pay_unseen(state: GameState, hand, cost, player: int, what: str) -> None:
    """A seat whose hand is hidden pays `cost` to the bank. Only the count can
    be checked; that it held the types is the table's word."""
    if len(hand) < sum(cost):
        raise ValueError(f"player {player} holds {len(hand)} cards, too few for {what}")
    for r, n in enumerate(cost):
        if n:
            hand.move(r, -n)
            state.bank[r] += n


def trade_ratios(state: GameState, player: int) -> list[int]:
    """The cheapest rate this player can trade each resource at. A port is
    usable once the player has a building on either of its two vertices; a
    generic port improves every resource, a specific port its own."""
    ratios = [BASE_TRADE_RATIO] * NUM_RESOURCES
    for port in state.board.ports:
        if not any(state.vertex_owner[v] == player for v in port.vertices):
            continue
        if port.resource is None:
            ratios = [min(r, port.ratio) for r in ratios]
        else:
            ratios[port.resource] = min(ratios[port.resource], port.ratio)
    return ratios


def bank_trade(state: GameState, player: int, give: Resource, receive: Resource) -> None:
    """Trade `player`'s `give`, at its `trade_ratios` rate, to the bank for one
    `receive`. Raises `ValueError` when either names no resource, `give` is
    `receive`, the hand holds too few `give`, or the bank has no `receive`."""
    check_resource(give)
    check_resource(receive)
    if give == receive:
        raise ValueError("cannot trade a resource for itself")
    ratio = trade_ratios(state, player)[give]
    hand = state.hands[player]
    if is_hidden(hand):
        if state.bank[receive] < 1:
            raise ValueError(f"bank has no {receive.name}")
        cost = [0] * NUM_RESOURCES
        cost[give] = ratio
        _pay_unseen(state, hand, cost, player, f"{ratio} {give.name}")
        hand.move(receive, 1)
        state.bank[receive] -= 1
        return
    if hand[give] < ratio:
        raise ValueError(f"player {player} needs {ratio} {give.name} to trade")
    if state.bank[receive] < 1:
        raise ValueError(f"bank has no {receive.name}")

    hand[give] -= ratio
    state.bank[give] += ratio
    hand[receive] += 1
    state.bank[receive] -= 1


def distribute(state: GameState, roll: int) -> list[list[int]]:
    """Pay out production for `roll`, applying the bank shortage rule: if the
    bank cannot cover every claim on a resource a lone claimant takes what is
    left, but when several are owed nobody receives any. Hence not per hex."""
    gross = production(state, roll)
    granted = [[0] * NUM_RESOURCES for _ in range(state.num_players)]

    for r in range(NUM_RESOURCES):
        claimants = [p for p in range(state.num_players) if gross[p][r]]
        if not claimants:
            continue
        demand = sum(gross[p][r] for p in claimants)
        if demand <= state.bank[r]:
            for p in claimants:
                granted[p][r] = gross[p][r]
        elif len(claimants) == 1:
            granted[claimants[0]][r] = state.bank[r]

    for p, gains in enumerate(granted):
        hand = state.hands[p]
        unseen = is_hidden(hand)
        for r, n in enumerate(gains):
            if not n:
                continue
            if unseen:
                hand.move(r, n)     # production is public: by name, into the count
            else:
                hand[r] += n
            state.bank[r] -= n

    return granted


def total_in_play(state: GameState) -> int:
    """Resource cards in the bank and every hand together, for comparing with
    `expected_total()`."""
    return sum(state.bank) + sum(
        hand_size(state, p) for p in range(state.num_players)
    )


def expected_total() -> int:
    """How many resource cards the game holds: `BANK_PER_RESOURCE` of each."""
    return BANK_PER_RESOURCE * NUM_RESOURCES
