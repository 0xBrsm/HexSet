from __future__ import annotations

from enum import IntEnum

from .board.ports import BASE_TRADE_RATIO
from .board.terrain import NUM_RESOURCES, Resource
from .state import BANK_PER_RESOURCE, GameState, production

BANK_TRADE_RATIO = BASE_TRADE_RATIO


class Purchase(IntEnum):
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
    return sum(state.hands[player])


def can_afford(state: GameState, player: int, purchase: Purchase) -> bool:
    hand = state.hands[player]
    return all(hand[r] >= n for r, n in enumerate(COSTS[purchase]))


def pay(state: GameState, player: int, purchase: Purchase) -> None:
    if not can_afford(state, player, purchase):
        raise ValueError(f"player {player} cannot afford {purchase.name}")
    hand = state.hands[player]
    for r, n in enumerate(COSTS[purchase]):
        hand[r] -= n
        state.bank[r] += n


def trade_ratios(state: GameState, player: int) -> list[int]:
    """The cheapest rate this player can trade each resource at.

    A port is usable once the player has any building on either of its two
    vertices; a generic port improves every resource, a specific port only its
    own.
    """
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
    if give == receive:
        raise ValueError("cannot trade a resource for itself")
    ratio = trade_ratios(state, player)[give]
    if state.hands[player][give] < ratio:
        raise ValueError(f"player {player} needs {ratio} {give.name} to trade")
    if state.bank[receive] < 1:
        raise ValueError(f"bank has no {receive.name}")

    state.hands[player][give] -= ratio
    state.bank[give] += ratio
    state.hands[player][receive] += 1
    state.bank[receive] -= 1


def distribute(state: GameState, roll: int) -> list[list[int]]:
    """Pay out production for `roll`, applying the bank shortage rule.

    Official rule: if the bank cannot cover every claim on a resource, a lone
    claimant takes what is left, but when several players are owed nobody
    receives any of it. That is why this cannot be resolved per hex.
    """
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
        for r, n in enumerate(gains):
            state.hands[p][r] += n
            state.bank[r] -= n

    return granted


def total_in_play(state: GameState) -> int:
    return sum(state.bank) + sum(sum(hand) for hand in state.hands)


def expected_total() -> int:
    return BANK_PER_RESOURCE * NUM_RESOURCES
