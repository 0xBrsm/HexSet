from __future__ import annotations

import random

from .board.terrain import NUM_RESOURCES, Resource
from .cards import (
    PLAYABLE,
    YEAR_OF_PLENTY_RESOURCES,
    DevCard,
)
from .economy import Purchase, can_afford, pay
from .robber import move_robber, steal
from .state import GameState


def can_buy(state: GameState, player: int) -> bool:
    return bool(state.deck) and can_afford(state, player, Purchase.DEV_CARD)


def buy(state: GameState, player: int) -> DevCard:
    """Buy the top card. It is held aside until the turn ends."""
    if not state.deck:
        raise ValueError("the development deck is empty")
    pay(state, player, Purchase.DEV_CARD)
    card = DevCard(state.deck.pop())
    state.new_dev_cards[player][card] += 1
    return card


def mature(state: GameState, player: int) -> None:
    """Make this turn's purchases playable. Call when the turn ends."""
    bought = state.new_dev_cards[player]
    for card, count in enumerate(bought):
        state.dev_cards[player][card] += count
        bought[card] = 0


def holdings(state: GameState, player: int) -> list[int]:
    """Every card held, playable or not — what the player would reveal on winning."""
    return [
        held + fresh
        for held, fresh in zip(state.dev_cards[player], state.new_dev_cards[player])
    ]


def can_play(state: GameState, player: int, card: DevCard) -> bool:
    return card in PLAYABLE and state.dev_cards[player][card] > 0


def spend_card(state: GameState, player: int, card: DevCard) -> None:
    if not can_play(state, player, card):
        raise ValueError(f"player {player} cannot play {card.name}")
    state.dev_cards[player][card] -= 1


def play_knight(
    state: GameState,
    player: int,
    target: int,
    victim: int | None = None,
    rng: random.Random | None = None,
) -> Resource | None:
    spend_card(state, player, DevCard.KNIGHT)
    state.knights_played[player] += 1
    move_robber(state, target)
    if victim is None:
        return None
    return steal(state, player, victim, rng or random.Random())


def play_year_of_plenty(state: GameState, player: int, resources: list[Resource]) -> None:
    if len(resources) != YEAR_OF_PLENTY_RESOURCES:
        raise ValueError(f"choose exactly {YEAR_OF_PLENTY_RESOURCES} resources")
    wanted = [0] * NUM_RESOURCES
    for resource in resources:
        wanted[resource] += 1
    if any(n > state.bank[r] for r, n in enumerate(wanted)):
        raise ValueError("the bank cannot supply that")

    spend_card(state, player, DevCard.YEAR_OF_PLENTY)
    for resource, count in enumerate(wanted):
        state.bank[resource] -= count
        state.hands[player][resource] += count


def play_monopoly(state: GameState, player: int, resource: Resource) -> int:
    spend_card(state, player, DevCard.MONOPOLY)
    taken = 0
    for other in range(state.num_players):
        if other == player:
            continue
        taken += state.hands[other][resource]
        state.hands[other][resource] = 0
    state.hands[player][resource] += taken
    return taken
