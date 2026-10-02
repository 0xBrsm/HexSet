# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from .board.terrain import NUM_RESOURCES, Resource, check_resource
from .cards import (
    NUM_DEV_CARDS,
    PLAYABLE,
    YEAR_OF_PLENTY_RESOURCES,
    DevCard,
)
from .economy import Purchase, can_afford, pay
from .state import GameState, HiddenRead, is_hidden, pile_size

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .chance import Chance

__all__ = [
    "can_buy",
    "buy",
    "buy_drawn",
    "mature",
    "dev_count",
    "holdings",
    "can_play",
    "check_play",
    "spend_card",
    "play_knight",
    "play_year_of_plenty",
    "play_monopoly",
]


def can_buy(state: GameState, player: int) -> bool:
    """Whether the deck has a card and `player` can pay for it. With cards in the
    deck it reads the hand's types, so a hidden hand raises `HiddenRead`."""
    return bool(state.deck) and can_afford(state, player, Purchase.DEV_CARD)


def buy(state: GameState, player: int) -> DevCard:
    """Buy the top card. It is held aside until the turn ends."""
    if not state.deck:
        raise ValueError("the development deck is empty")
    if is_hidden(state.deck):
        raise HiddenRead("cannot draw an unknown development card; sample a concrete state first")
    pay(state, player, Purchase.DEV_CARD)
    card = DevCard(state.deck.pop())
    state.new_dev_cards[player][card] += 1
    _age_bought(state, player)
    return card


def _age_bought(state: GameState, player: int) -> None:
    ages = state.dev_ages[player]
    if ages is not None:
        ages.append(0)


def buy_drawn(state: GameState, player: int, card: int) -> DevCard | int:
    """Buy from a deck known only by its length -- an observed state, whose
    host drew the card. `card` is what it drew for a seat whose cards this
    state shows, and `chance.UNSEEN` for one whose it hides: that seat's
    fresh cards grow by one, of no type anyone here can read."""
    if not state.deck:
        raise ValueError("the development deck is empty")
    fresh = state.new_dev_cards[player]
    if not is_hidden(fresh) and not 0 <= card < NUM_DEV_CARDS:
        raise ValueError(f"seat {player}'s cards are seen here: which card did it draw?")
    pay(state, player, Purchase.DEV_CARD)
    state.deck.add(-1)
    _age_bought(state, player)
    if is_hidden(fresh):
        fresh.add(1)
        return card
    fresh[card] += 1
    return DevCard(card)


def mature(state: GameState, player: int) -> None:
    """Make this turn's purchases playable, and every card this seat holds a
    turn older. Call when the turn ends."""
    ages = state.dev_ages[player]
    if ages is not None:
        ages[:] = [age + 1 for age in ages]
    bought = state.new_dev_cards[player]
    if is_hidden(bought):
        state.dev_cards[player].add(len(bought))
        bought.add(-len(bought))
        return
    for card, count in enumerate(bought):
        state.dev_cards[player][card] += count
        bought[card] = 0


def _age_played(state: GameState, player: int) -> None:
    """A play takes off the youngest card that could be played: this turn's
    purchases are the last entries, the matured ones before them."""
    ages = state.dev_ages[player]
    if ages is None:
        return
    at = len(ages) - pile_size(state.new_dev_cards[player]) - 1
    if at >= 0:
        del ages[at]


def dev_count(state: GameState, player: int) -> int:
    """How many development cards a seat holds, matured and fresh together.
    The size-only counterpart to `holdings`: public in the rules, so it reads
    on an observed state too."""
    return pile_size(state.dev_cards[player]) + pile_size(state.new_dev_cards[player])


def holdings(state: GameState, player: int) -> list[int]:
    """Every card held, playable or not — what the player would reveal on
    winning. An *identity* read: on an observed state a hidden holding raises
    `state.HiddenRead`. For the number alone, call `dev_count`."""
    return [
        held + fresh
        for held, fresh in zip(state.dev_cards[player], state.new_dev_cards[player])
    ]


def can_play(state: GameState, player: int, card: DevCard) -> bool:
    """Whether `card` is a playable kind and `player` holds one bought before
    this turn. An identity read: for a playable kind a hidden holding raises
    `HiddenRead`, which `check_play` avoids by checking the count."""
    return card in PLAYABLE and state.dev_cards[player][card] > 0


def check_play(state: GameState, player: int, card: DevCard) -> None:
    """Raise `ValueError` unless `player` may spend `card`; writes nothing.
    Of a hidden holding only the count can be checked: the card is played
    face up, so it is public from here, but the holding it came out of
    never was."""
    held = state.dev_cards[player]
    if is_hidden(held):
        if card not in PLAYABLE or len(held) < 1:
            raise ValueError(f"player {player} cannot play {card.name}")
    elif not can_play(state, player, card):
        raise ValueError(f"player {player} cannot play {card.name}")


def spend_card(state: GameState, player: int, card: DevCard) -> None:
    """Take one `card` out of `player`'s playable holding, applying none of its
    effect. Raises `ValueError` as `check_play` does."""
    check_play(state, player, card)
    _age_played(state, player)
    held = state.dev_cards[player]
    if is_hidden(held):
        held.add(-1)
    else:
        held[card] -= 1


def play_knight(state: GameState, player: int) -> None:
    """Spend the card and credit the play towards Largest Army eligibility.
    Moving the robber and stealing happen afterwards, through the same robber
    phase a seven uses."""
    spend_card(state, player, DevCard.KNIGHT)
    state.knights_played[player] += 1


def play_year_of_plenty(state: GameState, player: int, resources: list[Resource]) -> None:
    """Spend the card and take the named `resources`, repeats allowed, from the
    bank. Raises `ValueError`, writing nothing, unless exactly
    `YEAR_OF_PLENTY_RESOURCES` are named, the card may be played and the bank
    holds them all."""
    check_play(state, player, DevCard.YEAR_OF_PLENTY)
    if len(resources) != YEAR_OF_PLENTY_RESOURCES:
        raise ValueError(f"choose exactly {YEAR_OF_PLENTY_RESOURCES} resources")
    wanted = [0] * NUM_RESOURCES
    for resource in resources:
        check_resource(resource)
        wanted[resource] += 1
    if any(n > state.bank[r] for r, n in enumerate(wanted)):
        raise ValueError("the bank cannot supply that")

    spend_card(state, player, DevCard.YEAR_OF_PLENTY)
    state.dev_cards_played[DevCard.YEAR_OF_PLENTY] += 1
    hand = state.hands[player]
    for resource, count in enumerate(wanted):
        if not count:
            continue
        state.bank[resource] -= count
        if is_hidden(hand):
            hand.move(resource, count)
        else:
            hand[resource] += count


def play_monopoly(
    state: GameState, player: int, resource: Resource, chance: "Chance | None" = None
) -> int:
    """Every other seat hands over all of `resource`. How many a hidden hand
    held is not this state's to read: on an observed state `chance` is the
    host's (`chance.Hosted`), which says what each such seat surrendered --
    publicly, as it was surrendered at the table.

    Every surrender is settled before any card moves, so a play the host
    cannot account for leaves the state as it was."""
    check_play(state, player, DevCard.MONOPOLY)
    check_resource(resource)
    surrendered: list[tuple[int, int]] = []
    for other in range(state.num_players):
        if other == player:
            continue
        hand = state.hands[other]
        if is_hidden(hand):
            if chance is None:
                raise HiddenRead(
                    f"seat {other}'s hand is hidden: the host must say what it surrendered"
                )
            given = chance.surrender(hand, int(resource))
            if not 0 <= given <= len(hand):
                raise ValueError(f"seat {other} holds {len(hand)} cards, not {given}")
        else:
            given = hand[resource]
        surrendered.append((other, given))

    spend_card(state, player, DevCard.MONOPOLY)
    state.dev_cards_played[DevCard.MONOPOLY] += 1
    taken = 0
    for other, given in surrendered:
        hand = state.hands[other]
        if is_hidden(hand):
            hand.move(resource, -given)
        else:
            hand[resource] = 0
        taken += given
    mine = state.hands[player]
    if is_hidden(mine):
        mine.move(resource, taken)
    else:
        mine[resource] += taken
    return taken
