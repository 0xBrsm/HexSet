# SPDX-License-Identifier: GPL-3.0-only
"""An observed game (`game.observe`) played in lockstep with the game it
observes stays equal to what that seat can see of it -- every public field,
the ledger exactly, and its own legal actions.

This is the whole contract of the stepping half of `state.Hidden`: a seat at
a table somebody else hosts keeps one `Game` and applies the table's events
to it through the engine's own transitions, with the host supplying what the
seat cannot draw. The lockstep below plays the true game forward and feeds
each observer exactly what that seat would have been shown: the dice, a
steal's card only to the two seats party to it, a purchase's card only to the
buyer, and every seat's Monopoly surrender.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.chance import UNSEEN, ChanceExhausted, ChanceMismatch, Hosted
from hexset.game import (
    UNSTRUCTURED_TURN_CAP,
    Phase,
    is_over,
    observe,
    players_owing_discards,
    start,
    to_move,
)
from hexset.state import HiddenHand, HiddenRead, is_hidden, observed_by, pile_size
from hexset.trading import execute_agreed
from hexset.victory import public_victory_points


def _pick(options, rng):
    """A random legal action, ending the turn only now and then, so a game
    builds, buys and trades enough to exercise every transition."""
    doing = [a for a in options if a.type is not ActionType.END_TURN]
    if doing and rng.random() < 0.85:
        return rng.choice(doing)
    return rng.choice(options)


def _feed(hands_before, fresh_before, true, action, actor, observers):
    """Queue on every observer what its seat would be shown of `action`,
    which has just been applied to `true`."""
    state = true._state
    kind = action.type
    for seat, seen in observers.items():
        host = seen.chance
        if kind is ActionType.ROLL:
            host.expect("roll", true.last_roll)
        elif kind is ActionType.MOVE_ROBBER and action.b < state.num_players:
            victim = action.b
            lost = [b - a for a, b in zip(state.hands[victim], hands_before[victim])]
            if any(lost):
                card = lost.index(1)
                host.expect("steal", card if seat in (actor, victim) else UNSEEN)
        elif kind is ActionType.BUY_DEV_CARD:
            drawn = [b - a for a, b in zip(fresh_before, state.new_dev_cards[actor])]
            host.expect("draw", drawn.index(1) if seat == actor else UNSEEN)
        elif kind is ActionType.PLAY_MONOPOLY:
            for other in range(state.num_players):
                if other in (actor, seat):
                    continue
                host.expect("surrender", hands_before[other][action.a])


def _check(true, seen, seat):
    """`seen` is exactly `observed_by(true, seat)`, plus the same game fields
    and the same ledger -- the ledger is common knowledge, identical from
    every seat."""
    want = observed_by(true._state, seat)
    got = seen._state
    for name in ("vertex_owner", "vertex_building", "edge_owner", "robber", "bank",
                 "knights_played", "dev_cards_played", "road_lengths",
                 "longest_road_holder", "largest_army_holder"):
        assert getattr(got, name) == getattr(want, name), name
    for pile in ("hands", "dev_cards", "new_dev_cards"):
        for s in range(true.num_players):
            mine, theirs = getattr(got, pile)[s], getattr(want, pile)[s]
            assert is_hidden(mine) == is_hidden(theirs), (pile, s)
            assert (pile_size(mine) if is_hidden(mine) else list(mine)) == (
                pile_size(theirs) if is_hidden(theirs) else list(theirs)), (pile, s)
    assert len(got.deck) == len(want.deck)
    for name in ("phase", "current_player", "setup_step", "last_settlement", "last_roll",
                 "dev_card_played", "discard_quota", "free_roads", "resume_phase", "turns"):
        assert getattr(seen, name) == getattr(true, name), name
    assert seen.ledger == true.ledger
    if not is_over(true):
        assert legal_actions(seen, seat) == legal_actions(true, seat)


def _lockstep(seed: int, players: int, max_steps: int = 2500, trades: bool = True):
    rng = random.Random(seed)
    true = start(random_base_board(rng), players, rng, turn_cap=UNSTRUCTURED_TURN_CAP)
    observers = {seat: observe(true, seat) for seat in range(players)}
    for seat, seen in observers.items():
        _check(true, seen, seat)
    for _ in range(max_steps):
        if is_over(true):
            break
        if true.phase is Phase.DISCARD:
            actor = rng.choice(players_owing_discards(true))
        else:
            actor = to_move(true)
        action = _pick(legal_actions(true, actor), rng)
        hands_before = [hand[:] for hand in true._state.hands]
        fresh_before = true._state.new_dev_cards[actor][:]
        apply(true, action, seat=actor)
        _feed(hands_before, fresh_before, true, action, actor, observers)
        for seat, seen in observers.items():
            apply(seen, action, seat=actor)
            assert not seen.chance.pending, f"seat {seat} left {seen.chance.pending} unread"
        if trades and true.phase is Phase.MAIN and rng.random() < 0.3:
            _random_trade(true, observers, rng)
        for seat, seen in observers.items():
            if is_over(true) and not is_over(seen):
                # Won on victory point cards nobody else can count: the host
                # announces it. Only a hidden holding can do that.
                winner = true.won_by
                assert winner != seat
                assert public_victory_points(true._state, winner) < (
                    true._state.rules.winning_points)
                continue
            _check(true, seen, seat)
    return true


def _random_trade(true, observers, rng):
    """A trade the table made, between the current player and anyone who can
    cover a one-for-one: public, and applied to every observer as the host's
    word, with nobody asked."""
    state = true._state
    me = true.current_player
    others = [s for s in range(true.num_players) if s != me]
    rng.shuffle(others)
    for them in others:
        mine = [r for r in range(NUM_RESOURCES) if state.hands[me][r]]
        theirs = [r for r in range(NUM_RESOURCES) if state.hands[them][r]]
        pairs = [(g, t) for g in mine for t in theirs if g != t]
        if not pairs:
            continue
        give, take = rng.choice(pairs)
        received = [0] * NUM_RESOURCES
        received[give] -= 1
        received[take] += 1
        execute_agreed(true, me, them, tuple(received), ask_actor=False, ask_counterparty=False)
        for seen in observers.values():
            execute_agreed(seen, me, them, tuple(received), ask_actor=False,
                           ask_counterparty=False)
        return


@pytest.mark.parametrize("seed, players", [(3, 2), (2, 4)])
def test_an_observed_game_stays_what_its_seat_sees_of_the_true_one(seed, players):
    true = _lockstep(seed, players)
    assert true.turns > 10, "the lockstep should have played a real game"


def test_the_lockstep_reaches_every_transition():
    """Guard on the property above: the games it plays do roll sevens, steal,
    buy, play every kind of card and trade -- or the equality would hold
    over a game that never touched a hidden pile."""
    seen = set()
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        true = start(random_base_board(rng), 4, rng, turn_cap=UNSTRUCTURED_TURN_CAP)
        for _ in range(2500):
            if is_over(true):
                break
            actor = (rng.choice(players_owing_discards(true))
                     if true.phase is Phase.DISCARD else to_move(true))
            action = _pick(legal_actions(true, actor), rng)
            seen.add(action.type)
            apply(true, action, seat=actor)
    for kind in (ActionType.DISCARD, ActionType.MOVE_ROBBER, ActionType.BUY_DEV_CARD,
                 ActionType.PLAY_KNIGHT, ActionType.BANK_TRADE, ActionType.BUILD_CITY):
        assert kind in seen, kind


def test_a_monopoly_takes_what_each_hidden_seat_surrendered():
    rng = random.Random(4)
    true = start(random_base_board(rng), 3, rng)
    seen = observe(true, 0)
    for game in (true, seen):
        game.phase = Phase.MAIN
        game.current_player = 0
    true._state.hands[1][2] = 3
    true._state.hands[2][2] = 1
    true._state.bank[2] -= 4
    true._state.dev_cards[0][4] = 1         # a Monopoly
    seen._state.dev_cards[0][4] = 1
    for other, n in ((1, 3), (2, 1)):
        seen._state.hands[other] = HiddenHand(n)
    seen._state.bank[2] -= 4
    seen.chance.expect("surrender", 3)
    seen.chance.expect("surrender", 1)

    apply(true, Action(ActionType.PLAY_MONOPOLY, 2))
    apply(seen, Action(ActionType.PLAY_MONOPOLY, 2))

    assert seen._state.hands[0][2] == true._state.hands[0][2] == 4
    assert [len(seen._state.hands[s]) for s in (1, 2)] == [0, 0]
    # Public: the ledger knows each seat gave them up.
    assert seen.ledger.seats[0].known[2] == 4


def test_a_hosted_source_refuses_what_it_was_not_told():
    host = Hosted()
    with pytest.raises(ChanceExhausted):
        host.roll()
    host.expect("steal", UNSEEN)
    with pytest.raises(ChanceMismatch):
        host.roll()
    with pytest.raises(ValueError):
        host.expect("deck", 3)


def test_another_seats_hidden_hand_is_still_unreadable():
    """Taking public changes by name does not make the pile readable: the
    flow is bookkeeping for the ledger, not a composition."""
    hand = HiddenHand(2)
    hand.move(1, 1)
    assert len(hand) == 3
    with pytest.raises(HiddenRead):
        hand[1]
    with pytest.raises(HiddenRead):
        sum(hand)
    assert hand == HiddenHand(3)
    with pytest.raises(ValueError):
        hand.move(0, -4)


def test_an_observer_does_not_name_a_steal_it_was_not_party_to():
    """Between two other seats a steal moves one card each way in the counts
    and nothing in the ledger but the common-knowledge `steal`."""
    rng = random.Random(9)
    true = start(random_base_board(rng), 3, rng)
    seen = observe(true, 0)
    seen._state.hands[2] = HiddenHand(2)
    seen.ledger.seats[2].unknown = 2
    for game in (seen,):
        game.phase = Phase.ROBBER
        game.current_player = 1
    target = next(h for h in range(seen._state.board.num_hexes) if h != seen._state.robber)
    # Seat 2 sits on the hex, so the move has to rob it.
    corner = seen._state.board.topology.hex_vertices[target][0]
    seen._state.vertex_owner[corner] = 2
    seen._state.vertex_building[corner] = 1
    seen.chance.expect("steal", UNSEEN)
    from hexset.game import move_robber_to

    move_robber_to(seen, target, victim=2)
    assert len(seen._state.hands[1]) == 1 and len(seen._state.hands[2]) == 1
    assert seen.ledger.seats[1].unknown == 1 and seen.ledger.seats[2].unknown == 1
