"""`NetworkBot.gains_across`: `gains_many` at several positions in one forward."""

from __future__ import annotations

import random
from dataclasses import dataclass

from hexset.actions import ActionSpace, build_space
from hexset.board.board import random_base_board
from hexset.clients.netbot import bot_for
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.play import step_randomly
from hexset.trading import UNLIMITED, TradeParams

PLAYERS = 4


@dataclass
class HandPolicy:
    """A value head that reads each seat's hand as the row's position holds
    it: a card of resource r is worth (r + 1) hundredths, a hidden card one,
    so a one-for-one swap moves the value and the gain is not trivially 0."""

    space: ActionSpace
    calls: int = 0

    def value_rows(self, rows):
        self.calls += 1
        out = []
        for game, _ in rows:
            vals = []
            for hand in game._state.hands:
                if isinstance(hand, list):
                    vals.append(sum(0.01 * (r + 1) * n for r, n in enumerate(hand)))
                else:
                    vals.append(0.01 * hand.count)
            out.append(tuple(vals))
        return out

    def act_rows(self, rows):
        return [min(options, key=self.space.index) for _, _, options in rows]


@dataclass(frozen=True)
class Checkpoint:
    policy: HandPolicy
    space: ActionSpace
    players: int = PLAYERS
    trade_floor: float = UNLIMITED.trade_floor
    gate_plies: int = UNLIMITED.gate_plies
    trade_params: TradeParams | None = None


def positions(wanted=6):
    """Main-phase positions, `(game, seat)`, half with `seat` on turn (its own
    frame) and half with another seat on turn, each holding cards to give."""
    rng = random.Random(7)
    game = start(random_base_board(rng), PLAYERS, rng)
    own, other = [], []
    for _ in range(4000):
        if is_over(game) or (len(own) >= wanted // 2 and len(other) >= wanted // 2):
            break
        if game.phase is Phase.MAIN:
            mover = to_move(game)
            for seat in range(PLAYERS):
                if sum(game._state.hands[seat]) >= 2:
                    bucket = own if seat == mover else other
                    if len(bucket) < wanted // 2:
                        bucket.append((imagine(game, random.Random(1), randomize_deck=False), seat))
                    break
        step_randomly(game, rng)
    return own + other


def swaps(game, seat):
    """Every one-for-one exchange `seat` could give for, to each other seat."""
    hand = game.state(seat).known[seat]
    received, counterparties = [], []
    for y in range(5):
        if hand[y] <= 0:
            continue
        for r in range(5):
            if r == y:
                continue
            for them in range(PLAYERS):
                if them != seat:
                    received.append(tuple(1 if i == r else -1 if i == y else 0 for i in range(5)))
                    counterparties.append(them)
    return received, counterparties


def test_gains_across_prices_each_position_as_gains_many_does_in_one_forward():
    found = positions()
    board = found[0][0]._state.board
    space = build_space(board.topology.num_vertices, board.topology.num_edges,
                        board.topology.num_hexes, PLAYERS)
    bot = bot_for(Checkpoint(policy=HandPolicy(space), space=space), rng=random.Random(0))
    home = found[0][0]
    bot.seat_at(home)
    for seat in range(PLAYERS):
        group = [(g, *swaps(g, s)) for g, s in found if s == seat]
        if not group:
            continue
        expected = []
        for g, received, counterparties in group:
            bot.seat_at(g)
            expected.append(bot.gains_many(g.state(seat), received, counterparties))
        bot.seat_at(home)
        bot.policy.calls = 0
        own = sum(1 for g, *_ in group if to_move(g) == seat)
        across = bot.gains_across(seat, group)
        assert across == expected
        assert any(x not in (-1.0, 0.0) for row in across for x in row)
        assert bot._seated is home                      # left seated where it was
        # One forward for every own-frame position; the others are priced on their own.
        assert bot.policy.calls == (1 if own else 0) + (len(group) - own)
