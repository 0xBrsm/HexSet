# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes` asks every bot of a class that chooses for many
(`hexset.bots.ChoosesMany`) in one call a tick, and the games are the ones
the same bots play asked one at a time."""

from __future__ import annotations

import random

from hexset.actions import options_for
from hexset.bots import chooses_many
from hexset.game import UNSTRUCTURED_TURN_CAP, Phase, to_move
from hexset.gym.lanes import BoardBots, LaneEnv as _LaneEnv
from hexset.trading import TradeParams

SEED = 11


def LaneEnv(*args, **kwargs):
    kwargs.setdefault("turn_cap", UNSTRUCTURED_TURN_CAP)
    return _LaneEnv(*args, **kwargs)


def key(action):
    return (int(action.type), action.a, action.b)


class Alone:
    """Chooses off its own stream and how many moves it has been shown, so a
    bot asked at the wrong position, or drawing another's numbers, plays
    another game."""

    trade_floor = 0.0
    trade_params = TradeParams(max_offers=1)

    def __init__(self, rng: random.Random) -> None:
        self.rng = random.Random(rng.getrandbits(64))
        self.shown = 0

    def seen(self, game, actor, action, viewer):
        self.shown += 1

    def choose(self, game):
        options = options_for(game)
        return options[(self.shown + self.rng.randrange(1 << 16)) % len(options)]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


class Together(Alone):
    """`Alone`, asked for many at once: each call is kept, and its bots are
    answered last to first to show the order does not matter."""

    calls: list[list[tuple[int, int]]] = []

    @classmethod
    def choose_many(cls, pairs):
        asked = []
        for bot, game in pairs:
            assert isinstance(bot, cls)
            seat = to_move(game)
            if game.phase is Phase.DISCARD:
                # Asked as `choose` is asked: an out-of-order discard's seat is
                # the one `to_move` names.
                assert game.discard_quota[seat] > 0
            asked.append((id(game), seat))
        cls.calls.append(asked)
        return [bot.choose(game) for bot, game in reversed(pairs)][::-1]


def bench(kind) -> BoardBots:
    return BoardBots(lambda board: kind(random.Random(0)), seat_spawn=lambda board, rng: kind(rng))


def play(kind, lanes: int, *, games: int = 4, first_game: int = 0, stride: int = 1):
    env = LaneEnv(4, SEED, lanes, deal=games, action_cap=800, first_game=first_game, stride=stride,
                  bots={0: bench(kind), 2: lambda board: Alone(random.Random(1))},
                  caster=lambda i: (1, 0, 2, 0) if i % 2 else (0, 0, 1, 0), discard_order="random")

    def answer(requests):
        return [r.options[r.step % len(r.options)] if r.policy == 1 else None for r in requests]

    return {e.index: [(d.seat, key(d.action)) for d in e.stream()] for e in env.drain(answer)}


def test_the_hook_is_found_on_the_class():
    assert chooses_many(Together(random.Random(0)))
    assert not chooses_many(Alone(random.Random(0)))


def test_asked_together_the_games_are_the_ones_played_one_at_a_time():
    """Every lane count and worker split: the same moves in every game, the
    out-of-order discards (`discard_order="random"`) included."""
    alone = play(Alone, 1)
    Together.calls = []
    for lanes in (2, 4):
        assert play(Together, lanes) == alone, f"{lanes} lanes"
    halves = {**play(Together, 3, games=2, first_game=0, stride=2),
              **play(Together, 3, games=2, first_game=1, stride=2)}
    assert halves == alone
    assert any(len(call) > 1 for call in Together.calls), "no tick asked more than one bot"


def test_one_call_a_tick_each_bot_at_its_own_game():
    Together.calls = []
    env = LaneEnv(4, SEED, 4, deal=4, action_cap=600, bots={0: bench(Together)})
    ticks = 0
    while env.running:
        env.step([None] * len(env.requests()))
        ticks += 1
    assert len(Together.calls) == ticks
    for call in Together.calls:
        games = [game for game, _ in call]
        assert len(games) == len(set(games))
