# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes` gives every bot seat a bot of its own for its game,
built off a stream keyed by the run's seed, the game and the seat
(`hexset.arena.seat_key`): a bot that draws never repeats another game's or
seat's numbers, and draws the same ones whatever the lane count or worker
split."""

from __future__ import annotations

import random

from hexset.actions import options_for
from hexset.arena import seat_key
from hexset.bench.versus import BotPolicy, compete_batched
from hexset.board.board import random_base_board
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.gym.lanes import BoardBots, LaneEnv as _LaneEnv
from hexset.record import actions_of
from journals import fresh

SEED = 11
CAP = 160


def LaneEnv(*args, **kwargs):
    kwargs.setdefault("turn_cap", UNSTRUCTURED_TURN_CAP)
    return _LaneEnv(*args, **kwargs)


class Drawer:
    """A bot without `seen` that draws every move off its stream, and keeps
    each draw: what a sampling opponent does."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.draws: list[int] = []

    def choose(self, game):
        options = options_for(game)
        pick = self.rng.randrange(1 << 30)
        self.draws.append(pick)
        return options[pick % len(options)]


def drawers(made: list) -> BoardBots:
    def seat_spawn(board, rng):
        bot = Drawer(rng)
        made.append(bot)
        return bot

    return BoardBots(lambda board: seat_spawn(board, random.Random(SEED)), seat_spawn=seat_spawn)


def seated(env):
    """Drain `env`, keeping each game's seat bots by `(index, seat)`, and its
    episodes."""
    bots: dict[tuple[int, int], Drawer] = {}
    episodes = []
    while env.running:
        for lane in env._lanes:
            if lane is not None:
                for seat, bot in enumerate(lane.seat_bots):
                    assert bots.setdefault((lane.index, seat), bot) is bot
        episodes.extend(env.step([None] * len(env.requests())))
    return bots, episodes


def test_a_drawing_bot_draws_its_own_stream_in_every_game_and_seat():
    """One board pinned for every lane, where a bot shared per board would
    serve every seat of every game off one stream."""
    made: list[Drawer] = []
    env = LaneEnv(4, SEED, 2, deal=4, action_cap=CAP,
                  board=random_base_board(random.Random(0)), bots={0: drawers(made)})
    bots, _ = seated(env)
    assert len(bots) == len(made) == 4 * 4
    assert len({id(bot) for bot in bots.values()}) == 16, "a bot of its own per seat per game"
    openings = {tuple(bot.draws[:4]) for bot in bots.values()}
    assert len(openings) == 16, "no two games or seats draw alike"
    for (index, seat), bot in bots.items():
        stream = random.Random(seat_key(SEED, index, seat))
        assert bot.draws == [stream.randrange(1 << 30) for _ in bot.draws]


def test_the_same_game_and_seat_draw_the_same_whatever_the_lanes_or_workers():
    """One lane, three, and two workers sharding the indices: the same game
    and seat, the same draws and the same record."""
    runs = {}
    for name, shards in (("one", [(1, 0, 1)]), ("three", [(3, 0, 1)]),
                         ("workers", [(2, 0, 2), (2, 1, 2)])):
        draws, records = {}, {}
        for lanes, first, stride in shards:
            env = LaneEnv(4, SEED, lanes, deal=6 // stride, action_cap=CAP, first_game=first,
                          stride=stride, bots={0: drawers([])})
            bots, episodes = seated(env)
            draws.update((key, tuple(bot.draws)) for key, bot in bots.items())
            records.update((e.index, e.record) for e in episodes)
        assert sorted(records) == list(range(6))
        runs[name] = (draws, records)
    assert len(runs["one"][0]) == 6 * 4
    assert runs["one"] == runs["three"] == runs["workers"]


def test_a_deterministic_bot_plays_the_same_games_shared_or_its_own():
    """A bot that draws nothing plays the same moves whether a factory hands
    every seat one object or each seat its own: the records do not move."""

    class First:
        def choose(self, game):
            return options_for(game)[0]

    shared = First()
    one = LaneEnv(4, SEED, 2, deal=3, action_cap=CAP, bots={0: lambda board: shared}).drain()
    own = LaneEnv(4, SEED, 2, deal=3, action_cap=CAP, bots={0: lambda board: First()}).drain()
    assert {e.index: e.record for e in one} == {e.index: e.record for e in own}


def test_a_duel_plays_a_bot_policy_off_the_lanes_streams():
    """`compete_batched` hands a `BotPolicy` to the lanes: each of its seats
    a bot of its own per game off `seat_spawn`, and the same games as a lane
    driven directly."""
    made: list[Drawer] = []
    bench = drawers(made)
    verdict = compete_batched(
        {0: BotPolicy(bench.spawn, seat_spawn=bench.seat_spawn),
         1: BotPolicy(bench.spawn, seat_spawn=bench.seat_spawn)},
        4, players=4, seed=SEED, lanes=2, action_cap=CAP, turn_cap=UNSTRUCTURED_TURN_CAP,
        antithetic=False, episodes=True, journal=fresh(),
    )
    assert len(made) == 4 * 4
    assert len({tuple(bot.draws[:4]) for bot in made}) == 16
    alone = LaneEnv(4, SEED, 1, deal=4, action_cap=CAP, bots={0: drawers([])}).drain()
    assert ({e.index: list(actions_of(e.record)) for e in verdict.episodes}
            == {e.index: list(actions_of(e.record)) for e in alone})
