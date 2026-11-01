# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes` shows every move to a bot that reads the table
(`hexset.bots.watch`) as `hexset.arena.run_seated` does, and gives each of
its seats a bot of its own for the game (as every bot seat gets,
`test_lanes_streams`)."""

from __future__ import annotations

import random

import pytest

from hexset import arena
from hexset import decisions as D
from hexset.actions import ActionType, options_for
from hexset.arena import Entrant, _play_one
from hexset.board.board import random_base_board
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.gym.lanes import BoardBots, LaneEnv as _LaneEnv
from hexset.rules import STANDARD_GAME
from hexset.trading import TradeParams
from journals import fresh

SEED = 11
CAP = 600


def LaneEnv(*args, **kwargs):
    kwargs.setdefault("turn_cap", UNSTRUCTURED_TURN_CAP)
    return _LaneEnv(*args, **kwargs)


def key(action):
    return (int(action.type), action.a, action.b)


class Plain:
    """A deterministic bot with a trade gate and no `seen`: one object may
    serve every seat."""

    trade_floor = 0.0
    trade_params = TradeParams(max_offers=1)

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


class Watcher(Plain):
    """Keeps every move it is shown, `(actor, action, viewer)`, and chooses
    by how many it has been shown and the stream it was built off, so a move
    it missed, an extra one or another seat's stream changes its play."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.salt = 0 if rng is None else rng.getrandbits(16)
        self.calls: list[tuple[int, tuple, int]] = []
        self.chose: list[tuple] = []

    def seen(self, game, actor, action, viewer):
        self.calls.append((actor, key(action), viewer))

    def choose(self, game):
        options = options_for(game)
        picked = options[(len(self.calls) * 7 + self.salt) % len(options)]
        self.chose.append(key(picked))
        return picked


def bench(made: list) -> BoardBots:
    """A `BoardBots` of `Watcher`s, each kept in `made`."""

    def spawn(board, rng=None):
        bot = Watcher(rng)
        made.append(bot)
        return bot

    return BoardBots(spawn, seat_spawn=spawn)


def by_viewer(bots):
    """Each watching bot by the one seat it was shown moves from."""
    out = {}
    for bot in bots:
        if not bot.calls:
            continue
        viewers = {viewer for _, _, viewer in bot.calls}
        assert len(viewers) == 1, f"one bot was shown moves from seats {viewers}"
        (viewer,) = viewers
        assert viewer not in out, f"two bots watched from seat {viewer}"
        out[viewer] = bot
    return out


#: Games whose seating `hexset.arena` does not rotate (`seat_of`), so the
#: entrant in each seat draws the stream a lane hands that seat.
UNROTATED = (0, 4, 8)
#: Seat 1 plays without watching; the others watch.
LINEUP = ("watch", "plain", "watch", "watch")


@pytest.fixture(scope="module")
def played_both_ways():
    games = {}
    for index in UNROTATED:
        theirs: list[Watcher] = []

        def spawn_watcher(entrant, board, rng, theirs=theirs):
            bot = Watcher(rng)
            theirs.append(bot)
            return bot

        arena.register_entrant_kind("test-watch", spawn_watcher)
        arena.register_entrant_kind("test-plain", lambda entrant, board, rng: Plain())
        try:
            outcome = _play_one(
                (tuple(Entrant(name, f"test-{name}") for name in LINEUP), index, SEED, CAP, False,
                 "round", STANDARD_GAME, UNSTRUCTURED_TURN_CAP, "random"),
                fresh(),
            )
        finally:
            arena.unregister_entrant_kind("test-watch")
            arena.unregister_entrant_kind("test-plain")
        ours: list[Watcher] = []
        env = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, first_game=index,
                      caster=lambda i: tuple(0 if name == "watch" else 1 for name in LINEUP),
                      bots={0: bench(ours), 1: lambda board: Plain()})
        (episode,) = env.drain()
        games[index] = (outcome, by_viewer(theirs), episode, by_viewer(ours))
    return games


def test_a_lane_shows_a_watching_bot_what_the_arena_shows_it(played_both_ways):
    """The same calls in the same order, from the same seat, and the same
    choices, so the same game."""
    for index, (outcome, theirs, episode, ours) in played_both_ways.items():
        assert sorted(ours) == sorted(theirs) == [0, 2, 3]
        for seat in ours:
            assert ours[seat].calls, f"game {index}: seat {seat} was shown nothing"
            assert ours[seat].calls == theirs[seat].calls, f"game {index} seat {seat}"
            assert ours[seat].chose == theirs[seat].chose, f"game {index} seat {seat}"
        assert [(int(d.action.type), d.action.a, d.action.b) for d in episode.stream()] == list(
            outcome.record.actions
        )
        assert episode.trades == tuple(outcome.record.trades)


def test_every_move_is_shown_once_to_every_watcher_before_it_is_applied(played_both_ways):
    """Each watcher's calls are the game's whole move stream, the moves of
    the seat that does not watch included."""
    for _, _, episode, ours in played_both_ways.values():
        assert sorted(ours) == [0, 2, 3]
        stream = [(d.seat, key(d.action)) for d in episode.stream()]
        assert any(seat == 1 for seat, _ in stream)
        for seat, bot in ours.items():
            assert bot.calls == [(actor, action, seat) for actor, action in stream]


def test_a_caller_seat_and_out_of_order_discards_are_shown_too():
    """A caller's seat (policy 1, answered here) and a random discard order,
    which asks a bot to discard out of seat order (`LaneEnv._asked`)."""
    made: list[Watcher] = []
    env = LaneEnv(4, SEED, 3, deal=6, action_cap=1500, bots={0: bench(made)},
                  caster=lambda i: (1, 0, 0, 0), discard_order="random")

    def answer(requests):
        return [r.options[0] if r.policy == 1 else None for r in requests]

    episodes = env.drain(answer)
    assert len(episodes) == 6
    out_of_order = 0
    for episode in episodes:
        stream = episode.stream()
        for before, after in zip(stream, stream[1:]):
            if (before.action.type is ActionType.DISCARD and after.action.type is ActionType.DISCARD
                    and after.seat < before.seat):
                out_of_order += 1
    assert out_of_order, "no discard came out of seat order"
    # Three watching seats a game, each its own bot, each shown its own game.
    assert len(made) == 3 * len(episodes)
    streams = {tuple((d.seat, key(d.action)) for d in e.stream()) for e in episodes}
    shown = [bot for bot in made if bot.calls]
    assert len(shown) == 3 * len(episodes)
    for bot in shown:
        (viewer,) = {v for _, _, v in bot.calls}
        assert viewer in (1, 2, 3)
        assert tuple((a, k) for a, k, _ in bot.calls) in streams
    assert any(actor == 0 for bot in shown for actor, _, _ in bot.calls)


def test_each_part_of_a_compound_is_shown():
    """A seat answered in whole decisions (`decisions`): its compounds' parts
    are applied one at a time, and each is shown."""
    made: list[Watcher] = []
    env = LaneEnv(4, SEED, 2, deal=3, action_cap=1500, bots={1: bench(made)},
                  caster=lambda i: (0, 1, 1, 1), decisions={0})

    def answer(requests):
        out = []
        for r in requests:
            if r.decision is None:
                out.append(None)
                continue
            rng = random.Random(f"{r.index}:{r.step}")
            out.append(rng.choice(D.options(r.game, r.seat, double_turn=False)))
        return out

    episodes = env.drain(answer)
    streams = {tuple((d.seat, key(d.action)) for d in e.stream()) for e in episodes}
    shown = [bot for bot in made if bot.calls]
    assert len(shown) == 3 * len(episodes)
    for bot in shown:
        assert tuple((a, k) for a, k, _ in bot.calls) in streams
    # A setup turn is one compound, settlement then road: seat 0's moves
    # came in parts, each shown.
    for episode in episodes:
        ids = [d.decision_id for d in episode.decisions[0]]
        assert len(set(ids)) < len(ids)


def test_watching_bots_are_never_shared_across_seats_lanes_or_games():
    """One board pinned for every lane, where a bot without `seen` is one
    object for every seat of every lane."""
    made: list[Watcher] = []
    board = random_base_board(random.Random(0))
    env = LaneEnv(4, SEED, 2, deal=4, action_cap=300, board=board, bots={0: bench(made)})
    seated = []
    while env.running:
        seated.extend(lane.seat_bots for lane in env._lanes if lane is not None)
        env.step([None] * len(env.requests()))
    bots = {id(bot) for seats in seated for bot in seats}
    assert len(bots) == 4 * 4, "each seat of each game has a bot of its own"
    for bot in made:
        assert len({v for _, _, v in bot.calls}) == 1


def test_a_bot_that_does_not_watch_is_shown_nothing():
    """A factory that hands every seat one object (it draws nothing) is
    taken at its word, and a bot without `seen` is shown no move."""
    plain = Plain()
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=200, bots={0: lambda board: plain})
    gates = []
    while env.running:
        gates.extend(lane.game.gates for lane in env._lanes if lane is not None)
        assert all(not lane.watching for lane in env._lanes if lane is not None)
        env.step([None] * len(env.requests()))
    assert {id(gate) for seats in gates for gate in seats} == {id(plain)}


def test_seat_streams_are_the_games_and_seats_own():
    """`BoardBots.fresh` builds each seat's bot off a stream keyed by the
    run's seed, the game index and the seat: two lanes or one, the same."""
    salts = {}
    for lanes in (1, 3):
        made: list[Watcher] = []
        LaneEnv(4, SEED, lanes, deal=3, action_cap=120, bots={0: bench(made)}).drain()
        salts[lanes] = sorted(bot.salt for bot in made)
    assert salts[1] == salts[3]
    expected = sorted(random.Random(f"{SEED}:{i}:{s}").getrandbits(16) for i in range(3) for s in range(4))
    assert salts[1] == expected
