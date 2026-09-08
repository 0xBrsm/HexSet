# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes`: many games in flight, stepped in lockstep.

The claim under test is that a lane is not a second engine. A game played
through the lane environment must be the *same* game `hexset.arena` plays for
the same `(seed, index)` -- same actions, same cleared trades, same winner --
however many lanes are in flight, because both deal from the one game law
(`hexset.arena.deal_game`) and both seat a gate at every seat.

Imports nothing from `pettingzoo`/`gymnasium` and skips nothing: the lane
environment is engine-only, so it runs on a plain `pip install hexset`.
"""

from __future__ import annotations

import pytest

from hexset import arena
from hexset.arena import Entrant, _play_one
from hexset.bots.search2 import options_for
from hexset.gym.lanes import LaneEnv

SEED = 11
CAP = 2000


class Scripted:
    """A deterministic bot with a real trade gate, and no generator at all.

    Determinism is what lets one bot object serve every seat of a lane while
    `hexset.arena` spawns one per entrant and the two still play the same game:
    a bot that tie-broke from its own `random.Random` would play differently
    depending on how many of it there were.

    The gate wants one resource, chosen by the *view's* perspective, so the
    seats of a table sharing one bot object still want different things and
    trades clear between them. Its gain is strictly increasing in what it
    wants, which is what `hexset.trading.trade_event` requires of a gate.
    """

    trade_floor = 0.0

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


class Refuses:
    """A gate that declines everything, so its seat never trades."""

    trade_floor = 0.0

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def spawn(board):
    return Scripted()


def actions_of(episode):
    """The episode's decision stream in `hexset.record.Record`'s action shape."""
    return [
        (int(d.action.type), d.action.a, d.action.b) for d in episode.stream()
    ]


@pytest.fixture
def scripted_kind():
    """`Scripted` as an `arena` entrant kind, so `_play_one` can seat it."""
    arena.register_entrant_kind("test-scripted", lambda entrant, board, rng: Scripted())
    try:
        yield Entrant("scripted", "test-scripted")
    finally:
        arena._ENTRANT_KIND_FACTORIES.pop("test-scripted", None)


def played_by_the_arena(entrant, index, players=4, cap=CAP):
    """The same `(seed, index)` game, played by `hexset.arena` with a record."""
    return _play_one(((entrant,) * players, index, SEED, cap, False, True))


def test_a_lane_plays_the_arenas_game_for_the_same_seed_and_index(scripted_kind):
    """The whole point of the shared game law: one lane, one game, no drift."""
    index = 3
    theirs = played_by_the_arena(scripted_kind, index)

    env = LaneEnv(4, SEED, 3, deal=1, action_cap=CAP, first_game=index, bots={0: spawn})
    (episode,) = env.drain()

    assert actions_of(episode) == list(theirs.record.actions)
    assert episode.outcome.winner == theirs.seat
    assert episode.outcome.turns == theirs.turns
    assert episode.outcome.truncated is False
    assert episode.index == index and episode.seed == SEED


def test_lane_count_does_not_change_the_games():
    """A game is a function of `(seed, index)`, not of how it was scheduled."""
    played = {}
    for lanes in (1, 4, 16):
        env = LaneEnv(4, SEED, lanes, deal=4, action_cap=CAP, bots={0: spawn})
        played[lanes] = {
            e.index: (actions_of(e), e.trades, e.outcome) for e in env.drain()
        }
        assert sorted(played[lanes]) == [0, 1, 2, 3]
        assert env.running is False

    assert played[1] == played[4] == played[16]


def test_the_action_cap_truncates_and_says_so():
    """A capped game is stopped, not finished, and the outcome distinguishes them."""
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=40, bots={0: spawn})
    episodes = env.drain()

    assert [e.outcome.actions for e in episodes] == [40, 40]
    assert all(e.outcome.truncated for e in episodes)
    assert all(e.outcome.winner is None for e in episodes)
    assert all(len(e) == 40 for e in episodes)

    uncapped = LaneEnv(4, SEED, 2, deal=1, action_cap=CAP, bots={0: spawn}).drain()
    assert uncapped[0].outcome.truncated is False


def test_the_trade_census_matches_the_engines_own(scripted_kind):
    """Trades are not actions, so the census is the only record the stream has.

    Compared against `hexset.arena`'s independently written census of the same
    game (`_play_and_record`'s before/after diff around its own `apply`), not
    merely against itself.
    """
    index = 3
    theirs = played_by_the_arena(scripted_kind, index)

    env = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, first_game=index, bots={0: spawn})
    (episode,) = env.drain()

    assert episode.trades  # the gate above trades; a vacuous pass would not do
    assert tuple(episode.trades) == tuple(theirs.record.trades)
    assert episode.outcome.trades == len(episode.trades)
    assert all(step < episode.outcome.actions for step, _, _, _ in episode.trades)


def test_a_caster_routes_seats_and_seats_every_seats_gate():
    """Policy 0 is the caller's; 1..3 are bots, gated by the bots themselves."""
    bots: dict[int, Scripted] = {}

    def one_bot(board):
        return bots.setdefault(id(board), Scripted())

    refusals = Refuses()
    cast = (0, 1, 1, 1)
    env = LaneEnv(
        4,
        SEED,
        2,
        deal=2,
        action_cap=600,
        caster=lambda index: cast,
        bots={1: one_bot},
        gates={0: lambda game, seat: refusals},
    )

    episodes = []
    while env.running:
        requests = env.requests()
        for request in requests:
            assert request.policy == cast[request.seat]
            assert request.view.perspective == request.seat
            assert request.options
            gates = request.game.gates
            assert gates[0] is refusals
            assert all(gate in bots.values() for gate in gates[1:])
        # The mapping form: the caller answers only the seats it holds, and the
        # bot seats are played by their own bots.
        episodes.extend(
            env.step(
                {r.lane: r.options[0] for r in requests if r.policy == 0}
            )
        )

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.cast == cast
        assert episode.decisions[0], "the caller's seat took no decisions"
        assert all(d.policy == 0 for d in episode.decisions[0])
        assert episode.trades, "the bots' gates cleared nothing"
        # Seat 0 refuses everything, so every exchange is between two bot seats.
        assert all({a, b} <= {1, 2, 3} for _, a, b, _ in episode.trades)
