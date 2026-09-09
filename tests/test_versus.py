# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bench.versus`: the tournament's verdict over the lane driver.

The claim under test is that batching the policy call changed the plumbing
and nothing else. A duel run here must play the *same boards under the same
casts* as `hexset.arena.compete` running the same lineup at the same seed --
the pairing law is one law, not two that agree by inspection -- and the
numbers it reports must be the ones the episodes themselves say, computed
with `arena.wilson` on the same counts.

Everything is driven by a deterministic scripted bot, so a difference in a
number can only be a difference in the harness.
"""

from __future__ import annotations

import pytest

from hexset import arena
from hexset.arena import Entrant, compete, wilson
from hexset.bench.versus import BotPolicy, compete_batched
from hexset.bots.search2 import options_for
from hexset.record import replay

SEED = 11
CAP = 2000
GAMES = 8
LINEUP = (0, 0, 1, 1)


class Scripted:
    """A deterministic bot with a real trade gate, and no generator at all.

    Determinism is what lets `compete`'s four spawned bots and the lane
    driver's one-per-board bench play the same game: a bot that tie-broke
    from its own `random.Random` would play differently depending on how
    many of it there were and which stream each got.

    The gate wants one resource, chosen by the *view's* perspective, so the
    seats of a table sharing one bot object still want different things and
    trades clear between them -- a trade-free duel would pass this file's
    comparisons vacuously.
    """

    trade_floor = 0.0

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


def spawn(board):
    return Scripted()


def two_policies():
    """One `BotPolicy` per id: separate benches, identical behaviour."""
    return {0: BotPolicy(spawn), 1: BotPolicy(spawn)}


@pytest.fixture
def scripted_kind():
    """`Scripted` as an `arena` entrant kind, so `compete` can seat it."""
    arena.register_entrant_kind("test-scripted", lambda entrant, board, rng: Scripted())
    try:
        yield Entrant("scripted", "test-scripted")
    finally:
        arena._ENTRANT_KIND_FACTORIES.pop("test-scripted", None)


def tournament_boards(tournament):
    """`compete`'s games as `{(board index, cast): (winner seat, turns, points)}`.

    Reconstructed from what a `Tournament` reports rather than from the
    pairing law being tested: `seating` says which seat each entrant took,
    and antithetic pairing makes game `i`'s board the pair `i // 2`.
    """
    out = {}
    for game in range(tournament.games):
        seating = tournament.seating[game]
        cast = [0] * len(seating)
        points = [0] * len(seating)
        for entrant, pid in enumerate(LINEUP):
            cast[seating[entrant]] = pid
            points[seating[entrant]] = tournament.points[game][entrant]
        winner = tournament.winners[game]
        out[(game // 2, tuple(cast))] = (
            None if winner is None else seating[winner],
            tournament.turns[game],
            tuple(points),
        )
    return out


def test_the_pairing_law_is_the_tournaments_own(scripted_kind):
    """Same boards, same casts, same games -- one pairing law, not two.

    `compete` derives the pair inline from the game index; this derives it
    from a caster over the same index. The two must land on exactly the
    same set of (board, cast) plays, and each of those plays must be the
    same game: same winning seat, same turns, same terminal points.
    """
    tournament = compete(
        [scripted_kind] * 4, GAMES, seed=SEED, action_cap=CAP, antithetic=True
    )
    verdict = compete_batched(
        two_policies(),
        GAMES,
        players=4,
        seed=SEED,
        lanes=4,
        action_cap=CAP,
        episodes=True,
    )

    played = {
        (e.index, e.cast): (e.outcome.winner, e.outcome.turns, e.outcome.points)
        for e in verdict.episodes
    }
    assert len(played) == GAMES  # every (board, cast) distinct: 4 boards, both ways
    assert played == tournament_boards(tournament)
    # Both casts of a board are the same board under exchanged ids.
    for board in range(GAMES // 2):
        one, other = sorted(cast for index, cast in played if index == board)
        assert other == tuple(1 - pid for pid in one)


def test_the_verdict_is_what_the_episodes_say(scripted_kind):
    """Wins and the paired margin, recomputed by hand off the same games."""
    verdict = compete_batched(
        two_policies(),
        GAMES,
        players=4,
        seed=SEED,
        lanes=4,
        action_cap=CAP,
        episodes=True,
    )

    margins: dict[int, list[float]] = {}
    wins = 0
    for episode in verdict.episodes:
        mine = [s for s, pid in enumerate(episode.cast) if pid == 0]
        theirs = [s for s, pid in enumerate(episode.cast) if pid == 1]
        points = episode.outcome.points
        margins.setdefault(episode.index, []).append(
            sum(points[s] for s in mine) / len(mine)
            - sum(points[s] for s in theirs) / len(theirs)
        )
        wins += episode.outcome.winner in mine
    paired = [sum(pair) / 2 for pair in margins.values()]

    assert verdict.games == GAMES
    assert verdict.boards == len(paired) == GAMES // 2
    assert verdict.wins == wins
    assert verdict.win_rate == pytest.approx(wins / GAMES)
    assert verdict.paired_vp == pytest.approx(sum(paired) / len(paired))
    assert verdict.turns_max == max(e.outcome.turns for e in verdict.episodes)
    assert sum(s.wins for s in verdict.standings) == sum(
        1 for e in verdict.episodes if e.outcome.winner is not None
    )
    assert verdict.metrics()["paired_vp"] == verdict.paired_vp


def test_a_capped_game_is_truncated_and_not_exhausted():
    """Two different faults: the cap stopped it, the turns did not run out."""
    verdict = compete_batched(
        two_policies(), 4, players=4, seed=SEED, lanes=2, action_cap=40
    )

    assert verdict.games == 4 and verdict.boards == 2
    assert verdict.truncated == 4
    assert verdict.exhausted == 0
    assert verdict.wins == 0

    uncapped = compete_batched(
        two_policies(), 4, players=4, seed=SEED, lanes=2, action_cap=CAP
    )
    assert uncapped.truncated == 0


def test_lane_count_does_not_change_the_verdict():
    """A game is a function of `(seed, index)`, not of how it was scheduled."""
    one = compete_batched(
        two_policies(), 4, players=4, seed=SEED, lanes=1, action_cap=CAP
    )
    many = compete_batched(
        two_policies(), 4, players=4, seed=SEED, lanes=8, action_cap=CAP
    )

    # `seconds` is the wall clock, which is exactly what changing the lane
    # count is meant to change; every other metric is the verdict.
    def verdict_of(v):
        return {k: value for k, value in v.metrics().items() if k != "seconds"}

    assert verdict_of(one) == verdict_of(many)
    assert [s.wins for s in one.standings] == [s.wins for s in many.standings]


def test_the_win_interval_is_the_arenas_wilson():
    """No second interval: the same function on the same counts."""
    verdict = compete_batched(
        two_policies(), 4, players=4, seed=SEED, lanes=2, action_cap=CAP
    )

    assert (verdict.wilson_low, verdict.wilson_high) == wilson(
        verdict.wins, verdict.games
    )
    assert verdict.standings[0].interval() == wilson(
        verdict.standings[0].wins, verdict.games
    )


def test_records_ride_along_with_the_episodes():
    """`records=` reaches the lanes, so a duel's episodes carry their games.

    What `Tournament.records` is to `compete`: the games the verdict counted
    are the games the file holds, because both come out of the one run.
    """
    verdict = compete_batched(
        two_policies(), GAMES, seed=SEED, action_cap=CAP, episodes=True, records=True
    )

    assert len(verdict.episodes) == GAMES
    for episode in verdict.episodes:
        game = replay(episode.record)
        assert (game.won_by, game.turns) == (
            episode.outcome.winner,
            episode.outcome.turns,
        )

    # A record only reaches a caller on an episode, so asking for records
    # without them buys nothing and costs nothing.
    quiet = compete_batched(
        two_policies(), GAMES, seed=SEED, action_cap=CAP, records=True
    )
    assert quiet.episodes == ()
    assert quiet.metrics()["seconds"] == quiet.seconds
