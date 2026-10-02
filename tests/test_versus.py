# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bench.versus`: the tournament's verdict over the lane driver. A duel
must play the same boards under the same casts as `arena.compete` at the same
seed. Everything is driven by a deterministic bot, so a difference in a number
can only be a difference in the harness.
"""

from __future__ import annotations

import pytest

from hexset import arena
from hexset.game import NO_TURN_CAP
from hexset.arena import Entrant, compete
from hexset.bench.versus import BotPolicy, compete_batched
from hexset.actions import options_for
from hexset.record import replay
from hexset.trading import TradeParams

SEED = 11
# These bots always take the first legal action -- roll, then end the turn --
# so no game of theirs is ever won. The comparison here is about seat pairing,
# not outcomes, so a small action cap bounds every game and the turn cap is
# switched off rather than left to fire first.
CAP = 80
GAMES = 4
LINEUP = (0, 0, 1, 1)


class Scripted:
    """A deterministic bot with a real trade gate. Its gate wants one resource
    chosen by the *view's* perspective, so seats sharing one bot object still
    trade; a trade-free duel would pass this file's comparisons vacuously.

    It declares one offer a turn: a gate that declares nothing has no limit
    and keeps offering while its menu has an offer it has not made, and the
    offer menu asks for any cards -- far more rounds than a harness
    comparison needs.
    """

    trade_floor = 0.0
    trade_params = TradeParams(max_offers=1)

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


def spawn(board):
    return Scripted()


def two_policies():
    return {0: BotPolicy(spawn), 1: BotPolicy(spawn)}


@pytest.fixture
def scripted_kind():
    arena.register_entrant_kind("test-scripted", lambda entrant, board, rng: Scripted())
    try:
        yield Entrant("scripted", "test-scripted")
    finally:
        arena.unregister_entrant_kind("test-scripted")


def tournament_boards(tournament):
    """Reconstructed from what a `Tournament` reports, not from the law under test."""
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


def batched(**overrides):
    options = dict(players=4, turn_cap=NO_TURN_CAP, seed=SEED, lanes=2, action_cap=CAP)
    options.update(overrides)
    return compete_batched(two_policies(), GAMES, **options)


@pytest.fixture(scope="module")
def verdict():
    """One small duel, shared: every test below reads it rather than replaying it."""
    return batched(episodes=True, records=True)


def test_the_pairing_law_is_the_tournaments_own(scripted_kind, verdict):
    """`compete` derives the pair inline; this derives it from a caster."""
    tournament = compete(
        [scripted_kind] * 4, GAMES, seed=SEED, action_cap=CAP, antithetic=True,
        turn_cap=NO_TURN_CAP,
    )

    played = {
        (e.index, e.cast): (e.outcome.winner, e.outcome.turns, e.outcome.points)
        for e in verdict.episodes
    }
    assert len(played) == GAMES
    assert played == tournament_boards(tournament)
    for board in range(GAMES // 2):
        one, other = sorted(cast for index, cast in played if index == board)
        assert other == tuple(1 - pid for pid in one)


def test_the_verdict_is_what_the_episodes_say(verdict):
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


def test_a_capped_game_is_truncated_and_a_turn_capped_one_stops_the_run():
    """As under `arena.compete`: a game the action cap cut short is a reading,
    and one that ran out of turns with no winner is a defect, never scored as
    a loss for whichever side it stalled."""
    capped = batched(action_cap=40)
    assert capped.games == 4 and capped.boards == 2
    assert (capped.truncated, capped.wins) == (4, 0)
    assert "exhausted" not in capped.metrics()

    with pytest.raises(arena.Exhausted, match="no winner") as stopped:
        batched(turn_cap=8)
    assert stopped.value.turns >= 8 and stopped.value.seed == SEED


def test_lane_count_does_not_change_the_verdict(verdict):
    """A game is a function of `(seed, index)`, not of how it was scheduled."""
    one = batched(lanes=1)

    def verdict_of(v):
        return {k: value for k, value in v.metrics().items() if k != "seconds"}

    assert verdict_of(one) == verdict_of(verdict)
    assert [s.wins for s in one.standings] == [s.wins for s in verdict.standings]


def test_records_ride_along_with_the_episodes(verdict):
    assert len(verdict.episodes) == GAMES
    for episode in verdict.episodes:
        game = replay(episode.record)
        assert (game.won_by, game.turns) == (
            episode.outcome.winner,
            episode.outcome.turns,
        )

    quiet = batched(records=True, action_cap=20)
    assert quiet.episodes == ()
    assert quiet.metrics()["seconds"] == quiet.seconds


def test_an_odd_antithetic_collection_is_rejected():
    with pytest.raises(ValueError, match="even number"):
        compete_batched(two_policies(), 3)


def test_one_board_batched_metrics_are_strict_json():
    import json

    verdict = compete_batched(two_policies(), 2, lanes=1, action_cap=2)
    metrics = verdict.metrics()
    assert metrics["paired_vp_low"] is None
    assert metrics["paired_vp_high"] is None
    json.dumps(metrics, allow_nan=False)


def test_policy_adapter_gate_is_the_checkpoints_own_bot(monkeypatch):
    from hexset.bench.versus import PolicyPolicy

    seated = []

    class Gate:
        def seat_at(self, game):
            seated.append(game)

    def bot_for(checkpoint):
        return Gate()

    monkeypatch.setattr("hexset.clients.netbot.bot_for", bot_for)
    adapter = PolicyPolicy(None, object())
    gate = adapter.gate("a game", 0)
    assert isinstance(gate, Gate)
    assert seated == ["a game"]
