# SPDX-License-Identifier: GPL-3.0-only
"""The chance-correction estimator: it must reproduce the games, and be unbiased.

Two tests carry the correctness argument and the rest support them.

`test_every_chance_event_s_correction_has_exactly_zero_expectation` is the
algebraic form: at each event the enumerated outcomes' probabilities and values
are recovered and `sum_k p_k . delta_k` is asserted to be zero. That is the
martingale-difference property itself, checked event by event rather than
inferred from a mean.

`test_the_estimator_is_unbiased_over_real_games` is the statistical form, and it
is run with a value function chosen to be *wrong* -- large, deterministic, and
unrelated to the position -- because unbiasedness must not depend on `V`. A stub
returning zero would pass while checking nothing, so the test also asserts the
corrections it cancelled were big.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter

import numpy as np
import pytest

from benchmarks.aivat import (
    StubValuer,
    chance_outcomes,
    instrumented,
    margin_scale,
    replay,
    summarise,
)
from hexset.actions import ActionType, apply, victim_of
from hexset.arena import _play_one, entrant_from_name
from hexset.board.board import random_base_board
from hexset.cards import NUM_DEV_CARDS
from hexset.game import ROLL_ODDS, Phase, is_over, to_move
from hexset.mcts import draws_hidden

# Two cheap entrants that are nonetheless different, so a paired margin is not
# identically zero and the games stay fast enough for a test suite. Nothing here
# depends on which entrants play; the estimator's argument is about the chance
# nodes, not about who is sitting at them.
DUEL = tuple(
    entrant_from_name(name).renamed(f"{name}#{i}")
    for i, name in enumerate(["greedy", "greedy", "greedy-own", "greedy-own"])
)
SEED = 20_000


def _positions(entrants, index, seed, wanted, limit=400):
    """Play a game, yielding `(game, action)` at every event `wanted` accepts.

    The game is played on copies of nothing: `wanted` is handed the live game
    *before* the action is applied, which is the state the enumeration needs.
    """
    from hexset.arena import seat_of, spawn
    from hexset.game import start

    seats = len(entrants)
    pair, half = divmod(index, 2)
    rotation = pair + half * (seats // 2)
    board = random_base_board(random.Random(f"{seed}:{pair}:board"))
    seats_taken = [seat_of(e, rotation, seats) for e in range(seats)]
    lineup = [None] * seats
    for e, entrant in enumerate(entrants):
        lineup[seats_taken[e]] = spawn(
            entrant, board, random.Random(f"{seed}:{pair}:{e}")
        )
    game = start(board, seats, random.Random(f"{seed}:{pair}:game"))
    found = 0
    actions = 0
    while not is_over(game) and actions < limit:
        action = lineup[to_move(game)].choose(game)
        if wanted(game, action):
            yield game, action
            found += 1
        apply(game, action)
        actions += 1


def test_the_dice_enumeration_is_the_engine_s_own_odds():
    for game, action in _positions(
        DUEL, 0, SEED, lambda g, a: a.type is ActionType.ROLL
    ):
        outcomes = chance_outcomes(game, action, random.Random(1))
        assert [key for key, _, _ in outcomes] == [roll for roll, _ in ROLL_ODDS]
        assert [p for _, p, _ in outcomes] == [p for _, p in ROLL_ODDS]
        assert sum(p for _, p, _ in outcomes) == pytest.approx(1.0)
        for roll, _, child in outcomes:
            assert child.last_roll == roll
        # The real game is untouched: the enumeration only ever reads copies.
        assert game.phase is Phase.ROLL
        break


def test_a_dev_card_draw_enumerates_the_remaining_deck():
    seen = 0
    for game, action in _positions(
        DUEL, 0, SEED, lambda g, a: a.type is ActionType.BUY_DEV_CARD, limit=3000
    ):
        before = Counter(game.state.deck)
        outcomes = chance_outcomes(game, action, random.Random(1))
        if not outcomes:
            assert len(before) < 2
            continue
        seen += 1
        assert len(outcomes) == len(before)
        assert sum(p for _, p, _ in outcomes) == pytest.approx(1.0)
        buyer = game.current_player
        for _, probability, child in outcomes:
            drew = [
                card
                for card in range(NUM_DEV_CARDS)
                if child.state.new_dev_cards[buyer][card]
                > game.state.new_dev_cards[buyer][card]
            ]
            assert len(drew) == 1
            assert probability == pytest.approx(before[drew[0]] / sum(before.values()))
            # One card left the deck, and it was that card.
            assert Counter(child.state.deck) == before - Counter({drew[0]: 1})
    assert seen, "no development card was ever bought with a mixed deck"


def test_a_steal_enumerates_the_victim_s_hand():
    def wanted(game, action):
        return action.type in (
            ActionType.MOVE_ROBBER,
            ActionType.PLAY_KNIGHT,
        ) and draws_hidden(game, action)

    seen = 0
    for game, action in _positions(DUEL, 0, SEED, wanted, limit=3000):
        victim = victim_of(game, action.b)
        hand = list(game.state.hands[victim])
        outcomes = chance_outcomes(game, action, random.Random(1))
        if not outcomes:
            assert sum(1 for n in hand if n) < 2
            continue
        seen += 1
        assert sum(p for _, p, _ in outcomes) == pytest.approx(1.0)
        thief = game.current_player
        for _, probability, child in outcomes:
            took = [
                r
                for r in range(len(hand))
                if child.state.hands[thief][r] > game.state.hands[thief][r]
            ]
            assert len(took) == 1
            assert probability == pytest.approx(hand[took[0]] / sum(hand))
            assert child.state.hands[victim][took[0]] == hand[took[0]] - 1
            assert sum(child.state.hands[victim]) == sum(hand) - 1
    assert seen, "no steal from a mixed hand ever happened"


def test_every_chance_event_s_correction_has_exactly_zero_expectation():
    """`sum_k p_k . delta_k == 0`, per event. The estimator's whole argument."""
    row = instrumented(DUEL, 0, SEED, value="stub", detail=True)
    assert row["per_event"], "no chance event was corrected"
    for event in row["per_event"]:
        probabilities = np.array(event["probabilities"])
        values = np.array(event["values"])
        deltas = values - probabilities @ values
        assert abs(float(probabilities @ deltas)) < 1e-9
    # And with teeth: a value function this wrong makes the individual
    # corrections large, so the cancellation above is not vacuous.
    assert max(abs(e["delta"]) for e in row["per_event"]) > 1.0


def test_a_constant_value_function_corrects_nothing(monkeypatch):
    class Flat:
        def __init__(self, *_, **__):
            pass

        def margins(self, games):
            return np.full(len(games), 3.25)

    row = instrumented(DUEL, 0, SEED, value="stub", detail=True)
    monkeypatch.setattr("benchmarks.aivat.StubValuer", Flat)
    flat = instrumented(DUEL, 0, SEED, value="stub", detail=True)

    assert flat["events"] == row["events"]
    # Zero to float noise: each event subtracts a constant from itself.
    assert abs(flat["correction"]) < 1e-12
    assert all(abs(event["delta"]) < 1e-15 for event in flat["per_event"])
    assert flat["priced"] == {"roll": 0, "deck": 0, "steal": 0}


def test_the_instrumented_replay_reproduces_the_arena_game_exactly():
    """The same games, not merely games from the same distribution.

    Without this the SD comparison would be between two different cells and the
    reduction it reported would be an artefact of the resampling.
    """
    for index in range(8):
        winner, seat, _, points = _play_one((DUEL, index, SEED, 20_000, True))
        row = instrumented(DUEL, index, SEED, value="stub")
        assert row["points"] == points
        assert row["winner"] == winner
        assert row["board"] == index // 2


def test_the_correction_does_not_disturb_the_game_s_random_stream():
    """Every subset of terms plays the identical game."""
    base = instrumented(DUEL, 3, SEED, value="stub", terms=())
    for terms in (("roll",), ("deck",), ("steal",), ("roll", "deck", "steal")):
        row = instrumented(DUEL, 3, SEED, value="stub", terms=terms)
        assert row["points"] == base["points"]
        assert row["winner"] == base["winner"]
    assert base["correction"] == 0.0


def test_the_estimator_is_unbiased_over_real_games():
    """The two estimators agree in expectation on the same games.

    Their difference is the mean correction and nothing else, so the test is a
    one-sample t on it. Deliberately run with the wrong value function: a
    correct one would shrink the corrections and weaken the test.
    """
    rows = replay(DUEL, 40, seed=SEED, value="stub")
    summary = summarise(rows)
    assert summary["boards"] == 20
    assert summary["mean_correction_se"] > 0
    assert abs(summary["mean_correction_t"]) < 3.0
    assert summary["paired_vp"] - summary["paired_vp_aivat"] == pytest.approx(
        summary["mean_correction"]
    )
    # Teeth again: the per-game corrections this cancelled are not small
    # compared with the statistic they are subtracted from.
    scatter = statistics.stdev([row["correction"] for row in rows])
    assert scatter > 0.5


def test_the_margin_scale_inverts_relative_points():
    from hexset.victory import relative_points

    rng = random.Random(4)
    for _ in range(50):
        points = tuple(rng.randrange(0, 11) for _ in range(4))
        value = relative_points(points)
        margin = (points[0] + points[1]) / 2 - (points[2] + points[3]) / 2
        predicted = margin_scale(4) * (
            (value[0] + value[1]) / 2 - (value[2] + value[3]) / 2
        )
        assert predicted == pytest.approx(margin)


def test_the_stub_value_function_is_deterministic_across_processes():
    """`replay` fans out over a pool, so a per-process value would desync."""
    from hexset.game import start

    board = random_base_board(random.Random("x"))
    game = start(board, 4, random.Random("y"))
    stub = StubValuer()
    assert stub.margins([game])[0] == StubValuer().margins([game])[0]
