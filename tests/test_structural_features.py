from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.bots.heximax.evaluate import Survey
from hexset.bots.heximax.structural_features import (
    AWARD_FRAGILITY_COEFFICIENT,
    BACKUP_PROGRESS_COEFFICIENT,
    StructuralEvaluator, StructuralHeximax,
    backup_progress,
)
from hexset.catanatron.structuralcfg import (
    ARMS, GATES, SCREEN_SEEDS, SCREEN_WORKERS, lineup, manifest, screen_jobs,
    source_fingerprint,
)
from hexset.state import new_game
from hexset.game import start


def _fixture_survey() -> Survey:
    # W,B,S,Wh,O: no settlement spot, and city/development/road remain open.
    return Survey(
        rate=0.0, kinds=0, scarce=0, buildings=0, port_gain=0,
        settlements=1, cities=0, roads=0, spots=0, ratios=(4, 4, 4, 4, 4),
    )


def test_backup_progress_reserves_cards_without_double_counting():
    walk = _fixture_survey()
    # The city is the existing best at 4/5=.8.  Its wheat and ore are
    # committed; only sheep remains for development, and only brick remains
    # for road in the second row.
    assert backup_progress((0, 0, 1, 1, 3), walk, deck_left=1) == pytest.approx(.4 / 3)
    assert backup_progress((0, 1, 0, 1, 3), walk, deck_left=1) == pytest.approx(.35 / 2)


def test_structural_scalar_and_batch_scores_agree():
    state = new_game(random_base_board(random.Random(7)), 3, random.Random(8))
    evaluator = StructuralEvaluator(
        state.board, backup=True, fragility=True,
    )
    hands = np.asarray([
        [[0, 0, 1, 1, 3], [1, 1, 0, 0, 0], [0, 1, 1, 0, 1]],
        [[1, 0, 1, 1, 2], [0, 2, 0, 1, 0], [1, 0, 1, 0, 1]],
    ], dtype=float)
    batch = evaluator.score_many(state, 0, hands)
    for row in range(len(hands)):
        for seat in range(state.num_players):
            scalar = evaluator.score(state, seat, hands[row, seat], knower=0)
            assert batch[row, seat] == pytest.approx(scalar, abs=1e-12)


def test_award_fragility_uses_public_holder_and_ties():
    state = new_game(random_base_board(random.Random(9)), 3, random.Random(10))
    evaluator = StructuralEvaluator(state.board, fragility=True)
    # A current largest-army holder remains vulnerable on a tie: denominator 1.
    state.largest_army_holder = 0
    state.knights_played[:] = [3, 3, 0]
    assert evaluator._award_fragility(state, 0) == pytest.approx(2.0)
    assert evaluator._award_fragility(state, 1) == pytest.approx(0.0)
    # No current holder means no hidden or inferred award is counted.
    state.largest_army_holder = -1
    evaluator.clear_structural_cache()
    assert evaluator._award_fragility(state, 0) == 0.0
    # Ownership changes invalidate the cache and transfer the vulnerability.
    state.largest_army_holder = 1
    assert evaluator._award_fragility(state, 1) == pytest.approx(2.0)


def test_award_fragility_clamps_negative_gap_monotonically(monkeypatch):
    state = new_game(random_base_board(random.Random(91)), 3, random.Random(92))
    evaluator = StructuralEvaluator(state.board, fragility=True)
    state.longest_road_holder = 0

    for gap in (-2, -1, 0, 1, 2):
        evaluator.clear_structural_cache()
        monkeypatch.setattr(
            "hexset.bots.heximax.structural_features.longest_road",
            lambda _state, seat, gap=gap: 5 + gap if seat == 0 else 5,
        )
        expected = 2.0 / (1.0 + max(0, gap))
        assert evaluator._award_fragility(state, 0) == pytest.approx(expected)


def test_award_fragility_stale_holder_fixture_is_finite(monkeypatch):
    state = new_game(random_base_board(random.Random(93)), 3, random.Random(94))
    evaluator = StructuralEvaluator(state.board, fragility=True)
    state.longest_road_holder = 0
    # This is the stale award-holder state that previously reached 2/(1-1).
    monkeypatch.setattr(
        "hexset.bots.heximax.structural_features.longest_road",
        lambda _state, seat: 4 if seat == 0 else 5,
    )
    assert evaluator._award_fragility(state, 0) == pytest.approx(2.0)


def test_structural_native_constructors_and_manifest():
    board = random_base_board(random.Random(11))
    for label, backup, fragility in (
        ("backup", True, False), ("fragility", False, True), ("combined", True, True),
    ):
        name = f"heximax-structural-{label}"
        bot = spawn(PRESETS[name], board, random.Random(12))
        assert bot.evaluator.backup_enabled is backup
        assert bot.evaluator.fragility_enabled is fragility
        assert bot.depth == 2 and bot.width == 6 and bot.max_nodes == 600 and bot.k == 1
        assert bot.max_trades == 0
        assert bot.evaluator.weights.buy_progress == pytest.approx(.45)


def test_structural_registration_does_not_change_ordinary_constructor():
    from hexset.bots.heximax.search import Heximax
    board = random_base_board(random.Random(16))
    ordinary = spawn(PRESETS["heximax-notrade"], board, random.Random(17))
    structural = spawn(PRESETS["heximax-structural-backup"], board, random.Random(18))
    assert type(ordinary) is Heximax
    assert type(structural) is StructuralHeximax


def test_screen_plan_has_four_arms_and_eight_disjoint_jobs():
    jobs = screen_jobs()
    assert ARMS == ("control", "backup", "fragility", "combined")
    assert len(jobs) == 8
    assert {(job["arm"], job["gate"]) for job in jobs} == {
        (arm, gate) for arm in ARMS for gate in GATES
    }
    assert len({job["seed"] for job in jobs}) == 8
    assert all(job["games"] == 120 and job["workers"] == 30 for job in jobs)
    assert AWARD_FRAGILITY_COEFFICIENT == -.25
    assert BACKUP_PROGRESS_COEFFICIENT == .45
    assert source_fingerprint()


def test_structural_artifact_validator_is_strict(tmp_path):
    from hexset.catanatron.structural_validation import validate_artifact
    good = {
        "family": "heximax-structural-future", "source_hash": source_fingerprint(),
        "phase": "confirmation", "arm": "backup", "candidate": "backup",
        "role": "candidate", "candidate_color": "Color.RED",
        "gate": "ab2", "games": 1024,
        "workers": SCREEN_WORKERS, "seed": 420_000_000,
        "players": lineup("backup", "ab2"),
        "phenotype": manifest()["candidates"]["backup"],
        "wins": {"Color.RED": 1024, "Color.WHITE": 0,
                 "Color.BLUE": 0, "Color.ORANGE": 0},
        "points": {"Color.RED": [5] * 1024, "Color.WHITE": [5] * 1024,
                   "Color.BLUE": [5] * 1024, "Color.ORANGE": [5] * 1024},
    }
    # The normalized schema must account for every game and identity must be
    # tied to the selected arm before an artifact is eligible.
    path = tmp_path / "good.json"
    path.write_text(__import__("json").dumps(good))
    assert validate_artifact(path, phase="confirmation")["arm"] == "backup"
    bad = dict(good, source_hash="deadbeef")
    path.write_text(__import__("json").dumps(bad))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_artifact(path, phase="confirmation")


def test_structural_validation_rejects_protocol_reuse(tmp_path):
    from hexset.catanatron.structural_validation import validate_artifact
    doc = {
        "family": "heximax-structural-future", "source_hash": source_fingerprint(),
        "phase": "screen", "arm": "backup", "candidate": "backup",
        "role": "candidate", "candidate_color": "Color.RED",
        "gate": "ab2", "games": 120,
        "workers": SCREEN_WORKERS, "seed": SCREEN_SEEDS["ab2"] + 10_000,
        "players": lineup("backup", "ab2"),
        "phenotype": manifest()["candidates"]["backup"],
        "wins": {"Color.RED": 120, "Color.WHITE": 0,
                 "Color.BLUE": 0, "Color.ORANGE": 0},
        "points": {"Color.RED": [5] * 120, "Color.WHITE": [5] * 120,
                   "Color.BLUE": [5] * 120, "Color.ORANGE": [5] * 120},
    }
    path = tmp_path / "protocol.json"
    import json
    path.write_text(json.dumps(doc))
    assert validate_artifact(path, phase="screen")["seed"] == 400_010_000
    for field, value in (("candidate", "fragility"), ("players", lineup("fragility", "ab2")),
                         ("workers", SCREEN_WORKERS + 1), ("seed", 130_000_000),
                         ("games", 119)):
        bad = dict(doc, **{field: value})
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            validate_artifact(path, phase="screen")


def test_choose_clears_structural_cache_with_evaluator_caches():
    from unittest.mock import patch
    board = random_base_board(random.Random(13))
    game = start(board, 3, random.Random(14))
    bot = spawn(PRESETS["heximax-structural-backup"], board, random.Random(15))
    bot.evaluator._structural_cache[("stale", 0)] = (1.0, 2.0)
    with patch.object(bot.evaluator, "clear_structural_cache",
                      wraps=bot.evaluator.clear_structural_cache) as clear:
        bot.choose(game)
    clear.assert_called_once_with()
    assert isinstance(bot, StructuralHeximax)
    assert not bot.evaluator._structural_cache


def test_structural_validation_accepts_draws_and_rejects_bad_counts(tmp_path):
    from hexset.catanatron.structural_validation import validate_artifact
    import json
    doc = {
        "family": "heximax-structural-future", "source_hash": source_fingerprint(),
        "phase": "screen", "arm": "backup", "candidate": "backup",
        "role": "candidate", "candidate_color": "Color.RED",
        "gate": "ab2", "games": 120,
        "workers": SCREEN_WORKERS, "seed": SCREEN_SEEDS["ab2"] + 10_000,
        "players": lineup("backup", "ab2"),
        "phenotype": manifest()["candidates"]["backup"],
        "wins": {"Color.RED": 0, "Color.WHITE": 0,
                 "Color.BLUE": 0, "Color.ORANGE": 0},
        "points": {"Color.RED": [], "Color.WHITE": [],
                   "Color.BLUE": [], "Color.ORANGE": []},
    }
    path = tmp_path / "draw.json"
    path.write_text(json.dumps(doc))
    assert validate_artifact(path, phase="screen")["arm"] == "backup"

    bad = dict(doc, wins={"Color.RED": 121, "Color.WHITE": 0,
                          "Color.BLUE": 0, "Color.ORANGE": 0})
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exceed"):
        validate_artifact(path, phase="screen")

    bad = dict(doc, points={"Color.RED": [], "Color.WHITE": [], "Color.BLUE": []})
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly"):
        validate_artifact(path, phase="screen")

    bad = dict(doc, candidate_color="Color.WHITE")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="candidate color"):
        validate_artifact(path, phase="screen")


def test_structural_disabled_features_match_honest_evaluator():
    from hexset.bots.heximax.evaluate import HonestEvaluator, NO_TRADE_WEIGHTS
    state = new_game(random_base_board(random.Random(21)), 3, random.Random(22))
    structural = StructuralEvaluator(state.board, backup=False, fragility=False)
    honest = HonestEvaluator(state.board, NO_TRADE_WEIGHTS)
    hands = np.asarray([
        [[0, 0, 1, 1, 3], [1, 1, 0, 0, 0], [0, 1, 1, 0, 1]],
        [[1, 0, 1, 1, 2], [0, 2, 0, 1, 0], [1, 0, 1, 0, 1]],
    ], dtype=float)
    np.testing.assert_allclose(
        structural.score_many(state, 0, hands), honest.score_many(state, 0, hands),
        rtol=0.0, atol=1e-12,
    )
    for row in hands:
        for seat in range(state.num_players):
            assert structural.score(state, seat, row[seat], knower=0) == pytest.approx(
                honest.score(state, seat, row[seat], knower=0), abs=1e-12,
            )
