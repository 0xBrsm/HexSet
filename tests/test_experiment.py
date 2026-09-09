# SPDX-License-Identifier: GPL-3.0-only
"""Reproducible experiment artifacts and portable worker initialization."""
import hashlib
import json
from dataclasses import replace

import pytest

from hexset.arena import Entrant, compete, lineup_from_names
from hexset.bots.heximax import TRADING_WEIGHTS
from hexset.experiment import provenance, result_document
from hexset.record import replay


def register_test_runtime(kind):
    # Module-scope initializer and factory can be imported by spawned workers.
    from hexset.arena import register_entrant_kind
    register_entrant_kind(kind, make_test_bot)


def make_test_bot(entrant, board, rng):
    from hexset.bots import RandomBot
    return RandomBot(rng)


@pytest.fixture
def result():
    entrants = lineup_from_names(["random", "random"])
    return entrants, compete(entrants, 2, seed=9, action_cap=16, records=True)


def test_result_round_trip_keeps_settings_raw_outcomes_and_replay(tmp_path, result):
    entrants, tournament = result
    doc = result_document(tournament, entrants, seed=9, action_cap=16,
                          run_provenance={"commit": "test", "checkpoints": {}})
    path = tmp_path / "run" / "result.json"
    path.parent.mkdir()
    path.write_text(json.dumps(doc, allow_nan=False))
    saved = json.loads(path.read_text())
    assert saved == doc
    assert saved["settings"]["action_cap"] == 16
    assert [row["board_index"] for row in saved["outcomes"]] == [0, 0]
    for row, record in zip(saved["outcomes"], tournament.records):
        assert row["points"] == list(tournament.points[row["index"]])
        assert row["seating"] == list(tournament.seating[row["index"]])
        assert row["winner"] is None
        assert replay(record).turns == row["turns"]


def test_checkpoint_capture_detects_file_replacement(tmp_path, result):
    _, tournament = result
    checkpoint = tmp_path / "policy.bin"
    checkpoint.write_bytes(b"first policy")
    entrants = [Entrant("policy", kind="network", weights=str(checkpoint)),
                Entrant("heximax", weights=TRADING_WEIGHTS)]
    tournament = replace(tournament, standings=tuple(replace(s, name=e.name) for s, e in zip(tournament.standings, entrants)))
    before = provenance(entrants)
    checkpoint.write_bytes(b"different policy")
    doc = result_document(tournament, entrants, seed=9, run_provenance=before)
    assert doc["checkpoints"][str(checkpoint)]["sha256"] == hashlib.sha256(b"first policy").hexdigest()
    assert doc["checkpoints_changed"] is True
    assert doc["checkpoint_capture"] == "before_run"
    assert doc["settings"]["entrants"][1]["weights"]["robber_risk"] == TRADING_WEIGHTS.robber_risk


def test_result_refuses_incomplete_rows_and_nonserializable_settings(result):
    entrants, tournament = result
    with pytest.raises(ValueError, match="every game"):
        result_document(replace(tournament, seating=()), entrants, seed=9)
    with pytest.raises(TypeError, match="runtime object"):
        result_document(tournament, [replace(entrants[0], weights=object()), entrants[1]], seed=9)


@pytest.mark.parametrize("kwargs", [{"games": 0}, {"workers": 0}, {"action_cap": 0}])
def test_invalid_run_budgets_are_refused(kwargs):
    options = {"games": 2, **kwargs}
    with pytest.raises(ValueError, match="positive"):
        compete(lineup_from_names(["random"] * 2), **options)


def test_odd_seat_pairing_requires_and_completes_two_rotations():
    entrants = lineup_from_names(["random"] * 3)
    with pytest.raises(ValueError, match="twice the seat"):
        compete(entrants, 3, action_cap=1)
    tournament = compete(entrants, 6, action_cap=1)
    for e in range(3):
        seats = [row[e] for row in tournament.seating]
        assert [seats.count(s) for s in range(3)] == [2, 2, 2]


def test_runtime_initializer_works_with_spawned_workers():
    entrants = [Entrant("portable", kind="test-portable")] * 2
    options = dict(games=2, seed=3, action_cap=12,
                   worker_initializer=register_test_runtime,
                   worker_initargs=("test-portable",))
    from hexset import arena
    try:
        serial = compete(entrants, workers=1, **options)
        spawned = compete(entrants, workers=2, start_method="spawn", **options)
        assert (serial.winners, serial.points, serial.seating) == (spawned.winners, spawned.points, spawned.seating)
    finally:
        arena._ENTRANT_KIND_FACTORIES.pop("test-portable", None)


def failing_initializer():
    raise RuntimeError("runtime dependency unavailable")


def test_a_failed_runtime_initializer_is_reported():
    from concurrent.futures.process import BrokenProcessPool
    with pytest.raises(BrokenProcessPool):
        compete(lineup_from_names(["random"] * 2), 2, workers=2, action_cap=1,
                start_method="spawn", worker_initializer=failing_initializer)


@pytest.mark.parametrize("change", ["lineup", "seating", "points", "winner"])
def test_artifact_rejects_ambiguous_entrant_mapping(result, change):
    entrants, tournament = result
    if change == "lineup":
        entrants = [replace(e, name="wrong") for e in entrants]
    elif change == "seating":
        tournament = replace(tournament, seating=((0, 0), (1, 0)))
    elif change == "points":
        tournament = replace(tournament, points=((1,), (2,)))
    else:
        tournament = replace(tournament, winners=(3, None))
    with pytest.raises(ValueError):
        result_document(tournament, entrants, seed=9)


def test_installed_package_does_not_claim_an_enclosing_repositories_revision(tmp_path, monkeypatch):
    from hexset import experiment, _source
    fake = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "hexset" / "experiment.py"
    monkeypatch.setattr(experiment, "__file__", str(fake))
    monkeypatch.setattr(_source.subprocess, "run", lambda *a, **k: pytest.fail("installed package must not query ancestor Git"))
    assert experiment.provenance()["commit"] is None
