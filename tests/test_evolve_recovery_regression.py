"""Regression contract for recovering the interrupted g00 evolve campaign.

These tests intentionally exercise the on-disk game checkpoint contract.  They
never import Catanatron or launch a worker; the fake dispatcher only materializes
requested confirmation records so the controller's scheduling/identity logic
is tested independently of game policy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hexset.catanatron import evolve


COUNT = 12
DISCOVERY = 120
TOTAL = 300


def _checkpoint(tmp_path: Path):
    candidates = evolve.proposals(0, COUNT, 0)
    # Give p06 the best (still below-target) dual-gate margin, then p07-p09.
    # p00 is deliberately weak so accidental ``previous[:1]`` fallback is
    # visible in generation-one parent IDs.
    rates = {
        "g00-p00": (40, 20), "g00-p01": (41, 21), "g00-p02": (42, 22),
        "g00-p03": (43, 23), "g00-p04": (44, 24), "g00-p05": (45, 25),
        "g00-p06": (49, 29), "g00-p07": (48, 28), "g00-p08": (47, 27),
        "g00-p09": (46, 26), "g00-p10": (44, 22), "g00-p11": (43, 21),
    }
    root = tmp_path / "campaign"
    game_root = root / "games" / "generation-00"
    game_root.mkdir(parents=True)
    source = evolve._source_hash()
    for candidate in candidates:
        cid = candidate["id"]
        ab2, shipped = rates[cid]
        for gate, wins in (("vs-ab2", ab2), ("vs-shipped", shipped)):
            for index in range(DISCOVERY):
                path = game_root / f"{cid}-{gate}-discovery-00-{index:04d}.json"
                path.write_text(json.dumps({
                    "candidate": cid, "gate": gate, "stage": "discovery",
                    "generation": 0, "game_index": index,
                    "seed": evolve.paired_seed(gate, "discovery", 0, index),
                    "wins": {"Color.RED": int(index < wins)},
                    "candidate_wins": int(index < wins),
                    # Provenance is part of the raw-reuse contract.
                    "controller": "evolve.py", "source_hash": source,
                }))
    manifest = {
        "schema": 1, "controller": "evolve.py", "source_hash": source,
        "files": {},
    }
    for path in sorted(game_root.glob("*.json")):
        manifest["files"][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "raw-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state = {
        "protocol": 1, "source_hash": source, "seed_families": evolve.SEED_FAMILIES,
        "raw_manifest": str(manifest_path), "raw_manifest_sha256": manifest_digest,
        "config": {"count": COUNT, "games": DISCOVERY, "total_games": TOTAL,
                    "promote_top": 4, "generations": 1, "seed": 0},
        "generations": [{
            "generation": 0, "candidates": candidates,
            # This is the observed interrupted checkpoint defect.
            "summary": [{"candidate": c["id"], "rows": []} for c in candidates],
            "promoted": [], "complete": True,
        }],
    }
    checkpoint = root / "state.json"
    checkpoint.write_text(json.dumps(state))
    return checkpoint, root, candidates


def _materialize_confirmation(root: Path, jobs: list[tuple]):
    """Fake only the controller's requested records; no game is played."""
    for cid, gate, stage, generation, index, seed in jobs:
        path = evolve._game_path(root, generation, cid, gate, stage, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "candidate": cid, "gate": gate, "stage": stage,
            "generation": generation, "game_index": index, "seed": seed,
            "wins": {"Color.RED": 0}, "candidate_wins": 0,
            "controller": "evolve.py", "source_hash": evolve._source_hash(),
        }))


def _run_args(checkpoint: Path):
    return ["evolve", "--checkpoint", str(checkpoint), "--count", str(COUNT),
            "--games", str(DISCOVERY), "--total-games", str(TOTAL),
            "--promote-top", "4", "--generations", "1", "--seed", "0", "--run"]


def test_recovery_reuses_2880_and_schedules_exactly_1440(tmp_path, monkeypatch):
    checkpoint, root, _ = _checkpoint(tmp_path)
    calls = []

    def dispatch(path, jobs, generation):
        if jobs:
            calls.append(list(jobs))
        _materialize_confirmation(root, jobs)

    monkeypatch.setattr(evolve, "_run_jobs", dispatch)
    monkeypatch.setattr("sys.argv", _run_args(checkpoint))
    evolve.main()

    # Discovery files are complete and must be reused.  Only four candidates
    # receive 180 additional games at each of two gates.
    assert len(calls) == 1
    assert len(calls[0]) == 4 * 2 * (TOTAL - DISCOVERY)
    assert all(job[2] == "confirm" for job in calls[0])
    assert {job[0] for job in calls[0]} == {
        "g00-p06", "g00-p07", "g00-p08", "g00-p09",
    }

    state = json.loads(checkpoint.read_text())
    generation = state["generations"][0]
    assert set(generation["promoted"]) == {"g00-p06", "g00-p07", "g00-p08", "g00-p09"}
    by_id = {item["candidate"]: item for item in generation["summary"]}
    assert len(by_id["g00-p06"]["rows"]) == 2
    assert all(len(by_id[cid]["rows"]) == 1 for cid in (
        "g00-p00", "g00-p01", "g00-p02", "g00-p03", "g00-p04", "g00-p05",
        "g00-p10", "g00-p11",
    ))


def test_recovery_parent_selection_uses_actual_300_game_rows(tmp_path, monkeypatch):
    checkpoint, root, _ = _checkpoint(tmp_path)
    state = json.loads(checkpoint.read_text())
    state["config"]["generations"] = 2
    checkpoint.write_text(json.dumps(state))

    def dispatch(path, jobs, generation):
        _materialize_confirmation(root, jobs)

    monkeypatch.setattr(evolve, "_run_jobs", dispatch)
    args = _run_args(checkpoint)
    args[args.index("--generations") + 1] = "2"
    monkeypatch.setattr("sys.argv", args)
    # The test asks the controller to build the next generation after g00;
    # discovery/confirmation records are still supplied by the fake dispatcher.
    evolve.main()
    state = json.loads(checkpoint.read_text())
    assert len(state["generations"]) == 2
    parents = {
        candidate["parent_id"]
        for candidate in state["generations"][1]["candidates"]
    }
    assert parents <= {"g00-p06", "g00-p07", "g00-p08", "g00-p09"}
    assert "g00-p00" not in parents


def test_recovery_rerun_with_all_artifacts_schedules_zero_jobs(tmp_path, monkeypatch):
    checkpoint, root, _ = _checkpoint(tmp_path)
    calls = []

    def dispatch(path, jobs, generation):
        calls.append(list(jobs))
        _materialize_confirmation(root, jobs)

    monkeypatch.setattr(evolve, "_run_jobs", dispatch)
    monkeypatch.setattr("sys.argv", _run_args(checkpoint))
    evolve.main()
    calls.clear()
    evolve.main()
    assert calls == []


def test_missing_or_corrupt_dual_gate_evidence_fails_closed(tmp_path):
    checkpoint, root, candidates = _checkpoint(tmp_path)
    # Missing shipped gate means no complete dual-gate row can be promoted.
    for path in (root / "games" / "generation-00").glob("*-vs-shipped-*.json"):
        path.unlink()
    summary = evolve._summarize(root, 0, candidates, DISCOVERY, DISCOVERY)
    assert all(len(item["rows"]) <= 1 for item in summary)
    assert evolve.select_promotions(summary) == []

    corrupt = next((root / "games" / "generation-00").glob("g00-p00-vs-ab2-*.json"))
    corrupt.write_text("{broken")
    with pytest.raises(RuntimeError, match="corrupt game checkpoint"):
        evolve._summarize(root, 0, candidates, DISCOVERY, DISCOVERY)


def test_raw_reuse_requires_immutable_manifest_provenance(tmp_path, monkeypatch):
    checkpoint, root, _ = _checkpoint(tmp_path)
    raw = next((root / "games" / "generation-00").glob("g00-p06-vs-ab2-*.json"))
    value = json.loads(raw.read_text())
    # A forged per-record controller field must not authorize reuse: the
    # immutable sidecar digest must cover the actual raw bytes and closed
    # controller/source manifest.
    value["controller"] = "other-controller.py"
    raw.write_text(json.dumps(value))
    monkeypatch.setattr(evolve, "_run_jobs", lambda *args: None)
    monkeypatch.setattr("sys.argv", _run_args(checkpoint))
    with pytest.raises((RuntimeError, ValueError), match="(provenance|controller|source|manifest)"):
        evolve.main()
