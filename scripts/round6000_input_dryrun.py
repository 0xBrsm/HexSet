#!/usr/bin/env python3
"""Build and dry-validate the isolated round-6000 input population.

This script imports the certified generation-3000 checkpoint data locally,
replaces only ``next_candidates`` in a disposable checkpoint, and intercepts
the certified controller's ``_run_jobs`` before any game is launched.  It
never writes to the source or to a remote directory.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import multiprocessing
import shutil
from pathlib import Path

import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.catanatron import evolve

REVIEW_ROOT = Path("/data/data/com.termux/files/usr/tmp/round6000-review")
ORIGINAL = REVIEW_ROOT / "original-g3000-checkpoint.json"
ORIGINAL_SHA = "7ef82a1b06b8454c2d333fb56c3b5623bf3e6074cf5a431fd9ce65f1494d5876"
SOURCE_FILE = ROOT / "src/hexset/catanatron/evolve.py"
SOURCE_FILE_SHA = "9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c"
HISTORICAL_READINESS_SOURCE_TREE_SHA = "9d1bf3f060ace5cdfc31ed65ac58fc98a8e9611e37146b5bd7068c4dcd816b4e"
PRISTINE_INPUT_SHA = "9e3fe29f39f079272b1fbfc772e881a7b6dc2e3504fd6b5fd2df924315d60596"
CONTROLLER_SOURCE_HASH = "11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
PARENT_SIDECAR = Path("/data/data/com.termux/files/usr/tmp/luna-discovery-3000-pristine/g3000-selection-sidecar.json")
PARENT_SIDECAR_SHA = "e04e4f2345e1bd73a0326d243f45fd32c81dabd0cc07f1180ca068e3e8c3f57e"
IMAGE_SHA = "58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be"
BASELINE_SCARCE = float(NO_TRADE_WEIGHTS.scarce)

COMMON = {
    "depth": 2,
    "width": 6,
    "max_nodes": 600,
    "k": 1,
    "mode": "notrade",
    "max_trades": 0,
    "stance": "win",
    "temperature": 2.476644394795811,
}

ARM_DELTAS = {
    "g6000-p00": {},
    "g6000-p01": {"port": 2.0},
    "g6000-p02": {"port": 4.0},
    "g6000-p03": {"production": 0.85},
    "g6000-p04": {"production": 1.15},
    "g6000-p05": {"buy_progress": 0.75},
    "g6000-p06": {"buy_progress": 1.25},
    "g6000-p07": {"spare_card": 0.5},
    "g6000-p08": {"spare_card": 1.5},
    "g6000-p09": {"diversity": 1.25},
    "g6000-p10": {"knight": 1.5},
    "g6000-p11": {"port": 2.0, "buy_progress": 1.25, "spare_card": 0.5},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original() -> dict:
    if sha256(ORIGINAL) != ORIGINAL_SHA:
        raise AssertionError("original checkpoint digest mismatch")
    state = json.loads(ORIGINAL.read_text())
    if state.get("source_hash") != CONTROLLER_SOURCE_HASH:
        raise AssertionError("original source hash mismatch")
    if len(state.get("generations", [])) != 1:
        raise AssertionError("expected exactly one imported generation")
    record = state["generations"][0]
    if record.get("generation") != 3000 or record.get("complete") is not True:
        raise AssertionError("expected completed generation-3000 record")
    if len(record.get("candidates", [])) != 12 or len(record.get("summary", [])) != 12:
        raise AssertionError("expected twelve generation-3000 candidates and summaries")
    return state


def load_parent_sidecar(state: dict) -> dict:
    if sha256(PARENT_SIDECAR) != PARENT_SIDECAR_SHA:
        raise AssertionError("parent selection sidecar digest mismatch")
    sidecar = json.loads(PARENT_SIDECAR.read_text())
    selected = sidecar.get("selected", {})
    if selected.get("candidate") != "g3000-p06":
        raise AssertionError("parent sidecar does not select g3000-p06")
    p06 = next(c for c in state["generations"][0]["candidates"] if c["id"] == "g3000-p06")
    declared = dict(p06["weights"])
    effective = dict(sidecar.get("weights_effective", {}))
    declared["scarce"] = BASELINE_SCARCE
    if effective != declared:
        raise AssertionError("parent sidecar effective weights do not match g3000-p06 baseline-scarce vector")
    if sidecar.get("stance") != p06.get("stance") or sidecar.get("stance") != "win":
        raise AssertionError("parent sidecar stance mismatch")
    if sidecar.get("temperature") != p06.get("temperature") or sidecar.get("temperature") != COMMON["temperature"]:
        raise AssertionError("parent sidecar temperature mismatch")
    return sidecar


def build_candidates(state: dict) -> list[dict]:
    record = state["generations"][0]
    p06 = next(c for c in record["candidates"] if c["id"] == "g3000-p06")
    base = dict(p06["weights"])
    base["scarce"] = BASELINE_SCARCE
    expected_fields = tuple(base)
    candidates = []
    for candidate_id, deltas in ARM_DELTAS.items():
        weights = dict(base)
        for field, multiplier in deltas.items():
            weights[field] = base[field] * multiplier
        item = {"id": candidate_id, "weights": weights, **COMMON}
        if tuple(item["weights"]) != expected_fields:
            raise AssertionError(f"weight ordering changed for {candidate_id}")
        candidates.append(item)
    return candidates


def validate_population(candidates: list[dict], state: dict) -> None:
    if [c["id"] for c in candidates] != list(ARM_DELTAS):
        raise AssertionError("candidate IDs/order mismatch")
    p06 = next(c for c in state["generations"][0]["candidates"] if c["id"] == "g3000-p06")
    declared = p06["weights"]
    base = dict(declared)
    base["scarce"] = BASELINE_SCARCE
    fields = set(base)
    for item, deltas in zip(candidates, ARM_DELTAS.values()):
        if set(item) != {"id", "weights", *COMMON}:
            raise AssertionError(f"candidate metadata has extra/missing fields: {item['id']}")
        if set(item["weights"]) != fields:
            raise AssertionError(f"candidate weight fields mismatch: {item['id']}")
        if any(not math.isfinite(float(value)) for value in item["weights"].values()):
            raise AssertionError(f"non-finite candidate weight: {item['id']}")
        for field, expected_common in COMMON.items():
            value = item[field]
            if isinstance(expected_common, (int, float)):
                if not math.isfinite(float(value)) or value != expected_common:
                    raise AssertionError(f"candidate common metadata mismatch: {item['id']} {field}")
            elif value != expected_common:
                raise AssertionError(f"candidate common metadata mismatch: {item['id']} {field}")
        for field in fields:
            expected = base[field] * deltas[field] if field in deltas else base[field]
            if item["weights"][field] != expected:
                raise AssertionError(f"unexpected delta {item['id']} {field}")
        if item["weights"]["scarce"] != BASELINE_SCARCE:
            raise AssertionError(f"non-baseline scarcity in {item['id']}")
        if item["max_trades"] != 0 or item["mode"] != "notrade":
            raise AssertionError(f"trade phenotype mismatch in {item['id']}")


def make_input_checkpoint(state: dict, candidates: list[dict], out: Path) -> dict:
    imported = copy.deepcopy(state)
    original_record = copy.deepcopy(imported["generations"][0])
    imported["generations"][0]["next_candidates"] = copy.deepcopy(candidates)
    check_a = copy.deepcopy(original_record)
    check_b = copy.deepcopy(imported["generations"][0])
    check_a.pop("next_candidates")
    check_b.pop("next_candidates")
    if check_a != check_b:
        raise AssertionError("imported generation record changed outside next_candidates")
    imported["config"] = dict(imported["config"], generations=6001, seed=600000000)
    imported["input_protocol"] = {
        "name": "round6000-explicit-population-v1",
        "imported_from_checkpoint_sha256": ORIGINAL_SHA,
        "imported_generation": 3000,
        "proposal_override": "generations[0].next_candidates",
        "override_manifest": "round6000-input-manifest.json",
        "controller_source_hash": CONTROLLER_SOURCE_HASH,
        "controller_file_sha256": SOURCE_FILE_SHA,
        "historical_readiness_source_tree_sha256": HISTORICAL_READINESS_SOURCE_TREE_SHA,
        "parent_sidecar_sha256": PARENT_SIDECAR_SHA,
        "parent_sidecar_path": str(PARENT_SIDECAR),
    }
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(imported, indent=2, sort_keys=True) + "\n")
    return imported


def validate_pristine(imported: dict, candidates: list[dict], original: dict) -> None:
    if imported.get("source_hash") != CONTROLLER_SOURCE_HASH:
        raise AssertionError("pristine source hash mismatch")
    expected_config = {"count": 12, "games": 240, "total_games": 240,
                       "promote_top": 4, "generations": 6001, "seed": 600000000}
    if any(imported.get("config", {}).get(k) != v for k, v in expected_config.items()):
        raise AssertionError("pristine input config mismatch")
    protocol = imported.get("input_protocol", {})
    if protocol.get("name") != "round6000-explicit-population-v1":
        raise AssertionError("missing input-only protocol marker")
    if imported["generations"][0].get("next_candidates") != candidates:
        raise AssertionError("pristine override population mismatch")
    original_record = copy.deepcopy(original["generations"][0])
    imported_record = copy.deepcopy(imported["generations"][0])
    original_record.pop("next_candidates", None)
    imported_record.pop("next_candidates", None)
    if imported_record != original_record:
        raise AssertionError("imported generation-3000 body changed outside next_candidates")


def validate_isolation(root: Path, imported: dict) -> None:
    generations = imported.get("generations", [])
    if len(generations) != 1 or generations[0].get("generation") != 3000:
        raise AssertionError("pristine input must contain exactly one generation-3000 record")
    if any(g.get("generation") in {5999, 6000} for g in generations):
        raise AssertionError("preexisting generation-5999/6000 record")
    games = root / "games"
    if games.is_symlink():
        raise AssertionError("games directory is a symlink")
    if games.exists():
        if any(games.rglob("*.json")):
            raise AssertionError("preexisting current-root game records")


def _fork_resolve(name: str) -> dict:
    """Resolve a registered entrant in a forked worker without starting a game."""
    from hexset.arena import entrant_from_name
    entrant = entrant_from_name(name)
    return {
        "name": entrant.name,
        "weights": asdict(entrant.weights),
        "depth": entrant.depth,
        "width": entrant.width,
        "max_nodes": entrant.max_nodes,
        "k": entrant.k,
        "mode": entrant.mode,
        "max_trades": entrant.max_trades,
        "stance": entrant.stance,
        "temperature": entrant.temperature,
    }


def dry_run(pristine_checkpoint: Path, candidates: list[dict]) -> dict:
    dry_root = REVIEW_ROOT / "round6000-dryrun-nonexecutable"
    if dry_root.exists():
        shutil.rmtree(dry_root)
    dry_root.mkdir(parents=True)
    input_checkpoint = dry_root / "checkpoint.json"
    shutil.copy2(pristine_checkpoint, input_checkpoint)
    input_checkpoint.chmod(0o644)
    imported = json.loads(input_checkpoint.read_text())
    validate_isolation(dry_root, imported)
    calls: list[dict] = []

    def intercepted(root: Path, jobs: list[tuple], generation: int) -> None:
        calls.append({"root": str(root), "generation": generation, "jobs": list(jobs)})

    original_run_jobs = evolve._run_jobs
    original_register_preset = evolve.register_preset
    registrations: list[tuple[str, object]] = []

    def capture_registration(name: str, entrant: object) -> None:
        registrations.append((name, entrant))
        original_register_preset(name, entrant)

    evolve._run_jobs = intercepted
    evolve.register_preset = capture_registration
    argv = [
        "evolve.py", "--checkpoint", str(input_checkpoint), "--generation", "6000",
        "--count", "12", "--games", "240", "--total-games", "240",
        "--promote-top", "4", "--generations", "6001", "--seed", "600000000",
        "--source-hash", CONTROLLER_SOURCE_HASH, "--run",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            evolve.main()
    finally:
        sys.argv = old_argv
        evolve._run_jobs = original_run_jobs
        evolve.register_preset = original_register_preset
    expected_names = ["heximax-evolve-" + c["id"] for c in candidates]
    if [name for name, _ in registrations] != expected_names:
        raise AssertionError("controller did not register exactly the explicit population")
    expected_by_name = dict(zip(expected_names, candidates))
    for name, entrant in registrations:
        candidate = expected_by_name[name]
        expected_weights = dict(candidate["weights"])
        expected_weights["scarce"] = BASELINE_SCARCE
        if asdict(entrant.weights) != expected_weights:
            raise AssertionError(f"registered weight mismatch for {name}")
        for field in ("depth", "width", "max_nodes", "k", "mode", "max_trades", "stance", "temperature"):
            if getattr(entrant, field) != candidate[field]:
                raise AssertionError(f"registered metadata mismatch for {name}: {field}")
    default_start_method = multiprocessing.get_start_method()
    with multiprocessing.get_context("fork").Pool(min(4, len(expected_names))) as pool:
        forked = pool.map(_fork_resolve, expected_names)
    if [row["name"] for row in forked] != expected_names:
        raise AssertionError("forked worker did not inherit registered input entrants")
    for row in forked:
        expected = expected_by_name[row["name"]]
        if row["weights"] != dict(expected["weights"]):
            raise AssertionError("forked worker changed input entrant weights")
        for field in ("depth", "width", "max_nodes", "k", "mode", "max_trades", "stance", "temperature"):
            if row[field] != expected[field]:
                raise AssertionError(f"forked worker changed input metadata: {row['name']} {field}")
    if len(calls) != 2 or len(calls[0]["jobs"]) != 5760 or calls[1]["jobs"]:
        raise AssertionError(f"expected discovery batch plus empty confirmation batch, got {[len(c["jobs"]) for c in calls]}")
    call = calls[0]
    if call["generation"] != 6000:
        raise AssertionError("wrong generation passed to _run_jobs")
    jobs = call["jobs"]
    if len(jobs) != 12 * 2 * 240:
        raise AssertionError(f"expected 5760 jobs, got {len(jobs)}")
    by_candidate = {c["id"]: [j for j in jobs if j[0] == c["id"]] for c in candidates}
    for candidate_id, candidate_jobs in by_candidate.items():
        if len(candidate_jobs) != 480:
            raise AssertionError(f"wrong job count for {candidate_id}")
        for gate, expected_start in (("vs-ab2", 610_000_000), ("vs-shipped", 610_050_000)):
            rows = [j for j in candidate_jobs if j[1] == gate]
            if len(rows) != 240:
                raise AssertionError(f"wrong {gate} count for {candidate_id}")
            expected = list(range(expected_start, expected_start + 240))
            if [j[5] for j in sorted(rows, key=lambda j: j[4])] != expected:
                raise AssertionError(f"seed range mismatch for {candidate_id} {gate}")
            for job in rows:
                path = evolve._game_path(Path(call["root"]), 6000, candidate_id,
                                         gate, "discovery", job[4])
                if path.exists():
                    raise AssertionError("dry run unexpectedly created a game file")
    return {
        "calls": len(calls),
        "confirmation_jobs": len(calls[1]["jobs"]),
        "generation": call["generation"],
        "jobs": len(jobs),
        "candidate_count": len(by_candidate),
        "registrations": len(registrations),
        "fork_resolved": len(forked),
        "local_default_start_method": default_start_method,
        "games_per_candidate": 480,
        "seed_ranges": {
            "vs-ab2": [610_000_000, 610_000_239],
            "vs-shipped": [610_050_000, 610_050_239],
        },
    }


def main() -> None:
    if sha256(SOURCE_FILE) != SOURCE_FILE_SHA:
        raise AssertionError("local certified controller file digest mismatch")
    if Path(evolve.__file__).resolve() != SOURCE_FILE.resolve():
        raise AssertionError(f"executed evolve module mismatch: {evolve.__file__}")
    state = load_original()
    load_parent_sidecar(state)
    candidates = build_candidates(state)
    validate_population(candidates, state)
    pristine_root = REVIEW_ROOT / "round6000-input-pristine-rev2"
    pristine_checkpoint = pristine_root / "checkpoint.json"
    if pristine_checkpoint.exists():
        imported = json.loads(pristine_checkpoint.read_text())
        validate_isolation(pristine_root, imported)
        validate_pristine(imported, candidates, state)
        if PRISTINE_INPUT_SHA and sha256(pristine_checkpoint) != PRISTINE_INPUT_SHA:
            raise AssertionError("pristine input digest mismatch")
        if pristine_checkpoint.stat().st_mode & 0o777 != 0o444:
            raise AssertionError("pristine input is not immutable mode 0444")
    else:
        pristine_root.mkdir(parents=True, exist_ok=True)
        imported = make_input_checkpoint(state, candidates, pristine_checkpoint)
        pristine_checkpoint.chmod(0o444)
        validate_isolation(pristine_root, imported)
        validate_pristine(imported, candidates, state)
        if PRISTINE_INPUT_SHA and sha256(pristine_checkpoint) != PRISTINE_INPUT_SHA:
            raise AssertionError("created pristine input digest mismatch")
    validate_population(imported["generations"][0]["next_candidates"], state)
    pristine_sha = sha256(pristine_checkpoint)
    dry = dry_run(pristine_checkpoint, candidates)
    if sha256(pristine_checkpoint) != pristine_sha:
        raise AssertionError("dry run modified pristine input checkpoint")
    after = json.loads((REVIEW_ROOT / "round6000-dryrun-nonexecutable" / "checkpoint.json").read_text())
    imported_after = next(g for g in after["generations"] if g.get("generation") == 3000)
    imported_before = imported["generations"][0]
    for key in ("candidates", "summary", "promoted", "complete", "generation"):
        if imported_after.get(key) != imported_before.get(key):
            raise AssertionError(f"imported generation changed after dry run: {key}")
    if any(g.get("generation") == 5999 for g in after.get("generations", [])):
        raise AssertionError("dry input contains generation-5999")
    if not any(g.get("generation") == 6000 for g in after.get("generations", [])):
        raise AssertionError("dry controller did not append generation-6000 record")
    output = {
        "schema": 1,
        "status": "LOCAL_DRY_CONTROL_FLOW_PASS",
        "checkpoint": str(pristine_checkpoint),
        "checkpoint_sha256": pristine_sha,
        "pristine_input": True,
        "dry_run_checkpoint": str(REVIEW_ROOT / "round6000-dryrun-nonexecutable" / "checkpoint.json"),
        "dry_run_checkpoint_sha256": sha256(REVIEW_ROOT / "round6000-dryrun-nonexecutable" / "checkpoint.json"),
        "original_checkpoint_sha256": ORIGINAL_SHA,
        "controller_source_hash": CONTROLLER_SOURCE_HASH,
        "controller_file_sha256": SOURCE_FILE_SHA,
        "historical_readiness_source_tree_sha256": HISTORICAL_READINESS_SOURCE_TREE_SHA,
        "parent_sidecar_sha256": PARENT_SIDECAR_SHA,
        "parent_sidecar_path": str(PARENT_SIDECAR),
        "image_sha256": IMAGE_SHA,
        "population": candidates,
        "effective_phenotype": {
            "source_candidate": "g3000-p06",
            "parent_sidecar_verified": True,
            "scarce": BASELINE_SCARCE,
            "all_non_arm_fields_fixed": True,
            "child_weights_are_explicit": True,
            "candidate_constructor_used": False,
        },
        "input_config": {
            "generation_start": 6000,
            "generation_end_exclusive": 6001,
            "count": 12,
            "games": 240,
            "total_games": 240,
            "promote_top": 4,
            "seed": 600000000,
        },
        "resolved_spawn": {
            "kind": "heximax",
            "depth": 2,
            "width": 6,
            "max_nodes": 600,
            "k": 1,
            "mode": "notrade",
            "max_trades": 0,
            "stance": "win",
            "temperature": COMMON["temperature"],
            "entrant_placement": False,
            "placement": True,
            "placement_resolution": "heximax() default placement=True",
            "domestic_player_trades": False,
            "maritime_trade": True,
        },
        "dry_control_flow": dry,
        "worker_registration": {
            "source_pool": "Pool(min(30, len(jobs)))",
            "candidate_resolution": "_play_one parses heximax-evolve-<candidate_id>",
            "explicit_fork_probe": "PASS",
            "local_default_start_method": dry["local_default_start_method"],
            "pinned_runtime_requirement": "record runtime start method; certified source does not force fork",
        },
        "mock_artifacts": {
            "label": "NONEXECUTABLE_DRY_RUN_ONLY",
            "raw_games_created": 0,
            "checkpoint_may_contain_fake_complete_generation_6000": True,
            "must_not_upload_or_resume": True,
        },
        "post_validation_plan": {
            "raw_games": 5760,
            "promotion": "not used; this is an isolated one-generation input round",
            "native_point_promotions_used": False,
            "score_rule": "maximize min(rate_vs_ab2 - 0.50, rate_vs_shipped - 0.25); tie by ascending candidate ID",
            "required_gates": ["vs-ab2", "vs-shipped"],
            "required_workers": 30,
            "required_game_count_per_gate": 240,
        },
    }
    manifest = REVIEW_ROOT / "round6000-input-manifest.json"
    manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "input_checkpoint": str(pristine_checkpoint),
        "input_checkpoint_sha256": pristine_sha,
        "dry_run_checkpoint": str(REVIEW_ROOT / "round6000-dryrun-nonexecutable" / "checkpoint.json"),
        "dry_run_checkpoint_sha256": sha256(REVIEW_ROOT / "round6000-dryrun-nonexecutable" / "checkpoint.json"),
        "status": output["status"],
        "dry_control_flow": dry,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
