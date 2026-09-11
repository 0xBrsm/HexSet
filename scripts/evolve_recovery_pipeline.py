#!/usr/bin/env python3
"""Bounded evolve recovery -> strict fresh validation handoff.

The recovery controller is responsible for producing complete 300-game rows;
this wrapper selects only promoted, complete rows before calling the existing
``evolve_validation.run_fresh`` evaluator.  It never treats a discovery-only row
as a finalist or a non-PASS fresh result as a goal pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from hexset.catanatron import evolve_validation
from hexset.catanatron.stance_validation import GATES, THRESHOLDS

LEGACY_SOURCE = "11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
REPAIRED_CONTROLLER = "3b39bf32966aeacf6960019c95257fbb19b16179954c59fdca5f4c5b6cba5c13"


def _complete_promoted_candidates(state: dict, checkpoint: Path) -> dict:
    config = state.get("config", {})
    total_games = config.get("total_games", 300)
    if total_games != 300 or state.get("protocol") != 1:
        raise ValueError("unsupported evolve checkpoint protocol/config")
    if state.get("source_hash") != LEGACY_SOURCE:
        raise ValueError("legacy evolve source binding mismatch")
    completed_numbers = {generation.get("generation") for generation in state.get("generations", [])
                         if isinstance(generation, dict) and generation.get("complete")
                         and not generation.get("excluded_from_selection", False)}
    if not set(range(6)).issubset(completed_numbers):
        raise ValueError("six complete evolve generations are required")
    choices = []
    for generation in state.get("generations", []):
        if (not isinstance(generation, dict) or not generation.get("complete")
                or generation.get("excluded_from_selection", False)):
            continue
        promoted = set(generation.get("promoted", []))
        candidates = {c.get("id"): c for c in generation.get("candidates", [])
                      if isinstance(c, dict)}
        for item in generation.get("summary", []):
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate")
            rows = item.get("rows", [])
            by_gate = {row.get("gate"): row for row in rows if isinstance(row, dict)}
            if (not isinstance(cid, str) or cid not in promoted or len(rows) != 2 or
                    set(by_gate) != set(GATES) or
                    any(row.get("games") != total_games for row in by_gate.values())):
                continue
            candidate = candidates.get(cid)
            if (candidate is None or candidate.get("width") != 6 or
                    candidate.get("depth") != 2 or candidate.get("max_nodes") != 600 or
                    candidate.get("k") != 1 or candidate.get("mode") != "notrade" or
                    candidate.get("max_trades") != 0 or
                    item.get("stance", "win") != candidate.get("stance", "win") or
                    float(item.get("temperature", candidate.get("temperature", 0))) !=
                    float(candidate.get("temperature", 0)) or
                    not isinstance(item.get("weights"), dict)):
                continue
            if item.get("weights") != candidate.get("weights"):
                continue
            try:
                rates = {gate: evolve_validation._candidate_wins(by_gate[gate], cid) / total_games
                         for gate in GATES}
            except (KeyError, TypeError, ValueError):
                continue
            score = min(rates[gate] - THRESHOLDS[gate] for gate in GATES)
            choices.append((score, cid, generation.get("generation"), item))
    if not choices:
        raise ValueError("no complete promoted 300-game dual-gate candidate")
    score, cid, generation, record = max(choices, key=lambda value: (value[0], value[1]))
    # Summary rows omit some constructor fields; carry the exact registered
    # candidate phenotype into the fresh evaluator fingerprint and command.
    candidate = next(c for generation_item in state["generations"]
                     if generation_item.get("generation") == generation
                     for c in generation_item.get("candidates", [])
                     if c.get("id") == cid)
    record = dict(record)
    for field in ("depth", "width", "max_nodes", "k", "mode", "max_trades", "temperature"):
        record[field] = candidate[field]
    return {"candidate": cid, "generation": generation, "score": score,
            "candidate_record": record, "source_hash": state["source_hash"],
            "protocol": state["protocol"],
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def _verify_controller(controller: Path) -> None:
    actual_controller = hashlib.sha256(controller.read_bytes()).hexdigest()
    if actual_controller != REPAIRED_CONTROLLER:
        raise ValueError("repaired evolve controller hash mismatch")


def choose_candidate(checkpoint: Path, controller: Path) -> dict:
    state = json.loads(checkpoint.read_text())
    _verify_controller(controller)
    return _complete_promoted_candidates(state, checkpoint)


def _fresh_fingerprint(chosen: dict) -> str:
    return hashlib.sha256(json.dumps({
        "candidate": chosen["candidate"], "record": chosen["candidate_record"],
        "source": chosen.get("source_hash"), "protocol": chosen.get("protocol"),
    }, sort_keys=True).encode()).hexdigest()


def _check_fresh_artifacts(out: Path, chosen: dict) -> None:
    """Reject stale fresh artifacts before run_fresh can reuse them."""
    record = chosen["candidate_record"]
    expected_weights = record.get("weights", {})
    expected_stance = record.get("stance", "win")
    expected_temperature = record.get("temperature", 0)
    expected = [(out / "confirmation.json", 1024, "both", 50_000_000)]
    blocks = (4096, 4096, 8192)
    bases = {"vs-ab2": 60_000_000, "vs-shipped": 70_000_000}
    for look, block in enumerate(blocks):
        for gate in GATES:
            expected.append((out / f"holdout-{gate}-block-{look}.json", block, gate,
                             bases[gate] + sum(blocks[:look])))
    existing = [path for path, _, _, _ in expected if path.exists()]
    for path, games, gate, seed in expected:
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"malformed fresh artifact: {path}") from exc
        required = {"candidate", "games", "workers", "seed", "gate",
                    "weights", "stance", "temperature"}
        if not required.issubset(doc):
            raise ValueError(f"fresh artifact missing identity fields: {path}")
        if (doc["candidate"] != chosen["candidate"] or doc["games"] != games or
                doc["workers"] != 30 or doc["seed"] != seed or
                doc["gate"] != gate or doc["weights"] != expected_weights or
                doc["stance"] != expected_stance or
                float(doc["temperature"]) != float(expected_temperature)):
            raise ValueError(f"fresh artifact identity/source mismatch: {path}")
    manifest = out / "manifest.json"
    if existing:
        # The evaluator output intentionally has no invented source field.
        # Its provenance is the run_fresh execution manifest, whose fingerprint
        # covers candidate phenotype, checkpoint source, and protocol. Never
        # reuse a same-candidate artifact without that binding.
        if not manifest.exists():
            raise ValueError("fresh artifact source manifest missing")
        try:
            provenance = json.loads(manifest.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("malformed fresh source manifest") from exc
        if (provenance.get("candidate") != chosen["candidate"] or
                provenance.get("fingerprint") != _fresh_fingerprint(chosen) or
                provenance.get("checkpoint_sha256") != chosen["checkpoint_sha256"]):
            raise ValueError("fresh artifact source/phenotype manifest mismatch")


def run_pipeline(checkpoint: Path, controller: Path, out: Path, *, python: str = sys.executable,
                 runner: Callable | None = None, run_evolution: bool = False) -> dict:
    """Run evolution (when requested), then strict fresh validation atomically."""
    out.mkdir(parents=True, exist_ok=True)
    terminal = {"controller_sha256": REPAIRED_CONTROLLER,
                "legacy_source_hash": LEGACY_SOURCE,
                "checkpoint": str(checkpoint)}
    try:
        # Verify the exact file before allowing it to execute the evolution
        # subprocess. PYTHONPATH then supplies the deployed package imports.
        _verify_controller(controller)
        if run_evolution:
            command = [python, str(controller), "--run", "--generations", "6",
                       "--count", "12", "--games", "120", "--total-games", "300",
                       "--promote-top", "4", "--checkpoint", str(checkpoint),
                       "--seed", "10000000", "--source-hash", LEGACY_SOURCE]
            (runner or subprocess.run)(command, check=True)
        chosen = choose_candidate(checkpoint, controller)
        terminal["selection"] = chosen
        # run_fresh has its own durable manifest/artifact checks. Override only
        # its permissive selector with this prevalidated, complete-row selection.
        original_selector = evolve_validation.select_candidate
        evolve_validation.select_candidate = lambda _: chosen
        try:
            _check_fresh_artifacts(out, chosen)
            result = evolve_validation.run_fresh(checkpoint, out, python=python, runner=runner)
            _check_fresh_artifacts(out, chosen)
        finally:
            evolve_validation.select_candidate = original_selector
        terminal["fresh"] = result
        terminal["status"] = result.get("status")
        evolve_validation._atomic_json(out / "pipeline-verdict.json", terminal)
        if result.get("status") != "PASS":
            raise RuntimeError(f"fresh validation did not PASS: {result.get('status')}")
        return terminal
    except Exception as exc:
        terminal["status"] = "ERROR" if "fresh" not in terminal else terminal.get("fresh", {}).get("status", "ERROR")
        terminal["error"] = str(exc)
        evolve_validation._atomic_json(out / "pipeline-verdict.json", terminal)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-evolution", action="store_true",
                        help="run the six-generation evolve controller before fresh validation")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_pipeline(args.checkpoint, args.controller, args.out,
                                      run_evolution=args.run_evolution),
                         indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
