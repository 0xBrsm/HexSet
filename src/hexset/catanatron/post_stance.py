"""Resumable controller for post stance-search validation.

The controller deliberately treats JSON artifacts as an interface: every stage
is validated before the next subprocess is started, and a failed stage cannot
silently be promoted to a later one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .stance_validation import GATES, THRESHOLDS, aggregate

Z99 = 3.29052673149
SCREEN_GAMES = 120
CONFIRM_GAMES = 1024
HOLDOUT_GAMES = 4096
SCREEN_SEED = 10_000_000
CONFIRM_SEED = 21_000_00
HOLDOUT_SEEDS = {"vs-ab2": 22_000_00, "vs-shipped": 23_000_00}


def wilson_lower(wins: int, games: int, z: float = Z99) -> float:
    if not (0 <= wins <= games and games > 0):
        raise ValueError("invalid wins/games")
    p = wins / games
    d = 1 + z * z / games
    q = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    return (p + z * z / (2 * games) - q) / d


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _wins(row: dict, games: int, path: Path) -> int:
    if row.get("games") != games or not isinstance(row.get("wins"), dict):
        raise ValueError(f"invalid gate row: {path}")
    wins = row["wins"]
    key = "Color.RED" if "Color.RED" in wins else "0"
    if key not in wins:
        raise ValueError(f"missing Color.RED wins: {path}")
    try:
        values = [int(v) for v in wins.values()]
        result = int(wins[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-integer wins: {path}") from exc
    if result < 0 or result > games or any(v < 0 for v in values) or sum(values) != games:
        raise ValueError(f"wins do not total {games}: {path}")
    return result


def _document(path: Path, games: int, *, gates: set[str] | None = None) -> tuple[dict, dict[str, int]]:
    gates = set(GATES) if gates is None else gates
    doc = _load(path)
    if not isinstance(doc.get("candidate"), str) or not isinstance(doc.get("rows"), list):
        raise ValueError(f"missing candidate/rows: {path}")
    rows = {r.get("gate"): r for r in doc["rows"] if isinstance(r, dict)}
    if set(rows) != gates or len(doc["rows"]) != len(gates):
        raise ValueError(f"gate mismatch: {path}")
    return doc, {gate: _wins(rows[gate], games, path) for gate in gates}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: Sequence[str], runner: Callable | None = None) -> None:
    (runner or subprocess.run)(list(cmd), check=True)


def run_controller(root: Path, *, screen_dir: Path | None = None,
                   python: str = "python", evolve: Path | None = None,
                   checkpoint: Path | None = None,
                   runner: Callable | None = None, cheap_runner: Callable | None = None) -> dict:
    """Run/resume the campaign rooted at *root*, returning its verdict/status."""
    root = Path(root); screen_dir = Path(screen_dir or root / "screen")
    validation = root / "validation"; validation.mkdir(parents=True, exist_ok=True)
    manifest_path, progress_path = root / "manifest.json", root / "progress.log"
    _write(manifest_path, {"controller": "post-stance", "gates": list(GATES),
                           "screen_games": SCREEN_GAMES, "confirm_games": CONFIRM_GAMES,
                           "holdout_games": HOLDOUT_GAMES, "wilson_z": Z99})
    progress_path.open("a").write("start post-stance\n")

    files = sorted(screen_dir.glob("*.json")) if screen_dir.exists() else []
    try:
        if len(files) != 16:
            raise ValueError("stance screen requires exactly 16 files")
        selection = aggregate(files)
    except Exception as exc:
        # A complete but underperforming screen starts the bounded evolution
        # escape hatch; missing/corrupt input remains fail-closed.
        if len(files) == 16:
            return {"status": "FAIL", "stage": "screen", "error": str(exc)}
        result = {"status": "FAIL", "stage": "screen", "error": str(exc)}
        _write(validation / "status.json", result)
        return result

    def evolve_once(error: str) -> dict:
        nonlocal evolve, checkpoint
        # Run the bounded bank/DCL follow-on before adaptive evolution in live
        # mode. Injected runners used by protocol tests bypass this child.
        if cheap_runner is not None or runner is None:
            try:
                if cheap_runner is None:
                    from .cheap_followon import run as run_followon
                    follow = run_followon(validation / "cheap-followon", python)
                else:
                    follow = cheap_runner(validation / "cheap-followon", python)
                if follow.get("status") == "PASSED_VALIDATION":
                    gates = follow.get("gates", {})
                    # cheap_followon uses its CLI labels (ab2/shipped); normalize
                    # only after requiring the exact two-gate set.
                    if set(gates) == {"ab2", "shipped"}:
                        gates = {"vs-" + k: v for k, v in gates.items()}
                    valid = follow.get("candidate") and set(gates) == set(GATES)
                    for gate in GATES:
                        item = gates.get(gate, {})
                        try:
                            from .evolve_validation import wilson_lower
                            valid = valid and (item.get("candidate", follow.get("candidate")) == follow.get("candidate")) and item.get("games") == 4096 and wilson_lower(int(item.get("wins")), 4096) > THRESHOLDS[gate]
                        except (TypeError, ValueError):
                            valid = False
                    if valid:
                        _write(validation / "cheap-followon-result.json", follow)
                        return {"status": "PASS", "stage": "cheap-followon", "candidate": follow.get("candidate")}
                    follow = {"status": "INVALID_VALIDATION", "candidate": follow.get("candidate")}
            except Exception as follow_exc:
                result = {"status": "FAIL", "stage": "cheap-followon", "error": str(follow_exc)}
                _write(validation / "status.json", result)
                return result
        evolve = evolve or root / "LATEST_evolve.py"
        checkpoint = checkpoint or root / "evolve-checkpoint.json"
        try:
            _run([python, str(evolve), "--run", "--generations", "6", "--count", "12",
                  "--games", "120", "--total-games", "300", "--promote-top4",
                  "--checkpoint", str(checkpoint), "--seed", str(SCREEN_SEED)], runner)
        except Exception as evolve_exc:
            result = {"status": "FAIL", "stage": "evolve", "error": str(evolve_exc)}
            _write(validation / "status.json", result)
            return result
        if checkpoint.exists():
            try:
                from .evolve_validation import run_fresh
                result = run_fresh(checkpoint, validation / "evolved-fresh", python=python, runner=runner or subprocess.run)
                result["stage"] = "evolved-fresh"
                _write(validation / "status.json", result)
                return result
            except Exception as fresh_exc:
                result = {"status": "FAIL", "stage": "evolved-fresh", "error": str(fresh_exc), "checkpoint": str(checkpoint)}
                _write(validation / "status.json", result)
                return result
        result = {"status": "FAIL", "stage": "evolve", "error": "successful evolution produced no checkpoint",
                  "checkpoint": str(checkpoint)}
        _write(validation / "status.json", result)
        return result

    if not selection["ranking"] or not selection["ranking"][0]["positive_both"]:
        return evolve_once("screen candidate failed both point-estimate gates")

    candidate = selection["candidate"]
    _write(validation / "selection.json", selection)
    import hashlib as _hashlib
    source_digest = _hashlib.sha256()
    source_root = Path(__file__).resolve().parents[2]
    for source_file in sorted(source_root.rglob("*.py")):
        source_digest.update(source_file.relative_to(source_root).as_posix().encode()); source_digest.update(source_file.read_bytes())
    source_sha = source_digest.hexdigest()
    stance_fingerprint = _hashlib.sha256(json.dumps({"candidate": candidate, "selection": selection, "source": source_sha}, sort_keys=True).encode()).hexdigest()
    stance_manifest = validation / "stance-manifest.json"
    if stance_manifest.exists():
        prior = _load(stance_manifest)
        if prior.get("candidate") != candidate or prior.get("fingerprint") != stance_fingerprint:
            return {"status": "FAIL", "stage": "holdout", "error": "stance phenotype/source fingerprint mismatch"}
    else:
        _write(stance_manifest, {"candidate": candidate, "fingerprint": stance_fingerprint, "source_sha256": source_sha})
    confirm = validation / "confirmation.json"
    if confirm.exists():
        cdoc, cr = _document(confirm, CONFIRM_GAMES)
        if cdoc["candidate"] != candidate or not all(cr[g] / CONFIRM_GAMES > THRESHOLDS[g] for g in GATES):
            return {"status": "FAIL", "stage": "confirmation"}
    else:
        _run([python, "-m", "hexset.catanatron.stancecfg", "--candidate", candidate,
              "--gate", "both", "--games", str(CONFIRM_GAMES), "--workers", "30",
              "--seed", str(CONFIRM_SEED), "--out", str(confirm)], runner)
        cdoc, cr = _document(confirm, CONFIRM_GAMES)
        if cdoc["candidate"] != candidate or not all(cr[g] / CONFIRM_GAMES > THRESHOLDS[g] for g in GATES):
            return evolve_once("confirmation candidate failed both point-estimate gates")

    from .evolve_validation import continuation_decision
    totals = {g: 0 for g in GATES}
    blocks = (4096, 4096, 8192)
    for look, block in enumerate(blocks):
        for gate in GATES:
            out = validation / f"holdout-{gate}-block-{look}.json"
            if not out.exists():
                gate_arg = "ab2" if gate == "vs-ab2" else "shipped"
                _run([python, "-m", "hexset.catanatron.stancecfg", "--candidate", candidate,
                      "--gate", gate_arg, "--games", str(block), "--workers", "30",
                      "--seed", str(HOLDOUT_SEEDS[gate] + sum(blocks[:look])), "--out", str(out)], runner)
            hdoc, hw = _document(out, block, gates={gate})
            if hdoc["candidate"] != candidate:
                return {"status": "FAIL", "stage": "holdout", "gate": gate}
            totals[gate] += hw[gate]
        n = sum(blocks[:look + 1])
        decision = continuation_decision({g: (totals[g], n) for g in GATES})
        _write(validation / f"holdout-decision-{n}.json", {"candidate": candidate, "decision": decision, "counts": totals})
        if decision in ("PASS", "REJECTED_AT_LOOK", "UNRESOLVED_AT_CAP"):
            if decision != "PASS":
                return evolve_once("holdout continuation decision: " + decision)
            holdout_rates = {g: {"wins": totals[g], "games": n, "rate": totals[g] / n,
                                 "wilson_lower": wilson_lower(totals[g], n), "threshold": THRESHOLDS[g]} for g in GATES}
            break
    else:
        return evolve_once("holdout continuation exhausted")
    verdict = {"status": "PASS", "candidate": candidate,
               "z": Z99, "gates": holdout_rates,
               "screen_sha256": {p.name: _sha(p) for p in files}}
    _write(validation / "verdict.json", verdict)
    _write(validation / "goal-verdict.json", verdict)
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--screen-dir", type=Path)
    p.add_argument("--python", default="python")
    p.add_argument("--evolve", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args(argv)
    result = run_controller(args.root, screen_dir=args.screen_dir, python=args.python,
                            evolve=args.evolve, checkpoint=args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
