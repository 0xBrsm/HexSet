"""Fresh, resumable validation for the fixed ordinary-Heximax candidate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Callable
from .evolve_validation import continuation_decision, wilson_lower
from .fair_budget import (CANDIDATE, GATES, PHENOTYPES, SELF_OPPONENT,
                          SCREEN_GAMES, SCREEN_WORKERS, players_for,
                          source_fingerprint, _write)
THRESH = {"ab2": .50, "shipped": .25}
LOOKS = (4096, 8192, 16384)
BLOCKS = (4096, 4096, 8192)
CONFIRM_SEEDS = {"ab2": 90_000_000, "shipped": 90_100_000}
HOLDOUT_SEEDS = {"ab2": 100_000_000, "shipped": 110_000_000}
PROTOCOL = "fair-budget-validation-v1"

def _screen_rows(screen: dict[str, Any], expected_source: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = screen.get("manifest")
    rows = screen.get("results")
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise ValueError("complete fair-budget screen manifest/results required")
    if manifest.get("protocol") != "fair-budget-v1" or manifest.get("games") != 120 or manifest.get("workers") != 30:
        raise ValueError("invalid fair-budget screen manifest")
    if manifest.get("phenotypes") != PHENOTYPES or manifest.get("self_opponent") != SELF_OPPONENT:
        raise ValueError("screen phenotype mismatch")
    if manifest.get("source_hash") != (expected_source or source_fingerprint()):
        raise ValueError("screen source fingerprint mismatch")
    if len(rows) != 4:
        raise ValueError("fair-budget screen requires exactly four rows")
    manifest_jobs = manifest.get("jobs")
    jobs = {(j.get("role"), j.get("gate")): j for j in manifest_jobs if isinstance(j, dict)} if isinstance(manifest_jobs, list) else {}
    if not isinstance(manifest_jobs, list) or len(manifest_jobs) != 4 or len(jobs) != 4:
        raise ValueError("invalid fair-budget screen jobs")
    for row in rows:
        if not isinstance(row, dict) or (row.get("role"), row.get("gate")) not in jobs:
            raise ValueError("screen row/job mismatch")
        job = jobs[(row["role"], row["gate"])]
        for key in ("entrant", "config", "games", "workers", "seed", "players", "phenotype"):
            if row.get(key) != job.get(key):
                raise ValueError(f"screen row identity mismatch: {key}")
        wins = row.get("wins")
        values = list(wins.values()) if isinstance(wins, dict) else []
        if not wins or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values) or sum(values) != 120:
            raise ValueError("invalid screen wins")
    return manifest, rows

def validate_screen(screen: dict[str, Any], source_hash: str | None = None) -> bool:
    _, rows = _screen_rows(screen, source_hash)
    candidates = [r for r in rows if r.get("role") == "candidate"]
    controls = [r for r in rows if r.get("role") == "control"]
    if {r.get("gate") for r in candidates} != set(GATES) or len(candidates) != 2:
        raise ValueError("candidate dual-gate screen required")
    if {r.get("gate") for r in controls} != set(GATES) or len(controls) != 2:
        raise ValueError("control dual-gate screen required")
    for row in candidates:
        wins = int(row["wins"].get("Color.RED", row["wins"].get("RED", -1)))
        if wins / SCREEN_GAMES <= THRESH[row["gate"]]:
            raise ValueError("candidate screen does not clear target")
    return True

def _validation_manifest(screen: dict[str, Any], source_hash: str) -> dict[str, Any]:
    jobs = []
    for gate in GATES:
        jobs.append({"phase":"confirmation", "gate":gate, "games":1024,
                     "seed":CONFIRM_SEEDS[gate], "workers":30,
                     "players":players_for(CANDIDATE, gate), "candidate":CANDIDATE,
                     "phenotype":PHENOTYPES[CANDIDATE]})
    for look, games in enumerate(BLOCKS):
        offset = sum(BLOCKS[:look])
        for gate in GATES:
            jobs.append({"phase":"holdout", "look":look, "gate":gate,
                         "games":games, "seed":HOLDOUT_SEEDS[gate] + offset,
                         "workers":30, "players":players_for(CANDIDATE, gate),
                         "candidate":CANDIDATE, "phenotype":PHENOTYPES[CANDIDATE]})
    return {"schema":2, "protocol":PROTOCOL, "candidate":CANDIDATE,
            "source_hash":source_hash,
            "screen_fingerprint":hashlib.sha256(json.dumps(screen, sort_keys=True).encode()).hexdigest(),
            "phenotype":PHENOTYPES[CANDIDATE], "self_opponent":SELF_OPPONENT,
            "jobs":jobs}

def plan(root: Path, screen: dict[str, Any], source_hash: str | None = None) -> dict[str, Any]:
    source_hash = source_hash or source_fingerprint()
    validate_screen(screen, source_hash)
    root = Path(root)
    doc = _validation_manifest(screen, source_hash); path = root / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != doc:
            raise ValueError("validation manifest/source/phenotype mismatch")
    else:
        _write(path, doc)
    return doc

def decide(counts: dict[str, tuple[int, int]]) -> str:
    return continuation_decision({"vs-" + g: (counts[g][0], counts[g][1]) for g in GATES})

def _validate_artifact(job: dict[str, Any], result: dict[str, Any], source_hash: str) -> int:
    for key in ("protocol", "source_hash", "candidate", "gate", "games", "seed", "workers", "players", "phenotype"):
        expected = PROTOCOL if key == "protocol" else source_hash if key == "source_hash" else job.get(key)
        if result.get(key) != expected:
            raise ValueError(f"validation artifact identity mismatch: {key}")
    wins = result.get("candidate_wins")
    if isinstance(wins, bool) or not isinstance(wins, int) or not 0 <= wins <= job["games"]:
        raise ValueError("invalid candidate wins")
    totals = result.get("wins")
    values = list(totals.values()) if isinstance(totals, dict) else []
    if not totals or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values) or sum(values) != job["games"]:
        raise ValueError("invalid validation wins")
    red = totals.get("Color.RED", totals.get("RED"))
    if red is not None and int(red) != wins:
        raise ValueError("candidate win identity mismatch")
    return wins

def run(root: Path, screen: dict[str, Any], game_runner: Callable[[dict[str, Any]], dict[str, Any]], source_hash: str | None = None) -> dict[str, Any]:
    root = Path(root); source_hash = source_hash or source_fingerprint()
    doc = plan(root, screen, source_hash); counts = {}
    expected_files = {f"{j['phase']}-{j['gate']}-{j['games']}-{j.get('look', 'confirm')}.json" for j in doc["jobs"]}
    extras = {p.name for p in root.glob("*.json") if p.name != "manifest.json"} - expected_files
    if extras: raise ValueError(f"unexpected validation artifacts: {sorted(extras)}")
    cumulative = {g: [0, 0] for g in GATES}
    for job in doc["jobs"]:
        path = root / f"{job['phase']}-{job['gate']}-{job['games']}-{job.get('look', 'confirm')}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = game_runner(job)
            _validate_artifact(job, result, source_hash)
            _write(path, result)
        wins = _validate_artifact(job, result, source_hash)
        counts[job["gate"]] = (wins, job["games"])
        if job["phase"] == "confirmation" and wins / job["games"] <= THRESH[job["gate"]]:
            return {"status":"REJECTED", "stage":"confirmation", "counts":counts}
        if job["phase"] == "holdout":
            cumulative[job["gate"]][0] += wins; cumulative[job["gate"]][1] += job["games"]
            if job["gate"] == "shipped":
                result_status = decide({g: tuple(cumulative[g]) for g in GATES})
                if result_status in ("PASS", "REJECTED_AT_LOOK", "UNRESOLVED_AT_CAP"):
                    return {"status":result_status, "stage":"holdout",
                            "counts":{g:tuple(cumulative[g]) for g in GATES}}
    return {"status":"CONTINUE", "counts":counts}

def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True); p.add_argument("--screen",type=Path,required=True); p.add_argument("--run",action="store_true"); p.add_argument("--source-hash",default=None); a=p.parse_args(argv)
    screen=json.loads(a.screen.read_text())
    if not a.run: print(json.dumps(plan(a.root,screen,a.source_hash),indent=2)); return 0
    from .duel import run_duel
    runtime_source_hash = a.source_hash or source_fingerprint()
    def game_runner(job):
        gate=job["gate"]; lineup=job["players"]; duel=run_duel(lineup,job["games"],job["workers"],seed=job["seed"])
        wins={str(k):int(v) for k,v in duel.wins.items()}; red=wins.get("Color.RED",wins.get("RED"))
        return {"protocol":PROTOCOL,"source_hash":runtime_source_hash,"candidate":CANDIDATE,
                "gate":gate,"games":job["games"],"seed":job["seed"],"workers":job["workers"],
                "players":lineup,"phenotype":PHENOTYPES[CANDIDATE],"candidate_wins":red,
                "wins":wins,"points":{str(k):list(v) for k,v in duel.points.items()}}
    print(json.dumps(run(a.root,screen,game_runner,a.source_hash),indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
