"""Resumable opt-in continuation after an authoritative T1 screen.

This controller is deliberately separate from the T1 source and only accepts
an explicit non-passing prior verdict.  Each phase writes a manifest before
and after work, so an interrupted run can resume from its output root.
"""
from __future__ import annotations
import argparse, json, subprocess, math
from pathlib import Path

ALLOWED_PRIOR = frozenset({"DEMONSTRATED_POINT_REJECTION", "HOLDOUT_REJECTED", "UNRESOLVED_CAP", "REJECTED_AT_LOOK", "FAIL"})


def _read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e: raise ValueError(f"invalid JSON: {path}") from e


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _prior_status(prior):
    """Derive a closed non-pass verdict from a status file or artifact root."""
    path = Path(prior)
    if path.is_file():
        doc = _read(path)
        status = doc.get("status")
        if status == "PASS": return status, doc
        if status in ALLOWED_PRIOR: return status, doc
        raise ValueError("prior verdict is active, missing, or unsupported")
    if not path.is_dir(): raise ValueError("prior artifact root missing")
    manifest = path / "manifest.json"
    if not manifest.exists(): raise ValueError("prior manifest missing")
    md = _read(manifest)
    if md.get("status") in ("RUNNING", "ACTIVE"): raise ValueError("prior run is active")
    for name in ("confirmation.json", "holdout-decision-4096.json", "holdout-decision-8192.json", "holdout-decision-16384.json"):
        candidate = path / name
        if not candidate.exists(): continue
        doc = _read(candidate)
        if doc.get("status") == "PASS": return "PASS", doc
        if doc.get("status") in ("REJECTED_AT_LOOK", "HOLDOUT_REJECTED", "FAIL"):
            return "HOLDOUT_REJECTED", doc
        if doc.get("decision") in ("REJECTED_AT_LOOK", "UNRESOLVED_AT_CAP"):
            return ("UNRESOLVED_CAP" if doc["decision"] == "UNRESOLVED_AT_CAP" else "HOLDOUT_REJECTED"), doc
        rows = doc.get("rows")
        if candidate.name == "confirmation.json" and isinstance(rows, list) and len(rows) == 2:
            by = {r.get("gate"): r for r in rows if isinstance(r, dict)}
            if set(by) == {"vs-ab2", "vs-shipped"} and all(isinstance(r.get("games"), int) and r["games"] > 0 for r in by.values()):
                wins = {g: by[g].get("candidate_wins", by[g].get("wins")) for g in by}
                if all(isinstance(wins[g], int) and 0 <= wins[g] <= by[g]["games"] for g in by):
                    return "HOLDOUT_REJECTED", doc
    raise ValueError("prior artifacts are missing or incomplete")

def run_pipeline(*, prior, checkpoint, root, source_hash, cheap_run, validation_run, evolve_run):
    if isinstance(prior, (str, Path)):
        status, prior_doc = _prior_status(prior)
    else:
        prior_doc = prior
        status = prior_doc.get("status")
    if status == "PASS":
        return {"status": "PASS", "stage": "prior", "prior": prior_doc}
    if status not in ALLOWED_PRIOR:
        raise ValueError("prior verdict must be an explicit supported rejection or unresolved cap")
    if prior_doc.get("source_hash") is not None and prior_doc.get("source_hash") != source_hash:
        raise ValueError("prior source hash mismatch")
    rows = prior_doc.get("rows")
    if status == "FAIL" and not isinstance(rows, list):
        raise ValueError("FAIL prior lacks a complete closed game matrix")
    if rows is not None:
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError("prior gate matrix incomplete")
        gates = {r.get("gate") for r in rows if isinstance(r, dict)}
        if gates != {"vs-ab2", "vs-shipped"}:
            raise ValueError("prior gate identities incomplete")
        for row in rows:
            if not isinstance(row.get("games"), int) or row["games"] <= 0:
                raise ValueError("prior game count invalid")
            wins = row.get("candidate_wins", row.get("wins"))
            if not isinstance(wins, int) or not 0 <= wins <= row["games"]:
                raise ValueError("prior candidate wins invalid")
    checkpoint_path = Path(checkpoint)
    checkpoint_doc = _read(checkpoint_path) if checkpoint_path.exists() else {}
    if checkpoint_doc and checkpoint_doc.get("source_hash") != source_hash:
        raise ValueError("source hash mismatch")
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    _write(root / "manifest.json", {"schema": 1, "prior_status": status,
        "source_hash": source_hash, "checkpoint": str(checkpoint), "workers": 30,
        "max_trades": 0, "generations": 6})
    cheap_path = root / "cheap-result.json"
    cheap = _read(cheap_path) if cheap_path.exists() else cheap_run(root / "cheap")
    if not cheap_path.exists(): _write(cheap_path, cheap)
    if cheap.get("status") == "PASSED_VALIDATION":
        gates = cheap.get("gates", {})
        if set(gates) != {"ab2", "shipped"}:
            raise ValueError("cheap pass lacks both validated gates")
        for gate, row in gates.items():
            wins = row.get("wins"); games = row.get("games", 120)
            if not isinstance(wins, int) or not isinstance(games, int) or games != 4096 or not 0 <= wins <= games:
                raise ValueError("cheap pass requires 4096 fresh games per gate")
            threshold = {"ab2": .50, "shipped": .25}[gate]
            z = 3.29052673149; p = wins / games; d = 1 + z*z/games
            lower = (p + z*z/(2*games) - z*math.sqrt(p*(1-p)/games + z*z/(4*games*games))) / d
            if lower <= threshold:
                raise ValueError("cheap pass does not clear strict Wilson target")
        return {"status": "PASS", "stage": "cheap", "result": cheap}
    evolved = evolve_run(checkpoint, root / "evolve")
    _write(root / "evolve-result.json", evolved)
    # A completed evolution must always enter independent fresh validation.
    evolved_checkpoint = root / "evolve" / "checkpoint.json"
    validation = validation_run(evolved_checkpoint if evolved_checkpoint.exists() else checkpoint,
                                root / "validation")
    _write(root / "validation-result.json", validation)
    if validation.get("status") == "PASS":
        return {"status": "PASS", "stage": "validation", "result": validation,
                "evolve": evolved}
    return {"status": "FOLLOWON_COMPLETE", "stage": "evolve", "cheap": cheap,
            "validation": validation, "evolve": evolved}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--source-hash", required=True)
    p.add_argument("--python", default="python")
    a = p.parse_args(argv)
    from .cheap_followon import run as cheap_run
    from .evolve_validation import run_fresh
    def evolve_run(checkpoint, out):
        out.mkdir(parents=True, exist_ok=True)
        result = out / "checkpoint.json"
        cmd=[a.python,"-m","hexset.catanatron.evolve","--run","--checkpoint",str(result),"--generations","6","--count","12","--games","120","--total-games","300","--promote-top","4","--seed","10000000"]
        subprocess.run(cmd, check=True); return _read(result)
    result = run_pipeline(prior=a.prior, checkpoint=a.checkpoint, root=a.root,
        source_hash=a.source_hash,
        cheap_run=lambda out: cheap_run(out, a.python), validation_run=run_fresh,
        evolve_run=evolve_run)
    _write(a.root / "result.json", result); print(json.dumps(result, indent=2)); return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
