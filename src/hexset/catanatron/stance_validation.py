"""Fail-closed aggregation for the eight-arm stance screen."""
from __future__ import annotations
import json
from pathlib import Path

GATES = ("vs-ab2", "vs-shipped")
THRESHOLDS = {"vs-ab2": 0.50, "vs-shipped": 0.25}
CONTROL = "baseline-win-T2.4766"
EXPECTED_CANDIDATES = (CONTROL, "win-T1", "win-T1.5", "win-T4", "win-T6", "own", "relative", "paranoid")

def _red_wins(row: dict) -> int:
    wins = row.get("wins")
    if not isinstance(wins, dict): raise ValueError("missing wins")
    key = "Color.RED" if "Color.RED" in wins else "0"
    if key not in wins: raise ValueError("missing Color.RED wins")
    try: value = int(wins[key]); total = sum(int(v) for v in wins.values())
    except (TypeError, ValueError) as exc: raise ValueError("non-integer wins") from exc
    if value < 0 or any(int(v) < 0 for v in wins.values()) or total != 120: raise ValueError("wins do not total 120")
    return value

def _points_present(document: dict, row: dict) -> bool:
    """Accept document-level and stancecfg row-level point arrays strictly."""
    top = document.get("points")
    nested = row.get("points")
    if top is not None and (not isinstance(top, dict) or not top):
        raise ValueError("invalid points")
    if nested is not None and (not isinstance(nested, dict) or not nested):
        raise ValueError("invalid row points")
    return bool(top) or bool(nested)

def aggregate(paths: list[Path]) -> dict:
    if len(paths) != 16: raise ValueError("stance screen requires exactly 16 files")
    docs = []
    seen = set()
    for path in paths:
        d = json.loads(path.read_text())
        candidate = d.get("candidate")
        if candidate not in EXPECTED_CANDIDATES: raise ValueError(f"unexpected candidate: {path}")
        if d.get("games") != 120 or len(d.get("rows", [])) != 1: raise ValueError(f"invalid game/row count: {path}")
        row = d["rows"][0]; gate = row.get("gate")
        if gate not in GATES or (candidate, gate) in seen: raise ValueError(f"missing/duplicate gate: {path}")
        if row.get("games") != 120: raise ValueError(f"invalid row games: {path}")
        _red_wins(row)
        if not _points_present(d, row): raise ValueError(f"missing points: {path}")
        seen.add((candidate, gate)); docs.append(d)
    required = {(c, g) for c in EXPECTED_CANDIDATES for g in GATES}
    if seen != required: raise ValueError("incomplete stance candidate/gate matrix")
    by_candidate = {}
    for d in docs:
        c = d["candidate"]; row = d["rows"][0]; wins = _red_wins(row)
        by_candidate.setdefault(c, {})[row["gate"]] = wins / 120
    ranking = []
    for c in EXPECTED_CANDIDATES:
        if c == CONTROL: continue
        margins = {g: by_candidate[c][g] - THRESHOLDS[g] for g in GATES}
        ranking.append({"candidate": c, "rates": by_candidate[c], "margins": margins, "positive_both": all(margins[g] >= 0 for g in GATES), "score": min(margins.values())})
    ranking.sort(key=lambda x: (-x["score"], x["candidate"]))
    winner = ranking[0]
    return {"candidate": winner["candidate"], "ranking": ranking, "confidence": "screening-only; no holdout inference", "manifest": {"candidates": list(EXPECTED_CANDIDATES), "control": CONTROL, "games_per_gate": 120, "gates": list(GATES), "thresholds": THRESHOLDS}}

def write_selection(paths: list[Path], output: Path) -> dict:
    result = aggregate(paths); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n"); return result
