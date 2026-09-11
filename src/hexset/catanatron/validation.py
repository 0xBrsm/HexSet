"""Fail-closed selection and conservative holdout accounting for Luna."""
from __future__ import annotations
import json, math
from pathlib import Path

GATES = ("vs-ab2", "vs-shipped")
THRESHOLDS = {"vs-ab2": .50, "vs-shipped": .25}
Z99 = 2.5758293035489004

def lower(w: int, n: int, z: float = Z99) -> float:
    p = w / n; d = 1 + z*z/n
    q = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (p + z*z/(2*n) - q) / d

def read_screen(path: Path) -> dict:
    d = json.loads(path.read_text())
    if not isinstance(d.get("candidate"), str) or d["candidate"] == "baseline":
        raise ValueError(f"invalid/control candidate: {path}")
    if d.get("games") != 120 or len(d.get("rows", [])) != 2:
        raise ValueError(f"incomplete screen file: {path}")
    rows = {r.get("gate"): r for r in d["rows"]}
    if set(rows) != set(GATES): raise ValueError(f"missing/duplicate gates: {path}")
    for gate, r in rows.items():
        wins = r.get("wins", {}); key = "Color.RED" if "Color.RED" in wins else "0"
        if r.get("games") != 120 or key not in wins or sum(int(v) for v in wins.values()) != 120:
            raise ValueError(f"incomplete gate row: {path} {gate}")
    return d

def rank_screen(paths: list[Path]) -> list[str]:
    if len(paths) != 9: raise ValueError("expected exactly nine screen files")
    docs = [read_screen(p) for p in paths]
    names = [d["candidate"] for d in docs]
    if len(set(names)) != 9: raise ValueError("duplicate screen candidate")
    scored = []
    for d in docs:
        rates = {}
        for r in d["rows"]:
            wins = r["wins"]; key = "Color.RED" if "Color.RED" in wins else "0"
            rates[r["gate"]] = int(wins[key]) / 120
        scored.append((min(rates[g] - THRESHOLDS[g] for g in GATES), d["candidate"]))
    return [n for _, n in sorted(scored, key=lambda x: (-x[0], x[1]))[:3]]

def decide_holdout(rows: dict[str, tuple[int, int]]) -> bool:
    return all(lower(*rows[g]) > THRESHOLDS[g] for g in GATES)

__all__ = ["GATES", "THRESHOLDS", "lower", "read_screen", "rank_screen", "decide_holdout"]
