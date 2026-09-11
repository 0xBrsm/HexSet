"""Run the pre-evolution bank/DCL follow-on screen.

The screen is fail-closed and resumable.  It owns a separate seed family
starting at 3,000,000 and never changes the shipped or ordinary Heximax
presets.  A candidate must clear both point targets before confirmation or
holdout games are scheduled.
"""
from __future__ import annotations

import argparse, json, subprocess, math
from pathlib import Path

ARMS = ("bankext", "bankext-wide", "control", "control-wide", "dcp")
GATES = ("ab2", "shipped")
THRESHOLDS = {"ab2": .50, "shipped": .25}
GAMES = 120


def _run(python, root, arm, gate, games, seed):
    out = root / f"{arm}-{gate}-{games}-{seed}.json"
    if not out.exists():
        module = "bankcfg" if arm not in ("dcl", "dcp") else "dclcfg"
        subprocess.run([python, "-m", "hexset.catanatron." + module,
                        "--candidate", arm, "--gate", gate,
                        "--games", str(games), "--workers", "30",
                        "--seed", str(seed), "--out", str(out)], check=True)
    try:
        doc = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"malformed result: {out}") from exc
    if (doc.get("candidate") != arm or doc.get("gate") not in (gate, "vs-" + gate)
            or doc.get("games") != games or doc.get("workers") != 30
            or doc.get("seed") != seed):
        raise ValueError(f"result identity mismatch: {out}")
    if "max_trades" in doc and doc["max_trades"] != 0:
        raise ValueError(f"trading enabled in result: {out}")
    wins = doc.get("wins")
    points = doc.get("points")
    if not isinstance(wins, dict) or not isinstance(points, dict) or not points:
        raise ValueError(f"missing result totals: {out}")
    try:
        vals = [int(v) for v in wins.values()]
        if any(v < 0 for v in vals) or sum(vals) != games:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid result totals: {out}") from exc
    return doc


def _wins(doc):
    w = doc.get("wins", {}).get("Color.RED", doc.get("wins", {}).get("0"))
    if not isinstance(w, int) or not (0 <= w <= doc.get("games", 0)):
        raise ValueError("invalid result wins")
    return w

def _wilson_lower(wins, games, z=3.29052673149):
    p=wins/games; d=1+z*z/games
    q=z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))
    return (p+z*z/(2*games)-q)/d

def _oracle_ok(path):
    try: d=json.loads(path.read_text())
    except (OSError, ValueError): return False
    return (d.get("status") == "PASS" and isinstance(d.get("source_sha256"), str)
            and len(d["source_sha256"]) == 64 and d.get("schema") == 1 and d.get("source_sha256") == "f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea")


def run(root: Path, python="python"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    # DCL is enabled only after the independent oracle audit is recorded.
    oracle = root / "dcp-oracle.json"
    arms = ("bankext", "bankext-wide", "control", "control-wide", "dcp") if _oracle_ok(oracle) else ("bankext", "bankext-wide", "control", "control-wide")
    if "dcl" not in arms:
        (root / "dcp-skipped.json").write_text(json.dumps({"status":"SKIP", "reason":"oracle audit artifact absent"}, indent=2))
    rows = []
    for i, arm in enumerate(arms):
        rates = {}
        for j, gate in enumerate(GATES):
            doc = _run(python, root, arm, gate, GAMES, 3_000_000 + j * 100_000)
            rate = _wins(doc) / GAMES
            rates[gate] = rate
            rows.append({"arm": arm, "gate": gate, "games": GAMES, "wins": _wins(doc), "rate": rate})
        if all(rates[g] > THRESHOLDS[g] for g in GATES):
            # Confirmation and holdouts are delegated to the same evaluator,
            # with one gate per file and disjoint seed families.
            for j, gate in enumerate(GATES):
                c = _run(python, root, arm, gate, 1024, 4_000_000 + i * 10_000 + j * 100_000)
                if _wins(c) / 1024 <= THRESHOLDS[gate]:
                    break
            else:
                holdouts = {}
                for j, gate in enumerate(GATES):
                    h = _run(python, root, arm, gate, 4096, 5_000_000 + i * 1_000_000 + j * 100_000)
                    wins = _wins(h); holdouts[gate] = {"wins": wins, "games": 4096,
                        "wilson_lower_99_9": _wilson_lower(wins, 4096), "threshold": THRESHOLDS[gate]}
                status = "PASSED_VALIDATION" if all(v["wilson_lower_99_9"] > v["threshold"] for v in holdouts.values()) else "HOLDOUT_FAIL"
                if arm not in ("control", "control-wide"):
                    return {"status": status, "candidate": arm, "screen": rows, "gates": holdouts, "holdouts": holdouts}
                # Controls are calibration arms and can never be promoted.
    return {"status": "NO_CANDIDATE", "screen": rows, "arms": list(arms)}


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", default="python"); a = p.parse_args(argv)
    result = run(a.root, a.python)
    (a.root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0 if result["status"] == "PASSED_VALIDATION" else 1


if __name__ == "__main__": raise SystemExit(main())
