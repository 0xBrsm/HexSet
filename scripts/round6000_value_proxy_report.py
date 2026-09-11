#!/usr/bin/env python3
"""Validate and summarize a completed round6000 ValueFunction calibration.

This consumes the plan emitted by round6000_value_proxy_plan.py.  It never
starts games and requires one proxy record for every planned job, paired with
the archived AB:2 record for the same candidate and index.
"""
import argparse, json, math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def rank(values):
    return [k for k, _ in sorted(values.items(), key=lambda x: (-x[1], x[0]))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--proxy-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    plan = load(a.plan)
    if plan.get("status") != "PLANNED_NO_EXECUTION":
        raise ValueError("unexpected plan status")
    proxy_wins = {c: 0 for c in plan["population"]}
    ab2_wins = {c: 0 for c in plan["population"]}
    deltas = {c: [] for c in plan["population"]}
    wall = {c: [] for c in plan["population"]}
    seen = set()
    for job in plan["jobs"]:
        key = (job["candidate"], job["index"])
        if key in seen:
            raise ValueError(f"duplicate job {key}")
        seen.add(key)
        proxy_path = a.proxy_root / Path(job["out"]).name
        if not proxy_path.exists():
            raise ValueError(f"missing proxy record {proxy_path}")
        proxy, archived = load(proxy_path), load(job["archived_ab2_record"])
        for d, expected_gate in ((proxy, "vs-value"), (archived, "vs-ab2")):
            if d.get("candidate") != job["candidate"] or d.get("game_index") != job["index"]:
                raise ValueError(f"identity mismatch in {job['candidate']}:{job['index']}")
            if d.get("seed") != job["seed"] and d is proxy:
                raise ValueError(f"seed mismatch in {proxy_path}")
        if proxy.get("gate") not in ("vs-value", "value", "vs-f"):
            raise ValueError(f"unexpected proxy gate in {proxy_path}")
        pw, aw = proxy.get("candidate_wins"), archived.get("candidate_wins")
        if pw not in (0, 1) or aw not in (0, 1):
            raise ValueError(f"candidate_wins must be 0/1 in {proxy_path}")
        proxy_wins[job["candidate"]] += pw
        ab2_wins[job["candidate"]] += aw
        deltas[job["candidate"]].append(pw - aw)
        if "wall_seconds" in proxy:
            value = proxy["wall_seconds"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid wall_seconds in {proxy_path}")
            wall[job["candidate"]].append(value)
    expected = len(plan["population"]) * plan["games_per_candidate"]
    if len(seen) != expected:
        raise ValueError(f"expected {expected} records, found {len(seen)}")
    n = plan["games_per_candidate"]
    summary = {}
    for c in plan["population"]:
        summary[c] = {
            "proxy_wins": proxy_wins[c], "ab2_wins": ab2_wins[c],
            "proxy_rate": proxy_wins[c] / n, "ab2_rate": ab2_wins[c] / n,
            "win_rate_delta": (proxy_wins[c] - ab2_wins[c]) / n,
            "paired_delta_sum": sum(deltas[c]),
            "paired_delta_mean": sum(deltas[c]) / n,
            "wall_seconds_mean": (sum(wall[c]) / len(wall[c]) if wall[c] else None),
            "wall_records": len(wall[c]),
        }
    proxy_rank = rank({c: summary[c]["proxy_rate"] for c in summary})
    ab2_rank = rank({c: summary[c]["ab2_rate"] for c in summary})
    report = {
        "schema": 1, "status": "VALIDATED",
        "plan": str(a.plan), "records": len(seen),
        "proxy_rank": proxy_rank, "archived_ab2_rank": ab2_rank,
        "rank_order_exact": proxy_rank == ab2_rank,
        "summary": summary,
        "decision": "selection_proxy_requires_human_review; final AB2/self gates unchanged",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(a.out)
    print(json.dumps({k: report[k] for k in ("status", "records", "proxy_rank", "archived_ab2_rank", "rank_order_exact")}, indent=2))


if __name__ == "__main__":
    main()
