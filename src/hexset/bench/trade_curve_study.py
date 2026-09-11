# SPDX-License-Identifier: GPL-3.0-only
"""Checkpoint a fixed-sample zero-threshold trade-curve study on fresh boards.

The primary contrast is the change in the trading-vs-no-trade weight advantage
between opponent willingness 0 and 1. Midpoint contrasts are secondary. Read
the inferential result only after all planned blocks complete; progress is
not a significance-based stopping rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hexset.arena import mean_interval
from hexset.bench.metrics import json_metrics
from hexset.bench.trade_curve import run_cell


GRID = tuple((rate, alpha) for rate in (0.0, 0.5, 1.0) for alpha in (0.0, 0.5, 1.0))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_metrics(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def analyze(blocks: list[list[dict]]) -> dict:
    """Pair contrasts within the same game before clustering by board."""
    samples = {key: [] for key in GRID}
    vp_samples = {key: [] for key in GRID}
    counts = {key: dict(wins=0, games=0, unfinished=0, focal_trades=0, table_trades=0)
              for key in GRID}
    for block in blocks:
        cells = {(c["initiate"], c["alpha"]): c for c in block}
        if set(cells) != set(GRID):
            raise ValueError("analysis requires a complete grid in every block")
        reference = cells[GRID[0]]["experiment"]
        ref_settings = reference["settings"]
        for key in GRID:
            cell = cells[key]
            experiment = cell["experiment"]
            settings = experiment["settings"]
            if (cell["gate_floor"] != 0 or cell["respond"] != cell["initiate"]
                    or settings["seed"] != ref_settings["seed"]
                    or settings["games"] != ref_settings["games"]):
                raise ValueError("paired cells must share seed, game count, and zero threshold")
            rows = experiment["outcomes"]
            if [(r["index"], r["board_index"], r["seating"]) for r in rows] != [
                    (r["index"], r["board_index"], r["seating"])
                    for r in reference["outcomes"]]:
                raise ValueError("paired outcomes must share indices, boards, and seats")
            if len(rows) % 2:
                raise ValueError("every board must have two games")
            wins = [float(r["winner"] == 0) for r in rows]
            points = [r["points"][0] for r in rows]
            samples[key].extend((wins[i] + wins[i + 1]) / 2 for i in range(0, len(rows), 2))
            vp_samples[key].extend((points[i] + points[i + 1]) / 2
                                   for i in range(0, len(rows), 2))
            totals = counts[key]
            totals["wins"] += sum(wins)
            totals["games"] += len(rows)
            totals["unfinished"] += cell["unfinished"]
            totals["focal_trades"] += sum(cell["focal_trades_by_game"])
            totals["table_trades"] += sum(cell["table_trades_by_game"])

    def contrast(coefficients):
        return {
            label: asdict(mean_interval([
                sum(coefficient * values[key][i] for key, coefficient in coefficients.items())
                for i in range(len(values[GRID[0]]))
            ]))
            for label, values in (("win_rate", samples), ("focal_vp", vp_samples))
        }

    return {
        "completed_blocks": len(blocks),
        "boards": len(samples[GRID[0]]),
        "cells": [{"rate": r, "alpha": a, **counts[r, a],
                   "win_rate": asdict(mean_interval(samples[r, a]))}
                  for r, a in GRID],
        "primary_interaction": contrast({(1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1}),
        "primary_sign": "positive means trading weights gain relative to no-trade weights as willingness rises",
        "within_regime": [{"rate": r, "alpha": a,
                           **contrast({(r, a): 1, (r, 1): -1})}
                          for r in (0, 0.5, 1) for a in (0, 0.5)],
        "intervals": "95% normal intervals across independent board pairs; secondary contrasts are exploratory",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1024, help="per grid cell")
    parser.add_argument("--chunk-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=420000)
    parser.add_argument("--workers", type=int, default=30, help="Wintermute worker count")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint directory")
    args = parser.parse_args(argv)
    if (args.games <= 0 or args.chunk_games <= 0 or args.chunk_games % 4
            or args.games % args.chunk_games or args.workers <= 0):
        parser.error("positive game counts required; chunks must divide games and be multiples of four")
    args.out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        source_hash.update(str(path.relative_to(source_root)).encode() + b"\0" + path.read_bytes())
    plan = {"source_sha256": source_hash.hexdigest(), "games_per_cell": args.games, "chunk_games": args.chunk_games,
            "seed": args.seed, "grid": [list(key) for key in GRID], "gate_floor": 0,
            "primary": "(win[1,1] - win[1,0]) - (win[0,1] - win[0,0])",
            "stopping_rule": "fixed sample; no early stopping for significance"}
    plan_path = args.out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error("existing checkpoint plan differs; choose a new output directory")
    atomic_json(plan_path, plan)
    blocks = []
    for index in range(args.games // args.chunk_games):
        block = []
        for rate, alpha in GRID:
            path = args.out / f"block-{index:03d}-rate-{rate:g}-alpha-{alpha:g}.json"
            if path.exists():
                cell = json.loads(path.read_text())
            else:
                print(f"block={index + 1} rate={rate:g} alpha={alpha:g} start", flush=True)
                cell = run_cell(alpha, rate, rate, games=args.chunk_games,
                                seed=args.seed + index, workers=args.workers, gate_floor=0)
                atomic_json(path, cell)
                print(f"  {cell['wins']}/{cell['games']} wins; {cell['seconds']:.1f}s", flush=True)
            if (cell["alpha"] != alpha or cell["initiate"] != rate or cell["respond"] != rate
                    or cell["gate_floor"] != 0 or cell["games"] != args.chunk_games
                    or cell["experiment"]["settings"]["seed"] != args.seed + index):
                raise ValueError(f"checkpoint settings mismatch: {path}")
            block.append(cell)
        blocks.append(block)
        summary = analyze(blocks)
        summary["complete"] = len(blocks) == args.games // args.chunk_games
        summary["plan"] = plan
        atomic_json(args.out / "summary.json", summary)
        print(f"completed {len(blocks)} blocks / {summary['boards']} boards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
