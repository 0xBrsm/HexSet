# SPDX-License-Identifier: GPL-3.0-only
"""Select blends on early study blocks; validate on untouched later blocks.

Selection and confidence rules are recorded in the readout's holdout-plan.json.
Rates receive equal weight. Board samples average both seat assignments and
all three environments before estimating uncertainty. The two secondary
comparisons use Bonferroni-adjusted 97.5% intervals (95% family coverage).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist

from hexset.arena import mean_interval
from hexset.bench.trade_curve_study import GRID, analyze, atomic_json


def validate(blocks: list[list[dict]], split: int = 8) -> dict:
    if not 0 < split < len(blocks):
        raise ValueError("separate nonempty selection and validation blocks are required")
    analyze(blocks)  # Validate pairing and completeness before selecting.
    seeds = [b[0]["experiment"]["settings"]["seed"] for b in blocks]
    if len(set(seeds)) != len(seeds):
        raise ValueError("selection and validation blocks must use distinct seeds")
    wins = {key: 0 for key in GRID}
    for block in blocks[:split]:
        for cell in block:
            wins[cell["initiate"], cell["alpha"]] += sum(
                row["winner"] == 0 for row in cell["experiment"]["outcomes"])
    rates = (0.0, 0.5, 1.0)
    alphas = (0.0, 0.5, 1.0)
    static = max(alphas, key=lambda a: (sum(wins[r, a] for r in rates), a))
    adaptive = {
        r: max(alphas, key=lambda a: (wins[r, a], a == static, -abs(a - static), a))
        for r in rates
    }
    values = {"static_minus_trading_baseline": [], "adaptive_minus_static": []}
    held_out_wins = {"trading_baseline": 0, "static": 0, "adaptive": 0}
    games = 0
    for block in blocks[split:]:
        cells = {(c["initiate"], c["alpha"]): c["experiment"]["outcomes"] for c in block}
        count = len(cells[GRID[0]])
        for i in range(0, count, 2):
            board = {name: 0 for name in held_out_wins}
            for rate in rates:
                for j in (i, i + 1):
                    board["trading_baseline"] += cells[rate, 1.0][j]["winner"] == 0
                    board["static"] += cells[rate, static][j]["winner"] == 0
                    board["adaptive"] += cells[rate, adaptive[rate]][j]["winner"] == 0
            for name in held_out_wins:
                held_out_wins[name] += board[name]
            games += 6
            values["static_minus_trading_baseline"].append((board["static"] - board["trading_baseline"]) / 6)
            values["adaptive_minus_static"].append((board["adaptive"] - board["static"]) / 6)
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    return {
        "selection_blocks": split, "holdout_blocks": len(blocks) - split,
        "selected_static_alpha": static,
        "selected_adaptive_alphas": {str(r): a for r, a in adaptive.items()},
        "training_wins": [{"rate": r, "alpha": a, "wins": wins[r, a]} for r, a in GRID],
        "holdout_games_per_policy": games,
        "holdout_wins": held_out_wins,
        "comparisons": {name: asdict(mean_interval(v, z=z)) for name, v in values.items()},
        "confidence": "97.5% per contrast; Bonferroni family-wise 95% for two secondary comparisons",
        "population": "equal mixture of willingness 0, 0.5, and 1 against fixed opponents",
        "baseline_label": "trading move weights with zero-threshold gates, not the production preset",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--split", type=int, default=8)
    args = parser.parse_args(argv)
    plan = json.loads((args.study / "plan.json").read_text())
    count = plan["games_per_cell"] // plan["chunk_games"]
    blocks = [[json.loads((args.study / f"block-{i:03d}-rate-{r:g}-alpha-{a:g}.json").read_text())
               for r, a in GRID] for i in range(count)]
    result = validate(blocks, args.split)
    atomic_json(args.study / "holdout.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
