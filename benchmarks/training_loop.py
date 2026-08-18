"""Production-shape sync/async PPO timing for an otherwise idle GPU box.

The live training job owns both the GPU and most CPU cores, so this harness is
checked in rather than run against a contended box. It runs identical seeded
training jobs with and without ``--async-collect`` and reports the median time
between iteration records after the pipeline-fill iteration. That interval is
the number that captures overlap; adding logged collection and update durations
would double-count work that deliberately runs concurrently.

Run from ``src/`` inside the GPU image, after the training container stops:

    python -m benchmarks.training_loop

Four iterations take roughly seven minutes at the recorded 51 s baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from catan import train

from .throughput import environment


def _run(label: str, args, asynchronous: bool) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"catan-{label}-") as raw:
        directory = Path(raw)
        command = [
            "--device",
            args.device,
            "--lanes",
            str(args.lanes),
            "--iterations",
            str(args.iterations),
            "--games-per-iteration",
            str(args.games),
            "--action-cap",
            str(args.action_cap),
            "--max-offers",
            str(args.max_offers),
            "--seed",
            str(args.seed),
            "--width",
            str(args.width),
            "--rounds",
            str(args.rounds),
            "--epochs",
            str(args.epochs),
            "--minibatch",
            str(args.minibatch),
            "--collect-workers",
            str(args.collect_workers),
            "--checkpoint-dir",
            str(directory),
            "--checkpoint-every",
            "1000000",
            "--keep-every",
            "0",
        ]
        if asynchronous:
            command.append("--async-collect")

        started = time.perf_counter()
        train.main(command)
        wall = time.perf_counter() - started
        records = [
            json.loads(line)
            for line in (directory / "log.jsonl").read_text().splitlines()
        ]

    intervals = [
        current["elapsed"] - previous["elapsed"]
        for previous, current in zip(records, records[1:])
    ]
    return {
        "mode": label,
        "iterations": len(records),
        "positions": [record["positions"] for record in records],
        "collect_seconds_mean": round(
            statistics.mean(record["collect_seconds"] for record in records), 3
        ),
        "assemble_seconds_mean": round(
            statistics.mean(record["assemble_seconds"] for record in records), 3
        ),
        "update_seconds_mean": round(
            statistics.mean(record["update_seconds"] for record in records), 3
        ),
        "steady_intervals": [round(value, 3) for value in intervals],
        "steady_seconds_median": round(statistics.median(intervals), 3),
        "whole_run_seconds": round(wall, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=512)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--collect-workers", type=int, default=24)
    args = parser.parse_args(argv)
    if args.iterations < 3:
        parser.error("at least three iterations are needed after pipeline fill")
    if args.collect_workers <= 1:
        parser.error("the async comparison needs at least two collector workers")

    sync = _run("sync", args, False)
    asynchronous = _run("async", args, True)
    before = float(sync["steady_seconds_median"])
    after = float(asynchronous["steady_seconds_median"])
    result = {
        "environment": environment(),
        "settings": vars(args),
        "sync": sync,
        "async": asynchronous,
        "speedup": round(before / after, 3),
        "wall_time_reduction": round(1.0 - after / before, 3),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
