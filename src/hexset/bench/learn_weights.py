# SPDX-License-Identifier: GPL-3.0-only
"""Learn evaluation weights from recorded outcomes, then check them by playing.

Two numbers matter and they are not the same number. Held-out log loss says the
model predicts winners better; the duel says it plays better. A value function
can improve the first and not the second — ranking positions well on average is
not the same as ranking well among the handful of positions one move apart,
which is all the search ever compares. So this reports both and trusts the duel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from hexset.bench.throughput import default_workers, environment
from hexset.arena import Z_95, wilson
from hexset.dataset import base_rate, build, split_by_game
from hexset.bots.evaluate import TERM_NAMES, Weights
from hexset.fitting import accuracy, fit, log_loss
from hexset.record import read
from hexset.tuning import as_source, duel


def report(name, samples, coefficients, intercept):
    rows = [s.features for s in samples]
    labels = [s.won for s in samples]
    return {
        "split": name,
        "samples": len(samples),
        "games": len({s.game for s in samples}),
        "base_rate": round(base_rate(samples), 4),
        "log_loss": round(log_loss(rows, labels, coefficients, intercept), 4),
        "accuracy": round(accuracy(rows, labels, coefficients, intercept), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="JSON lines from hexset.bench.generate")
    parser.add_argument("--stride", type=int, default=8, help="actions between sampled positions")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--duel",
        type=int,
        default=400,
        help="games of fitted vs current defaults; 0 skips the check",
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples = build(read(args.records), stride=args.stride)
    if not samples:
        print("no labelled positions: were the recorded games all undecided?")
        return 1

    train, test = split_by_game(
        samples, holdout=args.holdout, seed=args.split_seed
    )
    result = fit(train, epochs=args.epochs, rate=args.rate, l2=args.l2)
    learned = result.weights()

    splits = [
        report("train", train, result.coefficients, result.intercept),
        report("test", test, result.coefficients, result.intercept),
    ]

    check = None
    if args.duel:
        wins, decided = duel(
            learned,
            Weights(),
            args.duel,
            seed=555_000,
            depth=1,
            width=None,
            workers=args.workers,
        )
        low, high = wilson(wins, decided, Z_95) if decided else (0.0, 1.0)
        check = {
            "wins": wins,
            "decided": decided,
            "win_rate": round(wins / decided, 4) if decided else 0.0,
            "interval_95": [round(low, 4), round(high, 4)],
            "better": low > 0.5,
        }
    elapsed = time.perf_counter() - started

    if args.json:
        print(
            json.dumps(
                {
                    "environment": environment(),
                    "settings": vars(args),
                    "seconds": round(elapsed, 1),
                    "splits": splits,
                    "coefficients": dict(zip(TERM_NAMES, result.coefficients)),
                    "weights": {n: getattr(learned, n) for n in TERM_NAMES},
                    "duel_vs_defaults": check,
                },
                indent=2,
            )
        )
        return 0

    env = environment()
    print(f"commit {env['commit']}  {env['machine']}  {elapsed:.0f}s")
    for split in splits:
        print(
            f"  {split['split']:<5} {split['samples']:>7} rows"
            f"  {split['games']:>4} games"
            f"  base {split['base_rate']:.3f}"
            f"  log loss {split['log_loss']:.4f}"
            f"  acc {split['accuracy']:.3f}"
        )
    if check is not None:
        verdict = "better" if check["better"] else "NOT shown better"
        print(
            f"  learned vs current defaults: {check['wins']}/{check['decided']}"
            f" = {check['win_rate']:.1%}  95% CI"
            f" [{check['interval_95'][0]:.1%}, {check['interval_95'][1]:.1%}]"
            f" — {verdict}"
        )
    print(as_source(learned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
