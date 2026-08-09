"""Fit the evaluation weights, printing each duel as it resolves.

Budget this before starting it. Each round plays `--games` games, and the
acceptance test needs enough of them to distinguish a real improvement from
noise: at 40 games a challenger must win about 65% to be accepted at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from benchmarks.throughput import default_workers, environment
from catan.tuning import ACCEPT_Z, Step, as_source, climb, confirm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--games", type=int, default=40, help="per duel; must be a multiple of 4"
    )
    parser.add_argument("--sigma", type=float, default=0.4, help="step size")
    parser.add_argument("--count", type=int, default=2, help="weights jittered per round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--z",
        type=float,
        default=ACCEPT_Z,
        help="acceptance strictness in standard errors; higher accepts less",
    )
    parser.add_argument(
        "--depth", type=int, default=1, help="1 fits the greedy bot, 2+ the search bot"
    )
    parser.add_argument("--width", type=int, default=6, help="beam, when depth is 2+")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--stance",
        default="relative",
        help="how a seat reads the per-seat vector; weights are fitted for one",
    )
    parser.add_argument(
        "--evaluator",
        default="default",
        choices=("default", "tiered"),
        help="which evaluation to fit; the two do not share a term set",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=400,
        help="games for the final fitted-vs-start duel; 0 skips it",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()

    def show(step: Step) -> None:
        mark = "accept" if step.accepted else "  keep"
        print(
            f"  round {step.round:>3}  {mark}  {step.wins:>3}/{step.decided}"
            f"  lower {step.lower:.3f}",
            flush=True,
        )

    best, history = climb(
        rounds=args.rounds,
        games=args.games,
        sigma=args.sigma,
        count=args.count,
        seed=args.seed,
        z=args.z,
        workers=args.workers,
        depth=args.depth,
        width=args.width if args.depth > 1 else None,
        stance=args.stance,
        evaluator=args.evaluator,
        report=None if args.json else show,
    )
    accepted = sum(1 for step in history if step.accepted)

    check = None
    if args.confirm:
        if not args.json:
            print(f"  confirming over {args.confirm} games...", flush=True)
        check = confirm(
            best,
            games=args.confirm,
            depth=args.depth,
            width=args.width if args.depth > 1 else None,
            workers=args.workers,
            stance=args.stance,
            evaluator=args.evaluator,
        )
    elapsed = time.perf_counter() - started

    if args.json:
        print(
            json.dumps(
                {
                    "environment": environment(),
                    "settings": vars(args),
                    "seconds": round(elapsed, 1),
                    "accepted": accepted,
                    "rounds": args.rounds,
                    "confirmation": None
                    if check is None
                    else {
                        "wins": check.wins,
                        "decided": check.decided,
                        "win_rate": round(check.win_rate, 3),
                        "interval_95": [round(check.lower, 3), round(check.upper, 3)],
                        "real": check.real,
                    },
                    "weights": {
                        name: getattr(best, name) for name in best.__dataclass_fields__
                    },
                },
                indent=2,
            )
        )
        return 0

    env = environment()
    print(f"commit {env['commit']}  {env['machine']}")
    print(
        f"{accepted}/{args.rounds} accepted in {elapsed:.0f}s"
        f"  ({args.games} games per duel, depth {args.depth}, {args.workers} worker(s))"
    )
    if check is not None:
        verdict = "a real gain" if check.real else "NOT distinguishable from noise"
        print(
            f"fitted vs start: {check.wins}/{check.decided} = {check.win_rate:.1%}"
            f"  95% CI [{check.lower:.1%}, {check.upper:.1%}] — {verdict}"
        )
    print(as_source(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
