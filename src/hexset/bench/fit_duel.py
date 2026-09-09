# SPDX-License-Identifier: GPL-3.0-only
"""Play a fitted `(weights, T)` against the shipped heximax, paired.

    python -m hexset.bench.fit_duel --fit fit.json --variant standard \
        --games 3072 --seed 42000 --workers 30 --out duel.json

The only test of a fit that counts. `hexset.bench.fit_weights` says whether
a vector predicts winners better on held-out games; this says whether the
bot that reads it wins more, on the same boards as the incumbent with seats
swapped (`hexset.bench.duel._via_arena`, `aabb` seating: two of each
side, antithetic pairs), reporting board-level win-share intervals and the
paired VP margin. Adoption is a decision made on this file, by hand.

The candidate is a heximax preset built from the fit: the variant's
`weights` and its `temperature`, everything else the shipped bot. A variant
with extra features (`leader_gap`, `remaining_production`) cannot be played
until the evaluator carries them, and is refused here rather than played
without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import hexset.bots  # noqa: F401 -- registers the heximax presets
from hexset.arena import Entrant, register_preset
from hexset.bench.duel import ARENA_GEOMETRY, _via_arena
from hexset.bench.throughput import default_workers, environment
from hexset.bots.evaluate import TERM_NAMES, Weights

CANDIDATE = "heximax-fitted"


def candidate_from(fit: dict, variant: str) -> tuple[Weights, float]:
    row = fit["variants"][variant]
    if row.get("extras"):
        raise SystemExit(
            f"variant {variant!r} carries extra features {sorted(row['extras'])} the "
            "evaluator does not have; it cannot be played as a Weights vector"
        )
    weights = Weights(**{name: float(row["weights"][name]) for name in TERM_NAMES})
    return weights, float(row["temperature"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, help="hexset.bench.fit_weights --out file")
    parser.add_argument("--variant", default="standard")
    parser.add_argument(
        "--against", default="heximax",
        help="the incumbent preset; `heximax-notrade` to read the no-trade table "
        "(the candidate then plays with trading off too)",
    )
    parser.add_argument("--games", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42000, help="duel seed")
    parser.add_argument(
        "--geometry", default=ARENA_GEOMETRY,
        help="seating, as `hexset.bench.duel.arena_lineup` spells it: a pattern "
        "of `a`/`b` letters, or a comma-separated lineup of entrant specs",
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    fit = json.loads(Path(args.fit).read_text())
    weights, temperature = candidate_from(fit, args.variant)
    notrade = args.against == "heximax-notrade"
    register_preset(
        CANDIDATE,
        Entrant(
            CANDIDATE, kind="heximax", depth=2, width=6, weights=weights,
            temperature=temperature,
            mode="notrade" if notrade else "honest", max_trades=0 if notrade else None,
        ),
    )
    ns = SimpleNamespace(
        a=CANDIDATE, b=args.against, games=args.games, duel_seed=args.seed,
        workers=args.workers, records=None,
    )
    result = _via_arena(ns, f"{CANDIDATE}:{args.variant}", args.against, geometry=args.geometry)
    result.update(
        {
            "environment": environment(),
            "fit": args.fit,
            "variant": args.variant,
            "candidate_weights": {name: getattr(weights, name) for name in TERM_NAMES},
            "candidate_temperature": temperature,
        }
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['a']} vs {result['b']}: {result['wins']}/{result['games']} = "
        f"{result['win_rate']:.1%} [{result['board_win_rate_low']:.1%}, {result['board_win_rate_high']:.1%}]"
        f"  paired VP {result['paired_vp']:+.3f} [{result['paired_vp_low']:+.3f}, "
        f"{result['paired_vp_high']:+.3f}]  unfinished {result['unfinished']}  "
        f"{result['seconds']:.0f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
