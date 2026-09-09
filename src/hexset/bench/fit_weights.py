# SPDX-License-Identifier: GPL-3.0-only
"""Fit heximax's weights and temperature from recorded games, then report.

    python -m hexset.bench.generate --bot heximax --games 5000 --workers 30 \
        --seed 300000 --out records.jsonl
    python -m hexset.bench.fit_weights --records records.jsonl --workers 30 \
        --out fit.json

Replays every record, reads each sampled position from every seat's own
information (`hexset.dataset`), and fits the conditional logit
(`hexset.fitting`) under several designs at once: the shipped term set, the
same with `robber_risk` pinned at zero, and the shipped set plus each of the
two cross-seat features. Every variant is scored by held-out log loss on
games the fit never saw, next to the shipped `(weights, T)` and the uniform
reading, so the table answers "does the fit predict winners better than what
ships" before a single duel is played. Which variant to *play* is then a
paired duel's question, not this script's.

Two numbers matter and they are not the same number. Held-out log loss says
the model predicts winners better; a duel says the bot plays better. A value
function can improve the first and not the second, since ranking positions
across games is not ranking the handful one move apart that the search
compares. This reports the first and stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from hexset.bench.throughput import default_workers, environment
from hexset.bots.evaluate import TERM_NAMES
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS
from hexset.bots.stances import WIN_TEMPERATURE
from hexset.dataset import ChoiceSet, samples_from, split_by_game
from hexset.fitting import (
    Design,
    base_features,
    fit_design,
    games_of,
    incumbent_beta,
    labels_of,
    log_loss,
    uniform_loss,
)
from hexset.record import read

VARIANTS: dict[str, dict] = {
    "standard": {},
    "no_risk": {"drop": ("robber_risk",)},
    "leader_gap": {"extras": ("leader_gap",)},
    "remaining": {"extras": ("remaining_production",)},
    "both": {"extras": ("leader_gap", "remaining_production")},
}


def _replay(job: tuple[int, object, int]) -> list[ChoiceSet]:
    game, record, stride = job
    return list(samples_from(record, game, stride=stride))


def summarise(fit, design: Design, test_X: np.ndarray, test_y: np.ndarray) -> dict:
    weights, temperature, extras = fit.weights()
    ratio_se = fit.ratio_se()
    out = {
        "columns": list(design.names),
        "beta": list(fit.beta),
        "beta_se_clustered": list(fit.se),
        "weights": {name: getattr(weights, name) for name in TERM_NAMES},
        "weights_se": {name: ratio_se.get(name) for name in TERM_NAMES},
        "extras": extras,
        "extras_se": {name: ratio_se.get(name) for name in extras},
        "temperature": temperature,
        "temperature_se": ratio_se["temperature"],
        "train_loss": fit.train_loss,
        "test_loss": log_loss(test_X, test_y, np.array(fit.beta)),
        "converged": fit.converged,
        "iterations": fit.iterations,
        "samples": fit.samples,
        "games": fit.games,
    }
    if fit.bootstrap:
        rows = []
        for beta in fit.bootstrap:
            w, t, e = design.weights(beta)
            rows.append([t] + [getattr(w, name) for name in TERM_NAMES] + list(e.values()))
        arr = np.array(rows)
        names = ["temperature", *TERM_NAMES, *extras]
        out["bootstrap"] = {
            "replicates": len(rows),
            "interval_95": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(
                    names, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
                )
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="JSON lines from hexset.bench.generate")
    parser.add_argument("--stride", type=int, default=8, help="actions between sampled positions")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help=f"comma-separated subset of {', '.join(VARIANTS)}",
    )
    parser.add_argument(
        "--profile", choices=("trading", "notrade"), default="trading",
        help="which shipped profile the incumbent row reads: `TRADING_WEIGHTS` "
        "for heximax records, `NO_TRADE_WEIGHTS` for heximax-notrade records",
    )
    parser.add_argument("--bootstrap", type=int, default=0, help="block-bootstrap replicates by game")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = list(read(args.records))
    jobs = [(i, r, args.stride) for i, r in enumerate(records)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            replayed = pool.map(_replay, jobs, chunksize=max(1, len(jobs) // (args.workers * 4)))
    else:
        replayed = [_replay(job) for job in jobs]
    samples = [s for game in replayed for s in game]
    if not samples:
        print("no labelled positions: were the recorded games all undecided?")
        return 1
    replay_seconds = time.perf_counter() - started

    train, test = split_by_game(samples, holdout=args.holdout, seed=args.split_seed)
    train_base = base_features(train)
    test_base = base_features(test)
    test_y = labels_of(test)
    seats = train_base.shape[1]

    incumbent = TRADING_WEIGHTS if args.profile == "trading" else NO_TRADE_WEIGHTS
    standard = Design.standard()
    shipped = incumbent_beta(standard, incumbent, WIN_TEMPERATURE)
    report: dict = {
        "environment": environment(),
        "settings": vars(args),
        "data": {
            "records": len(records),
            "decided": sum(1 for r in records if r.winner is not None),
            "choice_sets": len(samples),
            "train_choice_sets": len(train),
            "test_choice_sets": len(test),
            "train_games": len({s.game for s in train}),
            "test_games": len({s.game for s in test}),
            "replay_seconds": round(replay_seconds, 1),
        },
        "uniform_loss": uniform_loss(seats),
        "incumbent": {
            "profile": args.profile,
            "weights": {name: getattr(incumbent, name) for name in TERM_NAMES},
            "temperature": WIN_TEMPERATURE,
            "train_loss": log_loss(standard.matrix_from(train_base), labels_of(train), shipped),
            "test_loss": log_loss(standard.matrix_from(test_base), test_y, shipped),
        },
        "variants": {},
    }

    for name in args.variants.split(","):
        design = Design.standard(**VARIANTS[name])
        result = fit_design(
            design, train, base=train_base, bootstrap=args.bootstrap, seed=args.split_seed
        )
        report["variants"][name] = summarise(
            result, design, design.matrix_from(test_base), test_y
        )
        v = report["variants"][name]
        print(
            f"  {name:<12} test loss {v['test_loss']:.4f}  T={v['temperature']:.4f}"
            f"±{v['temperature_se']:.4f}  {'' if v['converged'] else 'NOT CONVERGED '}"
            f"({v['iterations']} it)",
            file=sys.stderr, flush=True,
        )

    report["seconds"] = round(time.perf_counter() - started, 1)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    d = report["data"]
    print(f"{d['decided']}/{d['records']} decided games -> {d['choice_sets']} choice sets "
          f"({d['train_games']} train / {d['test_games']} test games), replay {d['replay_seconds']}s")
    print(f"  uniform   test loss {report['uniform_loss']:.4f}")
    inc = report["incumbent"]
    print(f"  incumbent test loss {inc['test_loss']:.4f}  (T={inc['temperature']:.4f})")
    for name, v in report["variants"].items():
        print(f"  {name:<9} test loss {v['test_loss']:.4f}  T={v['temperature']:.4f}")
        print("    " + ", ".join(
            f"{k}={v['weights'][k]:.4g}±{(v['weights_se'][k] or 0):.2g}"
            for k in TERM_NAMES if k not in ("victory_point", "scarce")
        ))
        if v["extras"]:
            print("    " + ", ".join(
                f"{k}={val:.4g}±{(v['extras_se'][k] or 0):.2g}" for k, val in v["extras"].items()
            ))
    print(f"  {report['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
