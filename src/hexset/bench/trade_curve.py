# SPDX-License-Identifier: GPL-3.0-only
"""Sweep Heximax weight blends against opponents with controlled trade rates.

One focal player faces three fixed trading-profile Heximax opponents. Only
the focal move evaluator is blended; every trade evaluator stays fixed at
TRADING_WEIGHTS, with a zero gain threshold by default. Opponent willingness
is sampled once per game turn and role, independently of cards, candidate count, and the policy RNG. These are turn
participation probabilities, not probabilities per enumerated trade bundle.

This uses automatic arena clearing, not the server offer/response protocol.
The grid tests a line between existing profiles, not arbitrary optimal weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from hexset.arena import Entrant, compete, register_entrant_kind
from hexset.bench.metrics import json_metrics, paired_mean, side_metrics
from hexset.bots.evaluate import TERM_NAMES, Weights
from hexset.bots.heximax import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, heximax
from hexset.experiment import provenance, result_document


def probability(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities and blends must be finite and in [0, 1]")
    return value


def blend_weights(alpha: float) -> Weights:
    """0 is the no-trade profile; 1 is the trading profile."""
    probability(alpha)
    return Weights(**{
        name: (1 - alpha) * getattr(NO_TRADE_WEIGHTS, name)
        + alpha * getattr(TRADING_WEIGHTS, name)
        for name in TERM_NAMES
    })


@dataclass(frozen=True)
class CurveEntrant(Entrant):
    kind: str = "heximax-trade-curve"
    initiate: float = 1.0
    respond: float = 1.0
    gate_floor: float = 0.0


class WillingBot:
    """Benchmark-only wrapper; normal search, fixed trade valuation.

    All seats bind the live game during opening placement, before trading
    starts. Gates consult only its public turn/actor metadata. Search copies
    do not carry live gates, so hypothetical search cannot alter this schedule.
    """

    def __init__(self, bot, gate, seed: str, initiate: float, respond: float):
        self.bot = bot
        self.gate = gate
        self.trade_floor = gate.trade_floor
        self.seed = seed
        self.initiate = probability(initiate)
        self.respond = probability(respond)
        self.game = None

    def choose(self, game):
        self.game = game
        return self.bot.choose(game)

    def willing(self, turn: int, active: bool) -> bool:
        rate = self.initiate if active else self.respond
        if rate in (0.0, 1.0):
            return bool(rate)
        key = f"{self.seed}:{turn}:{int(active)}".encode()
        draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return draw / 2**64 < rate

    def gains_many(self, view, received, counterparties):
        if self.game is None:
            raise RuntimeError("trade gate must be bound through opening choose calls")
        active = self.game.current_player == view.perspective
        if not self.willing(self.game.turns, active):
            return [-1.0] * len(received)
        return self.gate.gains_many(view, received, counterparties)


def spawn_curve(entrant: CurveEntrant, board, rng: random.Random):
    # Read the initial RNG state without advancing the move-search stream.
    seed = hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
    bot = heximax(board, rng, weights=entrant.weights,
                  depth=entrant.depth, width=entrant.width)
    gate = heximax(board, weights=TRADING_WEIGHTS)
    if not math.isfinite(entrant.gate_floor) or entrant.gate_floor < 0:
        raise ValueError("trade floor must be finite and nonnegative")
    gate.trade_floor = entrant.gate_floor
    return WillingBot(bot, gate, seed, entrant.initiate, entrant.respond)


def register_curve() -> None:
    register_entrant_kind("heximax-trade-curve", spawn_curve)


def lineup(alpha: float, initiate: float, respond: float,
           gate_floor: float = 0.0) -> list[CurveEntrant]:
    common = dict(depth=2, width=6, gate_floor=gate_floor)
    focal = CurveEntrant("focal", weights=blend_weights(alpha), **common)
    return [focal] + [
        CurveEntrant(f"opponent-{i}", weights=TRADING_WEIGHTS,
                     initiate=probability(initiate), respond=probability(respond), **common)
        for i in range(3)
    ]


def run_cell(alpha: float, initiate: float, respond: float, *,
             games: int, seed: int, workers: int, gate_floor: float = 0.0) -> dict:
    entrants = lineup(alpha, initiate, respond, gate_floor)
    source = provenance(entrants)
    tournament = compete(entrants, games, seed=seed, workers=workers,
                         records=True, worker_initializer=register_curve)
    focal_trades = []
    all_trades = []
    for seating, cleared in zip(tournament.seating, tournament.cleared):
        seat = seating[0]
        focal_trades.append(sum(seat in (t.a, t.b) for t in cleared))
        all_trades.append(len(cleared))
    turns = sum(tournament.turns)
    return {
        "alpha": alpha, "initiate": initiate, "respond": respond,
        "gate_floor": entrants[0].gate_floor,
        **side_metrics(tournament, [0]),
        "seconds": tournament.seconds,
        "mean_turns": tournament.mean_turns,
        "focal_trades_per_game": sum(focal_trades) / games,
        "table_trades_per_game": sum(all_trades) / games,
        "focal_trades_per_elapsed_turn": sum(focal_trades) / turns if turns else 0.0,
        "table_trades_per_elapsed_turn": sum(all_trades) / turns if turns else 0.0,
        "focal_trades_by_game": focal_trades,
        "table_trades_by_game": all_trades,
        "experiment": result_document(tournament, entrants, seed=seed, workers=workers,
                                      run_provenance=source),
    }


def comparisons(cells: list[dict]) -> list[dict]:
    """Within-regime differences against alpha=1, paired across identical seeds.

    Cluster adjacent games by their shared board before estimating uncertainty.
    These are exploratory normal intervals, not multiplicity-adjusted tests.
    """
    output = []
    for cell in cells:
        baseline = next((c for c in cells if c["alpha"] == 1.0
                         and c["initiate"] == cell["initiate"]
                         and c["respond"] == cell["respond"]), None)
        if baseline is None or cell["alpha"] == 1.0:
            continue
        a = cell["experiment"]["outcomes"]
        b = baseline["experiment"]["outcomes"]
        wins = paired_mean([float(x["winner"] == 0) - float(y["winner"] == 0)
                            for x, y in zip(a, b)])
        vp = paired_mean([x["points"][0] - y["points"][0] for x, y in zip(a, b)])
        output.append({
            "alpha": cell["alpha"], "initiate": cell["initiate"],
            "respond": cell["respond"], "baseline_alpha": 1.0,
            "win_rate_difference": asdict(wins), "focal_vp_difference": asdict(vp),
        })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=128, help="per cell; positive multiple of 4")
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--rates", default="0,0.5,1", help="tied initiate/respond rates")
    parser.add_argument("--regimes", help="override rates: initiate:respond pairs, comma-separated")
    parser.add_argument("--trade-floor", type=float, default=0.0,
                        help="gain threshold for every gate; defaults to zero for this experiment")
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 4 or args.workers <= 0:
        parser.error("--games must be a positive multiple of 4; --workers must be positive")
    if not math.isfinite(args.trade_floor) or args.trade_floor < 0:
        parser.error("--trade-floor must be finite and nonnegative")
    try:
        alphas = list(dict.fromkeys(probability(float(x)) for x in args.alphas.split(",")))
        if args.regimes:
            regimes = [tuple(probability(float(x)) for x in r.split(":"))
                       for r in args.regimes.split(",")]
            if any(len(r) != 2 for r in regimes):
                raise ValueError("each regime must contain initiate:respond")
        else:
            rates = [probability(float(x)) for x in args.rates.split(",")]
            regimes = [(r, r) for r in rates]
        regimes = list(dict.fromkeys(regimes))
    except ValueError as error:
        parser.error(str(error))
    if 1.0 not in alphas:
        parser.error("--alphas must include the trading baseline, 1")
    report = {
        "settings": {**vars(args), "out": str(args.out)},
        "design": "one focal vs three fixed opponents; fixed trading-profile gates",
        "weights": {str(a): asdict(blend_weights(a)) for a in alphas},
        "interpretation": "exploratory grid; fresh-board confirmation required before adoption",
        "cells": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for initiate, respond in regimes:
        for alpha in alphas:
            print(f"start initiate={initiate:g} respond={respond:g} alpha={alpha:g}",
                  file=sys.stderr, flush=True)
            cell = run_cell(alpha, initiate, respond, games=args.games,
                            seed=args.seed, workers=args.workers, gate_floor=args.trade_floor)
            report["cells"].append(cell)
            report["comparisons"] = comparisons(report["cells"])
            report["seconds"] = time.perf_counter() - started
            temporary = args.out.with_suffix(args.out.suffix + ".tmp")
            temporary.write_text(json.dumps(json_metrics(report), indent=2, allow_nan=False) + "\n")
            temporary.replace(args.out)
            print(f"  wins={cell['wins']}/{args.games} "
                  f"focal trades/game={cell['focal_trades_per_game']:.2f} "
                  f"seconds={cell['seconds']:.1f}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
