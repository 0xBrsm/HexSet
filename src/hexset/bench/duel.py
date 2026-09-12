# SPDX-License-Identifier: GPL-3.0-only
"""Compare two arena entrants with paired boards and confidence intervals.

Run ``python -m hexset.bench.duel heximax heximax:pin-weights=0 --games 400``.
For batched model evaluation use ``hexset.bench.versus.compete_batched``.

A `network:`/`mcts:` entrant needs a runtime that can open its checkpoint,
which lives outside this distribution. Name it with ``--runtime``::

    python -m hexset.bench.duel --runtime hexn.netbot \
        network:runs/ppo/latest.pt heximax --games 400
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from dataclasses import replace

import hexset.bots  # noqa: F401 -- registers the heximax presets with hexset.arena
from hexset.game import MAX_TURNS

from hexset.bench.throughput import default_workers
from hexset.bench.metrics import json_metrics, paired_mean
from hexset.experiment import provenance, result_document

ARENA_GEOMETRY = "aabb"


def arena_lineup(a: str, b: str, geometry: str) -> tuple[list[str], list[int], list[int]]:
    """(entrant specs in lineup order, side-A slots, side-B slots) for a seating.

    `geometry` is either a pattern of `a`/`b` letters, one per seat and any
    length the arena can rotate (`aabb`, `abab`, `aab`, `aabbb`), or a
    comma-separated lineup whose entries are `a`, `b`, or any entrant spec
    the arena resolves -- which is how a duel gets run at a table that also
    seats bots on neither side. Side A is every slot holding `a`, side B
    every slot holding `b`.

    Slot lists index `Tournament.points`, which is in entrant order, so the
    paired split reads the right seats whichever order the lineup was built in.
    """
    slots = [s.strip() for s in geometry.split(",")] if "," in geometry else list(geometry)
    mine = [i for i, slot in enumerate(slots) if slot == "a"]
    theirs = [i for i, slot in enumerate(slots) if slot == "b"]
    if not mine or not theirs:
        raise ValueError(
            f"seating {geometry!r} must hold at least one 'a' slot and one 'b' "
            "slot -- those are the two sides the verdict is about"
        )
    specs = [a if slot == "a" else b if slot == "b" else slot for slot in slots]
    return specs, mine, theirs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="arena entrant name or registered checkpoint spec")
    p.add_argument("b", help="arena entrant name or registered checkpoint spec")
    p.add_argument("--pin-weights-a", type=int, choices=(0, 1), default=None,
                   help="pin side A Heximax to slider endpoint 0 or 1; trading is unchanged")
    p.add_argument("--pin-weights-b", type=int, choices=(0, 1), default=None,
                   help="pin side B Heximax to slider endpoint 0 or 1; trading is unchanged")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--duel-seed", type=int, default=20000)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--geometry", default=ARENA_GEOMETRY,
                   help="a/b seat pattern or comma-separated lineup, e.g. a,b,random,random")
    p.add_argument("--runtime", default=None,
                   help="import this module before spawning, so it can register the "
                        "entrant kinds it provides (e.g. hexn.netbot for network:/mcts:)")
    p.add_argument("--trade-mode", choices=("round", "auto"), default="round",
                   help="the table's bargaining rule: one propose-and-respond round a "
                        "turn (default), or the automatic clearing house studies before "
                        "HexSet 0.50 were recorded under")
    p.add_argument("--max-trades", type=int, default=1,
                   help="completed exchanges a turn; -1 for the unbounded clearing house")
    p.add_argument("--json", default=None, help="append verdict to this JSON Lines file")
    p.add_argument("--verdicts", default="runs/eval", help="default verdict directory")
    p.add_argument("--no-json", action="store_true", help="print without writing a verdict file")
    p.add_argument("--records", default=None, help="append replayable game records to this file")
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem
    result = _via_arena(args, label_a, label_b, args.geometry)

    destination = None
    if not args.no_json:
        destination = Path(args.json) if args.json else _verdict_path(args, label_a, label_b)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")

    print(json.dumps(result, indent=1, allow_nan=False))
    if destination is not None:
        print(f"\nappended to {destination}", file=sys.stderr)
    print(
        f"\n{label_a} vs {label_b}: {result['win_rate']*100:.1f}% "
        f"[{result['board_win_rate_low']*100:.1f}, {result['board_win_rate_high']*100:.1f}] "
        f"over {result['games']} games (board-level interval), paired VP {result['paired_vp']:+.2f}",
        file=sys.stderr,
    )
    return 0


def sides(lineup: list, label_a: str, label_b: str, mine=(0, 1), theirs=(2, 3)) -> list:
    """Rename a two-sided lineup so the two sides are distinguishable.

    Every `network:` spec is named "network" whatever checkpoint it carries, so
    a checkpoint-against-checkpoint duel arrives as four entrants of one name:
    `pooled` puts all four on one side and the paired-VP split has nobody to
    subtract from. Naming the sides after their labels is exact for any pair of
    entrants rather than only for the ones whose names happen to differ, and
    `spawn` reads `kind` and `weights`, never `name`.

    `mine`/`theirs` are the slots the two sides hold -- `[0, 1]`/`[2, 3]` for
    `aabb`, `[0, 2]`/`[1, 3]` for `abab`. Any remaining slot is on neither
    side and keeps the name its own entrant spec gave it, so a duel run at a
    table with third-party bots pools into three groups and the paired split
    still has exactly two to subtract. Verdicts use slot indices, so side
    labels and the first slot do not determine which outcomes count.
    """
    side_a, side_b = label_a, label_b
    if side_a == side_b:
        side_a, side_b = f"{label_a}-a", f"{label_b}-b"
    seen = {side_a: 0, side_b: 0}
    renamed = []
    for slot, entrant in enumerate(lineup):
        if slot not in mine and slot not in theirs:
            renamed.append(entrant)
            continue
        label = side_a if slot in mine else side_b
        renamed.append(entrant.renamed(f"{label}#{seen[label]}"))
        seen[label] += 1
    return renamed


def _verdict_path(args, label_a: str, label_b: str) -> Path:
    """Where a duel lands when the caller does not say.

    Named after the pairing rather than the caller, so the same comparison
    re-run later appends beside its predecessor instead of landing in whatever
    scratch file that session happened to use. Slashes in an entrant spec
    become dashes; a checkpoint path collapses to `<run>-<checkpoint>`.
    """

    def token(label: str, spec: str) -> str:
        if label:
            return label.replace("/", "-")
        parts = Path(spec).parts
        return "-".join(parts[-2:]).replace(".pt", "") if len(parts) > 1 else spec

    pair = f"{token(label_a, args.a)}__vs__{token(label_b, args.b)}"
    return Path(args.verdicts) / f"{pair}.json"


def _via_arena(args, label_a: str, label_b: str, geometry: str = ARENA_GEOMETRY) -> dict:
    """Two a side through `arena.compete`, sharded across `--workers`.

    Paired VP is recovered from the tournament's own per-game record rather than
    given up for the parallelism: `Tournament.points` carries every seat's
    terminal points per game in entrant order, so the within-game difference the
    single-process path reports can be rebuilt exactly.
    """
    from hexset.arena import compete, lineup_from_names, wilson
    from hexset.clients.netbot import load_runtime

    names, mine, theirs = arena_lineup(args.a, args.b, geometry)
    if args.games % 2:
        raise ValueError("paired evaluation requires an even number of games")
    if args.games % len(names):
        raise ValueError(
            f"{args.games} games does not divide evenly over the {len(names)} "
            f"seats of {geometry!r}: the arena rotates the lineup through every "
            "seat and an incomplete rotation leaves the seat bias in the verdict"
        )
    entrants = lineup_from_names(names)
    for side, slots in (("a", mine), ("b", theirs)):
        pin = getattr(args, f"pin_weights_{side}", None)
        if pin is None:
            continue
        for slot in slots:
            entrant = entrants[slot]
            if entrant.kind != "heximax" or entrant.weights is not None:
                raise ValueError(f"--pin-weights-{side} requires standard Heximax")
            if entrant.pin_weights is not None:
                raise ValueError(f"side {side} specifies a weight pin twice")
            entrants[slot] = replace(entrant, pin_weights=pin)
    lineup = sides(entrants, label_a, label_b, mine, theirs)

    run_provenance = provenance(lineup)
    started = time.monotonic()
    tournament = compete(
        lineup,
        args.games,
        seed=args.duel_seed,
        workers=args.workers,
        records=bool(args.records),
        # Runs once per worker, and in this process when there is only one:
        # `compete` owns both, so the runtime is registered exactly where an
        # entrant is spawned and nowhere else.
        worker_initializer=load_runtime if args.runtime else None,
        worker_initargs=(args.runtime,) if args.runtime else (),
        trade_mode=args.trade_mode,
        max_trades=args.max_trades,
    )
    seconds = time.monotonic() - started

    if args.records:
        from hexset.record import write

        Path(args.records).parent.mkdir(parents=True, exist_ok=True)
        written = write(args.records, tournament.records)
        print(f"appended {written} records to {args.records}", file=sys.stderr)

    wins = sum(winner in mine for winner in tournament.winners)
    low, high = wilson(wins, tournament.games)
    paired = [
        sum(points[i] for i in mine) / len(mine)
        - sum(points[i] for i in theirs) / len(theirs)
        for points in tournament.points
    ]
    # Adjacent games share a board and random streams; use boards as samples.
    margin = paired_mean(paired)
    win_share = paired_mean([float(w in mine) for w in tournament.winners])
    turns = tournament.turns
    # Exhausted: reached `MAX_TURNS` without a winner. Distinct from
    # `unfinished`, which also counts games `play`'s own action cap cut off
    # short of that -- those have `winner is None` too but never reach
    # `MAX_TURNS` turns.
    exhausted = sum(
        1 for winner, t in zip(tournament.winners, turns) if winner is None and t >= MAX_TURNS
    )
    return json_metrics({
        "experiment": result_document(tournament, lineup, seed=args.duel_seed,
                                      workers=args.workers, run_provenance=run_provenance),
        "a": label_a, "b": label_b, "a_path": args.a, "b_path": args.b,
        "games": tournament.games, "duel_seed": args.duel_seed,
        "workers": args.workers, "seconds": seconds, "via": "arena.compete",
        "geometry": geometry,
        "trade_mode": args.trade_mode, "max_trades": args.max_trades,
        "unfinished": tournament.unfinished,
        "wins": wins, "win_rate": wins / tournament.games if tournament.games else 0.0,
        "wilson_low": low, "wilson_high": high,
        "boards": margin.samples,
        "paired_vp": margin.mean,
        "paired_vp_low": margin.lower, "paired_vp_high": margin.upper,
        "board_win_rate_low": max(0.0, win_share.lower),
        "board_win_rate_high": min(1.0, win_share.upper),
        "interval_unit": "board",
        "win_rate_denominator": "all games, including unfinished",
        "wilson_assumption": "independent games; paired games are correlated",
        "truncated": tournament.unfinished - exhausted,
        "turns_mean": statistics.mean(turns) if turns else 0.0,
        "turns_median": statistics.median(turns) if turns else 0.0,
        "turns_max": max(turns) if turns else 0,
        "exhausted": exhausted,
    })


if __name__ == "__main__":
    raise SystemExit(main())
