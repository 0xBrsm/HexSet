# SPDX-License-Identifier: GPL-3.0-only
"""Compare two arena entrants with paired boards and confidence intervals.

Run ``python -m hexset.bench.duel heximax heximax-notrade --games 400``.
For batched model evaluation use ``hexset.bench.versus.compete_batched``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import hexset.bots  # noqa: F401 -- registers the heximax presets with hexset.arena
from hexset.game import MAX_TURNS

from hexset.bench.throughput import default_workers

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
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--duel-seed", type=int, default=20000)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--geometry", default=ARENA_GEOMETRY,
                   help="a/b seat pattern or comma-separated lineup, e.g. a,b,random,random")
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
            handle.write(json.dumps(result) + "\n")

    print(json.dumps(result, indent=1))
    if destination is not None:
        print(f"\nappended to {destination}", file=sys.stderr)
    print(
        f"\n{label_a} vs {label_b}: {result['win_rate']*100:.1f}% "
        f"[{result['wilson_low']*100:.1f}, {result['wilson_high']*100:.1f}] "
        f"over {result['games']} games, paired VP {result['paired_vp']:+.2f}",
        file=sys.stderr,
    )
    # The write happens once, above, whether the destination came from --json or
    # from the verdict default. A second append used to live here and survived
    # the change that introduced the default, so every duel passing --json
    # recorded itself twice.
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
    still has exactly two to subtract. Slot 0 is always side A, so `pooled`'s
    first group is side A under any seating.
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
    from hexset.arena import compete, lineup_from_names, pooled, wilson

    names, mine, theirs = arena_lineup(args.a, args.b, geometry)
    if args.games % len(names):
        raise ValueError(
            f"{args.games} games does not divide evenly over the {len(names)} "
            f"seats of {geometry!r}: the arena rotates the lineup through every "
            "seat and an incomplete rotation leaves the seat bias in the verdict"
        )
    lineup = sides(lineup_from_names(names), label_a, label_b, mine, theirs)

    started = time.monotonic()
    tournament = compete(
        lineup,
        args.games,
        seed=args.duel_seed,
        workers=args.workers,
        records=bool(args.records),
    )
    seconds = time.monotonic() - started

    if args.records:
        from hexset.record import write

        Path(args.records).parent.mkdir(parents=True, exist_ok=True)
        written = write(args.records, tournament.records)
        print(f"appended {written} records to {args.records}", file=sys.stderr)

    grouped = pooled(tournament.standings, tournament.games)
    wins = grouped[0].wins
    low, high = wilson(wins, tournament.games)
    paired = [
        sum(points[i] for i in mine) / len(mine)
        - sum(points[i] for i in theirs) / len(theirs)
        for points in tournament.points
    ]
    mean = sum(paired) / len(paired) if paired else 0.0
    spread = (
        1.96 * (sum((x - mean) ** 2 for x in paired) / (len(paired) - 1) / len(paired)) ** 0.5
        if len(paired) > 1
        else 0.0
    )
    turns = tournament.turns
    # Exhausted: reached `MAX_TURNS` without a winner. Distinct from
    # `unfinished`, which also counts games `play`'s own action cap cut off
    # short of that -- those have `winner is None` too but never reach
    # `MAX_TURNS` turns.
    exhausted = sum(
        1 for winner, t in zip(tournament.winners, turns) if winner is None and t >= MAX_TURNS
    )
    return {
        "a": label_a, "b": label_b, "a_path": args.a, "b_path": args.b,
        "games": tournament.games, "duel_seed": args.duel_seed,
        "workers": args.workers, "seconds": seconds, "via": "arena.compete",
        "geometry": geometry,
        "unfinished": tournament.unfinished,
        "wins": wins, "win_rate": wins / tournament.games if tournament.games else 0.0,
        "wilson_low": low, "wilson_high": high,
        "paired_vp": mean,
        "paired_vp_low": mean - spread, "paired_vp_high": mean + spread,
        "turns_mean": statistics.mean(turns) if turns else 0.0,
        "turns_median": statistics.median(turns) if turns else 0.0,
        "turns_max": max(turns) if turns else 0,
        "exhausted": exhausted,
    }


if __name__ == "__main__":
    raise SystemExit(main())
