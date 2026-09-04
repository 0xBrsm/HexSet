# SPDX-License-Identifier: GPL-3.0-only
"""Two checkpoints, head to head on identical boards, in paired terminal VP.

The in-loop ladder's 200-game rungs cannot resolve a slope: their scatter is the
binomial floor of the eval size, so a 150-iteration block's slope carries a 95%
half-width of ~9 points. This runs the instrument that *did* resolve ppo3's
selection — `train.versus` at 400 games — between any two checkpoints, so a
block's gain is measured as one high-resolution difference rather than fitted
through noise.

    python -m hexset.bench.duel /w/runs/ppo5/latest.pt /w/runs/ppo4/latest.pt \
        --games 400 --label-a ppo5 --label-b ppo4

Every verdict names its seat geometry -- see `GEOMETRIES` -- because the two
paths seat a 2v2 differently and the seating alone is worth ~0.35 VP.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import hexset.bots  # noqa: F401 -- registers the heximax presets with hexset.arena
from hexset.arena import NETWORK
from hexset.game import MAX_TURNS

# The `--workers 1` path (bare checkpoints, network-vs-network) runs through
# `hexnet.collect`/`hexnet.train`, which need torch -- so this module never
# imports them itself. `hexnet.duel` registers the runner here at import,
# which every HexNet entry point that wants this path pulls in; a hexset-only
# process gets a clear error naming the package instead of an ImportError deep
# inside `hexnet.train`.
_VERSUS_BACKEND: Callable[[argparse.Namespace, str, str], dict] | None = None


def register_versus_backend(runner: Callable[[argparse.Namespace, str, str], dict]) -> None:
    """Register the network-backed `--workers 1` runner: `hexnet.duel` calls
    this at import, wiring `_via_versus` (bare checkpoints, `hexnet.train.versus`)
    back into this module without it importing torch or hexnet itself."""
    global _VERSUS_BACKEND
    _VERSUS_BACKEND = runner

# The two ways four seats hold two sides, and which lineup slots each side owns.
# `arena._play_one` seats entrant `e` at `(e + rotation) % 4`, so `[a, a, b, b]`
# gives every copy one same-side neighbour -- blocked -- and `[a, b, a, b]` puts
# the copies opposite each other, each flanked by two opponents -- interleaved.
# `collect.alternating` seats the versus path on same-parity seats, so
# `--workers 1` has always played interleaved; `--workers >1` has always played
# blocked. On identical boards and dice the seating alone moves `lam095-805`
# vs `ppo4-585` from +0.08 to +0.43 VP (the harness-path check, addenda 6-8;
# `runs/eval/harness-seat-geometry.json`), so a verdict that does
# not record its geometry cannot be compared with one that does.
GEOMETRIES: dict[str, tuple[str, list[int], list[int]]] = {
    "blocked": ("aabb", [0, 1], [2, 3]),
    "interleaved": ("abab", [0, 2], [1, 3]),
}
# What every recorded arena verdict played, and the only seating `train.versus`
# can play. The arena default stays blocked so a default invocation reproduces
# the record bit for bit.
ARENA_GEOMETRY = "blocked"
VERSUS_GEOMETRY = "interleaved"


def arena_lineup(a: str, b: str, geometry: str) -> tuple[list[str], list[int], list[int]]:
    """(entrant specs in lineup order, side-A slots, side-B slots) for a seating.

    Slot lists index `Tournament.points`, which is in entrant order, so the
    paired split reads the right seats whichever order the lineup was built in.
    """
    order, mine, theirs = GEOMETRIES[geometry]
    return [a if slot == "a" else b for slot in order], list(mine), list(theirs)


def _is_bare_network(spec: str) -> bool:
    """A checkpoint played as a plain network -- the one entrant that batches.

    Either a raw path (what `side` and `train.versus` take) or the arena's
    `network:<path>` spelling of the same thing. Anything else -- a preset name
    or a `netsearch:`/`netgreedy:`/`mcts:` spec -- wraps the network in a search
    that cannot batch across lanes and gets one core in-process.
    """
    return Path(spec).exists() or spec.startswith(NETWORK)


def _default_workers(a: str, b: str) -> int:
    """1 for network-vs-network, which `--workers`' own help text prices at 400
    games in 134 s single-process; 26 for anything else, which the same help
    text prices at 200 games unfinished in 11 minutes at workers=1.
    """
    return 1 if _is_bare_network(a) and _is_bare_network(b) else 26


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="checkpoint path, or an arena entrant name")
    p.add_argument(
        "b", help="checkpoint path, or an arena entrant name, e.g. search2-notrade"
    )
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--lanes", type=int, default=512)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--device", default="cuda")
    # Board seed 0 matches the training runs; the duel seed is deliberately
    # *not* the in-loop ladder's `seed + 10_000` — those boards were already
    # used for monitoring and selection, so a verdict wants fresh ones. Keep
    # one value across every duel in a comparison: the boards then cancel
    # between the treatment and control differences too, not just within one.
    p.add_argument("--board-seed", type=int, default=0)
    p.add_argument("--duel-seed", type=int, default=20_000)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="above 1, run through `arena.compete`, which shards games across "
        "processes and is the path every recorded duel took. "
        "`train.versus` (workers=1) batches network inference across lanes in ONE "
        "process, which is ideal for network-vs-network — 400 games in 134 s — and "
        "catastrophic against a scripted bot, whose search cannot batch and gets "
        "one core: 200 games against search2 had not finished in 11 "
        "minutes, where the recorded arena run did 4000 in 607 s. Use workers for "
        "anything with a handcrafted bot on either side. Default: 1 when both "
        "entrants are bare network checkpoints, 26 otherwise -- see "
        "`_default_workers`",
    )
    p.add_argument(
        "--geometry",
        choices=sorted(GEOMETRIES),
        default=None,
        help="how the four seats hold the two sides on the arena path "
        f"(workers > 1). Default {ARENA_GEOMETRY!r}, the lineup `[a, a, b, b]` "
        "every recorded arena verdict played, so a default invocation "
        "reproduces the record exactly. 'interleaved' is `[a, b, a, b]`, each "
        "copy flanked by two opponents -- the seating `train.versus` plays and "
        "the only one it can play, so at workers=1 this may only name "
        f"{VERSUS_GEOMETRY!r}. Same boards and dice, the seating alone moves a "
        "pair by ~0.35 VP, and the verdict records which one it was",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch intra-op threads; 0 leaves the default. Set it whenever the "
        "container is capped with --cpus: torch sizes its pool from the host's "
        "core count, not the cgroup's, so a 6-CPU cap still spawns ~32 threads "
        "that then fight over 6 cores. Costs nothing when the box is idle and a "
        "great deal when it is not",
    )
    p.add_argument(
        "--json",
        default=None,
        help="append the result here instead of the default verdict path",
    )
    p.add_argument(
        "--verdicts",
        default="runs/eval",
        help="where a result lands when --json is not given. A duel that "
        "writes nowhere is the failure this default removes: a 400-game "
        "mcts-against-its-own-policy result was written up in prose and "
        "nowhere a tool could read it, so the ratings fit never saw it and "
        "placed that entrant half a VP wrong off a single unrelated duel",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="really write nothing, for a throwaway probe",
    )
    p.add_argument(
        "--records",
        default=None,
        help="append every game played as a v2 record (hexset.record.Record) "
        "here. Arena path only (--workers > 1): `train.versus` "
        "(--workers 1) plays through hexnet's own batched collector, which "
        "returns a verdict and no per-game history to record.",
    )
    args = p.parse_args(argv)

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    # Resolved rather than defaulted silently: 1 is right for network-vs-network
    # and catastrophic against anything that searches, so which one a run got
    # has to be visible in the run's own output, not inferred after the fact.
    if args.workers is None:
        args.workers = _default_workers(args.a, args.b)
        print(f"--workers not given; defaulting to {args.workers}", file=sys.stderr)
    else:
        print(f"--workers {args.workers}", file=sys.stderr)

    # The geometry is a factor of the same kind as `--workers`, and until it was
    # recorded the worker count chose it silently. It is printed for the same
    # reason the worker count is: which seating a verdict measured has to be
    # visible in the run's own output. The versus path cannot seat blocked --
    # `collect.alternating` is the interleaving -- so asking it to is an error,
    # not something to note and play anyway.
    if args.workers > 1:
        geometry = args.geometry or ARENA_GEOMETRY
        if args.geometry is None:
            print(f"--geometry not given; defaulting to {geometry}", file=sys.stderr)
        else:
            print(f"--geometry {geometry}", file=sys.stderr)
    else:
        if args.geometry not in (None, VERSUS_GEOMETRY):
            print(
                f"--geometry {args.geometry} is not available at --workers 1: "
                "`train.versus` seats the sides through `collect.alternating`, "
                f"which is always {VERSUS_GEOMETRY}. Use --workers 2 or more for "
                "the arena path, which can seat either way.",
                file=sys.stderr,
            )
            return 2
        geometry = VERSUS_GEOMETRY
        print(
            f"--geometry {geometry} (the only seating `train.versus` plays)",
            file=sys.stderr,
        )

    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem

    if args.records and args.workers <= 1:
        print(
            "--records needs the arena path (--workers > 1): `train.versus` "
            "(--workers 1) returns a verdict only, with no per-game history "
            "to record.",
            file=sys.stderr,
        )
        return 2

    if args.workers > 1:
        result = _via_arena(args, label_a, label_b, geometry)
    else:
        if _VERSUS_BACKEND is None:
            print(
                "the --workers 1 path (bare checkpoints, network-vs-network) "
                "needs the hexnet package: import hexnet.duel, or run "
                "`python -m hexnet.duel` instead of `python -m hexset.bench.duel`, "
                "or pass --workers 2 or more to use the arena path instead.",
                file=sys.stderr,
            )
            return 2
        result = _VERSUS_BACKEND(args, label_a, label_b)

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


def sides(lineup: list, label_a: str, label_b: str, mine=(0, 1)) -> list:
    """Rename a two-sided lineup so the two sides are distinguishable.

    Every `network:` spec is named "network" whatever checkpoint it carries, so
    a checkpoint-against-checkpoint duel arrives as four entrants of one name:
    `pooled` puts all four on one side and the paired-VP split has nobody to
    subtract from. Naming the sides after their labels is exact for any pair of
    entrants rather than only for the ones whose names happen to differ, and
    `spawn` reads `kind` and `weights`, never `name`.

    `mine` is which slots side A holds -- `[0, 1]` blocked, `[0, 2]` interleaved
    -- and every other slot is side B. Slot 0 is always side A, so `pooled`'s
    first group is side A under either seating.
    """
    side_a, side_b = label_a, label_b
    if side_a == side_b:
        side_a, side_b = f"{label_a}-a", f"{label_b}-b"
    seen = {side_a: 0, side_b: 0}
    renamed = []
    for slot, entrant in enumerate(lineup):
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
    lineup = sides(lineup_from_names(names), label_a, label_b, mine)

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
