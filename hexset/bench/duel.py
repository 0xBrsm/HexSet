# SPDX-License-Identifier: GPL-3.0-only
"""Compare two arena entrants with paired boards and confidence intervals.

``python -m hexset.bench.duel mybot random --runtime mybots --games 400``; for
batched model evaluation use ``hexset.bench.versus.compete_batched``. A bot
this distribution does not ship -- a `network:`/`mcts:` checkpoint, or any
bot package -- needs the runtime that registers it, named with ``--runtime``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from hexset.arena import RUNTIME_HELP
from hexset.board.board import BOARD_MODES
from hexset.game import MAX_TURNS, UNSTRUCTURED_TURN_CAP
from hexset.rules import GAME_TYPES, game_type

from hexset.bench.throughput import default_workers
from hexset.bench.metrics import json_metrics, paired_mean
from hexset.experiment import provenance, result_document

__all__ = [
    "ARENA_GEOMETRY",
    "arena_lineup",
    "side_swap",
    "add_table_options",
    "main",
    "sides",
]


ARENA_GEOMETRY = "aabb"


def arena_lineup(a: str, b: str, geometry: str) -> tuple[list[str], list[int], list[int]]:
    """(entrant specs in lineup order, side-A slots, side-B slots) for a seating.

    `geometry` is an `a`/`b` letter per seat (`aabb`, `abab`), or a comma-
    separated lineup whose entries are `a`, `b`, or any entrant spec -- which
    seats bots on neither side. Slots index `Tournament.points`.
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


def side_swap(mine, theirs, seats: int) -> tuple[int, ...] | None:
    """The antithetic complement for a seating (`arena.compete(complement=)`):
    the `i`-th `a` slot and the `i`-th `b` slot exchange seats and every slot
    on neither side keeps its own, so the second game of a board pair is the
    first with the two sides traded. For `aabb` and `ab` this is the
    arena's own `half_turn`.

    `None` where the sides differ in size (`aab`, `abbb`): no exchange of
    seats swaps them, and the pair is the board under `half_turn` instead --
    still a shared board, so the interval stays board-level, but the seat
    term cancels only across the rotation."""
    if len(mine) != len(theirs):
        return None
    swap = list(range(seats))
    for x, y in zip(mine, theirs):
        swap[x], swap[y] = y, x
    return tuple(swap)


def add_table_options(parser: argparse.ArgumentParser) -> None:
    """`--game-type`, `--board`, `--turn-cap` and `--trade-mode`: what every
    game of a run is dealt and played under, shared by every command that
    runs one (`args.game_type` names a `hexset.rules.GAME_TYPES` entry,
    `args.board` a `hexset.board.board.BOARD_MODES` one)."""
    parser.add_argument(
        "--game-type", default="standard", choices=sorted(GAME_TYPES),
        help="the contract every game is dealt under: its ruleset and the seat "
             "counts it is played at. Checked against the entrants that actually "
             "play, so a lineup `a b retired retired` under duel-variant is a "
             "two-seat game at a four-seat table")
    parser.add_argument(
        "--board", default="random", choices=sorted(BOARD_MODES),
        help="how every board is dealt: number discs in random order with 6 and "
             "8 apart (default), or in the rulebook's lettered spiral")
    parser.add_argument(
        "--turn-cap", type=int, default=MAX_TURNS,
        help="turns after which a game is abandoned and the run aborts. The "
             "default suits agents that are trying to win; raise it (say to "
             f"{UNSTRUCTURED_TURN_CAP}) to measure unstructured play such as random bots")
    parser.add_argument(
        "--trade-mode", choices=("round", "auto"), default="round",
        help="the bargaining mechanism: propose-and-respond rounds (default), or "
             "the automatic clearing house")


def main(argv: list[str] | None = None) -> int:
    """`python -m hexset.bench.duel`: compare two arena entrants with paired
    boards and confidence intervals."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="arena entrant name or registered checkpoint spec")
    p.add_argument("b", help="arena entrant name or registered checkpoint spec")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--duel-seed", type=int, default=20000)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--geometry", default=ARENA_GEOMETRY,
                   help="a/b seat pattern or comma-separated lineup, e.g. a,b,random,random. "
                        "With as many a slots as b slots, the second game of each board "
                        "pair swaps the two sides' seats")
    p.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    add_table_options(p)
    p.add_argument("--json", default=None, help="append verdict to this JSON Lines file")
    p.add_argument("--verdicts", default="runs/eval", help="default verdict directory")
    p.add_argument("--no-json", action="store_true", help="print without writing a verdict file")
    p.add_argument("--records", default=None, help="append replayable game records to this file")
    p.add_argument("--journal", default=None,
                   help="where each game is kept as it finishes (default: beside the verdict, "
                        "named for the pairing, seed and games; with --no-json, none)")
    p.add_argument("--resume", action="store_true",
                   help="continue the run the journal holds, playing only the games it lacks")
    p.add_argument("--progress-seconds", type=float, default=60.0,
                   help="print a running tally to stderr at most this often (0: after "
                        "every game)")
    p.add_argument("--no-progress", action="store_true", help="no running tally")
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


def _progress(label_a: str, label_b: str, mine, theirs, started: float, *,
              every: float = 60.0, workers: int = 1, out=None):
    """A `compete(progress=)` callback: one running-tally line on stderr at
    most every `every` seconds (every game at 0), so a long duel is never
    silent. Side-A win share and mean paired VP are over the games in so far.
    The time left is a straight extrapolation, and is withheld until `workers`
    games are in: until the pool has drained its first wave, elapsed time
    covers `workers` games in flight rather than the `done` that finished, so
    extrapolating from it overstates the remainder by about that factor."""
    stream = sys.stderr if out is None else out
    wins = 0
    margin = 0.0
    last = [float("-inf")]

    def report(done: int, games: int, outcome) -> None:
        nonlocal wins, margin
        wins += outcome.winner in mine
        margin += (sum(outcome.points[i] for i in mine) / len(mine)
                   - sum(outcome.points[i] for i in theirs) / len(theirs))
        now = time.monotonic()
        if now - last[0] < every and done < games:
            return
        last[0] = now
        elapsed = now - started
        left = (f"~{elapsed / done * (games - done):.0f}s left"
                if done >= workers else "time left unknown until the pool fills")
        print(
            f"{label_a} vs {label_b}: {done}/{games} games, "
            f"{label_a} {100 * wins / done:.1f}%, paired VP {margin / done:+.2f}, "
            f"{elapsed:.0f}s elapsed, {left}",
            file=stream, flush=True,
        )

    return report


def sides(lineup: list, label_a: str, label_b: str, mine=(0, 1), theirs=(2, 3)) -> list:
    """Rename a two-sided lineup so the two sides are distinguishable.

    Every `network:` spec is named "network", so unrenamed a checkpoint-vs-
    checkpoint duel pools into one side. Slots outside `mine`/`theirs` keep
    their own name and are on neither side.
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


def _journal_path(args, label_a: str, label_b: str) -> Path | None:
    """`--journal`, else one beside the verdict named for the pairing, the
    seed and the games -- the same run finds the same journal to resume."""
    if getattr(args, "journal", None):
        return Path(args.journal)
    if getattr(args, "no_json", False):
        return None
    if getattr(args, "json", None):
        verdict = Path(args.json)
    elif getattr(args, "verdicts", None):
        verdict = _verdict_path(args, label_a, label_b)
    else:
        return None  # a caller building its own namespace names its own journal
    return verdict.with_name(f"{verdict.stem}.s{args.duel_seed}.g{args.games}.games.jsonl")


def _verdict_path(args, label_a: str, label_b: str) -> Path:
    """Default verdict file, named after the pairing so re-runs append together."""

    def token(label: str, spec: str) -> str:
        if label:
            return label.replace("/", "-")
        parts = Path(spec).parts
        return "-".join(parts[-2:]).replace(".pt", "") if len(parts) > 1 else spec

    pair = f"{token(label_a, args.a)}__vs__{token(label_b, args.b)}"
    return Path(args.verdicts) / f"{pair}.json"


def _via_arena(args, label_a: str, label_b: str, geometry: str = ARENA_GEOMETRY) -> dict:
    """Two a side through `arena.compete`, sharded across `--workers`."""
    from hexset.arena import compete, lineup_from_names, load_runtime, wilson

    runtimes = tuple(args.runtime or ())
    load_runtime(*runtimes)

    names, mine, theirs = arena_lineup(args.a, args.b, geometry)
    if args.games % 2:
        raise ValueError("paired evaluation requires an even number of games")
    if args.games % len(names):
        raise ValueError(
            f"{args.games} games does not divide evenly over the {len(names)} "
            f"seats of {geometry!r}: the arena rotates the lineup through every "
            "seat and an incomplete rotation leaves the seat bias in the verdict"
        )
    lineup = sides(lineup_from_names(names), label_a, label_b, mine, theirs)

    run_provenance = provenance(lineup)
    journal = _journal_path(args, label_a, label_b)
    if journal is not None:
        print(f"keeping each game in {journal}", file=sys.stderr)
    started = time.monotonic()
    tournament = compete(
        lineup,
        args.games,
        seed=args.duel_seed,
        workers=args.workers,
        records=bool(args.records),
        worker_initializer=load_runtime if runtimes else None,
        worker_initargs=runtimes,
        trade_mode=args.trade_mode,
        game_type=game_type(getattr(args, "game_type", "standard")),
        turn_cap=getattr(args, "turn_cap", MAX_TURNS),
        board_mode=getattr(args, "board", "random"),
        complement=side_swap(mine, theirs, len(names)),
        journal=journal,
        resume=getattr(args, "resume", False),
        progress=None if getattr(args, "no_progress", False) else _progress(
            label_a, label_b, mine, theirs, started,
            every=getattr(args, "progress_seconds", 60.0), workers=args.workers,
        ),
    )
    seconds = time.monotonic() - started

    if args.records:
        # Each record is already in the journal from the moment its game
        # finished; this is the flat file replays read.
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
    # `unfinished` is the games the action cap cut short: one that reaches
    # the turn cap stops the run (`hexset.arena.Exhausted`) instead.
    return json_metrics({
        "experiment": result_document(tournament, lineup, seed=args.duel_seed,
                                      workers=args.workers, run_provenance=run_provenance),
        "a": label_a, "b": label_b, "a_path": args.a, "b_path": args.b,
        "games": tournament.games, "duel_seed": args.duel_seed,
        "workers": args.workers, "seconds": seconds, "via": "arena.compete",
        "geometry": geometry,
        "game_type": getattr(args, "game_type", "standard"),
        "board_mode": getattr(args, "board", "random"),
        "turn_cap": getattr(args, "turn_cap", MAX_TURNS),
        "trade_mode": args.trade_mode,
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
        "turns_mean": statistics.mean(turns) if turns else 0.0,
        "turns_median": statistics.median(turns) if turns else 0.0,
        "turns_max": max(turns) if turns else 0,
    })


if __name__ == "__main__":
    raise SystemExit(main())
