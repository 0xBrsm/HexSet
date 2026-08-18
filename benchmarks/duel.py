"""Two checkpoints, head to head on identical boards, in paired terminal VP.

The in-loop ladder's 200-game rungs cannot resolve a slope (see status.md, the
ppo4 amendment): their scatter is the binomial floor of the eval size, so a
150-iteration block's slope carries a 95% half-width of ~9 points. This runs
the instrument that *did* resolve ppo3's selection — `train.versus` at 400
games — between any two checkpoints, so a block's gain is measured as one
high-resolution difference rather than fitted through noise.

    python -m benchmarks.duel /w/runs/ppo5/latest.pt /w/runs/ppo4/latest.pt \
        --games 400 --label-a ppo5 --label-b ppo4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from catan.board.board import random_base_board
from catan.collect import frozen, named_opponent
from catan.train import versus


def side(spec: str, device: str, board, players: int, lanes: int, seed: int):
    """A checkpoint path or an arena entrant name, whichever `spec` names.

    Naming an entrant is how a checkpoint gets scored against `search2-offers3`,
    which is measured at parity with catanatron's `AB:2`, lives here, and plays
    the trading game catanatron does not model. A path that exists is a
    checkpoint; anything else is handed to the arena's entrant table, which
    raises on a name it does not know.
    """
    if Path(spec).exists():
        return frozen(spec, device, board, players)
    return named_opponent(spec, seed, lanes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="checkpoint path, or an arena entrant name")
    p.add_argument(
        "b", help="checkpoint path, or an arena entrant name, e.g. search2-offers3"
    )
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--lanes", type=int, default=512)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--max-offers", type=int, default=3)
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
        default=1,
        help="above 1, run through `arena.compete`, which shards games across "
        "processes and is the path every recorded duel in status.md used. "
        "`train.versus` (workers=1) batches network inference across lanes in ONE "
        "process, which is ideal for network-vs-network — 400 games in 134 s — and "
        "catastrophic against a scripted bot, whose search cannot batch and gets "
        "one core: 200 games against search2-offers3 had not finished in 11 "
        "minutes, where the recorded arena run did 4000 in 607 s. Use workers for "
        "anything with a handcrafted bot on either side",
    )
    p.add_argument("--json", default=None, help="append the result to this file")
    args = p.parse_args(argv)

    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem

    if args.workers > 1:
        result = _via_arena(args, label_a, label_b)
    else:
        result = _via_versus(args, label_a, label_b)

    print(json.dumps(result, indent=1))
    print(
        f"\n{label_a} vs {label_b}: {result['win_rate']*100:.1f}% "
        f"[{result['wilson_low']*100:.1f}, {result['wilson_high']*100:.1f}] "
        f"over {result['games']} games, paired VP {result['paired_vp']:+.2f}",
        file=sys.stderr,
    )
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    return 0


def _via_arena(args, label_a: str, label_b: str) -> dict:
    """Two a side through `arena.compete`, sharded across `--workers`.

    Paired VP is recovered from the tournament's own per-game record rather than
    given up for the parallelism: `Tournament.points` carries every seat's
    terminal points per game in entrant order, so the within-game difference the
    single-process path reports can be rebuilt exactly.
    """
    from catan.arena import base_name, compete, lineup_from_names, pooled, wilson

    lineup = lineup_from_names([args.a, args.a, args.b, args.b])
    mine = [i for i, e in enumerate(lineup) if base_name(e.name) == base_name(lineup[0].name)]
    theirs = [i for i in range(len(lineup)) if i not in mine]

    started = time.monotonic()
    tournament = compete(
        lineup, args.games, seed=args.duel_seed, workers=args.workers
    )
    seconds = time.monotonic() - started

    sides = pooled(tournament.standings, tournament.games)
    wins = sides[0].wins
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
    return {
        "a": label_a, "b": label_b, "a_path": args.a, "b_path": args.b,
        "games": tournament.games, "duel_seed": args.duel_seed,
        "workers": args.workers, "seconds": seconds, "via": "arena.compete",
        "unfinished": tournament.unfinished,
        "wins": wins, "win_rate": wins / tournament.games if tournament.games else 0.0,
        "wilson_low": low, "wilson_high": high,
        "paired_vp": mean,
        "paired_vp_low": mean - spread, "paired_vp_high": mean + spread,
    }


def _via_versus(args, label_a: str, label_b: str) -> dict:
    board = random_base_board(random.Random(args.board_seed))
    a = side(args.a, args.device, board, args.players, args.lanes, args.duel_seed + 1)
    b = side(args.b, args.device, board, args.players, args.lanes, args.duel_seed + 2)

    started = time.monotonic()
    result = versus(
        a,
        b,
        games=args.games,
        lanes=args.lanes,
        players=args.players,
        seed=args.duel_seed,
        max_offers=args.max_offers,
    )
    return {
        "a": label_a,
        "b": label_b,
        "a_path": args.a,
        "b_path": args.b,
        "games": args.games,
        "duel_seed": args.duel_seed,
        "seconds": time.monotonic() - started,
        "via": "train.versus",
        **result,
    }


if __name__ == "__main__":
    raise SystemExit(main())
