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
from catan.collect import frozen
from catan.train import versus


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="checkpoint under test")
    p.add_argument("b", help="reference checkpoint")
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
    p.add_argument("--json", default=None, help="append the result to this file")
    args = p.parse_args(argv)

    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem

    board = random_base_board(random.Random(args.board_seed))
    a = frozen(args.a, args.device, board, args.players)
    b = frozen(args.b, args.device, board, args.players)

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
    result = {
        "a": label_a,
        "b": label_b,
        "a_path": args.a,
        "b_path": args.b,
        "games": args.games,
        "duel_seed": args.duel_seed,
        "seconds": time.monotonic() - started,
        **result,
    }

    print(json.dumps(result, indent=1))
    print(
        f"\n{label_a} vs {label_b}: {result['win_rate']*100:.1f}% "
        f"[{result['wilson_low']*100:.1f}, {result['wilson_high']*100:.1f}] "
        f"over {args.games} games, paired VP {result['paired_vp']:+.2f}",
        file=sys.stderr,
    )
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
