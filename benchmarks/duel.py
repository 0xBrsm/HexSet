"""Two checkpoints, head to head on identical boards, in paired terminal VP.

The in-loop ladder's 200-game rungs cannot resolve a slope: their scatter is the
binomial floor of the eval size, so a 150-iteration block's slope carries a 95%
half-width of ~9 points. This runs the instrument that *did* resolve ppo3's
selection — `train.versus` at 400 games — between any two checkpoints, so a
block's gain is measured as one high-resolution difference rather than fitted
through noise.

    python -m benchmarks.duel /w/runs/ppo5/latest.pt /w/runs/ppo4/latest.pt \
        --games 400 --label-a ppo5 --label-b ppo4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from catan.arena import NETWORK
from catan.board.board import random_base_board
from catan.collect import frozen, named_opponent
from catan.train import versus


def entrant_seed(base: int, spec: str, slot: int, other: str) -> int:
    """A stochastic entrant's stream, keyed to the entrant and not to its slot.

    `side` used to take `duel_seed + 1` for argument A and `+ 2` for B, which
    means swapping the arguments swapped which random stream each agent got.
    That is invisible for a bare checkpoint -- `collect.frozen` takes no seed --
    but every named entrant is spawned with `random.Random(seed)`, and `mcts:`
    samples its rollouts from it while the handcrafted bots break ties with it.
    So a swapped duel involving a named entrant was not measuring the same two
    agents twice.

    **This is not the cause of the order asymmetry measured on 2026-08-24.** All
    six audited pairs were passed as bare paths, so they resolved through
    `frozen` and never consumed a seed at all -- including the two largest
    asymmetries, 0.159 VP on base450/lr15h4 and 0.146 on facAB/facnone. Whatever
    drives those is still unidentified. This fix removes a real positional
    dependence that would have contaminated any future duel naming a searcher or
    a bot; it does not explain what was already seen, and the arena path
    (`workers > 1`) has the same defect in `arena._play_one`, which seeds by
    lineup index and is left alone here because changing it would break the exact
    reproducibility of 76,460 games of recorded arena results.

    Hashing the spec fixes the versus path: an entrant plays the same wherever it
    sits, and a swapped duel measures the swap rather than the reseeding.

    The one case a hash cannot separate is a self-duel, where both specs are
    identical and there is no property to distinguish them by. There the slot is
    the only tiebreak available, and it is used deliberately -- giving both
    sides one stream would have them search in lockstep, which is a worse
    artefact than the one being removed.
    """
    if spec == other:
        return base + 1 + slot
    digest = hashlib.blake2s(spec.encode(), digest_size=4).digest()
    return base + 1 + int.from_bytes(digest, "big") % 1_000_003


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
        default=None,
        help="above 1, run through `arena.compete`, which shards games across "
        "processes and is the path every recorded duel took. "
        "`train.versus` (workers=1) batches network inference across lanes in ONE "
        "process, which is ideal for network-vs-network — 400 games in 134 s — and "
        "catastrophic against a scripted bot, whose search cannot batch and gets "
        "one core: 200 games against search2-offers3 had not finished in 11 "
        "minutes, where the recorded arena run did 4000 in 607 s. Use workers for "
        "anything with a handcrafted bot on either side. Default: 1 when both "
        "entrants are bare network checkpoints, 26 otherwise -- see "
        "`_default_workers`",
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
        "writes nowhere is the failure this default removes: the 400-game "
        "mcts-against-its-own-policy result lived only as prose in status.md, "
        "so `rank_checkpoints.py` could not see it and placed that entrant "
        "half a VP wrong off a single unrelated duel",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="really write nothing, for a throwaway probe",
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

    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem

    if args.workers > 1:
        result = _via_arena(args, label_a, label_b)
    else:
        result = _via_versus(args, label_a, label_b)

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


def sides(lineup: list, label_a: str, label_b: str) -> list:
    """Rename a `[a, a, b, b]` lineup so the two sides are distinguishable.

    Every `network:` spec is named "network" whatever checkpoint it carries, so
    a checkpoint-against-checkpoint duel arrives as four entrants of one name:
    `pooled` puts all four on one side and the paired-VP split has nobody to
    subtract from. Naming the sides after their labels is exact for any pair of
    entrants rather than only for the ones whose names happen to differ, and
    `spawn` reads `kind` and `weights`, never `name`.
    """
    side_a, side_b = label_a, label_b
    if side_a == side_b:
        side_a, side_b = f"{label_a}-a", f"{label_b}-b"
    return [
        lineup[0].renamed(f"{side_a}#0"),
        lineup[1].renamed(f"{side_a}#1"),
        lineup[2].renamed(f"{side_b}#0"),
        lineup[3].renamed(f"{side_b}#1"),
    ]


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


def _via_arena(args, label_a: str, label_b: str) -> dict:
    """Two a side through `arena.compete`, sharded across `--workers`.

    Paired VP is recovered from the tournament's own per-game record rather than
    given up for the parallelism: `Tournament.points` carries every seat's
    terminal points per game in entrant order, so the within-game difference the
    single-process path reports can be rebuilt exactly.
    """
    from catan.arena import compete, lineup_from_names, pooled, wilson

    lineup = sides(lineup_from_names([args.a, args.a, args.b, args.b]), label_a, label_b)
    mine, theirs = [0, 1], [2, 3]

    started = time.monotonic()
    tournament = compete(
        lineup, args.games, seed=args.duel_seed, workers=args.workers
    )
    seconds = time.monotonic() - started

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
    a = side(
        args.a, args.device, board, args.players, args.lanes,
        entrant_seed(args.duel_seed, args.a, 0, args.b),
    )
    b = side(
        args.b, args.device, board, args.players, args.lanes,
        entrant_seed(args.duel_seed, args.b, 1, args.a),
    )

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
        "workers": args.workers,
        "seconds": time.monotonic() - started,
        "via": "train.versus",
        **result,
    }


if __name__ == "__main__":
    raise SystemExit(main())
