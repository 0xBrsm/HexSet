# SPDX-License-Identifier: GPL-3.0-only
"""Runs a catanatron duel sharded across worker processes.

catanatron-play is single-process -- confirmed by reading it, there is no
multiprocessing anywhere in `cli/play.py` -- so a duel of any real size needs
this instead. Reports Wilson intervals rather than a raw win count, reusing
`hexset.arena.wilson` directly since dev-catan is already a dependency here
and the formula is right there.

Usage, matching catanatron's own `--players` syntax:

    python -m hexset.catanatron.duel --players=DC:search2-notrade,AB:2,AB:2,AB:2 \\
        --num=400 --workers=8

`main()` re-execs itself with `PYTHONHASHSEED=0` pinned before anything else
runs -- see `_ensure_pythonhashseed_zero` below for why a running process
cannot fix this in place.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from multiprocessing import Pool

from hexset.arena import wilson

from catanatron.cli.cli_players import parse_cli_string, register_cli_player
from catanatron.cli.play import play_batch
from catanatron.models.player import Color

from .player import DevCatanPlayer

register_cli_player("DC", DevCatanPlayer)

_MODULE = "hexset.catanatron.duel"


def _ensure_pythonhashseed_zero(argv=None, env=None, execve=os.execve) -> bool:
    """Pins `PYTHONHASHSEED=0` before any game is played.

    catanatron's own tie-breaks (`players/tree_search_utils.py`'s
    `max(robber_moves, key=impact)`, a max over a `set` of `Action`s) resolve
    via set iteration order, which is `PYTHONHASHSEED`-sensitive even for
    enum-only sets: string hashing at interpreter start-up shifts internal
    dict-resize patterns, which shifts where later `Enum` singletons land, so
    their id-based hash moves with the seed too (see
    `agents/reference/heximax.md`, "R-H1c take 2" -- localized to
    `prune_robber_actions` inside the installed catanatron package, not this
    adapter). Two runs of the same arms/seeds/game-count can therefore play
    different games unless the hash seed is fixed.

    `PYTHONHASHSEED` only takes effect at interpreter start-up, so a process
    already running with a different (or unset) value cannot fix this for
    itself -- it has to restart with the seed pinned before doing anything
    else. `multiprocessing.Pool`'s workers inherit this process's environment
    either way (fork copies it directly; spawn launches a fresh interpreter
    with the parent's environ), so pinning it once here, before the `Pool` is
    created, is enough for every shard.

    Returns `False` if the seed was already pinned (nothing to do). Otherwise
    re-execs via `execve` (real default: `os.execve`, which replaces this
    process and never returns; injectable so tests can observe the call
    instead of actually re-executing).
    """
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    if env.get("PYTHONHASHSEED") == "0":
        return False
    new_env = dict(env)
    new_env["PYTHONHASHSEED"] = "0"
    execve(sys.executable, [sys.executable, "-m", _MODULE, *argv[1:]], new_env)
    return True  # pragma: no cover — unreachable once the real os.execve runs


def shard_plan(num_games: int, workers: int) -> tuple[int, int]:
    """(shard_size, shard_count) exactly as `run_duel` computes them.

    Factored out so the report can state the shard count without restating the
    arithmetic, since the shard count is what selects the games.
    """
    shard_size = -(-num_games // workers)
    return shard_size, -(-num_games // shard_size)


def provenance() -> str:
    """Which catanatron this number came from, and which games were played.

    `pyproject.toml` pins catanatron to a git URL with no revision, so the
    version alone does not identify the code: two installs weeks apart can both
    call themselves 3.3.0 and differ. PEP 610 writes the resolved commit into
    the dist-info as `direct_url.json`, so read it back and stamp it on every
    report. A recorded eval without this cannot be reproduced or compared.

    `seed` and `workers` belong here for the same reason and it is the less
    obvious half: `run_duel` derives one seed per *shard* as `seed + i`, so the
    shard count -- and therefore the actual set of games played -- is a function
    of `--workers`. Two runs at the same `--seed` and different `--workers` play
    different games. A 500-game pair measured this way differed by 4 points on
    one checkpoint, which is why this line exists.

    `PYTHONHASHSEED` is the third thing that has to match for a *reproducibility*
    check (not the reading itself -- see `_ensure_pythonhashseed_zero`) to be
    meaningful, so its actual runtime value is stamped too, not merely assumed.
    """
    try:
        release = version("catanatron")
    except PackageNotFoundError:
        return "catanatron NOT INSTALLED"
    commit = None
    try:
        raw = distribution("catanatron").read_text("direct_url.json")
        if raw:
            commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except Exception:
        commit = None
    hashseed = os.environ.get("PYTHONHASHSEED", "unset")
    return (
        f"catanatron {release} @ {commit[:12] if commit else 'unknown-commit'} "
        f"| PYTHONHASHSEED={hashseed}"
    )


@dataclass(frozen=True)
class DuelResult:
    players_spec: str
    games: int
    seconds: float
    labels: dict[Color, str]
    seed: int
    workers: int
    wins: dict[Color, int]
    points: dict[Color, list[int]]

    def report(self) -> str:
        lines = [
            f"{self.games} games, {self.seconds:.1f}s "
            f"({self.games / self.seconds:.2f} games/sec)",
            f"  {provenance()} | seed {self.seed} | workers {self.workers} "
            f"| {shard_plan(self.games, self.workers)[1]} shards "
            f"of {shard_plan(self.games, self.workers)[0]}, seeds "
            f"{self.seed}-{self.seed + shard_plan(self.games, self.workers)[1] - 1}",
        ]
        for color, label in self.labels.items():
            w = self.wins.get(color, 0)
            pts = self.points.get(color, [])
            lo, hi = wilson(w, self.games)
            avg_vp = sum(pts) / len(pts) if pts else 0.0
            lines.append(
                f"  {label:<28} {w:>4}/{self.games} = {w / self.games:6.1%} "
                f"[{lo:5.1%}, {hi:5.1%}]   avg VP {avg_vp:.2f}"
            )
        return "\n".join(lines)


def _play_chunk(args: tuple[str, int, int]) -> tuple[dict, dict]:
    players_spec, num_games, seed = args
    random.seed(seed)
    players = parse_cli_string(players_spec)
    wins, results_by_player, _games = play_batch(num_games, players, quiet=True)
    return dict(wins), dict(results_by_player)


def run_duel(players_spec: str, num_games: int, workers: int, seed: int = 0) -> DuelResult:
    parts = players_spec.split(",")
    colors = list(Color)[: len(parts)]
    labels = {color: f"{i}:{part}" for i, (color, part) in enumerate(zip(colors, parts))}

    shard_size, _ = shard_plan(num_games, workers)
    chunks = []
    remaining = num_games
    i = 0
    while remaining > 0:
        n = min(shard_size, remaining)
        chunks.append((players_spec, n, seed + i))
        remaining -= n
        i += 1

    start = time.time()
    with Pool(len(chunks)) as pool:
        shard_results = pool.map(_play_chunk, chunks)
    elapsed = time.time() - start

    wins: dict[Color, int] = {c: 0 for c in colors}
    points: dict[Color, list[int]] = {c: [] for c in colors}
    for shard_wins, shard_points in shard_results:
        for color, count in shard_wins.items():
            wins[color] = wins.get(color, 0) + count
        for color, vps in shard_points.items():
            points.setdefault(color, []).extend(vps)

    return DuelResult(
        players_spec=players_spec,
        games=num_games,
        seconds=elapsed,
        labels=labels,
        seed=seed,
        workers=workers,
        wins=wins,
        points=points,
    )


def main() -> None:
    # Must run before anything else: fixes this process's own hash seed by
    # re-exec if needed, so every shard `Pool` forks or spawns below inherits
    # it. See `_ensure_pythonhashseed_zero`'s docstring for why a check alone,
    # this late, cannot substitute for the seed having been pinned at
    # start-up.
    _ensure_pythonhashseed_zero()
    assert os.environ.get("PYTHONHASHSEED") == "0", (
        "PYTHONHASHSEED is not pinned after _ensure_pythonhashseed_zero() -- "
        "the re-exec should have fixed this or replaced the process entirely."
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", required=True, help="catanatron --players syntax, e.g. DC:search2-notrade,AB:2,AB:2,AB:2")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_duel(args.players, args.num, args.workers, args.seed)
    print(result.report())


if __name__ == "__main__":
    main()