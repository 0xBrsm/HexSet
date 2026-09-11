# SPDX-License-Identifier: GPL-3.0-only
"""Runs a catanatron duel across worker processes.

catanatron-play is single-process -- confirmed by reading it, there is no
multiprocessing anywhere in `cli/play.py` -- so a duel of any real size needs
this instead. Reports Wilson intervals rather than a raw win count, reusing
`hexset.arena.wilson` directly since dev-catan is already a dependency here
and the formula is right there.

Usage, matching catanatron's own `--players` syntax:

    python -m hexset.catanatron.duel --players=DC:heximax-notrade,AB:2,AB:2,AB:2 \\
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
from hexset.rules import GAME_TYPES

from catanatron.cli.play import GameConfigOptions, play_batch
from catanatron.models.player import Color
from catanatron.players import register_builtins
from catanatron.registry import REGISTRY, SpecError

from .player import DevCatanPlayer

register_builtins()  # idempotent; seats AB/R/F/... by key below
REGISTRY.register("DC", DevCatanPlayer, replace=True)

_MODULE = "hexset.catanatron.duel"


def game_config_for(game_type: str) -> GameConfigOptions:
    """The catanatron game config for a HexSet game type.

    Raises `ValueError` on an unknown name -- fail fast at the CLI, not
    after the first shard is already playing the wrong game.
    """
    try:
        rules = GAME_TYPES[game_type]
    except KeyError:
        raise ValueError(
            f"unknown game type: {game_type!r} (choose from {sorted(GAME_TYPES)})"
        )
    return GameConfigOptions(
        discard_limit=rules.discard_limit, vps_to_win=rules.winning_points
    )


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
    """(shard_size, shard_count) for the optional static scheduler.

    Factored out so the report can state the shard count without restating the
    arithmetic. Shards determine process layout; game indices select seeds.
    """
    shard_size = -(-num_games // workers)
    return shard_size, -(-num_games // shard_size)


def provenance() -> str:
    """Which catanatron this number came from, and which games were played.

    `pyproject.toml` pins catanatron to a git revision, but the
    version alone does not identify the code: two installs weeks apart can both
    call themselves 3.3.0 and differ. PEP 610 writes the resolved commit into
    the dist-info as `direct_url.json`, so read it back and stamp it on every
    report. A recorded eval without this cannot be reproduced or compared.

    `seed` and `workers` belong here for the same reason. `seed` still selects
    the games: game `g` (0-based over the duel) always starts from
    `random.seed(seed + g)`, so the set of games played no longer depends on
    `--workers` -- the same `--seed` plays the same games at any worker count.
    `workers` is stamped anyway: it still decides the shard plan, hence the
    wall-clock time and the exact process layout the games ran under.

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
    game_type: str
    wins: dict[Color, int]
    points: dict[Color, list[int]]
    speedups: str = "off"
    scheduling: str = "dynamic"
    worker_seconds: float = 0.0

    def report(self) -> str:
        rules = GAME_TYPES[self.game_type]
        task_plan = (f"{self.games} dynamically assigned games" if self.scheduling == "dynamic"
                     else f"{shard_plan(self.games, self.workers)[1]} static shards "
                          f"of {shard_plan(self.games, self.workers)[0]}")
        lines = [
            f"{self.games} games, {self.seconds:.1f}s "
            f"({self.games / self.seconds:.2f} games/sec)",
            f"  {provenance()} | seed {self.seed} | workers {self.workers} "
            f"| {task_plan}; "
            f"game g plays seed {self.seed}+g",
            f"  game type {self.game_type}: "
            f"{rules.winning_points} VP to win, "
            f"discard over {rules.discard_limit}",
            f"  Catanatron speedups: {self.speedups}",
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


def build_players(players_spec: str) -> list:
    """One catanatron `Player` per comma-separated `--players` entry.

    Everything but `DC` goes through the shared player registry
    (`REGISTRY.build`, the replacement for the removed `parse_cli_string`).
    `DC` seats are built directly instead: the registry splits a spec on
    every `:`, but an entrant spec may itself contain colons
    (`network:<path>`, `mcts:<path>@N`), so the whole tail after `DC:` is
    the entrant.
    """
    parts = [part.strip() for part in players_spec.split(",") if part.strip()]
    if not 2 <= len(parts) <= 4:
        raise SpecError(f"a game needs 2 to 4 players, got {len(parts)}")
    players = []
    for part, color in zip(parts, Color):
        if part.split(":")[0].upper() == "DC":
            tail = part.split(":", 1)[1] if ":" in part else ""
            players.append(
                DevCatanPlayer(
                    color, DevCatanPlayer.Params(entrant=tail or "heximax-notrade")
                )
            )
        else:
            players.append(REGISTRY.build(part, color))
    return players


def _play_chunk(args: tuple[str, int, int, int, str, str]) -> tuple[dict, dict]:
    from .speedups import catanatron_speedups

    *game_args, speedups = args
    with catanatron_speedups(speedups):
        return _play_chunk_native(tuple(game_args))


def _play_chunk_native(args: tuple[str, int, int, int, str]) -> tuple[dict, dict]:
    """Play games [start, start + count) with one rules variant.

    Game g starts from random.seed(seed + g), independent of worker count.
    Each game receives the selected config; progress is reported between games.
    """
    players_spec, start, count, seed, game_type = args
    players = build_players(players_spec)
    config = game_config_for(game_type)
    wins: dict[Color, int] = {}
    points: dict[Color, list[int]] = {}
    # Progress goes to stderr, never stdout: the parent's report is stdout's
    # only job (it is usually redirected to a file), while a silent shard is
    # indistinguishable from a dead one on a 45-minute run. Flush every line
    # -- worker stderr is block-buffered and a heartbeat that arrives late is
    # no heartbeat at all.
    t0 = time.time()
    last_beat = t0

    def heartbeat(done: int) -> None:
        elapsed = time.time() - t0
        print(
            f"[duel] games {start}-{start + count - 1}, base seed {seed}: "
            f"{done}/{count} done, "
            f"{elapsed:.0f}s elapsed",
            file=sys.stderr,
            flush=True,
        )

    # A single-game task is reported by the parent to avoid worker log spam.
    if count > 1:
        heartbeat(0)
    for g in range(start, start + count):
        random.seed(seed + g)
        game_wins, game_points, _games = play_batch(
            1, players, game_config=config, quiet=True
        )
        for color, n in game_wins.items():
            wins[color] = wins.get(color, 0) + n
        for color, vps in game_points.items():
            points.setdefault(color, []).extend(vps)
        done = g - start + 1
        now = time.time()
        # Check after each game: report every 25 games or after 60 seconds.
        # A single long game can delay a heartbeat beyond that interval.
        if count > 1 and (done % 25 == 0 or now - last_beat >= 60):
            heartbeat(done)
            last_beat = now
    if count > 1:
        heartbeat(count)
    return wins, points


def _play_job(args):
    started = time.perf_counter()
    wins, points = _play_chunk(args)
    return args[1], args[2], wins, points, time.perf_counter() - started


def run_duel(
    players_spec: str,
    num_games: int,
    workers: int,
    seed: int = 0,
    game_type: str = "standard",
    speedups: str = "off",
    scheduling: str = "dynamic",
) -> DuelResult:
    if num_games < 1 or workers < 1:
        raise ValueError("games and workers must be positive")
    if scheduling not in ("dynamic", "static"):
        raise ValueError(f"unknown scheduling: {scheduling!r}")
    parts = players_spec.split(",")
    colors = list(Color)[: len(parts)]
    labels = {color: f"{i}:{part}" for i, (color, part) in enumerate(zip(colors, parts))}
    # Fail fast on a bad game type, before any shard starts playing.
    game_config_for(game_type)
    from .speedups import MODES, verify_runtime
    if speedups not in MODES:
        raise ValueError(f"unknown Catanatron speedups: {speedups!r}")
    if speedups != "off":
        verify_runtime()

    shard_size = 1 if scheduling == "dynamic" else shard_plan(num_games, workers)[0]
    chunks = []
    start = 0
    while start < num_games:
        n = min(shard_size, num_games - start)
        # The chunk carries the duel seed and the game-index range; each game
        # re-derives its own seed inside `_play_chunk`, so the games played
        # do not move when `--workers` re-shards them.
        chunks.append((players_spec, start, n, seed, game_type, speedups))
        start += n

    started = time.perf_counter()
    last_beat = started
    completed = 0
    worker_seconds = 0.0
    indexed_results = {}
    print(f"[duel] 0/{num_games} done ({scheduling})", file=sys.stderr, flush=True)
    with Pool(min(workers, len(chunks))) as pool:
        for index, count, wins, points, duration in pool.imap_unordered(_play_job, chunks, chunksize=1):
            indexed_results[index] = (wins, points)
            previous = completed
            completed += count
            worker_seconds += duration
            now = time.perf_counter()
            if completed // 25 > previous // 25 or now - last_beat >= 60 or completed == num_games:
                print(f"[duel] {completed}/{num_games} done, {now - started:.0f}s elapsed",
                      file=sys.stderr, flush=True)
                last_beat = now
    elapsed = time.perf_counter() - started
    # Arrival order depends on runtime. Restore game-index order before
    # aggregation so points remain reproducible across schedules/workers.
    shard_results = [indexed_results[i] for i in sorted(indexed_results)]

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
        game_type=game_type,
        wins=wins,
        points=points,
        speedups=speedups,
        scheduling=scheduling,
        worker_seconds=worker_seconds,
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
    parser.add_argument("--players", required=True, help="catanatron --players syntax, e.g. DC:heximax-notrade,AB:2,AB:2,AB:2")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--game-type",
        default="standard",
        choices=sorted(GAME_TYPES),
        help="rules variant: standard (10 VP, discard over 7) or "
        "colonist-1v1 (15 VP, discard over 9)",
    )
    parser.add_argument(
        "--catanatron-speedups", choices=("off", "basic", "cached", "fast"), default="off",
        help="optional reference-engine acceleration; recorded in the report",
    )
    parser.add_argument("--scheduling", choices=("dynamic", "static"), default="dynamic",
                        help="assign games as workers finish, or retain fixed shards")
    args = parser.parse_args()

    result = run_duel(args.players, args.num, args.workers, args.seed, args.game_type,
                      args.catanatron_speedups, args.scheduling)
    print(result.report())


if __name__ == "__main__":
    main()