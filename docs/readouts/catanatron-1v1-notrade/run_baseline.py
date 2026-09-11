"""Capture the standard Catanatron duel, without modifying either policy."""
import json
import os
from pathlib import Path
import random
import time
from multiprocessing import Pool

from catanatron.cli.play import OutputOptions, get_actual_victory_points
from hexset.catanatron.duel import DuelResult, parse_cli_string, play_batch, provenance
from catanatron.models.player import Color

SPEC = "DC:heximax-notrade,AB:2"
SEED = 20260910
WORKERS = 30
GAMES_PER_WORKER = 10
OUT = Path("results")


def chunk(index):
    seed = SEED + index
    random.seed(seed)
    players = parse_cli_string(SPEC)
    directory = OUT / f"shard-{index:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    wins, points, games = play_batch(
        GAMES_PER_WORKER, players, quiet=True,
        output_options=OutputOptions(output=str(directory), output_format="json"),
    )
    result = {
        "shard": index, "seed": seed, "requested_games": GAMES_PER_WORKER,
        "seconds": time.monotonic() - started,
        "wins": {c.value: n for c, n in wins.items()},
        "points": {c.value: v for c, v in points.items()},
        "decisions": players[0].decisions, "fallbacks": players[0].fallbacks,
        "games": [{"id": g.id, "seed": g.seed,
                   "seating": [c.value for c in g.state.colors],
                   "winner": g.winning_color().value,
                   "turns": g.state.num_turns,
                   "actions": len(g.state.action_records),
                   "points": {c.value: get_actual_victory_points(g.state, c)
                              for c in g.state.colors}} for g in games],
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0"
    OUT.mkdir(exist_ok=False)
    plan = {"players": SPEC, "seed": SEED, "workers": WORKERS,
            "games_per_worker": GAMES_PER_WORKER, "games": 300,
            "provenance": provenance(), "player_trading": False}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan), flush=True)
    started = time.monotonic()
    shards = []
    with Pool(WORKERS) as pool:
        for result in pool.imap_unordered(chunk, range(WORKERS)):
            shards.append(result)
            print(f"Completed {len(shards)}/30 shards: {result['shard']} {result['wins']}", flush=True)
    shards.sort(key=lambda s: s["shard"])
    colors = list(Color)[:2]
    wins = {c: sum(s["wins"].get(c.value, 0) for s in shards) for c in colors}
    points = {c: [v for s in shards for v in s["points"].get(c.value, [])] for c in colors}
    duel = DuelResult(SPEC, 300, time.monotonic() - started,
                      {c: f"{i}:{p}" for i, (c, p) in enumerate(zip(colors, SPEC.split(',')))},
                      SEED, WORKERS, wins, points)
    summary = {**plan, "seconds": duel.seconds,
               "completed_games": sum(len(s["games"]) for s in shards),
               "wins": {c.value: n for c, n in wins.items()},
               "points": {c.value: v for c, v in points.items()},
               "fallbacks": sum(s["fallbacks"] for s in shards),
               "decisions": sum(s["decisions"] for s in shards),
               "shards": shards}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "report.txt").write_text(duel.report() + "\n")
    print(duel.report(), flush=True)
    print(f"Fallbacks: {summary['fallbacks']}/{summary['decisions']}", flush=True)
    assert summary["completed_games"] == 300, "Some games did not finish; inspect raw records"


if __name__ == "__main__":
    main()
