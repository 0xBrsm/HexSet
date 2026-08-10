"""Report what our bot does, in the shape the published human aggregates use.

The right-hand column is a third-party summary of Colonist.io games that cannot
be verified — no corpus was ever published with it. Treat divergence as a
question worth asking, not as an error to correct.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.throughput import environment
from catan.arena import Z_95, wilson
from catan.behaviour import (
    by_count,
    by_timing,
    per_game,
    seat_win_rates,
    walks,
)
from catan.record import read

# Quoted from the dataset repo's FINDINGS_SUMMARY.md, over ~41k Colonist games.
# Unverified: kept for comparison only.
HUMAN_KNIGHT_WIN_RATE = {0: 0.182, 1: 0.188, 2: 0.167, 3: 0.423, 4: 0.36}
HUMAN_SEAT_WIN_RATE = [0.2491, 0.2495, 0.2518, 0.2495]


def show(title, grouped, reference=None):
    print(f"  {title}")
    for key, (games, wins) in grouped.items():
        rate = wins / games if games else 0.0
        low, high = wilson(wins, games, Z_95)
        note = ""
        if reference is not None and key in reference:
            note = f"   human {reference[key]:.1%}"
        print(
            f"    {key:>3}  {games:>6} seats  {rate:6.1%}"
            f"  95% CI [{low:.1%}, {high:.1%}]{note}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    walked = walks(read(args.records))
    if not walked:
        print("no decided games in that file")
        return 1

    rates = {kind: per_game(walked, kind) for kind in (
        "knight", "monopoly", "road_building", "year_of_plenty",
        "bought", "settlement", "city", "road", "bank_trade",
    )}
    knights = by_count(walked, "knight")
    seats = seat_win_rates(walked)

    if args.json:
        print(json.dumps({
            "environment": environment(),
            "games": len(walked),
            "per_player_game": {k: round(v, 3) for k, v in rates.items()},
            "knight_win_rate": {str(k): v for k, v in knights.items()},
            "monopoly_win_rate": {str(k): v for k, v in by_count(walked, "monopoly", 2).items()},
            "road_building_timing": {str(k): v for k, v in by_timing(walked, "road_building").items()},
            "seat_win_rate": {str(k): v for k, v in seats.items()},
        }, indent=2))
        return 0

    env = environment()
    print(f"commit {env['commit']}  {env['machine']}  {len(walked)} games")
    print("  per player-game:")
    for kind, value in rates.items():
        print(f"    {kind:<16} {value:6.2f}")
    print()
    show("win rate by knights played", knights, HUMAN_KNIGHT_WIN_RATE)
    print()
    show("win rate by monopolies played", by_count(walked, "monopoly", 2))
    print()
    show("win rate by fifth of game the first road building fell in",
         by_timing(walked, "road_building"))
    print()
    print("  win rate by seat (human: 24.9 / 25.0 / 25.2 / 25.0)")
    for seat, (games, wins) in seats.items():
        low, high = wilson(wins, games, Z_95)
        print(
            f"    seat {seat}  {games:>6} games  {wins / games:6.1%}"
            f"  95% CI [{low:.1%}, {high:.1%}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
