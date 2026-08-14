"""Summarise a corpus of recorded human games into one row per game.

The logs are state-delta streams, not action streams: an event carries the
seconds the player spent (``input.deltaS``) and the fields of the game state
that changed, never the move itself.  Everything here is therefore read off the
deltas rather than replayed, which keeps a single pass over the archive enough.

Emits JSONL so the aggregation in ``report`` can be re-run without re-reading
the archive, which takes tens of minutes on a phone.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from typing import Any, Iterator

# victoryPointsState buckets, inferred from totals reconciling to the win
# threshold across sampled games.
VP_WEIGHTS = {"0": 1, "1": 2, "2": 1, "3": 2, "4": 2}


def vp_total(state: dict[str, int]) -> int:
    return sum(VP_WEIGHTS.get(key, 1) * count for key, count in state.items())


def summarise(raw: dict[str, Any]) -> dict[str, Any] | None:
    root = raw.get("data", raw)
    history = root.get("eventHistory")
    if not history:
        return None
    end = history.get("endGameState") or {}
    players = end.get("players") or {}
    if not players:
        return None

    settings = root.get("gameSettings") or {}
    users = root.get("playerUserStates") or {}
    if isinstance(users, dict):
        users = list(users.values())
    seats = sorted(players)

    winner = next((pid for pid, p in players.items() if p.get("winningPlayer")), None)

    vp: dict[str, dict[str, int]] = {pid: {} for pid in seats}
    initial = history.get("initialState") or {}
    for pid, state in (initial.get("playerStates") or {}).items():
        vp.setdefault(pid, {}).update(state.get("victoryPointsState") or {})

    turn = 0
    think: list[float] = []
    think_by_turn: list[tuple[int, float]] = []
    trajectory: list[list[int]] = []
    last_recorded = -1

    for event in history.get("events") or ():
        change = event.get("stateChange") or {}
        current = change.get("currentState") or {}
        if "completedTurns" in current:
            turn = current["completedTurns"]

        delta = (event.get("input") or {}).get("deltaS")
        if isinstance(delta, (int, float)) and delta >= 0:
            think.append(float(delta))
            think_by_turn.append((turn, float(delta)))

        touched = False
        for pid, state in (change.get("playerStates") or {}).items():
            points = state.get("victoryPointsState")
            if points:
                vp.setdefault(pid, {}).update(points)
                touched = True
        if touched and turn != last_recorded:
            trajectory.append([turn] + [vp_total(vp.get(pid, {})) for pid in seats])
            last_recorded = turn
        elif touched and trajectory:
            trajectory[-1] = [turn] + [vp_total(vp.get(pid, {})) for pid in seats]

    activity = end.get("activityStats") or {}
    resources = end.get("resourceStats") or {}

    return {
        "seats": seats,
        "winner": winner,
        "bots": sum(1 for u in users if u.get("isBot")),
        "humans": sum(1 for u in users if not u.get("isBot")),
        "game_type": settings.get("gameType"),
        "scenario": settings.get("scenarioSetting"),
        "map": settings.get("mapSetting"),
        "vp_to_win": settings.get("victoryPointsToWin"),
        "elo_type": settings.get("eloType"),
        "turns": end.get("totalTurnCount"),
        "duration_ms": end.get("gameDurationInMS"),
        "events": len(history.get("events") or ()),
        "dice": end.get("diceStats"),
        "final_vp": {pid: vp_total(players[pid].get("victoryPoints") or {}) for pid in seats},
        "observed_vp": {pid: vp_total(vp.get(pid, {})) for pid in seats},
        "rank": {pid: players[pid].get("rank") for pid in seats},
        "trajectory": trajectory,
        "think": {
            "n": len(think),
            "sum": round(sum(think), 1),
            "mean": round(statistics.fmean(think), 3) if think else None,
            "median": round(statistics.median(think), 3) if think else None,
            "p90": round(sorted(think)[int(0.90 * len(think))], 3) if think else None,
            "p99": round(sorted(think)[int(0.99 * len(think))], 3) if think else None,
            "max": round(max(think), 1) if think else None,
            "over_10s": sum(1 for d in think if d >= 10.0),
            "under_1s": sum(1 for d in think if d < 1.0),
            "opening": round(sum(d for t, d in think_by_turn if t <= 2), 1),
        },
        "activity": {
            pid: {
                "proposed_trades": a.get("proposedTrades"),
                "successful_trades": a.get("successfulTrades"),
                "dev_bought": a.get("devCardsBought"),
                "dev_used": a.get("devCardsUsed"),
                "blocked": a.get("resourceIncomeBlocked"),
            }
            for pid, a in activity.items()
        },
        "resources": {
            pid: {
                "rolling": r.get("rollingIncome"),
                "trade_in": r.get("tradeIncome"),
                "trade_out": r.get("tradeLoss"),
                "robbed": r.get("robbingLoss"),
                "discarded": r.get("rollingLoss"),
                "total_in": r.get("totalResourceIncome"),
            }
            for pid, r in resources.items()
        },
    }


def extract(archive: str, limit: int | None) -> Iterator[dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as tar:
        seen = 0
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                row = summarise(json.loads(handle.read()))
            except (ValueError, KeyError, TypeError) as exc:
                sys.stderr.write(f"skip {member.name}: {exc}\n")
                continue
            if row is None:
                continue
            row["id"] = member.name.rsplit("/", 1)[-1].removesuffix(".json")
            yield row
            seen += 1
            if limit and seen >= limit:
                return


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "sd": round(statistics.pstdev(ordered), 3) if len(ordered) > 1 else 0.0,
        "p10": round(pick(0.10), 3),
        "median": round(pick(0.50), 3),
        "p90": round(pick(0.90), 3),
    }


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-game rows.

    The headline question is the one the value head keeps failing at: how early
    is the eventual winner identifiable from the public score alone.
    """
    settings = Counter(
        (r["game_type"], r["scenario"], r["map"], r["vp_to_win"], len(r["seats"]))
        for r in rows
    )
    composition = Counter((r["humans"], r["bots"]) for r in rows)

    # Restrict the curve to the dominant configuration so turn indices compare.
    main = [
        r
        for r in rows
        if len(r["seats"]) == 4 and r["vp_to_win"] == 10 and r["scenario"] == 0 and r["map"] == 0
    ]

    leader_hits: dict[int, list[int]] = defaultdict(list)
    tie_share: dict[int, list[int]] = defaultdict(list)
    vp_at_turn: dict[int, list[int]] = defaultdict(list)
    # Keyed by (turn bucket, lead over the best rival): did the leader win.
    margin_hits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in main:
        winner = row["winner"]
        if winner is None or not row["trajectory"]:
            continue
        seats = row["seats"]
        index = seats.index(winner) if winner in seats else None
        if index is None:
            continue
        cursor = 0
        totals = [0] * len(seats)
        turns = row["turns"] or 0
        for turn in range(0, min(turns, 120) + 1):
            while cursor < len(row["trajectory"]) and row["trajectory"][cursor][0] <= turn:
                totals = row["trajectory"][cursor][1:]
                cursor += 1
            if not any(totals):
                continue
            best = max(totals)
            leaders = [i for i, v in enumerate(totals) if v == best]
            leader_hits[turn].append(1 if index in leaders and len(leaders) == 1 else 0)
            tie_share[turn].append(1 if len(leaders) > 1 else 0)
            vp_at_turn[turn].append(totals[index])
            if len(leaders) == 1:
                rival = max(v for i, v in enumerate(totals) if i != leaders[0])
                margin = min(4, best - rival)
                margin_hits[(10 * (turn // 10), margin)].append(1 if index == leaders[0] else 0)

    curve = []
    for turn in sorted(leader_hits):
        hits = leader_hits[turn]
        if len(hits) < 200:
            continue
        curve.append(
            {
                "turn": turn,
                "games": len(hits),
                "leader_is_winner": round(statistics.fmean(hits), 4),
                "tied": round(statistics.fmean(tie_share[turn]), 4),
                "sole_leader_is_winner": round(
                    statistics.fmean(hits) / (1.0 - statistics.fmean(tie_share[turn])), 4
                )
                if statistics.fmean(tie_share[turn]) < 1.0
                else None,
                "winner_vp": round(statistics.fmean(vp_at_turn[turn]), 2),
            }
        )

    margin_curve = [
        {
            "turns": bucket,
            "lead": margin,
            "n": len(hits),
            "leader_wins": round(statistics.fmean(hits), 4),
        }
        for (bucket, margin), hits in sorted(margin_hits.items())
        if len(hits) >= 300
    ]

    think_all = [r["think"] for r in rows if r["think"]["n"]]
    dice = [0] * 11
    for row in rows:
        if row["dice"] and len(row["dice"]) == 11:
            for i, count in enumerate(row["dice"]):
                dice[i] += count
    dice_total = sum(dice) or 1

    trades = [
        a["proposed_trades"]
        for r in rows
        for a in r["activity"].values()
        if a["proposed_trades"] is not None
    ]
    accepted = [
        a["successful_trades"]
        for r in rows
        for a in r["activity"].values()
        if a["successful_trades"] is not None
    ]
    hidden = [
        r["final_vp"][pid] - r["observed_vp"].get(pid, 0)
        for r in rows
        for pid in r["seats"]
        if pid in r["final_vp"]
    ]

    return {
        "games": len(rows),
        "settings": [{"key": list(k), "n": v} for k, v in settings.most_common(12)],
        "composition": [{"humans": k[0], "bots": k[1], "n": v} for k, v in composition.most_common(8)],
        "main_config_games": len(main),
        "turns": quantiles([r["turns"] for r in rows if r["turns"]]),
        "minutes": quantiles([r["duration_ms"] / 60000 for r in rows if r["duration_ms"]]),
        "events": quantiles([float(r["events"]) for r in rows]),
        "decision_seconds": {
            "per_game_mean": quantiles([t["mean"] for t in think_all if t["mean"] is not None]),
            "per_game_median": quantiles([t["median"] for t in think_all if t["median"] is not None]),
            "per_game_p90": quantiles([t["p90"] for t in think_all if t["p90"] is not None]),
            "decisions_per_game": quantiles([float(t["n"]) for t in think_all]),
            "share_over_10s": round(
                sum(t["over_10s"] for t in think_all) / max(1, sum(t["n"] for t in think_all)), 4
            ),
            "share_under_1s": round(
                sum(t["under_1s"] for t in think_all) / max(1, sum(t["n"] for t in think_all)), 4
            ),
            "opening_share_of_total": round(
                sum(t["opening"] for t in think_all) / max(1.0, sum(t["sum"] for t in think_all)), 4
            ),
        },
        "dice": {
            str(face): round(count / dice_total, 4) for face, count in zip(range(2, 13), dice)
        },
        "dice_expected": {
            str(face): round((6 - abs(7 - face)) / 36, 4) for face in range(2, 13)
        },
        "trades": {
            "proposed_per_seat": quantiles([float(t) for t in trades]),
            "accepted_per_seat": quantiles([float(t) for t in accepted]),
            "acceptance_rate": round(sum(accepted) / max(1, sum(trades)), 4),
        },
        "hidden_vp_at_end": quantiles([float(h) for h in hidden]),
        "leader_curve": curve,
        "margin_curve": margin_curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="path to the .tar.gz of game logs")
    parser.add_argument("--rows", help="JSONL of per-game rows to write or read")
    parser.add_argument("--report", help="path to write the aggregate JSON")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.archive:
        if not args.rows:
            parser.error("--archive needs --rows to write to")
        count = 0
        with open(args.rows, "w", encoding="utf-8") as out:
            for row in extract(args.archive, args.limit or None):
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                if args.progress and count % args.progress == 0:
                    sys.stderr.write(f"{count}\n")
                    sys.stderr.flush()
        sys.stderr.write(f"extracted {count}\n")

    if args.report:
        if not args.rows:
            parser.error("--report needs --rows")
        with open(args.rows, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        summary = report(rows)
        with open(args.report, "w", encoding="utf-8") as out:
            json.dump(summary, out, indent=2)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
