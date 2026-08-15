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
import math
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
    # Per turn: decisions, seconds spent, decisions touching a trade.
    profile: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    # Corners claimed by a settlement, in order; the first two per seat are setup.
    claims: list[list[int]] = []

    for event in history.get("events") or ():
        change = event.get("stateChange") or {}
        current = change.get("currentState") or {}
        if "completedTurns" in current:
            turn = current["completedTurns"]

        corners = (change.get("mapState") or {}).get("tileCornerStates") or {}
        for corner, built in corners.items():
            if built.get("buildingType") == 1 and "owner" in built:
                claims.append([built["owner"], int(corner)])

        slot = profile[min(turn, 120)]
        slot[0] += 1
        if "tradeState" in change:
            slot[2] += 1

        delta = (event.get("input") or {}).get("deltaS")
        if isinstance(delta, (int, float)) and delta >= 0:
            think.append(float(delta))
            think_by_turn.append((turn, float(delta)))
            slot[1] += float(delta)

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
    board = initial.get("mapState") or {}
    hexes = board.get("tileHexStates") or {}
    corners = board.get("tileCornerStates") or {}

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
        "play_order": root.get("playOrder"),
        "setup": claims[: 2 * len(seats)],
        "hexes": [
            [h.get("x"), h.get("y"), h.get("type"), h.get("diceNumber")]
            for _, h in sorted(hexes.items(), key=lambda kv: int(kv[0]))
        ],
        "corners": [
            [c.get("x"), c.get("y"), c.get("z")]
            for _, c in sorted(corners.items(), key=lambda kv: int(kv[0]))
        ],
        "trajectory": trajectory,
        "profile": [
            [turn, int(slot[0]), round(slot[1], 1), int(slot[2])]
            for turn, slot in sorted(profile.items())
        ],
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


# The six (dx, dy, z) offsets from a hex to its corners, fitted exhaustively
# against one board and verified: they are the only six that land every hex on a
# recorded corner, and they cover all 54 with 114 incidences (24 corners on three
# hexes, 12 on two, 18 on one).
CORNER_OFFSETS = ((-1, 1, 0), (0, -1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, -1, 1))
# The two resources with three hexes rather than four, so brick and ore.
SCARCE = frozenset({2, 5})


def pips(number: int | None) -> int:
    return 0 if not number or number == 7 else 6 - abs(7 - number)


def opening(row: dict[str, Any]) -> list[dict[str, int]] | None:
    """Pip count and resource mix of each seat's two setup settlements."""
    order, winner = row.get("play_order"), row.get("winner")
    hexes, corners, setup = row.get("hexes"), row.get("corners"), row.get("setup")
    if not order or winner is None or len(order) != 4:
        return None
    if not hexes or not corners or len(hexes) != 19 or len(corners) != 54 or len(setup or ()) != 8:
        return None

    index = {(c[0], c[1], c[2]): i for i, c in enumerate(corners)}
    touching: dict[int, list[int]] = defaultdict(list)
    for position, (hx, hy, _, _) in enumerate(hexes):
        for dx, dy, z in CORNER_OFFSETS:
            corner = index.get((hx + dx, hy + dy, z))
            if corner is not None:
                touching[corner].append(position)
    if len(touching) != 54:
        return None

    claimed: dict[int, list[int]] = defaultdict(list)
    for owner, corner in setup:
        claimed[owner].append(corner)
    if len(claimed) != 4 or any(len(v) != 2 for v in claimed.values()):
        return None

    seats = []
    for owner, cs in claimed.items():
        # Duplicates are kept: two setup settlements can share a hex, and that
        # hex then pays twice on its number.
        adjacent = [h for c in cs for h in touching[c]]
        kinds = {hexes[h][2] for h in adjacent if hexes[h][2] != 0}

        # Cards per roll is k[n] when n comes up, so the mean is fixed by pips
        # and the spare dimension is how concentrated k is across numbers.
        cards: dict[int, int] = defaultdict(int)
        for h in adjacent:
            number = hexes[h][3]
            if number and number != 7:
                cards[number] += 1
        mean = sum(k * pips(n) / 36 for n, k in cards.items())
        second = sum(k * k * pips(n) / 36 for n, k in cards.items())
        live = sum(pips(n) / 36 for n in cards)

        seats.append(
            {
                "pips": sum(pips(hexes[h][3]) for h in adjacent),
                "resources": len(kinds),
                "scarce": len(kinds & SCARCE),
                "numbers": len(cards),
                "dead": round(1.0 - live, 4),
                "variance": round(second - mean * mean, 4),
                "pick": order.index(owner) if owner in order else -1,
                "win": 1 if int(winner) == owner else 0,
            }
        )
    return seats


def rate(seats: list[dict[str, int]]) -> dict[str, Any]:
    n = len(seats)
    p = sum(s["win"] for s in seats) / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return {"n": n, "win": round(p, 4), "lo": round(p - half, 4), "hi": round(p + half, 4)}


def placement(path: str) -> dict[str, Any]:
    """What separates a good opening from a bad one, over every recorded seat.

    Observational, and the confound is the obvious one: stronger players choose
    better corners *and* play better afterwards, so every figure here is an
    upper bound on what the corner itself is worth.  The within-game ranks are
    the partial control, since all four seats share a board.
    """
    seats: list[dict[str, int]] = []
    ranks: dict[str, dict[int, list[dict[str, int]]]] = {
        "pips": defaultdict(list),
        "resources": defaultdict(list),
    }
    games = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if len(row["seats"]) != 4 or row["vp_to_win"] != 10:
                continue
            if row["scenario"] != 0 or row["map"] != 0:
                continue
            found = opening(row)
            if not found:
                continue
            games += 1
            seats.extend(found)
            for key, bucket in ranks.items():
                for position, seat in enumerate(sorted(found, key=lambda s: -s[key])):
                    bucket[position].append(seat)

    def bands(field: str, edges: list[tuple[int, int]], within: list[dict[str, int]] | None = None):
        source = seats if within is None else within
        out = []
        for lo, hi in edges:
            chosen = [s for s in source if lo <= s[field] < hi]
            if len(chosen) >= 300:
                out.append({"band": f"{lo}-{hi - 1}", **rate(chosen)})
        return out

    fixed_pips = [s for s in seats if 20 <= s["pips"] < 22]
    fixed_res = [s for s in seats if s["resources"] == 4]
    return {
        "games": games,
        "seat_observations": len(seats),
        "by_pips": bands("pips", [(lo, lo + 2) for lo in range(12, 30, 2)]),
        "by_resources": bands("resources", [(k, k + 1) for k in range(1, 6)]),
        "by_scarce": bands("scarce", [(k, k + 1) for k in range(0, 3)]),
        "rank_by_pips": [
            {"rank": p + 1, **rate(v)} for p, v in sorted(ranks["pips"].items())
        ],
        "rank_by_resources": [
            {"rank": p + 1, **rate(v)} for p, v in sorted(ranks["resources"].items())
        ],
        "by_numbers": bands("numbers", [(k, k + 1) for k in range(2, 8)]),
        "numbers_at_fixed_pips": bands("numbers", [(k, k + 1) for k in range(3, 8)], fixed_pips),
        "dead_at_fixed_pips": [
            {"band": f"{lo:.2f}-{lo + 0.05:.2f}", **rate(chosen)}
            for lo in [x / 100 for x in range(30, 75, 5)]
            if len(chosen := [s for s in fixed_pips if lo <= s["dead"] < lo + 0.05]) >= 300
        ],
        "variance_at_fixed_pips": [
            {"band": f"{lo:.2f}-{lo + 0.1:.2f}", **rate(chosen)}
            for lo in [x / 10 for x in range(2, 16)]
            if len(chosen := [s for s in fixed_pips if lo <= s["variance"] < lo + 0.1]) >= 300
        ],
        "diversity_at_fixed_pips": bands("resources", [(k, k + 1) for k in range(3, 6)], fixed_pips),
        "pips_at_fixed_diversity": bands("pips", [(lo, lo + 2) for lo in range(16, 26, 2)], fixed_res),
        "pick_position": [
            {
                "pick": k + 1,
                "mean_pips": round(
                    statistics.fmean([s["pips"] for s in seats if s["pick"] == k]), 2
                ),
                "mean_resources": round(
                    statistics.fmean([s["resources"] for s in seats if s["pick"] == k]), 2
                ),
                **rate([s for s in seats if s["pick"] == k]),
            }
            for k in range(4)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="path to the .tar.gz of game logs")
    parser.add_argument("--rows", help="JSONL of per-game rows to write or read")
    parser.add_argument("--report", help="path to write the aggregate JSON")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--placement", help="path to write the opening analysis")
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

    if args.placement:
        if not args.rows:
            parser.error("--placement needs --rows")
        summary = placement(args.rows)
        with open(args.placement, "w", encoding="utf-8") as out:
            json.dump(summary, out, indent=2)
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")

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
