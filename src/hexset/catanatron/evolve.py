"""Resumable, bounded proposal/racing controller for Heximax weights.

The default command emits a deterministic proposal plan; ``--run`` executes the
required before invoking the duel harness; this keeps campaign planning from
starting workers accidentally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from multiprocessing import Pool
from dataclasses import asdict, replace
from pathlib import Path

from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
import hexset.bots
from hexset.arena import Entrant, register_preset
from hexset.bots.stances import WIN_TEMPERATURE
from hexset.bots.evaluate import ROLLS

WEIGHT_FIELDS = tuple(NO_TRADE_WEIGHTS.__dataclass_fields__)
MUTABLE_FIELDS = tuple(f for f in WEIGHT_FIELDS if f != "victory_point")
BOUNDS = {
    "production": (1.0, 5.5), "diversity": (0.0, 1.0),
    "scarce": (0.0, 2.0), "buy_progress": (0.35, 2.5),
    "road": (0.0, 0.35), "knight": (0.0, 0.35),
    "spare_card": (0.0, 0.35), "robber_risk": (-0.75, 0.0),
    "port": (0.0, 0.30), "temperature": (0.75, 8.0),
}
SEED_FAMILIES = {"discovery": 10_000_000, "confirm": 20_000_000, "holdout": 30_000_000}


def _source_hash() -> str:
    """Hash the runnable campaign code, excluding checkpoints and results."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _clip(field: str, value: float) -> float:
    lo, hi = BOUNDS[field]
    return max(lo, min(hi, value))


def candidate(label: str, values: dict[str, float], *, stance: str = "win") -> dict:
    """Return a serializable candidate, deriving scarcity from production."""
    vals = {k: float(v) for k, v in values.items() if k in WEIGHT_FIELDS}
    vals["victory_point"] = 1.0
    vals["scarce"] = 0.91 * vals["production"] / ROLLS
    return {"id": label, "weights": vals, "stance": stance,
            "temperature": float(values.get("temperature", WIN_TEMPERATURE)),
            "depth": 2, "width": 6, "max_nodes": 600, "k": 1,
            "mode": "notrade", "max_trades": 0}


def proposals(generation: int, count: int = 12, seed: int = 0) -> list[dict]:
    """Generate deterministic clipped mutations around incumbent/screen seeds."""
    base = asdict(NO_TRADE_WEIGHTS)
    seeds = [base,
             {**base, "production": base["production"] * 1.41, "road": 0.0},
             {**base, "robber_risk": base["robber_risk"] * .5,
              "buy_progress": base["buy_progress"] * 1.41, "road": 0.0},
             {**base, "production": base["production"] * .5, "road": 0.0},
             {**base, "robber_risk": base["robber_risk"] * 1.41,
              "buy_progress": base["buy_progress"] * 1.41, "road": 0.0}]
    rng = random.Random(seed + generation)
    out = []
    for i in range(count):
        v = dict(seeds[i % len(seeds)])
        for field in MUTABLE_FIELDS:
            if field == "scarce":
                continue
            scale = max(abs(v[field]), 0.1)
            v[field] = _clip(field, v[field] + rng.gauss(0.0, 0.12 * scale))
        v["temperature"] = _clip("temperature", v.get("temperature", WIN_TEMPERATURE))
        out.append(candidate(f"g{generation:02d}-p{i:02d}", v))
    return out


def seed_for(stage: str, generation: int, game_index: int) -> int:
    """Shared per-game seed; deliberately independent of candidate ID."""
    if stage not in SEED_FAMILIES:
        raise ValueError(stage)
    return SEED_FAMILIES[stage] + generation * 100_000 + game_index


def paired_seed(gate: str, stage: str, generation: int, game_index: int) -> int:
    """Stable per-game seed shared by candidates, with disjoint gate families."""
    if gate not in ("vs-ab2", "vs-shipped"):
        raise ValueError(gate)
    return seed_for(stage, generation, game_index) + (50_000 if gate == "vs-shipped" else 0)


def select_promotions(results: list[dict], limit: int = 4) -> list[str]:
    """Select complete candidates by dual gate margins (AB2 .50, self .25)."""
    scored = []
    for item in results:
        margins = {}
        for row in item.get("rows", []):
            games = row.get("games", 0); wins = row.get("wins", {})
            if games:
                candidate_wins = row.get("candidate_wins")
                if candidate_wins is None:
                    candidate_wins = next((wins[key] for key in ("Color.RED", "RED", "red") if key in wins), None)
                    if candidate_wins is None:
                        candidate_wins = (next(iter(wins.values())) if len(wins) == 1 else
                                          wins.get(item.get("candidate"), 0))
                rate = candidate_wins / games
                target = 0.50 if row.get("gate") == "vs-ab2" else 0.25
                margins[row.get("gate")] = rate - target
        if len(margins) == 2 and min(margins.values()) >= 0:
            scored.append((min(margins.values()), item["candidate"]))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]


def next_generation(previous: list[dict], generation: int, count: int = 12, seed: int = 0) -> list[dict]:
    """Mutate/recombine the best completed vectors, retaining an incumbent anchor."""
    if not previous:
        return proposals(generation, count, seed)
    ranked = select_promotions(previous, limit=min(2, len(previous)))
    by_id = {c.get("id", c.get("candidate")): c for c in previous}
    parents = [by_id[x] for x in ranked if x in by_id]
    if not parents:
        parents = previous[:1]
    rng = random.Random(seed + generation * 7919)
    out = []
    for i in range(count):
        parent = parents[i % len(parents)]
        vals = dict(parent["weights"])
        for field in MUTABLE_FIELDS:
            if field == "scarce": continue
            vals[field] = _clip(field, vals[field] + rng.gauss(0.0, max(abs(vals[field]), .1) * .1))
        parent_id = parent.get("id", parent.get("candidate"))
        item = candidate(f"g{generation:02d}-p{i:02d}-from-{parent_id}", vals, stance=parent.get("stance", "win"))
        item["parent_id"] = parent_id
        out.append(item)
    return out


def save_checkpoint(path: Path, payload: dict) -> None:
    """Atomically replace a checkpoint, retaining append-only result records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True); f.flush()
        tmp = Path(f.name)
    tmp.replace(path)


def _play_one(job: tuple[str, str, str, int, int, int]) -> dict:
    """Run exactly one seeded game; seed omits candidate for pairing."""
    candidate_id, gate, stage, generation, game_index, seed = job
    import random
    # Import the project adapter: it registers the DC player before parsing the
    # lineup. Directly importing catanatron's parser leaves DC unregistered.
    from hexset.catanatron.duel import parse_cli_string, play_batch
    from catanatron.cli.play import get_actual_victory_points
    name = "heximax-evolve-" + candidate_id
    players_spec = (f"DC:{name},AB:2,AB:2,AB:2" if gate == "vs-ab2" else
                    f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
    random.seed(seed)
    player_objects = parse_cli_string(players_spec)
    candidate_player = player_objects[0]
    wins, points, games = play_batch(1, player_objects, quiet=True)
    candidate_color = getattr(candidate_player, "color", None)
    game_manifest = []
    candidate_wins = 0
    for game in games:
        colors = list(game.state.colors)
        # Catanatron assigns colors to player objects; state.colors is seating
        # order and may be rotated independently of the player specification.
        actual_candidate_color = candidate_color
        if actual_candidate_color is None:
            raise RuntimeError("candidate player has no assigned color")
        winner = game.winning_color()
        candidate_wins += int(winner == actual_candidate_color)
        game_manifest.append({
            "id": game.id, "seed": game.seed,
            "seating": [getattr(color, "value", str(color)) for color in colors],
            "candidate_color": getattr(actual_candidate_color, "value", str(actual_candidate_color)),
            "winner": getattr(winner, "value", str(winner)),
            "points": {getattr(color, "value", str(color)): get_actual_victory_points(game.state, color)
                       for color in colors},
        })
    evolve_player = player_objects[0]
    return {"candidate": candidate_id, "gate": gate, "stage": stage,
            "generation": generation, "game_index": game_index, "seed": seed,
            "players": players_spec, "wins": {str(k): v for k, v in wins.items()},
            "candidate_wins": candidate_wins,
            "points": {str(k): v for k, v in points.items()},
            "games": game_manifest,
            "decisions": getattr(evolve_player, "decisions", 0),
            "fallbacks": getattr(evolve_player, "fallbacks", 0)}


def _game_path(root: Path, generation: int, candidate_id: str, gate: str,
               stage: str, game_index: int) -> Path:
    return (root / "games" / f"generation-{generation:02d}" /
            f"{candidate_id}-{gate}-{stage}-{generation:02d}-{game_index:04d}.json")


def _read_game(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"corrupt game checkpoint: {path}") from exc
    required = {"candidate", "gate", "stage", "generation", "game_index", "wins"}
    if not required.issubset(value):
        raise RuntimeError(f"incomplete game checkpoint: {path}")
    return value


def _run_jobs(root: Path, jobs: list[tuple], generation: int) -> None:
    if not jobs:
        return
    with Pool(min(30, len(jobs))) as pool:
        for record in pool.imap_unordered(_play_one, jobs):
            out = _game_path(root, generation, record["candidate"], record["gate"],
                             record["stage"], record["game_index"])
            if not out.exists():
                save_checkpoint(out, record)


def _summarize(root: Path, generation: int, candidates: list[dict],
               games: int, total_games: int) -> list[dict]:
    summary = []
    for c in candidates:
        rows = []
        for gate in ("vs-ab2", "vs-shipped"):
            records = []
            for stage, begin, end in (("discovery", 0, games), ("confirm", games, total_games)):
                for i in range(begin, end):
                    path = _game_path(root, generation, c["id"], gate, stage, i)
                    if path.exists():
                        record = _read_game(path)
                        if (record.get("candidate") != c["id"] or record.get("gate") != gate or
                                record.get("stage") != stage or record.get("generation") != generation or
                                record.get("game_index") != i):
                            raise RuntimeError(f"game checkpoint identity mismatch: {path}")
                        records.append(record)
            if len(records) != total_games:
                continue
            wins = {}
            candidate_wins = None
            for record in records:
                if "candidate_wins" in record:
                    candidate_wins = (candidate_wins or 0) + record["candidate_wins"]
                for key, value in record.get("wins", {}).items():
                    wins[key] = wins.get(key, 0) + value
            row = {"gate": gate, "games": len(records), "wins": wins}
            if candidate_wins is not None:
                row["candidate_wins"] = candidate_wins
            rows.append(row)
        summary.append({"candidate": c["id"], "weights": c["weights"],
                        "stance": c.get("stance", "win"), "rows": rows})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--total-games", type=int, default=300)
    p.add_argument("--promote-top", type=int, default=4)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-hash", default=_source_hash())
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.count < 1 or a.count > 12:
        p.error("count must be 1..12")
    if a.generations < 1:
        p.error("generations must be at least 1")
    if a.games < 1 or a.total_games < a.games:
        p.error("total-games must be >= games >= 1")

    root = a.checkpoint.parent
    if a.checkpoint.exists():
        old = json.loads(a.checkpoint.read_text())
        config = old.get("config", {})
        expected = {"count": a.count, "games": a.games, "total_games": a.total_games,
                    "promote_top": a.promote_top, "generations": a.generations,
                    "seed": a.seed}
        if (old.get("source_hash") != a.source_hash or old.get("protocol") != 1 or
                any(config.get(k) != v for k, v in expected.items())):
            p.error("checkpoint source/protocol/config mismatch; refusing resume")
        state = old
    else:
        state = {"protocol": 1, "source_hash": a.source_hash,
                 "seed_families": SEED_FAMILIES, "config": {
                     "count": a.count, "games": a.games, "total_games": a.total_games,
                     "promote_top": a.promote_top, "generations": a.generations,
                     "seed": a.seed}, "generations": []}

    if not a.run:
        if not state["generations"]:
            state["candidates"] = proposals(a.generation, a.count, a.seed)
        save_checkpoint(a.checkpoint, state)
        print(json.dumps(state, indent=2, sort_keys=True)); return

    completed = {g["generation"]: g for g in state.get("generations", [])
                 if g.get("complete")}
    for generation in range(a.generation, a.generations):
        if generation in completed:
            # A completion marker does not make damaged game records invisible.
            _summarize(root, generation, completed[generation]["candidates"],
                       a.games, a.total_games)
            continue
        previous = completed.get(generation - 1)
        if generation == a.generation and not previous and not state.get("generations"):
            candidates = proposals(generation, a.count, a.seed)
        elif previous:
            candidates = next_generation(previous["summary"], generation, a.count, a.seed)
        else:
            prior = state.get("generations", [])[-1]
            candidates = prior["next_candidates"]

        for c in candidates:
            name = "heximax-evolve-" + c["id"]
            register_preset(name, Entrant(name, kind="heximax", weights=replace(
                NO_TRADE_WEIGHTS, **{k: v for k, v in c["weights"].items() if k != "scarce"}),
                depth=2, width=6, max_nodes=600, k=1, mode="notrade", max_trades=0,
                stance=c["stance"], temperature=c["temperature"]))

        jobs = []
        for c in candidates:
            for gate in ("vs-ab2", "vs-shipped"):
                for i in range(a.games):
                    path = _game_path(root, generation, c["id"], gate, "discovery", i)
                    if not path.exists():
                        jobs.append((c["id"], gate, "discovery", generation, i,
                                     paired_seed(gate, "discovery", generation, i)))
        _run_jobs(root, jobs, generation)
        discovery = _summarize(root, generation, candidates, a.games, a.games)
        promoted = select_promotions(discovery, a.promote_top)

        jobs = []
        for c in candidates:
            if c["id"] not in promoted:
                continue
            for gate in ("vs-ab2", "vs-shipped"):
                for i in range(a.games, a.total_games):
                    path = _game_path(root, generation, c["id"], gate, "confirm", i)
                    if not path.exists():
                        jobs.append((c["id"], gate, "confirm", generation, i,
                                     paired_seed(gate, "confirm", generation, i)))
        _run_jobs(root, jobs, generation)
        summary = _summarize(root, generation, candidates, a.games, a.total_games)
        next_candidates = next_generation(summary, generation + 1, a.count, a.seed)
        record = {"generation": generation, "candidates": candidates, "summary": summary,
                  "promoted": promoted, "next_candidates": next_candidates, "complete": True}
        state["generations"] = [g for g in state.get("generations", []) if g.get("generation") != generation]
        state["generations"].append(record)
        state["generations"].sort(key=lambda g: g["generation"])
        save_checkpoint(a.checkpoint, state)
        completed[generation] = record

    state["generation"] = a.generations - 1
    save_checkpoint(a.checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
