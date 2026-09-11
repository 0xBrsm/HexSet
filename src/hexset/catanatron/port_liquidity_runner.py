"""Checkpointed, paired screen runner for the opt-in port-liquidity family."""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing, random, tempfile
from pathlib import Path
from .port_liquiditycfg import GATES, SCREEN_GAMES, SCREEN_SEEDS, SCREEN_WORKERS, arm_manifest, lineup, source_fingerprint
from hexset.bots.heximax.port_liquidity import ARM_LABELS

def paired_seed(gate: str, index: int) -> int:
    if gate not in GATES or not 0 <= index < SCREEN_GAMES: raise ValueError("invalid gate or game index")
    return SCREEN_SEEDS[gate] + index

def _record_path(root: Path, arm: str, gate: str, index: int) -> Path:
    return root / "games" / gate / arm / f"{index:04d}.json"

def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); tmp = Path(f.name)
    tmp.replace(path)

def _play_one(job: tuple[str, str, int, int]) -> dict:
    arm, gate, index, seed = job
    import hexset.bots  # noqa: F401
    from hexset.bots.heximax import port_liquidity  # noqa: F401
    from .duel import parse_cli_string, play_batch
    from catanatron.cli.play import get_actual_victory_points
    random.seed(seed)
    players_spec = lineup(arm, gate); players = parse_cli_string(players_spec); candidate = players[0]
    wins, points, games = play_batch(1, players, quiet=True)
    color = getattr(candidate, "color", None)
    if color is None or len(games) > 1: raise RuntimeError("invalid game result count")
    manifests = []
    if games:
        game = games[0]; winner = game.winning_color(); colors = list(game.state.colors)
        actual_seed = int(game.seed)
        manifests.append({"id": str(game.id), "seed": actual_seed,
                          "requested_seed": seed, "seating": [str(c) for c in colors],
                          "candidate_color": str(color), "winner": str(winner),
                          "points": {str(c): get_actual_victory_points(game.state, c) for c in colors}})
        candidate_wins = int(winner == color)
        actual_points = manifests[0]["points"]
    else:
        winner = None; candidate_wins = 0; actual_points = {}
    return {"family":"heximax-port-liquidity-future","source_hash":source_fingerprint(),"arm":arm,"gate":gate,
            "index":index,"seed":seed,"requested_games":1,"games":manifests,"workers":SCREEN_WORKERS,"players":players_spec,
            "candidate_color":str(color),"candidate_wins":candidate_wins,"winner":str(winner) if winner is not None else None,
            "wins":{str(k):int(v) for k,v in wins.items()},"points":{str(k):list(v) for k,v in points.items()},
            "actual_points":actual_points,"decisions":int(getattr(candidate,"decisions",0)),
            "fallbacks":int(getattr(candidate,"fallbacks",0)),"phenotype":arm_manifest(arm)}

def validate_record(record: dict, arm: str, gate: str, index: int) -> dict:
    if record.get("family") != "heximax-port-liquidity-future": raise ValueError("wrong artifact family")
    if record.get("source_hash") != source_fingerprint(): raise ValueError("source fingerprint mismatch")
    if record.get("arm") != arm or record.get("gate") != gate: raise ValueError("arm/gate mismatch")
    if record.get("index") != index or record.get("seed") != paired_seed(gate,index): raise ValueError("seed/index mismatch")
    if record.get("players") != lineup(arm,gate): raise ValueError("lineup mismatch")
    if record.get("phenotype") != arm_manifest(arm): raise ValueError("phenotype mismatch")
    if record.get("requested_games") != 1 or record.get("workers") != SCREEN_WORKERS: raise ValueError("runtime identity mismatch")
    if record.get("candidate_color") not in ("Color.RED", "RED"):
        raise ValueError("candidate must remain the first RED player")
    candidate_wins = record.get("candidate_wins")
    if candidate_wins not in (0, 1): raise ValueError("invalid candidate result")
    games = record.get("games")
    if not isinstance(games, list) or len(games) > 1: raise ValueError("invalid game manifest")
    wins = record.get("wins")
    if not isinstance(wins, dict) or any(not isinstance(v, int) or v < 0 for v in wins.values()):
        raise ValueError("invalid winner counts")
    if sum(wins.values()) not in (0, 1): raise ValueError("wins are not one-hot")
    key = record["candidate_color"]
    if wins.get(key, wins.get("RED", 0)) != candidate_wins:
        raise ValueError("candidate win count mismatch")
    if not games:
        if candidate_wins or sum(wins.values()) or record.get("winner") is not None: raise ValueError("draw has winner")
    else:
        game = games[0]
        if game.get("requested_seed") != record["seed"] or game.get("candidate_color") != record["candidate_color"]:
            raise ValueError("game metadata mismatch")
        if record.get("winner") != game.get("winner") or (record.get("winner") == record.get("candidate_color")) != bool(candidate_wins):
            raise ValueError("candidate result does not match winner")
        seating = game.get("seating")
        if not isinstance(seating, list) or len(seating) != 4 or len(set(seating)) != 4:
            raise ValueError("invalid seating metadata")
        if record.get("winner") not in seating or record.get("candidate_color") not in seating:
            raise ValueError("winner or candidate absent from seating")
        if sum(wins.values()) != 1 or wins.get(record["winner"], 0) != 1:
            raise ValueError("wins do not identify the recorded winner")
    return record

def run_screen(arm: str, gate: str, root: Path, *, workers: int=SCREEN_WORKERS, games: int=SCREEN_GAMES, pool_factory=None) -> dict:
    if arm not in ARM_LABELS or gate not in GATES: raise ValueError("unknown arm or gate")
    if workers != SCREEN_WORKERS: raise ValueError(f"workers must be {SCREEN_WORKERS}")
    if games != SCREEN_GAMES: raise ValueError(f"games must be {SCREEN_GAMES}")
    jobs=[]; records=[]
    for index in range(games):
        path=_record_path(root,arm,gate,index)
        if path.exists(): records.append(validate_record(json.loads(path.read_text()),arm,gate,index))
        else: jobs.append((arm,gate,index,paired_seed(gate,index)))
    if jobs:
        factory = pool_factory or (lambda n: multiprocessing.get_context("spawn").Pool(n))
        with factory(SCREEN_WORKERS) as pool:
            for record in pool.imap_unordered(_play_one,jobs):
                validate_record(record,arm,gate,record["index"]); _atomic(_record_path(root,arm,gate,record["index"]),record); records.append(record)
    records.sort(key=lambda value:value["index"])
    summary={"family":"heximax-port-liquidity-future","source_hash":source_fingerprint(),"arm":arm,"gate":gate,
             "games":len(records),"workers":workers,"seed":SCREEN_SEEDS[gate],"players":lineup(arm,gate),
             "phenotype":arm_manifest(arm),"candidate_wins":sum(r["candidate_wins"] for r in records),
             "records_sha256":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
    _atomic(root/"summaries"/f"{gate}-{arm}.json",summary); return summary

def main(argv=None) -> int:
    if __import__("os").environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required")
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--arm",choices=ARM_LABELS,required=True)
    p.add_argument("--gate",choices=GATES,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=SCREEN_WORKERS); p.add_argument("--games",type=int,default=SCREEN_GAMES); a=p.parse_args(argv)
    print(json.dumps(run_screen(a.arm,a.gate,a.out,workers=a.workers,games=a.games),indent=2,sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
