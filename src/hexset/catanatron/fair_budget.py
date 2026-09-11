"""Run the fixed four-job ordinary-Heximax budget screen."""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from hexset.arena import Entrant, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
GATES = ("ab2", "shipped")
SCREEN_SEEDS = {"ab2": 80_000_000, "shipped": 80_100_000}
SCREEN_GAMES = 120
SCREEN_WORKERS = 30
CONTROL = "heximax-notrade-wide"
CANDIDATE = "heximax-fair-k4"
SELF_OPPONENT = "heximax-notrade"
PHENOTYPES: dict[str, dict[str, Any]] = {
    CONTROL: {"kind":"heximax","mode":"notrade","depth":2,"width":12,"max_nodes":2400,"max_trades":0,"k":1,"weights":asdict(NO_TRADE_WEIGHTS)},
    CANDIDATE: {"kind":"heximax","mode":"notrade","depth":2,"width":12,"max_nodes":9600,"max_trades":0,"k":4,"weights":asdict(NO_TRADE_WEIGHTS)},
}
def _entrant(name: str) -> Entrant:
    c=PHENOTYPES[name]
    return Entrant(name, kind="heximax", weights=NO_TRADE_WEIGHTS, depth=c["depth"], width=c["width"], max_nodes=c["max_nodes"], max_trades=0, mode="notrade", k=c["k"])
register_preset(CONTROL, _entrant(CONTROL))
register_preset(CANDIDATE, _entrant(CANDIDATE))
def players_for(entrant: str, gate: str) -> str:
    if entrant not in PHENOTYPES or gate not in GATES: raise ValueError("unknown entrant or gate")
    opponents="AB:2,AB:2,AB:2" if gate=="ab2" else ",".join(f"DC:{SELF_OPPONENT}" for _ in range(3))
    return f"DC:{entrant},{opponents}"
def source_fingerprint() -> str:
    here=Path(__file__).resolve(); root=here.parents[1]
    files=(here,here.with_name("fair_budget_validation.py"),root/"arena.py",root/"bots"/"heximax"/"presets.py",root/"bots"/"heximax"/"search.py",root/"bots"/"heximax"/"evaluate.py",here.with_name("duel.py"),here.with_name("player.py"))
    d=hashlib.sha256()
    for f in files: d.update(f.relative_to(root).as_posix().encode()); d.update(f.read_bytes())
    return d.hexdigest()
def _write(path: Path, value: object) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
def _manifest(games: int, workers: int, source_hash: str) -> dict[str,Any]:
    jobs=[]
    for index, (role,entrant) in enumerate((("control",CONTROL),("control",CONTROL),("candidate",CANDIDATE),("candidate",CANDIDATE))):
        gate = GATES[index % 2]
        jobs.append({"config":"k1" if role=="control" else "k4","role":role,"entrant":entrant,"gate":gate,"games":games,"workers":workers,"seed":SCREEN_SEEDS[gate] + index * 10_000,"players":players_for(entrant,gate),"phenotype":PHENOTYPES[entrant]})
    return {"schema":3,"protocol":"fair-budget-v1","games":games,"workers":workers,"source_hash":source_hash,"phenotypes":PHENOTYPES,"self_opponent":SELF_OPPONENT,"jobs":jobs}
def plan(root: Path, games: int=SCREEN_GAMES, workers: int=SCREEN_WORKERS, source_hash: str|None=None) -> dict[str,Any]:
    if games!=SCREEN_GAMES or workers!=SCREEN_WORKERS: raise ValueError("fair budget screen requires exactly 120 games and 30 workers")
    root=Path(root); expected=_manifest(games,workers,source_hash or source_fingerprint()); path=root/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())!=expected: raise ValueError("screen manifest/source/phenotype mismatch")
    else: _write(path,expected)
    return expected
def _validate_result(job: dict[str,Any], result: dict[str,Any], source_hash: str) -> None:
    fields=("protocol","source_hash","config","role","entrant","gate","games","workers","seed","players","phenotype")
    expected={"protocol":"fair-budget-v1","source_hash":source_hash,**{k:job[k] for k in fields if k in job}}
    for k,v in expected.items():
        if result.get(k)!=v: raise ValueError(f"screen artifact identity mismatch: {k}")
    wins=result.get("wins")
    if not isinstance(wins,dict): raise ValueError("screen artifact has invalid wins")
    values = list(wins.values()) if isinstance(wins, dict) else []
    valid = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values) and sum(values) == job["games"]
    if not valid: raise ValueError("screen artifact has invalid wins")
def run(root: Path, source_hash: str|None=None) -> dict[str,Any]:
    from hexset.catanatron.duel import run_duel
    source_hash=source_hash or source_fingerprint(); manifest=plan(root,source_hash=source_hash); root=Path(root)
    expected_files = {f"{j['config']}-{j['role']}-{j['gate']}.json" for j in manifest["jobs"]}
    extras = {p.name for p in root.glob("*.json") if p.name != "manifest.json"} - expected_files
    if extras: raise ValueError(f"unexpected screen artifacts: {sorted(extras)}")
    results=[]
    for job in manifest["jobs"]:
        path=root/f"{job['config']}-{job['role']}-{job['gate']}.json"
        if path.exists(): doc=json.loads(path.read_text()); _validate_result(job,doc,source_hash)
        else:
            duel=run_duel(job["players"],job["games"],job["workers"],seed=job["seed"])
            doc={"protocol":"fair-budget-v1","source_hash":source_hash,**{k:job[k] for k in ("config","role","entrant","gate","games","workers","seed","players","phenotype")},"wins":{str(k):int(v) for k,v in duel.wins.items()},"points":{str(k):list(v) for k,v in duel.points.items()}}
            _validate_result(job,doc,source_hash); _write(path,doc)
        results.append(doc)
    return {"status":"COMPLETE","manifest":manifest,"results":results}
def entrant_config(label: str) -> Entrant:
    if label=="k1": return _entrant(CONTROL)
    if label=="k4": return _entrant(CANDIDATE)
    raise ValueError(label)
def main(argv: list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True); p.add_argument("--run",action="store_true"); p.add_argument("--source-hash",default=None); a=p.parse_args(argv)
    print(json.dumps(run(a.root,a.source_hash) if a.run else plan(a.root,source_hash=a.source_hash),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
