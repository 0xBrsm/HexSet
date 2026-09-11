"""Fail closed selection and fresh validation of completed evolve checkpoints.

Only complete generation summaries with both gate rows participate.  This
module is intentionally separate from the exploratory evolve runner.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, tempfile
from pathlib import Path
from .stance_validation import GATES, THRESHOLDS

Z99 = 3.29052673149

def wilson_lower(wins: int, games: int, z: float = Z99) -> float:
    if games <= 0 or wins < 0 or wins > games: raise ValueError("invalid wins/games")
    p=wins/games; d=1+z*z/games
    return (p+z*z/(2*games)-z*math.sqrt(p*(1-p)/games+z*z/(4*games*games)))/d


CONTINUATION_LOOKS = (4096, 8192, 16384)
CONTINUATION_ALPHA = 0.001 / 3

def binomial_upper_tail(wins: int, games: int, p: float) -> float:
    """Exact P[X >= wins] for X~Binomial(games,p), via log-PMF recurrence."""
    if not (0 <= wins <= games and 0 < p < 1): raise ValueError("invalid binomial inputs")
    import math
    logp=math.lgamma(games+1)-math.lgamma(wins+1)-math.lgamma(games-wins+1)+wins*math.log(p)+(games-wins)*math.log1p(-p)
    term=math.exp(logp); total=term
    for x in range(wins, games):
        term *= (games-x)/(x+1) * p/(1-p); total += term
    return min(1.0,total)

def continuation_decision(counts: dict[str, tuple[int,int]]) -> str:
    """Decide a fixed-phenotype continuation look across both gates."""
    if set(counts) != set(GATES): raise ValueError("both gates required")
    games={n for _,n in counts.values()}
    n=games.pop() if len(games)==1 else 0
    if n not in CONTINUATION_LOOKS: raise ValueError("invalid cumulative look")
    for gate in GATES:
        wins,total=counts[gate]
        if not (0 <= wins <= total == n): raise ValueError("invalid counts")
        if wins / n <= THRESHOLDS[gate]: return "REJECTED_AT_LOOK"
    if all(binomial_upper_tail(counts[g][0], n, THRESHOLDS[g]) < CONTINUATION_ALPHA for g in GATES): return "PASS"
    return "CONTINUE" if n != CONTINUATION_LOOKS[-1] else "UNRESOLVED_AT_CAP"

def _candidate_wins(row: dict, candidate: str) -> int:
    if not isinstance(row, dict) or row.get("gate") not in GATES: raise ValueError("invalid gate row")
    n=row.get("games"); w=row.get("candidate_wins")
    if not isinstance(n,int) or n <= 0: raise ValueError("invalid games")
    if w is None:
        wins=row.get("wins", {}); w=wins.get("Color.RED", wins.get("RED"))
    if not isinstance(w,int) or not 0 <= w <= n: raise ValueError("invalid candidate wins")
    return w

def select_candidate(checkpoint: Path) -> dict:
    try: state=json.loads(Path(checkpoint).read_text())
    except (OSError, ValueError) as e: raise ValueError("malformed checkpoint") from e
    choices=[]
    for gen in state.get("generations", []):
        if not isinstance(gen,dict) or not gen.get("complete"): continue
        for item in gen.get("summary", []):
            cid=item.get("candidate"); rows=item.get("rows",[])
            if not isinstance(cid,str) or len(rows)!=2: continue
            by={r.get("gate"):r for r in rows if isinstance(r,dict)}
            if set(by)!=set(GATES): continue
            try:
                rates={g:_candidate_wins(by[g],cid)/by[g]["games"] for g in GATES}
            except (KeyError,ValueError,TypeError): continue
            score=min(rates[g]-THRESHOLDS[g] for g in GATES)
            choices.append((score, cid, gen.get("generation"), item))
    if not choices: raise ValueError("no complete dual-gate candidates")
    score,cid,gen,item=max(choices,key=lambda x:(x[0],x[1]))
    return {"candidate":cid,"generation":gen,"score":score,"candidate_record":item,
            "source_hash":state.get("source_hash"),"protocol":state.get("protocol"),
            "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}

def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".stage-", dir=path.parent)
    import os
    with os.fdopen(fd, "w") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n")
    Path(name).replace(path)

def _run_fresh_legacy(checkpoint: Path, out_dir: Path, *, python="python", evaluator="hexset.catanatron.evolve_candidate_eval", runner=subprocess.run) -> dict:
    """Run/resume preregistered confirmation then independent single-gate holdouts."""
    chosen=select_candidate(checkpoint); candidate=chosen["candidate"]; record=chosen["candidate_record"]
    out_dir.mkdir(parents=True, exist_ok=True); confirm=out_dir/"confirmation.json"
    def invoke(gate, games, seed, out):
        cmd=[python,"-m",evaluator,"--candidate-id",candidate,"--weights-json",json.dumps(record.get("weights",{}),sort_keys=True),"--stance",record.get("stance","win"),"--temperature",str(record.get("temperature",0)),"--gate",gate,"--games",str(games),"--workers","30","--seed",str(seed),"--out",str(out)]
        runner(cmd,check=True)
    if not confirm.exists(): invoke("both",1024,50_000_000,confirm)
    doc, wins = _read_fresh(confirm, 1024, {"vs-ab2","vs-shipped"}, candidate)
    if not all(wins[g]/1024 > THRESHOLDS[g] for g in GATES):
        return {"status":"NEEDS_FRESH_VALIDATION","stage":"confirmation","candidate":candidate}
    rows={}
    for i,g in enumerate(GATES):
        out=out_dir/f"holdout-{g}.json"
        if not out.exists(): invoke(g,4096,60_000_000+i*10_000_000,out)
        _, w=_read_fresh(out,4096,{g},candidate); rows[g]={"wins":w[g],"games":4096,"rate":w[g]/4096,"wilson_lower":wilson_lower(w[g],4096),"threshold":THRESHOLDS[g]}
    result={"status":"PASS" if all(v["wilson_lower"]>v["threshold"] for v in rows.values()) else "FAIL","candidate":candidate,"z":Z99,"gates":rows,"checkpoint_sha256":chosen["checkpoint_sha256"]}
    _atomic_json(out_dir/"goal-verdict.json",result); return result

def run_fresh(checkpoint: Path, out_dir: Path, *, python="python", evaluator="hexset.catanatron.evolve_candidate_eval", runner=None) -> dict:
    chosen=select_candidate(checkpoint); candidate=chosen["candidate"]; record=chosen["candidate_record"]; out_dir.mkdir(parents=True,exist_ok=True)
    import hashlib
    fingerprint=hashlib.sha256(json.dumps({"candidate":candidate,"record":record,"source":chosen.get("source_hash"),"protocol":chosen.get("protocol")},sort_keys=True).encode()).hexdigest()
    manifest=out_dir/"manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get("candidate") != candidate or old.get("fingerprint") != fingerprint: raise ValueError("fresh phenotype fingerprint mismatch")
    else: _atomic_json(manifest,{"candidate":candidate,"fingerprint":fingerprint,"checkpoint_sha256":chosen["checkpoint_sha256"]})
    def invoke(gate,games,seed,out):
        cmd=[python,"-m",evaluator,"--candidate-id",candidate,"--weights-json",json.dumps(record.get("weights",{}),sort_keys=True),"--stance",record.get("stance","win"),"--temperature",str(record.get("temperature",0)),"--gate",gate,"--games",str(games),"--workers","30","--seed",str(seed),"--out",str(out)]
        (runner or __import__('subprocess').run)(cmd,check=True)
    cpath=out_dir/"confirmation.json"
    if not cpath.exists(): invoke("both",1024,50_000_000,cpath)
    _,cw=_read_fresh(cpath,1024,set(GATES),candidate)
    if not all(cw[g]/1024>THRESHOLDS[g] for g in GATES): return {"status":"NEEDS_FRESH_VALIDATION","stage":"confirmation","candidate":candidate}
    totals={g:0 for g in GATES}; blocks=(4096,4096,8192); bases={"vs-ab2":60_000_000,"vs-shipped":70_000_000}
    for look,block in enumerate(blocks):
        for g in GATES:
            path=out_dir/f"holdout-{g}-block-{look}.json"
            if not path.exists(): invoke(g,block,bases[g]+sum(blocks[:look]),path)
            _,w=_read_fresh(path,block,{g},candidate); totals[g]+=w[g]
        decision=continuation_decision({g:(totals[g],sum(blocks[:look+1])) for g in GATES})
        _atomic_json(out_dir/f"decision-{sum(blocks[:look+1])}.json",{"candidate":candidate,"decision":decision,"counts":totals})
        if decision in ("PASS","REJECTED_AT_LOOK","UNRESOLVED_AT_CAP"):
            result={"status":decision,"candidate":candidate,"counts":totals,"checkpoint_sha256":chosen["checkpoint_sha256"]}; _atomic_json(out_dir/"goal-verdict.json",result); return result
    return {"status":"UNRESOLVED_AT_CAP","candidate":candidate,"counts":totals}

def _read_fresh(path, games, gates, candidate):
    d=json.loads(path.read_text())
    if d.get("candidate") != candidate or d.get("games") != games: raise ValueError("fresh artifact identity mismatch")
    rows={r.get("gate"):r for r in d.get("rows",[]) if isinstance(r,dict)}
    if set(rows)!=set(gates) or len(rows)!=len(gates): raise ValueError("fresh gate matrix mismatch")
    out={};
    for g,r in rows.items():
        w=r.get("candidate_wins",r.get("wins",{}).get("Color.RED")); n=r.get("games")
        if not isinstance(w,int) or n!=games or w<0 or w>n: raise ValueError("invalid fresh counts")
        out[g]=w
    return d,out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv); result=select_candidate(a.checkpoint); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
