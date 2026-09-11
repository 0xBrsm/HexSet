"""Evaluate one evolved phenotype against a frozen gate.

Parsing is dependency-free; the optional Catanatron imports occur only after
argument validation so orchestration and parser probes work without it.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--candidate-id",required=True); p.add_argument("--weights-json",required=True); p.add_argument("--stance",required=True); p.add_argument("--temperature",required=True); p.add_argument("--gate",choices=("both","vs-ab2","vs-shipped"),required=True); p.add_argument("--games",type=int,required=True); p.add_argument("--workers",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(argv)
 if a.games<1 or a.workers<1: p.error("games/workers must be positive")
 weights=json.loads(a.weights_json)
 if not isinstance(weights,dict): p.error("weights must be object")
 try:
  import hexset.bots
  from hexset.arena import Entrant, register_preset
  from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
  from dataclasses import replace
  from .duel import run_duel
  name="heximax-evolve-"+a.candidate_id
  register_preset(name,Entrant(name,kind="heximax",weights=replace(NO_TRADE_WEIGHTS,**{k:float(v) for k,v in weights.items() if k!="scarce"}),depth=2,width=6,max_nodes=600,k=1,mode="notrade",max_trades=0,stance=a.stance,temperature=float(a.temperature)))
  gs=("vs-ab2","vs-shipped") if a.gate=="both" else (a.gate,)
  rows=[]
  for g in gs:
   lineup=f"DC:{name},AB:2,AB:2,AB:2" if g=="vs-ab2" else f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"
   r=run_duel(lineup,a.games,a.workers,seed=a.seed); rows.append({"gate":g,"games":a.games,"workers":a.workers,"seed":a.seed,"players":lineup,"wins":{str(k):v for k,v in r.wins.items()},"points":{str(k):v for k,v in r.points.items()},"report":r.report()})
 except ImportError as e:
  raise SystemExit(f"Catanatron evaluator unavailable: {e}")
 doc={"candidate":a.candidate_id,"weights":weights,"stance":a.stance,"temperature":float(a.temperature),"games":a.games,"workers":a.workers,"seed":a.seed,"gate":a.gate,"rows":rows}; a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_suffix(a.out.suffix+".tmp"); tmp.write_text(json.dumps(doc,indent=2)+"\n"); tmp.replace(a.out); print(json.dumps(doc,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
