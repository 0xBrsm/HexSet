"""Second-phase joint weight interaction screen for the Luna campaign."""
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
import hexset.bots
from hexset.arena import Entrant, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.bots.evaluate import ROLLS
from .duel import run_duel

BASE=NO_TRADE_WEIGHTS
CHANGES={
 "progress141-road0":{"buy_progress":1.41,"road":0},
 "progress141-roadhalf":{"buy_progress":1.41,"road":.5},
 "progress141-prodhalf":{"buy_progress":1.41,"production":.5},
 "progress141-riskhalf":{"buy_progress":1.41,"robber_risk":.5},
 "progress141-spare141":{"buy_progress":1.41,"spare_card":1.41},
 "prodhalf-road0":{"production":.5,"road":0},
 "prod141-road0":{"production":1.41,"road":0},
 "prodhalf-progress141-road0":{"production":.5,"buy_progress":1.41,"road":0},
 "riskhalf-progress141-road0":{"robber_risk":.5,"buy_progress":1.41,"road":0},
 "risk141-progress141-road0":{"robber_risk":1.41,"buy_progress":1.41,"road":0},
 "spare141-progress141-road0":{"spare_card":1.41,"buy_progress":1.41,"road":0},
 "all-joint":{"production":.5,"buy_progress":1.41,"robber_risk":.5,"road":0,"spare_card":1.41},
}
def make(v):
 d={t:getattr(BASE,t)*f for t,f in v.items()}
 if "production" in d: d["scarce"] = .91 * d["production"] / ROLLS
 return replace(BASE,**d)
WEIGHTS={k:make(v) for k,v in CHANGES.items()}
for k,w in WEIGHTS.items(): register_preset("heximax-joint-"+k,Entrant("heximax-joint-"+k,kind="heximax",depth=2,width=6,max_trades=0,mode="notrade",weights=w))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--candidate',choices=sorted(WEIGHTS),required=True); p.add_argument('--games',type=int,required=True); p.add_argument('--workers',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); name='heximax-joint-'+a.candidate; rows=[]
 for gate,players in (("vs-ab2",f"DC:{name},AB:2,AB:2,AB:2"),("vs-shipped",f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")):
  r=run_duel(players,a.games,a.workers,a.seed); rows.append({"gate":gate,"games":a.games,"wins":{str(k):v for k,v in r.wins.items()},"report":r.report(),"players":players})
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({"candidate":a.candidate,"games":a.games,"workers":a.workers,"seed":a.seed,"rows":rows},indent=2)); print(a.out)
if __name__=='__main__': main()
