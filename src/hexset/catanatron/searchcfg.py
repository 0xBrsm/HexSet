"""Screen Heximax search budgets against the two frozen four-player gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import hexset.bots
from hexset.arena import Entrant,register_preset
from .duel import run_duel

CONFIGS={"nodes1200-width8":(1200,8,1),"nodes2400-width12":(2400,12,1),"nodes2400-full":(2400,None,1),"nodes600-k4":(600,6,4),"nodes2400-k4":(2400,6,4),"depth3-width6":(600,6,1),"depth3-width6-2400":(2400,6,1),"depth3-width6-4800":(4800,6,1),"port-aware":(600,6,1)}
for n,(nodes,width,k) in CONFIGS.items():
 register_preset("heximax-search-"+n,Entrant("heximax-search-"+n,kind="heximax",depth=3 if n.startswith("depth3") else 2,width=width,max_nodes=nodes,k=k,max_trades=0,mode="notrade",port_aware=n=="port-aware"))
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',choices=sorted(CONFIGS),required=True);p.add_argument('--games',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gate',choices=('vs-ab2','vs-shipped'),default=None);a=p.parse_args(); name='heximax-search-'+a.candidate; rows=[]
 gates=(("vs-ab2",f"DC:{name},AB:2,AB:2,AB:2"),("vs-shipped",f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"))
 if a.gate is not None: gates=tuple(x for x in gates if x[0]==a.gate)
 for gate,players in gates:
  r=run_duel(players,a.games,a.workers,a.seed); rows.append({"gate":gate,"games":a.games,"wins":{str(k):v for k,v in r.wins.items()},"report":r.report()})
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({"candidate":a.candidate,"games":a.games,"seed":a.seed,"rows":rows},indent=2));print(a.out)
if __name__=='__main__':main()
