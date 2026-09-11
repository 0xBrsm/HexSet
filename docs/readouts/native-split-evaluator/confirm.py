"""Fresh fixed confirmation of the selected split against shared trading."""
import argparse
from multiprocessing import get_context
from pathlib import Path
import json
import math
import statistics
import time
from run import base_identity, initialize, play, atomic
from hexset.arena import wilson


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 out=args.out;base=base_identity();n=512;seed=719000000
 candidates=['N:T:0.0','T:T:0.0'];jobs=[];start=time.perf_counter()
 for candidate in candidates:
  for index in range(n):
   identity={**base,'purpose':'fresh-split-confirmation','seed':seed,'index':index,
             'candidate':candidate,'opponent':'T:T:0.0','floor':0.0}
   jobs.append((identity,str(out/'games'/candidate.replace(':','-')/f'{index:05d}.json')))
 rows=[]
 with get_context('spawn').Pool(30,initializer=initialize) as pool:
  for row in pool.imap_unordered(play,jobs,chunksize=1):
   rows.append(row)
   if len(rows)%64==0:print(f'{len(rows)}/1024 confirmation games complete',flush=True)
 groups={c:{r['identity']['index']:r for r in rows if r['identity']['candidate']==c} for c in candidates}
 summary=[]
 for c,g in groups.items():
  assert set(g)==set(range(n)) and all(sum(r['candidate_seat']==s for r in g.values())==128 for s in range(4))
  wins=sum(r['winner']==0 for r in g.values())
  summary.append(dict(candidate=c,wins=wins,games=n,rate=wins/n,wilson95=wilson(wins,n),
                      domestic_trades=sum(r['domestic_trades'] for r in g.values()),
                      candidate_trades=sum(r['candidate_trades'] for r in g.values())))
 d=[int(groups[candidates[0]][i]['winner']==0)-int(groups[candidates[1]][i]['winner']==0) for i in range(n)]
 mean=statistics.mean(d);half=1.959964*statistics.stdev(d)/math.sqrt(n)
 result=dict(manifest=base,seed=seed,rows=summary,seconds=time.perf_counter()-start,
             primary=dict(contrast='N:T minus T:T',paired_win_rate_difference=mean,interval95=[mean-half,mean+half],
                          discordant_games=sum(v!=0 for v in d),improvement_confirmed=mean-half>0))
 atomic(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
