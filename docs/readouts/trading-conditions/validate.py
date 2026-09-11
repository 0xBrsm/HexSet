"""Frozen out-of-sample condition validation: moderate population dominates."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import statistics
from statistics import NormalDist
import time
import run_adaptive as r

REGIMES={
 'quiet':[.1,.1,.1], 'moderate':[.3,.3,.3], 'active':[.65,.65,.65],
 'mixed':[.05,.45,.85], 'rising':[.1,.1,.1], 'falling':[.75,.75,.75],
 'none':[0.,0.,0.], 'full':[1.,1.,1.],
}
CHANGES={'rising':dict(turn=40,after=[.75,.75,.75]),'falling':dict(turn=40,after=[.1,.1,.1])}
MOVES={k:['T','T','T'] for k in REGIMES}
MOVES.update(mixed=['N','M','T'],rising=['N','T','M'],falling=['M','N','T'],none=['O','O','O'])
PRIMARY=('quiet','moderate','active','mixed','rising','falling')
ORIGINAL_FACTORY=r.factory


def willing_gains(self,view,received,counterparties):
 if self.role>=0:
  assert self.game is not None
  rates=REGIMES[r.REGIME];change=CHANGES.get(r.REGIME)
  if change and self.game.turns>=change['turn']:rates=change['after']
  rate=rates[self.role];active=self.game.current_player==view.perspective
  key=f'{self.seed}:{self.game.turns}:{int(active)}'.encode()
  u=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big')/2**64
  if u>=rate:return [-1.0]*len(received)
 return r.SplitBot.gains_many(self,view,received,counterparties)


def factory(entrant,board,rng):
 role=r.SPAWN_INDEX-1
 if role>=0:entrant=replace(entrant,name=f'{MOVES[r.REGIME][role]}:T:0.0')
 return ORIGINAL_FACTORY(entrant,board,rng)


r.REGIMES=REGIMES;r.factory=factory;r.WillingBot.gains_many=willing_gains


def interval(groups,a,b,regimes,alpha=.025):
 means=[];variances=[]
 for regime in regimes:
  ga=groups[regime,a];gb=groups[regime,b];assert set(ga)==set(gb)
  d=[int(ga[i]['winner']==0)-int(gb[i]['winner']==0) for i in ga]
  means.append(statistics.mean(d));variances.append(statistics.variance(d)/len(d))
 mean=statistics.mean(means);half=NormalDist().inv_cdf(1-alpha/2)*math.sqrt(sum(variances))/len(means)
 return dict(a=a,b=b,regimes=list(regimes),alpha=alpha,difference=mean,interval=[mean-half,mean+half])


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
 base={**r.base_identity(),'validation_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'opponent_move_profiles':MOVES,'regime_changes':CHANGES,'primary_regimes':PRIMARY,
       'participation':'private stable per-turn role draws; change rules explicitly recorded'}
 jobs=[];started=time.perf_counter()
 for ri,regime in enumerate(REGIMES):
  n=256 if regime in PRIMARY else 128
  candidates=['M:T:0.0','A:T:0.0','T:T:0.0']+(['N:T:0.0'] if regime not in PRIMARY else [])
  for c in candidates:
   for index in range(n):
    identity={**base,'purpose':'fresh-condition-validation','seed':722000000+ri*10000,'index':index,
              'candidate':c,'opponent':'configured-move-profiles / T gates','floor':0.,'regime':regime}
    jobs.append((identity,str(out/'games'/regime/c.replace(':','-')/f'{index:05d}.json')))
 rows=[]
 with get_context('spawn').Pool(30,initializer=r.initialize) as pool:
  for row in pool.imap_unordered(r.play,jobs,chunksize=1):
   rows.append(row)
   if len(rows)%128==0:print(f'{len(rows)}/{len(jobs)} fresh validation games complete',flush=True)
 groups={};summary=[]
 for row in rows:
  i=row['identity'];groups.setdefault((i['regime'],i['candidate']),{})[i['index']]=row
 for (regime,c),g in sorted(groups.items()):
  n=256 if regime in PRIMARY else 128
  assert len(g)==n and set(g)==set(range(n))
  assert all(sum(x['candidate_seat']==s for x in g.values())==n//4 for s in range(4))
  wins=sum(x['winner']==0 for x in g.values())
  summary.append(dict(regime=regime,candidate=c,wins=wins,games=n,rate=wins/n,wilson95=r.arena.wilson(wins,n),
                      domestic_trades=sum(x['domestic_trades'] for x in g.values()),
                      candidate_trades=sum(x['candidate_trades'] for x in g.values())))
 contrasts=[interval(groups,'M:T:0.0','T:T:0.0',PRIMARY),interval(groups,'A:T:0.0','M:T:0.0',PRIMARY)]
 result=dict(manifest=base,rows=summary,primary=contrasts,seconds=time.perf_counter()-started,
             fixed_improvement_confirmed=contrasts[0]['interval'][0]>0,
             adaptive_improvement_confirmed=contrasts[1]['interval'][0]>0)
 r.atomic(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
