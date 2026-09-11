"""Frozen native adaptive-shape screen and independent fresh confirmation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import random
from statistics import mean, variance, NormalDist
import sys
import time
import numpy as np
import hexset
from hexset import arena
import hexset.game as engine
from hexset.bots.heximax import Heximax, TRADING_WEIGHTS, heximax
from hexset.bots.heximax.adaptive import TradeActivity, trading_profile
from hexset.bots.stances import WIN_TEMPERATURE

SOURCE='d2df60dd0abeb72f1bd0ae4222716ce1b4246ac249c64cd2d76aeb0aa611b001'
PARAMS=dict(mode='honest',max_trades=None,depth=2,width=6,max_nodes=600,k=1,
            stance='win',temperature=WIN_TEMPERATURE,placement=True,trade_floor=0.0)
REGIMES={
 'none':dict(rates=[0,0,0],moves=['0']*3),
 'moderate':dict(rates=[.3]*3,moves=['1']*3),
 'heavy':dict(rates=[.65]*3,moves=['1']*3),
 'full':dict(rates=[1]*3,moves=['1']*3),
 'mixed':dict(rates=[.05,.45,.85],moves=['0','1','T']),
 'rising':dict(rates=[.1]*3,moves=['0','1','T'],change=40,after=[.75]*3),
 'falling':dict(rates=[.75]*3,moves=['0','1','T'],change=40,after=[.1]*3),
}
CURVES=('square','linear','sqrt')
CHALLENGERS=('square','sqrt')
ROLE=0
REGIME='none'


def source_hash():
 root=Path(hexset.__file__).parent;h=hashlib.sha256()
 for p in sorted(root.rglob('*.py')):
  h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
 return h.hexdigest()


def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');tmp.replace(path)


def curve_value(profile, activity):
 if profile=='square':return activity*activity
 if profile=='sqrt':return math.sqrt(activity)
 if profile in ('linear','linear-probe'):return activity
 raise ValueError(profile)


class MappedActivity(TradeActivity):
 def __init__(self,profile):
  super().__init__();self.profile=profile
 @property
 def raw_mean(self):return super().mean
 @property
 def mean(self):return curve_value(self.profile,self.raw_mean)


def make(profile,board,rng):
 if profile in CURVES or profile=='linear-probe':
  bot=heximax(board,rng,**PARAMS,adaptive=True)
  if profile!='linear':bot.activity=MappedActivity(profile)
  return bot
 weights,expansion=(TRADING_WEIGHTS,0.) if profile=='T' else trading_profile(float(profile))
 return heximax(board,rng,**PARAMS,weights=weights,expansion_value=expansion,trade_weights=TRADING_WEIGHTS)


class ControlledBot:
 def __init__(self,bot,role,seed,profile):
  self.bot=bot;self.role=role;self.seed=seed;self.profile=profile;self.game=None
  self.max_trades=None;self.trade_floor=0.
  self.observer=bot.activity if role<0 else TradeActivity()
  self.events=[];self.trajectory=[]
 def choose(self,game):
  self.game=game
  alpha=self.observer.mean if self.role<0 else None
  activity=self.observer.raw_mean if isinstance(self.observer,MappedActivity) else self.observer.mean
  action=self.bot.choose(game)
  if self.role<0:
   expected,expansion=trading_profile(alpha)
   assert self.bot.evaluator.weights==expected and self.bot.expansion_value==expansion
   if not self.trajectory or self.trajectory[-1]['turn']!=game.turns:
    self.trajectory.append(dict(turn=game.turns,activity=activity,alpha=alpha))
  return action
 def gains_many(self,view,received,counterparties):
  if self.role>=0:
   assert self.game is not None
   regime=REGIMES[REGIME];rates=regime['rates']
   if 'change' in regime and self.game.turns>=regime['change']:rates=regime['after']
   active=self.game.current_player==view.perspective
   key=f'{self.seed}:{self.game.turns}:{int(active)}'.encode()
   u=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big')/2**64
   if u>=rates[self.role]:return [-1.]*len(received)
  return self.bot.gains_many(view,received,counterparties)
 def observe_trade(self,**public):
  if self.role<0:
   self.bot.observe_trade(**public)
   self.events.append(public)


def factory(entrant,board,rng):
 global ROLE
 role=ROLE-1;ROLE+=1
 profile=entrant.name if role<0 else REGIMES[REGIME]['moves'][role]
 seed=hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
 bot=make(profile,board,rng)
 assert bot.trade_floor==0 and bot.max_trades is None
 assert all(getattr(bot,k)==v for k,v in PARAMS.items())
 return ControlledBot(bot,role,seed,profile)


def initialize():arena.register_entrant_kind('slider-shape',factory)


def play(job):
 global ROLE,REGIME
 identity,path=job;ROLE=0;REGIME=identity['regime'];path=Path(path)
 if path.exists():
  row=json.loads(path.read_text());assert row['identity']==identity and row['complete'];return row
 trace=hashlib.sha256();actions=0;live=[];trades=[];focal=[]
 original_choose=Heximax.choose;original_play=arena.play_game;original_event=engine.trade_event
 def choose(bot,game):
  nonlocal actions
  assert type(game) is engine.Game and game.max_trades is None
  action=original_choose(bot,game)
  trace.update(f'{engine.to_move(game)}:{action!r}\n'.encode());actions+=1
  return action
 def seated(game,bots,**kwargs):
  live.append(game)
  assert all(b.max_trades is None and b.trade_floor==0 for b in bots)
  for b in bots:
   b.game=game
   if b.role<0:focal.append(b)
  return original_play(game,bots,**kwargs)
 def event(game,gate):
  result=original_event(game,gate)
  if live and game is live[0]:trades.extend(dict(turn=game.turns,**t._asdict()) for t in result)
  return result
 Heximax.choose=choose;arena.play_game=seated;engine.trade_event=event
 ent=lambda n:arena.Entrant(n,kind='slider-shape')
 started=time.perf_counter()
 try:
  outcome=arena._play_one(((ent(identity['candidate']),)+(ent('opponent'),)*3,identity['index'],identity['seed'],20000,False,False))
 finally:Heximax.choose=original_choose;arena.play_game=original_play;engine.trade_event=original_event
 assert len(live)==len(focal)==1 and outcome.winner is not None
 assert all(t['gain_a']>0 and t['gain_b']>0 for t in trades)
 seat=outcome.seating[0]
 row=dict(identity=identity,complete=True,winner=outcome.winner,seating=outcome.seating,candidate_seat=seat,
          turns=outcome.turns,points=outcome.points,actions=actions,action_sha256=trace.hexdigest(),
          seconds=time.perf_counter()-started,domestic_trades=len(trades),trades=trades,
          candidate_trades=sum(t['a']==seat or t['b']==seat for t in trades),
          public_events=focal[0].events,trajectory=focal[0].trajectory)
 atomic(path,row);return row


def identity_base():
 assert source_hash()==SOURCE
 return dict(protocol='adaptive-slider-shape-v1',host='hexset',source_sha256=SOURCE,
             runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             plan_sha256=hashlib.sha256(Path(__file__).with_name('PLAN.md').read_bytes()).hexdigest(),
             parameters=PARAMS,regimes=REGIMES,profiles={p:dict(weights=asdict(trading_profile(float(p))[0]),
             expansion=trading_profile(float(p))[1]) for p in ('0','1')},
             curves={'square':'activity*activity','linear':'activity','sqrt':'sqrt(activity)'},exchange_weights=asdict(TRADING_WEIGHTS),
             python=sys.version,numpy=np.__version__,information_model='native-public-history',
             antithetic=False,trade_census='all-live-trade-events')


def paired(groups,a,controls,z=None):
 if z is None:z=NormalDist().inv_cdf(.975)
 means=[];vs=[]
 for regime in REGIMES:
  ga=groups[regime,a];gb=groups[regime,controls[regime]]
  assert set(ga)==set(gb)
  assert all(ga[i]['seating']==gb[i]['seating'] and ga[i]['identity']['seed']==gb[i]['identity']['seed'] for i in ga)
  d=[int(ga[i]['winner']==0)-int(gb[i]['winner']==0) for i in sorted(ga)]
  means.append(mean(d));vs.append(variance(d)/len(d))
 estimate=mean(means);se=math.sqrt(sum(vs))/len(means)
 z95=NormalDist().inv_cdf(.975)
 return dict(difference=estimate,se=se,interval95=[estimate-z95*se,estimate+z95*se],lower=estimate-z*se)


def summarize(rows):
 groups={}
 for r in rows:
  i=r['identity'];g=groups.setdefault((i['regime'],i['candidate']),{});assert i['index'] not in g;g[i['index']]=r
 summaries=[]
 for (regime,c),g in sorted(groups.items()):
  n=len(g);assert set(g)==set(range(n)) and n%4==0
  assert all(sum(r['candidate_seat']==s for r in g.values())==n//4 for s in range(4))
  wins=sum(r['winner']==0 for r in g.values())
  summaries.append(dict(regime=regime,candidate=c,wins=wins,games=n,rate=wins/n,
                        wilson95=arena.wilson(wins,n),domestic_trades=sum(r['domestic_trades'] for r in g.values())))
 return groups,summaries


def selection(groups):
 controls={g:'linear' for g in REGIMES}
 winner=min(CHALLENGERS,key=lambda c:(-sum(r['winner']==0 for g in REGIMES for r in groups[g,c].values()),CHALLENGERS.index(c)))
 contrasts={c:paired(groups,c,controls,z=NormalDist().inv_cdf(.8)) for c in CHALLENGERS}
 for d in contrasts.values():d['lower80']=d.pop('lower')
 d=contrasts[winner]
 qualifies=d['difference']>=.01 and d['lower80']>0
 if not qualifies:
  return dict(challenger=None,advance=False,screen_best=winner,screen_contrasts=contrasts,confirmation_n=0)
 vs=[]
 for g in REGIMES:
  diffs=[int(groups[g,winner][i]['winner']==0)-int(groups[g,'linear'][i]['winner']==0) for i in groups[g,winner]]
  vs.append(variance(diffs))
 v=mean(vs);target=(NormalDist().inv_cdf(.95)+NormalDist().inv_cdf(.8))**2*v/(7*.02**2)
 n=min(768,max(256,64*math.ceil(target/64)))
 return dict(challenger=winner,advance=True,screen_contrasts=contrasts,confirmation_n=n,
             estimated_variance=v,uncapped_target_per_regime=target,power_cap_applied=target>768)


def synthetic_controls():
 for profile in (*CURVES,'linear-probe'):
  obs=MappedActivity(profile)
  assert obs.raw_mean==obs.mean==0
  for n in range(1,9):
   obs.observe(turn=n,actor=0,hand_sizes=(2,2,0,0),trade_participants=((0,1),))
   assert obs.raw_mean==n/8 and obs.mean==curve_value(profile,n/8)
  assert obs.mean==1
  obs.observe(turn=9,actor=0,hand_sizes=(1,2,0,0),trade_participants=())
  assert obs.mean==1
  for n in range(10,18):obs.observe(turn=n,actor=0,hand_sizes=(2,2,0,0),trade_participants=())
  assert obs.raw_mean==obs.mean==0
 print('Synthetic window and mapping controls passed.',flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirmation'],required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path)
 args=p.parse_args();base=identity_base();out=args.out;initialize();start=time.perf_counter()
 chosen=json.loads(args.selection.read_text())['selection'] if args.selection else None
 def job(regime,c,index,seed,purpose):
  identity={**base,'regime':regime,'candidate':c,'index':index,'seed':seed,'purpose':purpose}
  return identity,str(out/'games'/regime/c/f'{index:05d}.json')
 if args.stage=='screen':
  synthetic_controls()
  for regime,candidates in [('none',CURVES),('full',('linear','linear-probe'))]:
   for index in range(2):
    results=[]
    for c in candidates:
     identity,path=job(regime,c,index,726900000,'preflight')
     results.append(play((identity,str(out/'preflight'/regime/c/f'{index:05d}.json'))))
    for k in ('winner','seating','points','turns','actions','action_sha256','trades'):
     assert all(r[k]==results[0][k] for r in results),(regime,index,k)
  print('Ten behavior control games matched full actions and trade histories.',flush=True)
 else:assert chosen is not None and chosen['advance']
 jobs=[]
 for ri,regime in enumerate(REGIMES):
  candidates=CURVES if args.stage=='screen' else ('linear',chosen['challenger'])
  n=64 if args.stage=='screen' else chosen['confirmation_n']
  seed=(726000000 if args.stage=='screen' else 727000000)+ri*10000
  # Interleave matched policies so completion progress does not favor an arm.
  for index in range(n):
   for c in candidates:jobs.append(job(regime,c,index,seed,args.stage))
 rows=[]
 with get_context('spawn').Pool(30,initializer=initialize) as pool:
  for row in pool.imap_unordered(play,jobs,chunksize=1):
   rows.append(row)
   if len(rows)%128==0:print(f'{len(rows)}/{len(jobs)} {args.stage} games complete',flush=True)
 groups,summary=summarize(rows)
 for c in set(r['identity']['candidate'] for r in rows):
  assert all(groups['none',c][i]['action_sha256']==groups['none','linear'][i]['action_sha256'] for i in range(n))
 result=dict(manifest=base,stage=args.stage,rows=summary,games=len(rows),seconds=time.perf_counter()-start)
 if args.stage=='screen':result['selection']=selection(groups)
 else:
  result['selection']=chosen
  result['primary']=paired(groups,chosen['challenger'],{g:'linear' for g in REGIMES})
  result['primary'].pop('lower')
  primary=result['primary'];primary['lower95']=primary['difference']-NormalDist().inv_cdf(.95)*primary['se']
  result['meaningful_improvement_confirmed']=primary['lower95']>.01
 atomic(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('manifest','rows')}),flush=True)


if __name__=='__main__':main()
