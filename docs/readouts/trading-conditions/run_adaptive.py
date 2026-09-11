"""Native 2x2 evaluator screen and trace-checked correction of trade censuses."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys
import time
import numpy as np
import hexset
import hexset.arena as arena
import hexset.game as engine
from hexset.bots.heximax import heximax, NO_TRADE_WEIGHTS, TRADING_WEIGHTS
from hexset.bots.heximax.search import Heximax
from hexset.bots.evaluate import Weights, TERM_NAMES
from activity import TradeActivity
from hexset.bots.stances import WIN_TEMPERATURE

PARAMS=dict(mode='honest',max_trades=None,depth=2,width=6,max_nodes=600,k=1,
            stance='win',temperature=WIN_TEMPERATURE,placement=True)
PROFILES={
 'N':dict(weights=asdict(NO_TRADE_WEIGHTS),expansion_value=.25),
 'O':dict(weights=asdict(replace(NO_TRADE_WEIGHTS,road=.1237)),expansion_value=0.0),
 'T':dict(weights=asdict(TRADING_WEIGHTS),expansion_value=0.0),
}
PROFILES['M']=dict(weights={k:(PROFILES['N']['weights'][k]+PROFILES['T']['weights'][k])/2
                            for k in PROFILES['N']['weights']},expansion_value=.125)
PROFILES['H']=dict(weights={**PROFILES['N']['weights'],'production':7.067,'scarce':.91*7.067/36},expansion_value=.25)
REGIMES={
 'none': [0.0,0.0,0.0], 'quiet':[.15,.15,.15], 'moderate':[.4,.4,.4],
 'active':[.75,.75,.75], 'full':[1.0,1.0,1.0], 'mixed':[0.0,.3,.9],
 'rising':[.1,.1,.1],
}
REGIME='full'
SPAWN_INDEX=0
SOURCE='0254eb3dbe0d14bea1b2d9eb42ec18a553d38c2f9100dd3477edb62347778c3e'


def source_hash():
 root=Path(hexset.__file__).parent;h=hashlib.sha256()
 for p in sorted(root.rglob('*.py')):h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
 return h.hexdigest()


def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');tmp.replace(path)


def bot_for(profile,board,rng,floor):
 spec=PROFILES[profile]
 bot=heximax(board,rng,**PARAMS,weights=Weights(**spec['weights']),expansion_value=spec['expansion_value'])
 bot.trade_floor=floor
 assert {k:getattr(bot,k) for k in PARAMS}==PARAMS
 assert asdict(bot.evaluator.weights)==spec['weights'] and bot.expansion_value==spec['expansion_value']
 return bot


class SplitBot:
 def __init__(self,move,gate,floor):
  self.move=move;self.gate=gate;self.trade_floor=floor;self.max_trades=None
 def choose(self,game):return self.move.choose(game)
 def gains_many(self,view,received,counterparties):
  if self.gate is not self.move:
   # A dedicated gate never calls choose: keep its caches scoped to this
   # live valuation batch rather than retaining mutable views all game.
   e=self.gate.evaluator
   for name in ['_walk_cache','_expansion_cache','_belief_cache','_evaluate_cache']:
    getattr(e,name).clear()
  return self.gate.gains_many(view,received,counterparties)


class WillingBot(SplitBot):
 def __init__(self,move,gate,floor,role,seed):
  super().__init__(move,gate,floor);self.role=role;self.seed=seed;self.game=None
  self.activity=TradeActivity();self.rule=None;self.alpha_trace=[]
 def choose(self,game):
  self.game=game
  if self.role<0:
   alpha=self.activity.mean
   if self.rule in ('A','P'):
    values=adaptive_values(self.rule,alpha)
    e=self.move.evaluator;weights=Weights(**values['weights'])
    vector=tuple(getattr(weights,k) for k in TERM_NAMES)
    e.weights=e.inner.weights=weights;e.vector=e.inner.vector=vector;e.expansion_value=values['expansion_value']
   if not self.alpha_trace or self.alpha_trace[-1]['turn']!=game.turns:
    self.alpha_trace.append(dict(turn=game.turns,activity=alpha))
  return super().choose(game)
 def gains_many(self,view,received,counterparties):
  if self.role>=0:
   assert self.game is not None
   rate=REGIMES[REGIME][self.role]
   if REGIME=='rising' and self.game.turns>=48:rate=.8
   active=self.game.current_player==view.perspective
   key=f'{self.seed}:{self.game.turns}:{int(active)}'.encode()
   u=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big')/2**64
   if u>=rate:return [-1.0]*len(received)
  return super().gains_many(view,received,counterparties)


def adaptive_values(rule,alpha):
 assert 0<=alpha<=1
 if rule=='A':
  weights={k:(1-alpha)*PROFILES['N']['weights'][k]+alpha*PROFILES['T']['weights'][k] for k in PROFILES['N']['weights']}
  return dict(weights=weights,expansion_value=.25*(1-alpha))
 if rule=='P':
  production=(1-alpha)*2.785+alpha*7.067
  return dict(weights={**PROFILES['N']['weights'],'production':production,'scarce':.91*production/36},expansion_value=.25)
 raise ValueError(rule)


def factory(entrant,board,rng):
 global SPAWN_INDEX
 role=SPAWN_INDEX-1;SPAWN_INDEX+=1
 move,gate,floor=entrant.name.split(':');floor=float(floor)
 seed=hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
 m=bot_for('N' if move in ('A','P') else move,board,rng,floor)
 g=m if move==gate else bot_for(gate,board,random.Random(0),floor)
 bot=WillingBot(m,g,floor,role,seed);bot.rule=move;return bot


def initialize():arena.register_entrant_kind('native-split-profile',factory)


def play(job):
 global REGIME,SPAWN_INDEX
 identity,path=job;REGIME=identity['regime'];SPAWN_INDEX=0;path=Path(path)
 if path.exists():
  row=json.loads(path.read_text());assert row['identity']==identity and row['complete'];return row
 random.seed(identity['seed']+identity['index'])
 ent=lambda name:arena.Entrant(name,kind='native-split-profile')
 lineup=(ent(identity['candidate']),)+(ent(identity['opponent']),)*3
 trace=hashlib.sha256();actions=0;calls=[0]*4;live=[];trades=[];observations=[]
 original_choose=Heximax.choose;original_gains=Heximax.gains_many
 original_play=arena.play_game;original_event=engine.trade_event
 def choose(bot,game):
  nonlocal actions
  assert type(game) is engine.Game and game.max_trades is None and bot.max_trades is None
  assert bot.trade_floor==identity['floor']
  action=original_choose(bot,game);trace.update(f'{engine.to_move(game)}:{action!r}\n'.encode());actions+=1
  return action
 def gains(bot,view,received,counterparties):
  assert bot.max_trades is None and bot.trade_floor==identity['floor']
  calls[view.perspective]+=1;return original_gains(bot,view,received,counterparties)
 def run_game(game,bots,**kwargs):
  assert all(b.max_trades is None and b.trade_floor==identity['floor'] for b in bots)
  live.append(game);return original_play(game,bots,**kwargs)
 def event(game,gate):
  real=bool(live and game is live[0])
  # Hand sizes are public. Neither card identities nor private gains enter
  # the adaptive observer, which receives counts and trade participants only.
  sizes=tuple(sum(hand) for hand in game._state.hands) if real else ()
  result=original_event(game,gate)
  if live and game is live[0]:
   trades.extend(dict(turn=game.turns,**t._asdict()) for t in result)
   public=dict(turn=game.turns,actor=game.current_player,hand_sizes=sizes,
               trade_participants=[(t.a,t.b) for t in result])
   observations.append(public)
   for bot in game.gates:
    if bot.role<0:bot.activity.observe(**public)
  return result
 Heximax.choose=choose;Heximax.gains_many=gains;arena.play_game=run_game;engine.trade_event=event
 started=time.perf_counter()
 try:outcome=arena._play_one((lineup,identity['index'],identity['seed'],20000,False,False))
 finally:
  Heximax.choose=original_choose;Heximax.gains_many=original_gains;arena.play_game=original_play;engine.trade_event=original_event
 assert outcome.winner is not None and len(live)==1 
 assert all(t['gain_a']>identity['floor'] and t['gain_b']>identity['floor'] for t in trades)
 seat=outcome.seating.index(0)
 row=dict(identity=identity,complete=True,winner=outcome.winner,seating=outcome.seating,candidate_seat=seat,
          turns=outcome.turns,points=outcome.points,actions=actions,action_sha256=trace.hexdigest(),
          seconds=time.perf_counter()-started,trade_gate_calls_by_seat=calls,
          domestic_trades=len(trades),trades=trades,
          candidate_trades=sum(t['a']==seat or t['b']==seat for t in trades),
          public_trade_events=observations,
          activity_trace=next(b.alpha_trace for b in live[0].gates if b.role<0),
          final_turn_trades=[t._asdict() for t in live[0].trades])
 atomic(path,row);return row


def base_identity():
 assert source_hash()==SOURCE
 return dict(protocol='native-trading-conditions-v1',host='hexset',information_model='native-public-history',
             source_sha256=SOURCE,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             runtime=dict(python=sys.version,numpy=np.__version__),parameters=PARAMS,profiles=PROFILES,
             antithetic=False,rule_set='standard',trade_census='all-live-trade-events',regimes=REGIMES,
             activity_observer_sha256=hashlib.sha256(Path(__file__).with_name('activity.py').read_bytes()).hexdigest(),
             adaptive_rules=dict(A='alpha=public activity mean; interpolate N to T move profile',P='alpha=public activity mean; interpolate production/scarcity only'),
             activity_prior=[1,3],activity_half_life=8,
             participation='stable private per-turn per-role draws; rising .1 to .8 at turn 48')


def verify_old(row,path):
 old=json.loads(path.read_text())
 for key in ['winner','seating','candidate_seat','turns','points','actions','action_sha256','trade_gate_calls_by_seat']:
  assert row[key]==old[key],(path,key)
 assert row['final_turn_trades']==old['trades']


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
 base=base_identity();initialize();start=time.perf_counter();n=64
 for regime,ri in [('none',0),('moderate',2),('full',4),('mixed',5)]:
  c='N:T:0.0';index=0
  identity={**base,'purpose':'observer-control-preflight','seed':720000000+ri*10000,'index':index,
            'candidate':c,'opponent':'T:T:0.0','floor':0.0,'regime':regime}
  path=out/'preflight'/f'{regime}.json';play((identity,str(path)));row=json.loads(path.read_text())
  old=json.loads((Path('/old/screen/games')/regime/c.replace(':','-')/f'{index:05d}.json').read_text())
  keys=['winner','seating','candidate_seat','turns','points','actions','action_sha256','trades','trade_gate_calls_by_seat']
  assert {k:row[k] for k in keys}=={k:old[k] for k in keys}
 print('Four observer controls match full actions and trades exactly.',flush=True)
 jobs=[];candidates=['N:T:0.0','M:T:0.0','T:T:0.0','H:T:0.0','A:T:0.0','P:T:0.0']
 for ri,regime in enumerate(REGIMES):
  for c in candidates:
   for index in range(n):
    identity={**base,'purpose':'adaptive-screen','seed':721000000+ri*10000,'index':index,
              'candidate':c,'opponent':'T:T:0.0','floor':0.0,'regime':regime}
    jobs.append((identity,str(out/'games'/regime/c.replace(':','-')/f'{index:05d}.json')))
 rows=[]
 with get_context('spawn').Pool(30,initializer=initialize) as pool:
  for row in pool.imap_unordered(play,jobs,chunksize=1):
   rows.append(row)
   if len(rows)%128==0:print(f'{len(rows)}/{len(jobs)} adaptive-screen games complete',flush=True)
 summary=[]
 for regime in REGIMES:
  for c in candidates:
   g=[r for r in rows if r['identity']['regime']==regime and r['identity']['candidate']==c]
   assert len(g)==n and {r['identity']['index'] for r in g}==set(range(n))
   assert all(sum(r['candidate_seat']==s for r in g)==n//4 for s in range(4))
   wins=sum(r['winner']==0 for r in g)
   summary.append(dict(regime=regime,candidate=c,games=n,wins=wins,rate=wins/n,wilson95=arena.wilson(wins,n),
                       domestic_trades=sum(r['domestic_trades'] for r in g),candidate_trades=sum(r['candidate_trades'] for r in g),
                       final_activity=sum(r['activity_trace'][-1]['activity'] for r in g)/n))
 atomic(out/'summary.json',dict(manifest=base,rows=summary,seconds=time.perf_counter()-start))
 print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
