"""Independently audit records, paired inference and activity diagnostics."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, variance, NormalDist
import tarfile

REGIMES=('none','moderate','heavy','full','mixed','rising','falling')
CURVES=('square','linear','sqrt')
CHALLENGERS=('square','sqrt')
SOURCE='d2df60dd0abeb72f1bd0ae4222716ce1b4246ac249c64cd2d76aeb0aa611b001'
BINS=((0,20),(20,40),(40,60),(60,80),(80,100),(100,20000))


def difference(ga,gb):
 assert set(ga)==set(gb)
 assert all(ga[i]['seating']==gb[i]['seating'] and ga[i]['seed']==gb[i]['seed'] for i in ga)
 d=[int(ga[i]['winner']==0)-int(gb[i]['winner']==0) for i in sorted(ga)]
 estimate=mean(d);se=math.sqrt(variance(d)/len(d));z=NormalDist().inv_cdf(.975)
 return dict(difference=estimate,variance=variance(d),se=se,interval95=[estimate-z*se,estimate+z*se])


def wilson(w,n):
 z=NormalDist().inv_cdf(.975);p=w/n;denom=1+z*z/n
 center=(p+z*z/(2*n))/denom;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
 return [max(0.,center-half),min(1.,center+half)]


def audit_record(row):
 i=row['identity'];assert row['complete'] and row['winner'] in range(4)
 assert i['host']=='hexset' and i['source_sha256']==SOURCE and not i['antithetic']
 p=i['parameters'];assert p['trade_floor']==0 and p['max_trades'] is None
 assert (p['depth'],p['width'],p['max_nodes'],p['k'])==(2,6,600,1)
 assert row['candidate_seat']==row['seating'][0]
 trades=row['trades'];seat=row['candidate_seat']
 assert len(trades)==row['domestic_trades'] and all(t['gain_a']>0 and t['gain_b']>0 for t in trades)
 assert row['candidate_trades']==sum(t['a']==seat or t['b']==seat for t in trades)
 events=row['public_events'];trajectory=row['trajectory']
 assert trajectory and all(a['turn']<b['turn'] for a,b in zip(trajectory,trajectory[1:]))
 assert all(set(e)=={'turn','actor','hand_sizes','trade_participants'} for e in events)
 assert [(e['turn'],a,b) for e in events for a,b in e['trade_participants']]==[(t['turn'],t['a'],t['b']) for t in trades]
 history=deque([0]*8,maxlen=8);j=0;last=-1
 for state in trajectory:
  while j<len(events) and events[j]['turn']<state['turn']:
   e=events[j];j+=1
   assert set(e)=={'turn','actor','hand_sizes','trade_participants'}
   if e['turn']<=last:continue
   last=e['turn'];actor=e['actor'];sizes=e['hand_sizes']
   if sizes[actor]>=2 and any(n>=2 for s,n in enumerate(sizes) if s!=actor):
    history.append(int(any(actor in pair for pair in e['trade_participants'])))
  activity=sum(history)/8
  assert state['activity']==activity
  expected=activity*activity if i['candidate']=='square' else math.sqrt(activity) if i['candidate']=='sqrt' else activity
  assert state['alpha']==expected
 values=[s['alpha'] for s in trajectory]
 steps=[b-a for a,b in zip(values,values[1:])]
 signs=[1 if d>0 else -1 for d in steps if d]
 bins={f'{lo}-{hi-1}':[s['alpha'] for s in trajectory if lo<=s['turn']<hi] for lo,hi in BINS}
 return dict(seed=i['seed'],seating=row['seating'],winner=row['winner'],seat=seat,
             action_sha256=row['action_sha256'],
             domestic_trades=len(trades),candidate_trades=row['candidate_trades'],
             bins={k:mean(v) for k,v in bins.items() if v},mean_alpha=mean(values),
             mean_activity=mean(s['activity'] for s in trajectory),
             final_alpha=values[-1],mean_step=mean(map(abs,steps)) if steps else 0,
             reversals=sum(a!=b for a,b in zip(signs,signs[1:])),decisions=len(values),
             fraction_alpha_zero=mean(v==0 for v in values),fraction_alpha_one=mean(v==1 for v in values))


def analyze(archive,stage,screen=None):
 groups={};identities={};count=0;preflight={}
 here=Path(__file__).parent
 runner_hash=hashlib.sha256((here/'run.py').read_bytes()).hexdigest()
 plan_hash=hashlib.sha256((here/'PLAN.md').read_bytes()).hexdigest()
 with tarfile.open(archive) as t:
  summary=json.load(t.extractfile(stage+'/summary.json'))
  assert summary['stage']==stage
  manifest=summary['manifest']
  assert manifest['source_sha256']==SOURCE
  assert manifest['runner_sha256']==runner_hash and manifest['plan_sha256']==plan_hash
  assert manifest['protocol']=='adaptive-slider-shape-v1'
  for m in t:
   if not m.isfile() or not m.name.endswith('.json'):continue
   if m.name.startswith(stage+'/preflight/'):
    r=json.load(t.extractfile(m));i=r['identity']
    assert i['purpose']=='preflight' and i['seed']==726900000
    audit_record(r);preflight[i['regime'],i['candidate'],i['index']]=r
    continue
   if not m.name.startswith(stage+'/games/'):continue
   row=json.load(t.extractfile(m));i=row['identity'];key=(i['regime'],i['candidate'])
   assert all(i[k]==v for k,v in manifest.items())
   assert i['purpose']==stage
   assert i['seed']==(726000000 if stage=='screen' else 727000000)+10000*REGIMES.index(i['regime'])
   g=groups.setdefault(key,{});assert i['index'] not in g
   signature=json.dumps({k:v for k,v in i.items() if k!='index'},sort_keys=True)
   assert identities.setdefault(key,signature)==signature
   g[i['index']]=audit_record(row);count+=1
 chosen=summary['selection']
 if stage=='screen':
  candidates=CURVES;n=64
  expected={(regime,c,index) for regime,cs in [('none',CURVES),('full',('linear','linear-probe'))] for c in cs for index in range(2)}
  assert set(preflight)==expected
  for regime,cs in [('none',CURVES),('full',('linear','linear-probe'))]:
   for index in range(2):
    reference=preflight[regime,'linear',index]
    for c in cs:
     assert all(preflight[regime,c,index][k]==reference[k] for k in ('winner','seating','points','turns','actions','action_sha256','trades'))
 else:
  assert screen is not None and chosen==screen['selection'] and chosen['advance']
  candidates=('linear',chosen['challenger']);n=chosen['confirmation_n']
 assert set(groups)=={(g,c) for g in REGIMES for c in candidates}
 for g in groups.values():
  assert len(g)==n and set(g)==set(range(n))
  assert all(sum(r['seat']==s for r in g.values())==n//4 for s in range(4))
 assert count==summary['games']==n*len(REGIMES)*len(candidates)
 for c in candidates:
  assert all(groups['none',c][i]['action_sha256']==groups['none','linear'][i]['action_sha256'] for i in range(n))
  assert all(r['domestic_trades']==0 and r['mean_alpha']==0 for r in groups['none',c].values())
 rows=[]
 for (regime,c),g in sorted(groups.items()):
  wins=sum(r['winner']==0 for r in g.values())
  old=next(r for r in summary['rows'] if r['regime']==regime and r['candidate']==c)
  assert old['wins']==wins and old['games']==n
  assert old['domestic_trades']==sum(r['domestic_trades'] for r in g.values())
  rows.append(dict(regime=regime,candidate=c,wins=wins,games=n,rate=wins/n,wilson95=wilson(wins,n),
    mean_trades=old['domestic_trades']/n,mean_candidate_trades=mean(r['candidate_trades'] for r in g.values())))
 primary={};per_regime=[]
 for c in candidates:
  if c=='linear':continue
  contrasts=[]
  for regime in REGIMES:
   d=difference(groups[regime,c],groups[regime,'linear'])
   per_regime.append(dict(candidate=c,regime=regime,**d));contrasts.append(d)
  estimate=mean(d['difference'] for d in contrasts)
  se=math.sqrt(sum(d['se']**2 for d in contrasts))/len(REGIMES)
  z=NormalDist().inv_cdf(.975)
  primary[c]=dict(difference=estimate,se=se,interval95=[estimate-z*se,estimate+z*se],
    lower80=estimate-NormalDist().inv_cdf(.8)*se,lower95=estimate-NormalDist().inv_cdf(.95)*se,
    passes_meaningful_margin=estimate-NormalDist().inv_cdf(.95)*se>.01,
    challenger_wins=sum(r['winner']==0 for g in REGIMES for r in groups[g,c].values()),
    linear_wins=sum(r['winner']==0 for g in REGIMES for r in groups[g,'linear'].values()),
    games_per_policy=n*len(REGIMES))
 if stage=='screen':
  best=min(CHALLENGERS,key=lambda c:(-primary[c]['challenger_wins'],CHALLENGERS.index(c)))
  d=primary[best];advance=d['difference']>=.01 and d['lower80']>0
  assert chosen['advance']==advance and chosen['challenger']==(best if advance else None)
  for c in CHALLENGERS:
   for k in ('difference','se','lower80'):assert abs(chosen['screen_contrasts'][c][k]-primary[c][k])<1e-12
  if advance:
   v=mean(r['variance'] for r in per_regime if r['candidate']==best)
   target=(NormalDist().inv_cdf(.95)+NormalDist().inv_cdf(.8))**2*v/(7*.02**2)
   assert abs(chosen['estimated_variance']-v)<1e-12 and abs(chosen['uncapped_target_per_regime']-target)<1e-9
   assert chosen['confirmation_n']==min(768,max(256,64*math.ceil(target/64)))
   assert chosen['power_cap_applied']==(target>768)
  else:assert chosen['confirmation_n']==0 and chosen['screen_best']==best
 else:
  d=primary[chosen['challenger']]
  for k in ('difference','se','lower95'):assert abs(summary['primary'][k]-d[k])<1e-12
  assert summary['meaningful_improvement_confirmed']==d['passes_meaningful_margin']
 diagnostics=[]
 for regime in REGIMES:
  for c in candidates:
   g=groups[regime,c];bins={}
   for lo,hi in BINS:
    key=f'{lo}-{hi-1}';values=[r['bins'][key] for r in g.values() if key in r['bins']]
    bins[key]=dict(games=len(values),mean_alpha=mean(values) if values else None)
   paired_change=[r['bins']['60-79']-r['bins']['20-39'] for r in g.values() if '60-79' in r['bins'] and '20-39' in r['bins']]
   diagnostics.append(dict(regime=regime,candidate=c,bins=bins,
    mean_activity=mean(r['mean_activity'] for r in g.values()),mean_alpha=mean(r['mean_alpha'] for r in g.values()),
    mean_final_alpha=mean(r['final_alpha'] for r in g.values()),mean_absolute_step=mean(r['mean_step'] for r in g.values()),
    mean_reversals=mean(r['reversals'] for r in g.values()),
    fraction_alpha_zero=mean(r['fraction_alpha_zero'] for r in g.values()),
    fraction_alpha_one=mean(r['fraction_alpha_one'] for r in g.values()),
    paired_pre_post_change=mean(paired_change) if paired_change else None,paired_pre_post_games=len(paired_change)))
 return dict(stage=stage,archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
  records_audited=count,preflight_records_audited=len(preflight),
  public_trajectory_replay='all raw activities and mapped values match exactly',selection=chosen,
  primary=primary,per_regime=per_regime,rows=rows,diagnostics=diagnostics,seconds=summary['seconds'])


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--stage',required=True,choices=['screen','confirmation'])
 p.add_argument('--out',type=Path,required=True);p.add_argument('--screen',type=Path)
 a=p.parse_args();result=analyze(a.archive,a.stage,json.loads(a.screen.read_text()) if a.screen else None)
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','diagnostics','per_regime')},indent=2))
