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
FIXED=('0','.25','.5','.75','1')
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
  assert state['alpha']==(activity if i['candidate']=='A' else float(i['candidate']))
 values=[s['alpha'] for s in trajectory]
 steps=[b-a for a,b in zip(values,values[1:])]
 signs=[1 if d>0 else -1 for d in steps if d]
 bins={f'{lo}-{hi-1}':[s['alpha'] for s in trajectory if lo<=s['turn']<hi] for lo,hi in BINS}
 return dict(seed=i['seed'],seating=row['seating'],winner=row['winner'],seat=seat,
             domestic_trades=len(trades),candidate_trades=row['candidate_trades'],
             bins={k:mean(v) for k,v in bins.items() if v},mean_alpha=mean(values),
             final_alpha=values[-1],mean_step=mean(map(abs,steps)) if steps else 0,
             reversals=sum(a!=b for a,b in zip(signs,signs[1:])),decisions=len(values),
             fraction_alpha_zero=mean(v==0 for v in values),fraction_alpha_one=mean(v==1 for v in values))


def analyze(archive,stage):
 groups={};identities={};count=0
 with tarfile.open(archive) as t:
  for m in t:
   if not m.isfile() or not m.name.startswith(stage+'/games/') or not m.name.endswith('.json'):continue
   row=json.load(t.extractfile(m));i=row['identity'];key=(i['regime'],i['candidate'])
   g=groups.setdefault(key,{});assert i['index'] not in g
   signature=json.dumps({k:v for k,v in i.items() if k!='index'},sort_keys=True)
   assert identities.setdefault(key,signature)==signature
   assert i['seed']==(724000000 if stage=='screen' else 725000000)+10000*REGIMES.index(i['regime'])
   g[i['index']]=audit_record(row);count+=1
  summary=json.load(t.extractfile(stage+'/summary.json'))
 n=64 if stage=='screen' else summary['selection']['confirmation_n']
 for g in groups.values():
  assert len(g)==n and set(g)==set(range(n))
  assert all(sum(r['seat']==s for r in g.values())==n//4 for s in range(4))
 assert count==summary['games']
 def choose(regimes):
  return min(FIXED,key=lambda p:(-sum(r['winner']==0 for g in regimes for r in groups[g,p].values()),abs(float(p)-.5),float(p)))
 chosen=summary['selection']
 if stage=='screen':
  assert chosen['global_best']==choose(REGIMES)
  assert chosen['conditional_best']=={g:choose([g]) for g in REGIMES}
  assert len(groups)==len(REGIMES)*6
 else:
  assert set(groups)=={(g,c) for g in REGIMES for c in ('A',chosen['global_best'],chosen['conditional_best'][g])}
 rows=[]
 for (regime,c),g in sorted(groups.items()):
  wins=sum(r['winner']==0 for r in g.values())
  old=next(r for r in summary['rows'] if r['regime']==regime and r['candidate']==c)
  assert old['wins']==wins and old['domestic_trades']==sum(r['domestic_trades'] for r in g.values())
  rows.append(dict(regime=regime,candidate=c,wins=wins,games=n,rate=wins/n,wilson95=wilson(wins,n),
                   mean_trades=old['domestic_trades']/n,mean_candidate_trades=mean(r['candidate_trades'] for r in g.values())))
 primary={};per_regime=[]
 for name,controls in [('global',{g:chosen['global_best'] for g in REGIMES}),('conditional',chosen['conditional_best'])]:
  contrasts=[]
  for regime in REGIMES:
   d=difference(groups[regime,'A'],groups[regime,controls[regime]])
   per_regime.append(dict(comparator=name,regime=regime,control=controls[regime],**d));contrasts.append(d)
  estimate=mean(d['difference'] for d in contrasts)
  se=math.sqrt(sum(d['se']**2 for d in contrasts))/len(REGIMES);z=NormalDist().inv_cdf(.975)
  primary[name]=dict(difference=estimate,se=se,interval95=[estimate-z*se,estimate+z*se],
                     lower975=estimate-z*se,passes_margin=estimate-z*se>-.02,
                     adaptive_wins=sum(r['winner']==0 for g in REGIMES for r in groups[g,'A'].values()),
                     control_wins=sum(r['winner']==0 for g in REGIMES for r in groups[g,controls[g]].values()),
                     games_per_policy=n*len(REGIMES))
  if stage=='confirmation':
   old=summary['primary'][name]
   assert abs(old['difference']-estimate)<1e-12 and abs(old['se']-se)<1e-12
   assert abs(old['lower975']-primary[name]['lower975'])<1e-8
 if stage=='screen':
  v=max(mean(r['variance'] for r in per_regime if r['comparator']==name) for name in ('global','conditional'))
  target=(NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8))**2*v/(7*.02**2)
  expected_n=min(1024,max(256,64*math.ceil(target/64)))
  assert abs(chosen['estimated_variance']-v)<1e-12 and chosen['confirmation_n']==expected_n
  assert abs(chosen['uncapped_target_per_regime']-target)<1e-9
 if stage=='confirmation':assert summary['noninferiority_confirmed']==all(d['passes_margin'] for d in primary.values())
 diagnostics=[]
 for regime in REGIMES:
  g=groups[regime,'A'];bins={}
  for lo,hi in BINS:
   key=f'{lo}-{hi-1}';values=[r['bins'][key] for r in g.values() if key in r['bins']]
   bins[key]=dict(games=len(values),mean_alpha=mean(values) if values else None)
  paired_change=[r['bins']['60-79']-r['bins']['20-39'] for r in g.values() if '60-79' in r['bins'] and '20-39' in r['bins']]
  diagnostics.append(dict(regime=regime,bins=bins,mean_alpha=mean(r['mean_alpha'] for r in g.values()),
    mean_final_alpha=mean(r['final_alpha'] for r in g.values()),mean_absolute_step=mean(r['mean_step'] for r in g.values()),
    mean_reversals=mean(r['reversals'] for r in g.values()),
    fraction_alpha_zero=mean(r['fraction_alpha_zero'] for r in g.values()),
    fraction_alpha_one=mean(r['fraction_alpha_one'] for r in g.values()),
    paired_pre_post_change=mean(paired_change) if paired_change else None,paired_pre_post_games=len(paired_change)))
 return dict(stage=stage,archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
             records_audited=count,public_trajectory_replay='all values match exactly',selection=chosen,
             primary=primary,per_regime=per_regime,rows=rows,diagnostics=diagnostics,seconds=summary['seconds'])


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--stage',required=True,choices=['screen','confirmation']);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();result=analyze(a.archive,a.stage);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','diagnostics','per_regime')},indent=2))
