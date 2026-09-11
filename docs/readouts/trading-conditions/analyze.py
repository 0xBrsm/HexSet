"""Validate completed checkpoint tar and compute stratified paired contrasts."""
import argparse
import json
import math
from pathlib import Path
import statistics
import tarfile

INTERIOR=('quiet','moderate','active','mixed','rising')


def records(path,prefix):
 with tarfile.open(path) as tar:
  rows=[json.load(tar.extractfile(m)) for m in tar if m.isfile() and m.name.startswith(prefix+'/games/') and m.name.endswith('.json')]
 groups={}
 for r in rows:
  i=r['identity'];key=(i['regime'],i['candidate']);group=groups.setdefault(key,{})
  assert r['complete'] and r['winner'] is not None and i['host']=='hexset' and i['floor']==0
  assert i['index'] not in group and len(r['trades'])==r['domestic_trades']
  assert all(t['gain_a']>0 and t['gain_b']>0 for t in r['trades'])
  # Outcome.seating maps entrant -> seat. Frozen runners used index(0),
  # the inverse, for descriptive candidate counts; wins use entrant IDs.
  r['reported_candidate_seat']=r['candidate_seat']
  r['candidate_seat']=r['seating'][0]
  r['candidate_trades']=sum(t['a']==r['candidate_seat'] or t['b']==r['candidate_seat'] for t in r['trades'])
  group[i['index']]=r
 for key,g in groups.items():
  n=len(g);assert n%4==0 and set(g)==set(range(n))
  assert all(sum(r['candidate_seat']==s for r in g.values())==n//4 for s in range(4))
  identities={json.dumps({k:v for k,v in r['identity'].items() if k!='index'},sort_keys=True) for r in g.values()}
  assert len(identities)==1
 return groups


def contrast(groups,a,b,regimes):
 means=[];variances=[]
 for regime in regimes:
  ga=groups[regime,a];gb=groups[regime,b];assert set(ga)==set(gb)
  assert all((ga[i]['identity']['seed'],ga[i]['seating'])==(gb[i]['identity']['seed'],gb[i]['seating']) for i in ga)
  d=[int(ga[i]['winner']==0)-int(gb[i]['winner']==0) for i in sorted(ga)]
  means.append(statistics.mean(d));variances.append(statistics.variance(d)/len(d))
 mean=statistics.mean(means);half=1.959964*math.sqrt(sum(variances))/len(means)
 return dict(a=a,b=b,regimes=list(regimes),difference=mean,interval95=[mean-half,mean+half])


def analyze(path,prefix,regimes=INTERIOR):
 groups=records(path,prefix);candidates=sorted({c for r,c in groups})
 ranking=[]
 for candidate in candidates:
  rows=[r for regime in regimes for r in groups[regime,candidate].values()]
  wins=sum(r['winner']==0 for r in rows)
  ranking.append(dict(candidate=candidate,wins=wins,games=len(rows),rate=wins/len(rows)))
 ranking.sort(key=lambda x:(-x['rate'],x['candidate']))
 contrasts=[contrast(groups,c,'T:T:0.0',regimes) for c in candidates if c!='T:T:0.0']
 return dict(ranking=ranking,contrasts_against_trading=contrasts)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--prefix',required=True);p.add_argument('--out',type=Path)
 a=p.parse_args();result=analyze(a.archive,a.prefix);text=json.dumps(result,indent=2)+'\n'
 if a.out:a.out.write_text(text)
 print(text)
