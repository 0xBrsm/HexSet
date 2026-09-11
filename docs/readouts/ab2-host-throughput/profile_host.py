"""Profile the existing host benchmark, retaining native/adapter call costs."""
import argparse
import cProfile
from contextlib import contextmanager
import json
from pathlib import Path
import pstats
import sys

import worker

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--host',choices=['hexset','catanatron'],required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():p.error('output exists')
original=worker.catanatron_speedups
cache_stats={}
@contextmanager
def capture(mode):
    with original(mode) as cache:
        yield cache
        cache_stats.update(cache.stats())
worker.catanatron_speedups=capture
prof=cProfile.Profile()
prof.enable()
result=worker.play(a.host,'fast',a.seed)
prof.disable()
worker.catanatron_speedups=original
stats=pstats.Stats(prof)
functions=[]
for (file,line,name),(primitive,calls,self_s,cumulative,callers) in stats.stats.items():
    functions.append({'file':file,'line':line,'name':name,'primitive_calls':primitive,
                      'calls':calls,'self_seconds':self_s,'cumulative_seconds':cumulative})
result.update(cache=cache_stats,profile_total_seconds=stats.total_tt,python=sys.version,
              functions=sorted(functions,key=lambda r:r['cumulative_seconds'],reverse=True))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
prof.dump_stats(str(a.out.with_suffix('.prof')))
print(a.host,a.seed,result['seconds'],cache_stats,flush=True)
