"""Profile full Heximax self-play in HexSet only; run clean/profile separately."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time

import hexset
from hexset.bench import profile_heximax as harness
from hexset.victory import victory_points

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--preset', choices=['heximax', 'heximax-notrade'], required=True)
p.add_argument('--games', type=int, default=3)
p.add_argument('--seed', type=int, default=660000000)
p.add_argument('--profile', action='store_true')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.games < 1 or a.out.exists():
    p.error('positive games and a fresh output path are required')
a.out.parent.mkdir(parents=True, exist_ok=True)
root = Path(hexset.__file__).parent
source = hashlib.sha256()
for path in sorted(root.rglob('*.py')):
    source.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
rows = []
prof = cProfile.Profile() if a.profile else None
original = harness.TimedBot.choose
all_times = []
for i in range(a.games):
    digest = hashlib.sha256()
    def choose(self, game):
        action = original(self, game)
        digest.update(repr(action).encode()+b'\0')
        return action
    harness.TimedBot.choose = choose
    if prof:
        prof.enable()
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        game, times = harness.play_one_game(a.preset, a.seed, i)
    finally:
        seconds, cpu_seconds = time.perf_counter()-wall, time.process_time()-cpu
        if prof:
            prof.disable()
        harness.TimedBot.choose = original
    if game.won_by is None:
        raise RuntimeError('game did not reach a winner')
    all_times.extend(times)
    state = game.state(0, hidden=False)
    row = {'index': i, 'seconds': seconds, 'cpu_seconds': cpu_seconds,
           'decisions': len(times), 'decision_seconds': sum(times),
           'winner': game.won_by, 'turns': game.turns,
           'points': [victory_points(state, s) for s in range(4)],
           'action_sha256': digest.hexdigest()}
    rows.append(row)
    print(json.dumps(row), flush=True)
loaded = [n for n in sys.modules if n == 'catanatron' or n.startswith('catanatron.')]
assert not loaded, loaded
report = {'engine':'hexset', 'version':hexset.__version__, 'source_sha256':source.hexdigest(),
          'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'python':sys.version, 'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
          'preset':a.preset, 'seed':a.seed, 'profiled':a.profile, 'games':rows,
          'catanatron_modules_loaded':loaded,
          'decision_ms':{'mean':statistics.mean(all_times)*1000,
                         'p50':harness._pctile(all_times,.5)*1000,
                         'p95':harness._pctile(all_times,.95)*1000}}
if prof:
    prof.dump_stats(str(a.out.with_suffix('.prof')))
    stats = pstats.Stats(prof)
    records = []
    for (filename,line,name),(primitive,calls,self_s,cumulative,callers) in stats.stats.items():
        records.append({'file':filename,'line':line,'name':name,'primitive_calls':primitive,
                        'calls':calls,'self_seconds':self_s,'cumulative_seconds':cumulative})
    report['profile_total_seconds'] = stats.total_tt
    report['profile_functions'] = sorted(records,key=lambda r:r['cumulative_seconds'],reverse=True)
a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
