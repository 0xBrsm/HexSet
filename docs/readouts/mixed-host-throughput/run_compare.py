"""Reversed-order mixed-matchup host comparison, 120 games per host per round."""
import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import sys
import time
from worker import play


def job(seed):
    return play(os.environ['HOST'], seed)


def fingerprint(package):
    root = Path(package.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', choices=['hexset','catanatron'])
    p.add_argument('--round', type=int)
    a=p.parse_args()
    if a.host:
        import hexset, catanatron
        from hexset.catanatron.speedups import verify_runtime
        verify_runtime()
        out=Path(f'/out/{a.round}-{a.host}.json')
        if out.exists(): raise RuntimeError(f'{out} exists')
        started=time.perf_counter()
        with Pool(30) as pool:
            games=list(pool.imap_unordered(job, range(690200000,690200120), chunksize=1))
        elapsed=time.perf_counter()-started
        games.sort(key=lambda g:g['seed'])
        row={'host':a.host,'round':a.round,'seconds':elapsed,'games_per_second':len(games)/elapsed,
             'games':games,'python':sys.version,'hexset_source_sha256':fingerprint(hexset),
             'catanatron_source_sha256':fingerprint(catanatron)}
        out.write_text(json.dumps(row,indent=2)+'\n')
        print(f'{a.round} {a.host}: {elapsed:.3f}s, {len(games)/elapsed:.3f} games/s',flush=True)
    else:
        for rep,order in enumerate((('hexset','catanatron'),('catanatron','hexset'))):
            for host in order:
                print(f'Starting {rep} {host}',flush=True)
                subprocess.run([sys.executable,__file__,'--host',host,'--round',str(rep)],
                               env={**os.environ,'HOST':host},check=True)
        fields=('seed','winner','points','turns','actions','action_sha256','decisions','heximax_fallbacks')
        for host in ('hexset','catanatron'):
            first=json.loads(Path(f'/out/0-{host}.json').read_text())['games']
            second=json.loads(Path(f'/out/1-{host}.json').read_text())['games']
            for x,y in zip(first,second,strict=True):
                assert all(x[f]==y[f] for f in fields), (host,x['seed'])
        Path('/out/verified.json').write_text(json.dumps({'complete':True,'games':480,
            'matched_repeat_pairs':240,'fields':fields},indent=2)+'\n')
        print('All 240 repeat pairs match traces/outcomes.',flush=True)
