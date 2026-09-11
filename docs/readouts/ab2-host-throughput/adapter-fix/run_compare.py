"""Two reversed-order rounds, each with 120 games and 30 workers per arm."""
import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, '/study')
from worker import play


def job(seed):
    return play(os.environ['HOST'], 'fast', seed)


def fingerprint(package):
    root = Path(package.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['baseline', 'fixed', 'native'])
    p.add_argument('--round', type=int)
    a = p.parse_args()
    if a.arm:
        import hexset, catanatron
        from hexset.catanatron.speedups import verify_runtime
        verify_runtime()
        target = Path(f'/out/{a.round}-{a.arm}.json')
        if target.exists():
            raise RuntimeError(f'{target} exists')
        start = time.perf_counter()
        with Pool(30) as pool:
            games = list(pool.imap_unordered(job, range(680200000, 680200120), chunksize=1))
        elapsed = time.perf_counter()-start
        games.sort(key=lambda g: g['seed'])
        row = {'arm': a.arm, 'round': a.round, 'workers': 30, 'seconds': elapsed,
               'games_per_second': len(games)/elapsed, 'games': games,
               'hexset_source_sha256': fingerprint(hexset),
               'catanatron_source_sha256': fingerprint(catanatron), 'python': sys.version}
        target.write_text(json.dumps(row, indent=2)+'\n')
        print(f'{a.round} {a.arm}: {elapsed:.3f}s, {len(games)/elapsed:.3f} games/s', flush=True)
    else:
        for rep, order in enumerate((('baseline','native','fixed'), ('fixed','native','baseline'))):
            for arm in order:
                env = {**os.environ, 'HOST': 'catanatron' if arm == 'native' else 'hexset',
                       'PYTHONPATH': '/baseline/src' if arm == 'baseline' else '/study/src'}
                print(f'Starting {rep} {arm}', flush=True)
                subprocess.run([sys.executable, __file__, '--arm', arm, '--round', str(rep)],
                               env=env, check=True)
        fields = ('seed', 'winner', 'points', 'turns', 'actions', 'action_sha256')
        prior = json.loads(Path('/study/prior-pool.json').read_text())
        for rep in range(2):
            for arm in ('baseline', 'native', 'fixed'):
                row = json.loads(Path(f'/out/{rep}-{arm}.json').read_text())
                host = 'catanatron' if arm == 'native' else 'hexset'
                expected = next(r for r in prior['rows'] if r['host'] == host)
                for before, after in zip(expected['games'], row['games'], strict=True):
                    assert all(before[f] == after[f] for f in fields), (rep, arm, before['seed'])
        Path('/out/verified.json').write_text(json.dumps({'complete': True,
            'games': 720, 'matched_prior_traces': 720, 'fields': fields}, indent=2)+'\n')
        print('All 720 games matched prior traces and outcomes.', flush=True)
