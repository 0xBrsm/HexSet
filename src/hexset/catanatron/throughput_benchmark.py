# SPDX-License-Identifier: GPL-3.0-only
"""Compare duel modes/schedulers with per-game traces and a bounded worker pool.

Each trial has a fresh parent interpreter and pool. Trial order reverses on
alternate repetitions. All profiles play the same requested seeds; every
completed game's actual seed, ordered actions, wins and VP are checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .speed_benchmark import atomic_json


def traced_job(args):
    """Pool entry point; works with fork, spawn and forkserver."""
    from . import duel
    started = time.perf_counter()
    original = duel.play_batch
    records = []
    def traced_batch(*a, **kw):
        wins, points, games = original(*a, **kw)
        game = games[0]
        records.append({
            'index': args[1] + len(records), 'actual_seed': game.seed,
            'action_count': len(game.state.action_records),
            'action_sha256': hashlib.sha256(json.dumps(
                [repr(record) for record in game.state.action_records]
            ).encode()).hexdigest(),
            'wins': {str(c): wins.get(c, 0) for c in game.state.colors},
            'points': {str(c): list(v) for c, v in points.items()},
        })
        return wins, points, games
    duel.play_batch = traced_batch
    try:
        wins, points = duel._play_chunk(args)
    finally:
        duel.play_batch = original
    folder = Path(os.environ['HEXSET_THROUGHPUT_RECORDS'])
    atomic_json(folder / f'{args[1]:08d}.json', records)
    return args[1], args[2], wins, points, time.perf_counter() - started


def trial(args):
    from . import duel
    os.environ['HEXSET_THROUGHPUT_RECORDS'] = str(args.out.parent / (args.out.stem + '-games'))
    # Top-level function stays picklable in a freshly spawned worker.
    duel._play_job = traced_job
    mode, scheduling = args.profile.split(':')
    r = duel.run_duel(args.players, args.games, args.workers, args.seed,
                      speedups=mode, scheduling=scheduling)
    records = []
    for path in sorted(Path(os.environ['HEXSET_THROUGHPUT_RECORDS']).glob('*.json')):
        records.extend(json.loads(path.read_text()))
    records.sort(key=lambda x: x['index'])
    if [r['index'] for r in records] != list(range(args.games)):
        raise RuntimeError('missing or repeated game indices')
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob('*.py')):
        source.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    report = {
        'source_sha256': source.hexdigest(), 'python':sys.version,
        'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
        'profile': args.profile, 'games': args.games, 'workers': args.workers,
        'seed': args.seed, 'players': args.players, 'seconds': r.seconds,
        'worker_seconds': r.worker_seconds,
        'worker_utilization': r.worker_seconds / (min(args.games, args.workers) * r.seconds),
        'games_per_second': r.games / r.seconds, 'report': r.report(),
        'records': records,
    }
    atomic_json(args.out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--players', default='DC:heximax-notrade,AB:2,AB:2,AB:2')
    p.add_argument('--games', type=int, default=240)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seed', type=int, default=650200000)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--profiles', nargs='+', default=['cached:static','fast:static','fast:dynamic'])
    p.add_argument('--profile', default='cached:static', help=argparse.SUPPRESS)
    p.add_argument('--trial', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.trial:
        trial(args)
        return
    if args.out.exists():
        p.error('output already exists')
    if min(args.games, args.workers, args.repeats) < 1:
        p.error('games, workers and repeats must be positive')
    env = {**os.environ, 'PYTHONHASHSEED':'0', 'OMP_NUM_THREADS':'1',
           'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    rows = []
    reference = None
    for repeat in range(args.repeats):
        profiles = args.profiles if repeat % 2 == 0 else args.profiles[::-1]
        for profile in profiles:
            output = args.out.parent / (args.out.stem + '-trials') / f'{repeat}-{profile.replace(":","-")}.json'
            cmd = [sys.executable, '-m', __spec__.name, '--trial', '--profile', profile,
                   '--players', args.players, '--games', str(args.games), '--workers', str(args.workers),
                   '--seed', str(args.seed), '--out', str(output)]
            subprocess.run(cmd, env=env, check=True)
            row = json.loads(output.read_text())
            if reference is None:
                reference = row['records']
            equivalent = row['records'] == reference
            rows.append({'repeat': repeat, 'equivalent': equivalent, **row})
            atomic_json(args.out, {'complete':len(rows)==args.repeats*len(args.profiles),
                                  'equivalent':all(r['equivalent'] for r in rows), 'rows':rows})
            print(f"{profile} repeat {repeat}: {row['seconds']:.2f}s, "
                  f"{row['games_per_second']:.3f} games/s, "
                  f"{row['worker_utilization']:.1%} utilization, equivalent={equivalent}", flush=True)
            if not equivalent:
                raise RuntimeError('game trace or outcome differs; see trial records')


if __name__ == '__main__':
    main()
