# SPDX-License-Identifier: GPL-3.0-only
"""Fresh-process paired benchmark of optional native-engine speedups.

Each arm starts a new interpreter. Arm order rotates across seeds. Clean
measurements have no leaf instrumentation; --audit additionally checks every
leaf against the native scalar evaluator and counts deadline observations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def worker(args):
    from catanatron.models.player import Color
    from catanatron.players import value, minimax
    from .duel import build_players, game_config_for
    from .speedups import catanatron_speedups, verify_runtime
    from catanatron.cli.play import play_batch
    verify_runtime()
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        source.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    native_base = value.base_fn
    native_alpha = minimax.AlphaBetaPlayer.alphabeta
    leaf_digest = hashlib.sha256()
    leaves = 0
    deadlines = 0
    with catanatron_speedups(args.mode) as cache:
        if args.audit:
            installed_base = value.base_fn
            def trace_factory(params=value.DEFAULT_WEIGHTS):
                evaluate = installed_base(params)
                reference = native_base(params)
                def traced(game, color):
                    nonlocal leaves
                    actual = evaluate(game, color)
                    expected = reference(game, color)
                    if actual != expected:
                        raise AssertionError(f'leaf score mismatch: {actual!r} != {expected!r}')
                    leaves += 1
                    leaf_digest.update(float(actual).hex().encode() + b'\0')
                    return actual
                return traced
            def trace_alpha(self, game, depth, alpha, beta, deadline, node):
                nonlocal deadlines
                if depth > 0 and game.winning_color() is None and time.time() >= deadline:
                    deadlines += 1
                return native_alpha(self, game, depth, alpha, beta, deadline, node)
            value.base_fn = trace_factory
            minimax.AlphaBetaPlayer.alphabeta = trace_alpha
        try:
            random.seed(args.seed)
            players = build_players(args.players)
            t0, c0 = time.perf_counter(), time.process_time()
            wins, points, games = play_batch(1, players, game_config=game_config_for(args.game_type), quiet=True)
            cpu, wall = time.process_time() - c0, time.perf_counter() - t0
            game = games[0]
            actions = [repr(record) for record in game.state.action_records]
            result = {
                'mode': args.mode, 'seed': args.seed, 'game_seed': game.seed,
                'players': args.players, 'game_type': args.game_type, 'audit': args.audit,
                'source_sha256': source.hexdigest(), 'python': sys.version,
                'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                'native_deadline_seconds': minimax.MAX_SEARCH_TIME_SECS,
                'wall_seconds': wall, 'cpu_seconds': cpu,
                'wins': {str(c): wins.get(c, 0) for c in Color},
                'points': {str(c): list(v) for c, v in points.items()},
                'action_count': len(actions),
                'action_sha256': hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
                'leaves': leaves if args.audit else None,
                'leaf_sha256': leaf_digest.hexdigest() if args.audit else None,
                'deadline_observations': deadlines if args.audit else None,
                'cache': cache.stats() if cache else None,
            }
        finally:
            minimax.AlphaBetaPlayer.alphabeta = native_alpha
            if args.audit:
                value.base_fn = installed_base
    return result


def compare(rows, audit, baseline="off"):
    groups = defaultdict(dict)
    for row in rows:
        groups[row['seed']][row['mode']] = row
    fields = ['game_seed', 'wins', 'points', 'action_count', 'action_sha256', 'source_sha256']
    if audit:
        fields += ['leaves', 'leaf_sha256', 'deadline_observations']
    comparisons = []
    for seed, arms in groups.items():
        if baseline not in arms:
            continue
        for mode, row in arms.items():
            if mode == baseline:
                continue
            differences = [name for name in fields if row[name] != arms[baseline][name]]
            comparisons.append({'seed': seed, 'mode': mode, 'differences': differences})
    totals = {}
    for mode in ('off', 'basic', 'cached', 'fast'):
        if mode == baseline:
            continue
        pairs = [g for g in groups.values() if baseline in g and mode in g]
        if not pairs:
            continue
        base = sum(g[baseline]['wall_seconds'] for g in pairs)
        fast = sum(g[mode]['wall_seconds'] for g in pairs)
        totals[mode] = {'pairs': len(pairs), 'baseline_wall_seconds': base,
                        'candidate_wall_seconds': fast, 'speedup': base / fast,
                        'wall_reduction': 1 - fast / base}
    return {'comparisons': comparisons, 'totals': totals,
            'equivalent': bool(comparisons) and all(not c['differences'] for c in comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', default='DC:heximax-notrade,AB:2,AB:2,AB:2')
    parser.add_argument('--game-type', default='standard')
    parser.add_argument('--seed', type=int, default=630000000)
    parser.add_argument('--games', type=int, default=8, help='games per arm')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('off', 'basic', 'cached', 'fast'), default='off')
    parser.add_argument('--modes', nargs='+', choices=('off', 'basic', 'cached', 'fast'),
                        default=['off', 'basic', 'cached', 'fast'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        atomic_json(args.out, worker(args))
        return
    if args.games < 1:
        parser.error('--games must be positive')
    if args.out.exists():
        parser.error('output exists; choose a new output path')
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    rows = []
    modes = tuple(args.modes)
    if len(modes) < 2 or len(set(modes)) != len(modes):
        parser.error('--modes requires at least two distinct modes')
    total = len(modes) * args.games
    for i in range(args.games):
        order = modes[i % len(modes):] + modes[:i % len(modes)]
        for mode in order:
            path = args.out.parent / (args.out.stem + '-games') / f'{args.seed + i}-{mode}.json'
            cmd = [sys.executable, '-m', __spec__.name, '--worker', '--players', args.players,
                   '--game-type', args.game_type, '--seed', str(args.seed + i),
                   '--mode', mode, '--out', str(path)]
            if args.audit:
                cmd.append('--audit')
            subprocess.run(cmd, env=env, check=True)
            rows.append(json.loads(path.read_text()))
            summary = compare(rows, args.audit, baseline=modes[0])
            atomic_json(args.out, {'complete': len(rows) == total,
                                   'baseline': modes[0], 'rows': rows, **summary})
            print(f'{len(rows)}/{total}: seed {args.seed+i} {mode}', flush=True)
            if any(c['differences'] for c in summary['comparisons']):
                raise RuntimeError('paired equivalence failed; see output')
    if args.audit and any(row['deadline_observations'] for row in rows):
        raise RuntimeError('deadline-limited search observed; equivalence is not established')
    print(json.dumps(summary['totals'], indent=2))


if __name__ == '__main__':
    main()
