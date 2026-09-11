"""Fixed native HexSet benchmark for the current README; see PLAN.md."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
from importlib.metadata import distribution
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time

POLICY_COMMIT = '38ac537e44944a892b881b5c0152d4de6828b2a4'
REFERENCE_COMMIT = 'ecf931181b9a65bb4116a2153fb78c16f1438e00'
SEEDS = {2: 727300000, 4: 727400000}
PREFLIGHT_SEEDS = {2: 727920000, 4: 727930000}


def atomic(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(row, sort_keys=True, indent=2) + '\n')
    temp.replace(path)


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def runtime():
    import catanatron
    import hexset
    from hexset.catanatron.speedups import verify_runtime
    verify_runtime()
    direct = json.loads(distribution('catanatron').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id'] == REFERENCE_COMMIT, direct
    return dict(policy_commit=POLICY_COMMIT, catanatron_commit=REFERENCE_COMMIT,
                hexset_source_sha256=fingerprint(Path(hexset.__file__).parent),
                catanatron_source_sha256=fingerprint(Path(catanatron.__file__).parent),
                python=platform.python_version(), catanatron_path=catanatron.__file__,
                pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                threads={k: os.environ.get(k) for k in
                         ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def play(job):
    seats, index, seed, mode, destination = job
    import hexset.arena as arena
    from hexset.bots.heximax import Heximax, TRADING_WEIGHTS
    from hexset.bots.heximax.adaptive import trading_profile
    from hexset.catanatron.speedups import catanatron_speedups
    from hexset.record import to_json
    from catanatron.players.minimax import AlphaBetaPlayer
    path = Path(destination)
    identity = dict(seats=seats, index=index, seed=seed, mode=mode,
                    policy_commit=POLICY_COMMIT, catanatron_commit=REFERENCE_COMMIT)
    if path.exists():
        row = json.loads(path.read_text())
        assert row['identity'] == identity and row['complete']
        return row
    entrants = arena.lineup_from_names(['heximax'] + ['catanatron'] * (seats - 1))
    settings = [asdict(e) for e in entrants]
    alpha_seen = set()
    calls = 0
    deadline_hits = 0
    original_choose = Heximax.choose
    original_alpha = AlphaBetaPlayer.alphabeta

    def choose(bot, game):
        nonlocal calls
        assert bot.pin_weights is None and bot._adaptive
        assert (bot.depth, bot.width, bot.max_nodes, bot.k) == (2,6,600,1)
        assert bot.max_trades is None and bot.trade_floor == 0
        assert bot.trade_evaluator.weights == TRADING_WEIGHTS
        action = original_choose(bot, game)
        expected, expansion = trading_profile(bot.activity.mean)
        assert bot.evaluator.weights == expected and bot.expansion_value == expansion
        alpha_seen.add(bot.activity.mean)
        calls += 1
        return action

    def alphabeta(bot, game, depth, alpha, beta, deadline, node):
        nonlocal deadline_hits
        if depth > 0 and game.winning_color() is None and time.time() >= deadline:
            deadline_hits += 1
        return original_alpha(bot, game, depth, alpha, beta, deadline, node)

    started = time.perf_counter()
    with catanatron_speedups(mode):
        Heximax.choose = choose
        AlphaBetaPlayer.alphabeta = alphabeta
        try:
            outcome = arena._play_one((tuple(entrants), index, seed, arena.MAX_ACTIONS, False, True))
        finally:
            Heximax.choose = original_choose
            AlphaBetaPlayer.alphabeta = original_alpha
    record = json.loads(to_json(outcome.record))
    row = dict(identity=identity, complete=True, settings=settings,
               winner=outcome.winner, seating=outcome.seating, points=outcome.points,
               turns=outcome.turns, exchanges=len(outcome.cleared),
               heximax_decisions=calls, observed_alphas=sorted(alpha_seen),
               deadline_hits=deadline_hits, seconds=time.perf_counter()-started,
               record=record)
    atomic(path, row)
    return row


def stage(jobs, root, workers):
    started = time.monotonic()
    last_progress = started
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(play, job) for job in jobs}
        try:
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.append(future.result())
                    last_progress = time.monotonic()
                now = time.monotonic()
                atomic(root/'heartbeat.json', dict(status='running', complete=len(rows), total=len(jobs),
                       elapsed_seconds=now-started, seconds_since_progress=now-last_progress,
                       utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                if now-last_progress > 300:
                    raise RuntimeError('no completed game for 300 seconds')
            atomic(root/'heartbeat.json', dict(status='complete', complete=len(rows), total=len(jobs),
                   elapsed_seconds=time.monotonic()-started))
        except BaseException as exc:
            atomic(root/'heartbeat.json', dict(status='failed', error=repr(exc), complete=len(rows), total=len(jobs)))
            for future in pending: future.cancel()
            raise
    return rows, time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    info = runtime()
    info.update(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                workers=args.workers, host='HexSet', antithetic=False, games_per_format=400)
    manifest = args.out/'manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == info
    else: atomic(manifest,info)
    preflight = [(n,i,PREFLIGHT_SEEDS[n],mode,str(args.out/'preflight'/f'{n}-{i}-{mode}.json'))
                 for n in (2,4) for i in range(n) for mode in ('off','fast')]
    controls, seconds = stage(preflight,args.out/'preflight',min(args.workers,12))
    by_id = {(r['identity']['seats'],r['identity']['index'],r['identity']['mode']):r for r in controls}
    matches = all(by_id[n,i,'off']['record'] == by_id[n,i,'fast']['record'] and
                  by_id[n,i,'off']['winner'] == by_id[n,i,'fast']['winner']
                  for n in (2,4) for i in range(n))
    mode = 'fast' if matches and not any(r['deadline_hits'] for r in controls) else 'off'
    atomic(args.out/'preflight-summary.json',dict(matches=matches, selected_mode=mode, seconds=seconds,
           deadline_hits=sum(r['deadline_hits'] for r in controls), games=len(controls)))
    jobs = [(n,i,SEEDS[n],mode,str(args.out/'games'/f'{n}-{i:04d}.json')) for n in (2,4) for i in range(400)]
    rows, seconds = stage(jobs,args.out/'games',args.workers)
    from hexset.arena import wilson
    summary = dict(runtime=info, seconds=seconds, mode=mode, results=[])
    for n in (2,4):
        cohort = [r for r in rows if r['identity']['seats']==n]
        wins = sum(r['winner']==0 for r in cohort)
        summary['results'].append(dict(seats=n, games=len(cohort), heximax_wins=wins,
               win_rate=wins/len(cohort), wilson95=wilson(wins,len(cohort)),
               winners=dict(Counter(str(r['winner']) for r in cohort)),
               unfinished=sum(r['winner'] is None for r in cohort),
               focal_seats=dict(Counter(r['seating'][0] for r in cohort)),
               exchanges=sum(r['exchanges'] for r in cohort),
               observed_alphas=sorted({a for r in cohort for a in r['observed_alphas']}),
               deadline_hits=sum(r['deadline_hits'] for r in cohort)))
    atomic(args.out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
