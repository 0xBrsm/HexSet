"""Current stock Heximax automatic-trade behavior baseline."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time
import hexset
import hexset.arena as arena
from hexset.bots.heximax import heximax, Heximax, HonestEvaluator, TRADING_WEIGHTS
from hexset.bots.heximax.adaptive import trading_profile
from hexset.bots.evaluate import TERM_NAMES
from hexset.record import to_json

SOURCE = 'd9cd0c7c833130e401a8d0fbf468a7b6113824e6cb3a88310b47df90410411e2'
POLICY = '270b9270ea7684c9db60dd346b1f3444e1b9fd0c'
SEED = 732000000
GAMES = 400
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n'); temp.replace(path)


def fingerprint(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


def play(job):
    index, destination = job
    path = Path(destination)
    assert not path.exists(), 'No reruns or replacement games'
    entrants = arena.lineup_from_names(['heximax']*4)
    started = time.perf_counter()
    outcome = arena._play_one((tuple(entrants), index, SEED, arena.MAX_ACTIONS, False, True))
    assert not any(n == 'catanatron' or n.startswith('catanatron.') for n in sys.modules)
    row = dict(index=index, seed=SEED, policy_commit=POLICY,
        settings=[asdict(e) for e in entrants], winner=outcome.winner,
        seating=outcome.seating, points=outcome.points, turns=outcome.turns,
        exchanges=len(outcome.cleared), record=json.loads(to_json(outcome.record)),
        seconds=time.perf_counter()-started)
    atomic(path,row)
    return row


def stage(jobs, root, workers):
    started = last = time.monotonic(); rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(play,j) for j in jobs}
        try:
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.append(future.result()); last = time.monotonic()
                now = time.monotonic()
                atomic(root/'heartbeat.json', dict(status='running', complete=len(rows), total=len(jobs),
                    seconds=now-started, seconds_since_progress=now-last))
                if now-last > 300: raise RuntimeError('no completed game for 300 seconds')
        except BaseException as exc:
            atomic(root/'heartbeat.json', dict(status='failed', error=repr(exc), complete=len(rows)))
            for f in pending: f.cancel()
            raise
    seconds = time.monotonic()-started
    atomic(root/'heartbeat.json', dict(status='complete', complete=len(rows), total=len(jobs), seconds=seconds))
    return rows, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=30)
    args = parser.parse_args(); root = args.out
    assert not root.joinpath('manifest.json').exists()
    source = fingerprint(Path(hexset.__file__).parent); assert source == SOURCE
    from hexset.trading import MAX_TRADE_CARDS
    assert MAX_TRADE_CARDS == 3
    atomic(root/'manifest.json',dict(policy_commit=POLICY,source_sha256=source,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=SEED,games=GAMES,workers=args.workers,python=platform.python_version(),
        protocol='native automatic clearing',exchange_weights=asdict(TRADING_WEIGHTS),
        card_cap_per_side=MAX_TRADE_CARDS,antithetic=False,games_cap=400,
        evaluation_seconds_cap=300,adaptive_moves=True,trade_floor=0,max_trades=None))
    jobs=[(i,str(root/'games'/f'{i:04d}.json')) for i in range(GAMES)]
    rows,seconds=stage(jobs,root,args.workers)
    atomic(root/'completion.json',dict(games=len(rows),seconds=seconds,
        unfinished=sum(r['winner'] is None for r in rows),
        exchanges=sum(r['exchanges'] for r in rows)))


if __name__ == '__main__': main()
