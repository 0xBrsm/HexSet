"""Fixed 400-game native 2v2 exchange-weight experiment; see PLAN.md."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
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

SOURCE = 'c5b426a446f78e5686bca51c66f053bef44f58e924911bbb38a53eb1f743d9ef'
POLICY = '38ac537e44944a892b881b5c0152d4de6828b2a4'
SEED = 728000000
PREFLIGHT_SEED = 728900000
LIVE = []
VERIFY = False


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n'); temp.replace(path)


def fingerprint(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


class ExchangeBot:
    max_trades = None
    trade_floor = 0.0

    def __init__(self, board, rng, adaptive):
        self.bot = heximax(board, rng)
        self.board = board
        self.adaptive_exchange = adaptive
        self.moves = Counter(); self.gates = Counter(); self.observations = []
        self.checked_gains = 0
        assert (self.bot.depth, self.bot.width, self.bot.max_nodes, self.bot.k) == (2,6,600,1)
        assert self.bot.pin_weights is None and self.bot._adaptive
        assert self.bot.max_trades is None and self.bot.trade_floor == 0
        LIVE.append(self)

    def choose(self, game):
        assert game.max_trades is None
        action = self.bot.choose(game)
        weights, expansion = trading_profile(self.bot.activity.mean)
        assert self.bot.evaluator.weights == weights and self.bot.expansion_value == expansion
        self.moves[str(self.bot.activity.mean)] += 1
        return action

    def observe_trade(self, **event):
        self.bot.observe_trade(**event)
        self.observations.append(dict(**event, alpha=self.bot.activity.mean))

    def prepare_gate(self):
        alpha = self.bot.activity.mean
        weights = trading_profile(alpha)[0] if self.adaptive_exchange else TRADING_WEIGHTS
        gate = self.bot._trade_policy
        evaluator = gate.evaluator
        evaluator.weights = evaluator.inner.weights = weights
        evaluator.vector = evaluator.inner.vector = tuple(getattr(weights,k) for k in TERM_NAMES)
        assert evaluator.expansion_value == 0
        gate._clear_evaluation_caches()
        self.gates[str(alpha)] += 1
        return gate, weights

    def gains_many(self, view, received, counterparties):
        gate, weights = self.prepare_gate()
        values = gate.gains_many(view, received, counterparties)
        if VERIFY:
            independent = Heximax(HonestEvaluator(self.board, weights, expansion_value=0),
                stance=self.bot.stance, temperature=self.bot.temperature,
                rng=random.Random(0), trade_floor=0)
            expected = independent.gains_many(view, received, counterparties)
            assert values == expected
            self.checked_gains += len(values)
        return values

    def estimate_many(self, view, candidates):
        gate, _ = self.prepare_gate()
        return gate.estimate_many(view, candidates)

    def accepts(self, view, received, counterparty):
        return self.gains_many(view, [received], [counterparty])[0] > 0

    def report(self):
        return dict(adaptive_exchange=self.adaptive_exchange, move_alphas=dict(self.moves),
            gate_alphas=dict(self.gates), observations=self.observations,
            checked_gains=self.checked_gains)


def factory(entrant, board, rng):
    return ExchangeBot(board, rng, entrant.kind == 'exchange-adaptive')


arena.register_entrant_kind('exchange-adaptive', factory)
arena.register_entrant_kind('exchange-fixed', factory)


def play(job):
    global LIVE, VERIFY
    index, seed, variant, destination = job
    LIVE = []; VERIFY = variant == 'mixed-check'
    identity = dict(index=index, seed=seed, variant=variant, policy_commit=POLICY)
    path = Path(destination)
    if path.exists():
        row = json.loads(path.read_text()); assert row['complete'] and row['identity'] == identity
        return row
    if variant == 'stock':
        entrants = arena.lineup_from_names(['heximax']*4)
    else:
        kinds = ['exchange-fixed']*4 if variant == 'wrapped' else ['exchange-adaptive']*2+['exchange-fixed']*2
        entrants = [arena.Entrant(k, kind=k) for k in kinds]
    started = time.perf_counter()
    outcome = arena._play_one((tuple(entrants), index, seed, arena.MAX_ACTIONS, False, True))
    assert not any(n == 'catanatron' or n.startswith('catanatron.') for n in sys.modules)
    row = dict(identity=identity, complete=True, settings=[asdict(e) for e in entrants],
        winner=outcome.winner, seating=outcome.seating, points=outcome.points,
        turns=outcome.turns, exchanges=len(outcome.cleared),
        bots=[b.report() for b in LIVE], record=json.loads(to_json(outcome.record)),
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
    p = argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--workers',type=int,default=30)
    args = p.parse_args(); root = args.out
    source = fingerprint(Path(hexset.__file__).parent); assert source == SOURCE
    manifest = dict(policy_commit=POLICY, source_sha256=source,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), host='HexSet',
        python=platform.python_version(), workers=args.workers, seed=SEED, games=400, antithetic=False,
        pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        adaptive_move_both=True, exchange_expansion_both=0, floor_both=0,
        candidate='latest public activity -> current adaptive move coefficients for exchanges',
        control='stock fixed TRADING_WEIGHTS for exchanges')
    path = root/'manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == manifest
    else: atomic(path,manifest)
    jobs = [(i,PREFLIGHT_SEED,variant,str(root/'preflight'/f'{i}-{variant}.json'))
            for i in range(4) for variant in ('stock','wrapped','mixed-check')]
    checks, seconds = stage(jobs,root/'preflight',min(args.workers,12))
    lookup = {(r['identity']['index'],r['identity']['variant']):r for r in checks}
    for i in range(4):
        assert lookup[i,'stock']['record'] == lookup[i,'wrapped']['record']
        assert lookup[i,'stock']['winner'] == lookup[i,'wrapped']['winner']
        assert all(b['checked_gains'] > 0 for b in lookup[i,'mixed-check']['bots'])
    atomic(root/'preflight-summary.json',dict(passed=True, games=12, control_pairs=4,
        mixed_games_with_independent_gain_checks=4, seconds=seconds))
    jobs = [(i,SEED,'mixed',str(root/'games'/f'{i:04d}.json')) for i in range(400)]
    rows, seconds = stage(jobs,root/'games',args.workers)
    wins = sum(r['winner'] in (0,1) for r in rows)
    summary = dict(games=400, adaptive_exchange_wins=wins,
        fixed_exchange_wins=sum(r['winner'] in (2,3) for r in rows),
        unfinished=sum(r['winner'] is None for r in rows), win_rate=wins/400,
        wilson95=arena.wilson(wins,400), seconds=seconds,
        winners=dict(Counter(str(r['winner']) for r in rows)),
        exchanges=sum(r['exchanges'] for r in rows))
    atomic(root/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
