"""Bounded exchange spare-card range screen, runoff and fresh confirmation."""
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
SEED = 731000000
RUNOFF_SEED = 731100000
CONFIRM_SEED = 731200000
PREFLIGHT_SEED = 731900000
CANDIDATES = {f"spare-{value:g}": replace(TRADING_WEIGHTS, spare_card=value)
              for value in (0., .075, .1875, .225, .30, .45, .60, 1.20)}
PROFILES = {"control":TRADING_WEIGHTS, **CANDIDATES}
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

    def __init__(self, board, rng, profile):
        self.bot = heximax(board, rng)
        self.board = board
        self.profile = profile
        self.weights = PROFILES[profile]
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
        weights = self.weights
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
        return dict(profile=self.profile, exchange_weights=asdict(self.weights), move_alphas=dict(self.moves),
            gate_alphas=dict(self.gates), observations=self.observations,
            checked_gains=self.checked_gains)


def factory(entrant, board, rng):
    return ExchangeBot(board, rng, entrant.name)


arena.register_entrant_kind('exchange-sweep', factory)


def play(job):
    global LIVE, VERIFY
    index, seed, variant, candidate, opponent, destination = job
    LIVE = []; VERIFY = variant == 'candidate-check'
    identity = dict(index=index, seed=seed, variant=variant, candidate=candidate, opponent=opponent, policy_commit=POLICY)
    path = Path(destination)
    assert not path.exists(), 'No reruns or replacement games in this bounded study'
    if variant == 'stock':
        entrants = arena.lineup_from_names(['heximax']*4)
    else:
        profiles = ['control']*4 if variant == 'wrapped' else [candidate]*2+[opponent]*2
        entrants = [arena.Entrant(p, kind='exchange-sweep') for p in profiles]
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


def summarize(rows):
    wins = sum(r['winner'] in (0,1) for r in rows)
    return dict(games=len(rows), candidate_wins=wins, control_wins=sum(r['winner'] in (2,3) for r in rows),
        unfinished=sum(r['winner'] is None for r in rows), win_rate=wins/len(rows),
        wilson95=arena.wilson(wins,len(rows)), winners=dict(Counter(str(r['winner']) for r in rows)),
        exchanges=sum(r['exchanges'] for r in rows))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--workers',type=int,default=30)
    args = p.parse_args(); root = args.out
    assert not (root/'manifest.json').exists(), 'Single run; no automatic retries'
    started = time.monotonic()
    source = fingerprint(Path(hexset.__file__).parent); assert source == SOURCE
    manifest = dict(policy_commit=POLICY, source_sha256=source,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), host='HexSet',
        python=platform.python_version(), workers=args.workers, screen_seed=SEED,
        runoff_seed=RUNOFF_SEED,confirmation_seed=CONFIRM_SEED,antithetic=False,
        evaluation_seconds_cap=900,games_cap=1600,pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        adaptive_move_both=True,exchange_expansion_both=0,floor_both=0,
        profiles={k:asdict(v) for k,v in PROFILES.items()},candidate_order=list(CANDIDATES))
    atomic(root/'manifest.json',manifest)
    jobs = [(i,PREFLIGHT_SEED,v,'control','control',str(root/'preflight'/f'control-{i}-{v}.json'))
            for i in range(4) for v in ('stock','wrapped')]
    jobs += [(i,PREFLIGHT_SEED+100+j,'candidate-check',c,'control',str(root/'preflight'/f'{c}-{i}.json'))
             for j,c in enumerate(CANDIDATES) for i in range(2)]
    checks, seconds = stage(jobs,root/'preflight',min(args.workers,24))
    lookup = {(r['identity']['index'],r['identity']['variant']):r for r in checks if r['identity']['candidate']=='control'}
    for i in range(4):
        assert lookup[i,'stock']['record'] == lookup[i,'wrapped']['record']
        assert lookup[i,'stock']['winner'] == lookup[i,'wrapped']['winner']
    candidates_checked = [r for r in checks if r['identity']['variant']=='candidate-check']
    assert all(b['checked_gains'] > 0 for r in candidates_checked for b in r['bots'])
    atomic(root/'preflight-summary.json',dict(passed=True,games=len(checks),seconds=seconds,
        control_trace_pairs=4,candidate_check_games=16,
        independently_checked_gains=sum(b['checked_gains'] for r in candidates_checked for b in r['bots'])))
    jobs = [(i,SEED,'screen',c,'control',str(root/'screen'/c/f'{i:04d}.json')) for c in CANDIDATES for i in range(64)]
    rows, seconds = stage(jobs,root/'screen',args.workers)
    screen = {c:summarize([r for r in rows if r['identity']['candidate']==c]) for c in CANDIDATES}
    atomic(root/'screen-summary.json',dict(results=screen,seconds=seconds))
    ranked = sorted(CANDIDATES,key=lambda c:-screen[c]['candidate_wins'])
    selection = dict(runoff=ranked[:2],rule='screen wins descending; ties use registered order',
        screen_sha256=hashlib.sha256((root/'screen-summary.json').read_bytes()).hexdigest())
    atomic(root/'runoff-selection.json',selection)
    if screen[ranked[0]]['candidate_wins']<=32:
        atomic(root/'verdict.json',dict(status='complete',eligible_for_adoption=False,
            reason='No screen candidate exceeded 50%',games=536,seconds=time.monotonic()-started)); return
    a,b = ranked[:2]
    jobs = [(i,RUNOFF_SEED,'runoff',a,b,str(root/'runoff'/f'{i:04d}.json')) for i in range(256)]
    rows, seconds = stage(jobs,root/'runoff',args.workers)
    runoff = summarize(rows);runoff.update(candidate=a,opponent=b,seconds=seconds)
    atomic(root/'runoff-summary.json',runoff)
    candidate = a if runoff['candidate_wins']>=runoff['control_wins'] else b
    atomic(root/'selection.json',dict(candidate=candidate,rule='most runoff wins; ties retain screen leader',
        runoff_sha256=hashlib.sha256((root/'runoff-summary.json').read_bytes()).hexdigest()))
    jobs = [(i,CONFIRM_SEED,'confirmation',candidate,'control',str(root/'confirmation'/f'{i:04d}.json')) for i in range(800)]
    rows, seconds = stage(jobs,root/'confirmation',args.workers)
    confirmation = summarize(rows);confirmation.update(candidate=candidate,seconds=seconds)
    atomic(root/'confirmation-summary.json',confirmation)
    atomic(root/'verdict.json',dict(status='complete',eligible_for_adoption=confirmation['wilson95'][0]>.5,
        candidate=candidate,selected_at_upper_edge=candidate==list(CANDIDATES)[-1],
        rule='fresh two-sided 95% Wilson lower bound > 50%; independent audit required',
        games=1592,seconds=time.monotonic()-started))
    print(json.dumps(confirmation,indent=2),flush=True)


if __name__ == '__main__': main()
