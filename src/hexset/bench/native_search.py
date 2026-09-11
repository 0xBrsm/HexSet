# SPDX-License-Identifier: GPL-3.0-only
"""Checkpointed native HexSet candidate evaluation against frozen references."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys
import time

import hexset.bots
from hexset.arena import Entrant, _play_one, register_entrant_kind, spawn, wilson
from hexset.bots.evaluate import Weights
from hexset.bots.heximax import NO_TRADE_WEIGHTS
from hexset.bots.heximax.search import Heximax, heximax
from hexset.bots.stances import WIN_TEMPERATURE
from hexset.game import Game, to_move

PROTOCOL = 'native-hexset-public-ledger-v1-independent-boards'
FROZEN = dict(weights=asdict(NO_TRADE_WEIGHTS), depth=2, width=6, max_nodes=600,
              k=1, stance='win', temperature=WIN_TEMPERATURE, placement=True,
              mode='notrade', max_trades=0)
_CONFIG = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hash():
    import hexset
    root = Path(hexset.__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


def _candidate(entrant, board, rng):
    cfg = dict(_CONFIG[entrant.name])
    cfg['weights'] = Weights(**cfg['weights'])
    bot = heximax(board, rng, **cfg)
    actual = {key: asdict(bot.evaluator.inner.weights) if key == 'weights' else getattr(bot,key)
              for key in _CONFIG[entrant.name]}
    assert actual == _CONFIG[entrant.name], (actual, _CONFIG[entrant.name])
    return bot


def initialize(config):
    global _CONFIG
    _CONFIG = config
    register_entrant_kind('native-candidate', _candidate)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    tmp.replace(path)


def play_job(job):
    from hexset.catanatron.bot import CatanatronBot
    from hexset.catanatron.speedups import catanatron_speedups
    path = Path(job['path']); identity = job['identity']
    if path.exists():
        row = json.loads(path.read_text())
        if row.get('identity') != identity or row.get('complete') is not True:
            raise ValueError(f'incompatible or incomplete checkpoint: {path}')
        return row
    candidate, gate, seed, index = (identity[k] for k in ('candidate','gate','seed','index'))
    random.seed(seed + index)
    opponent = Entrant('frozen-shipped',kind='native-candidate') if gate=='shipped' else Entrant('ab2',kind='catanatron',depth=2)
    lineup = [Entrant(candidate,kind='native-candidate'), opponent, opponent, opponent]
    trace = hashlib.sha256(); decisions = 0; informed = 0
    original_h, original_a = Heximax.choose, CatanatronBot.choose

    def checked(original):
        def choose(bot, game):
            nonlocal decisions, informed
            assert type(game) is Game, 'real game must be native HexSet'
            seat = to_move(game)
            if identity['audit']:
                view = game.state(seat)
                assert view.ledger is game.ledger
                state = game.state(seat, hidden=False)
                for i,row in enumerate(game.ledger.seats):
                    assert row.total() == sum(state.hands[i])
                    assert all(0 <= a <= b for a,b in zip(row.known,state.hands[i]))
                    if i != seat:
                        assert view.known[i] == row.known
                if any(sum(row.known)>0 for i,row in enumerate(game.ledger.seats) if i!=seat):
                    informed += 1
            action = original(bot, game)
            trace.update(f'{seat}:{action!r}\n'.encode())
            decisions += 1
            return action
        return choose

    Heximax.choose, CatanatronBot.choose = checked(original_h), checked(original_a)
    started = time.perf_counter()
    try:
        with catanatron_speedups(identity['speedups']):
            outcome = _play_one((tuple(lineup), index, seed, identity['action_cap'], False, False))
    finally:
        Heximax.choose, CatanatronBot.choose = original_h, original_a
    row = dict(identity=identity, complete=True, seconds=time.perf_counter()-started,
               winner=outcome.winner, seat=outcome.seat, turns=outcome.turns,
               points=outcome.points, seating=outcome.seating, actions=decisions,
               action_sha256=trace.hexdigest(), informed_decisions=informed)
    if outcome.winner is None:
        raise RuntimeError(f'unfinished game: {identity}')
    if identity['audit'] and informed == 0:
        raise AssertionError('audit never observed public opponent hand information')
    atomic(path,row)
    return row


def run(config, candidates, gates, seed, games, workers, out, *, audit=False, speedups='fast'):
    from hexset.arena import MAX_ACTIONS
    from hexset.catanatron.speedups import verify_runtime
    verify_runtime()
    if games <= 0 or games % 4 or workers < 1:
        raise ValueError('positive workers and complete four-seat rotations required')
    config = {**config, 'frozen-shipped': FROZEN}
    import numpy, networkx
    runtime = dict(python=sys.version, numpy=numpy.__version__, networkx=networkx.__version__)
    sh = source_hash(); rows=[]; jobs=[]
    for candidate in candidates:
        for gate in gates:
            if gate not in ('ab2','shipped'): raise ValueError(gate)
            for index in range(games):
                identity=dict(protocol=PROTOCOL,runtime=runtime,host='hexset',information_model='native-public-history',
                    source_sha256=sh,catanatron_revision='d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816',
                    candidate=candidate,phenotype=config[candidate],frozen_shipped=FROZEN,
                    gate=gate,seed=seed,index=index,antithetic=False,rule_set='standard',
                    speedups=speedups,audit=audit,action_cap=MAX_ACTIONS)
                path=out/'games'/f'{candidate}-{gate}-{seed}-{index:05d}.json'
                jobs.append(dict(identity=identity,path=str(path)))
    initialize(config)
    started=time.perf_counter()
    if workers==1:
        rows=list(map(play_job,jobs))
    else:
        with get_context('spawn').Pool(workers, initializer=initialize,initargs=(config,)) as pool:
            for row in pool.imap_unordered(play_job,jobs,chunksize=1):
                rows.append(row)
                if len(rows)%32==0: print(f'{len(rows)}/{len(jobs)} checkpoints ready',flush=True)
    summary=[]
    for candidate in candidates:
        for gate in gates:
            group=sorted((r for r in rows if r['identity']['candidate']==candidate and r['identity']['gate']==gate),key=lambda r:r['identity']['index'])
            wins=sum(r['winner']==0 for r in group)
            summary.append(dict(candidate=candidate,gate=gate,games=len(group),wins=wins,
                                rate=wins/len(group),wilson95=wilson(wins,len(group)),
                                seconds=sum(r['seconds'] for r in group)))
    result=dict(protocol=PROTOCOL,source_sha256=sh,seed=seed,games_per_gate=games,workers=workers,
                audit=audit,speedups=speedups,seconds=time.perf_counter()-started,rows=summary)
    atomic(out/f'summary-{seed}-{games}-{digest([candidates,gates,audit,speedups])[:12]}.json',result)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--candidates',nargs='+',required=True)
    p.add_argument('--gates',nargs='+',default=['ab2','shipped'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--games',type=int,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',action='store_true');p.add_argument('--speedups',choices=['off','fast'],default='fast')
    a=p.parse_args()
    run(json.loads(a.manifest.read_text())['candidates'],a.candidates,a.gates,a.seed,a.games,a.workers,a.out,
        audit=a.audit,speedups=a.speedups)


if __name__=='__main__': main()
