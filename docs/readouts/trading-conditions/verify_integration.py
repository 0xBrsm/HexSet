"""Replay frozen validation traces through the public split-evaluator API."""
import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import validate as v
from hexset.bots.heximax import heximax

r=v.r
ORIGINAL_FACTORY=v.factory


def factory(entrant,board,rng):
    bot=ORIGINAL_FACTORY(entrant,board,rng)
    assert bot.rule not in ('A','P')
    same=bot.move is bot.gate
    integrated=heximax(board,rng,**r.PARAMS,
        weights=bot.move.evaluator.weights,expansion_value=bot.move.expansion_value,
        trade_weights=None if same else bot.gate.evaluator.weights,
        trade_expansion_value=bot.gate.expansion_value,trade_floor=0.0)
    bot.move=bot.gate=integrated
    return bot


r.factory=factory


def check(job):
    identity,path,old_path=job
    row=r.play((identity,path));old=json.loads(Path(old_path).read_text())
    # Gate calls include the new public method and its internal delegate;
    # candidate-seat metadata is a known reporting correction, not gameplay.
    keys=('winner','seating','turns','points','actions','action_sha256','trades',
          'public_trade_events','activity_trace','final_turn_trades')
    row=json.loads(json.dumps(row))
    for key in keys:assert row[key]==old[key],(old_path,key)
    return dict(regime=identity['regime'],candidate=identity['candidate'],index=identity['index'],
                action_sha256=row['action_sha256'],domestic_trades=row['domestic_trades'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();jobs=[]
    for regime in v.REGIMES:
        candidates=['M:T:0.0']+(['T:T:0.0'] if regime in ('moderate','full') else [])
        for c in candidates:
            for index in range(4):
                relative=Path('games')/regime/c.replace(':','-')/f'{index:05d}.json'
                old_path=a.old/relative;old=json.loads(old_path.read_text())
                identity={**old['identity'],'purpose':'split-api-behavior-replay',
                    'integrated_source_sha256':r.source_hash(),
                    'integration_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                jobs.append((identity,str(a.out/relative),str(old_path)))
    with get_context('spawn').Pool(30,initializer=r.initialize) as pool:
        rows=list(pool.imap_unordered(check,jobs))
    result=dict(replayed=len(rows),all_actions_and_trades_identical=True,rows=rows)
    r.atomic(a.out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
