"""Fixed native trading-enabled fair-share comparison; resumable per game."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys
import time
import numpy as np
import hexset
import hexset.arena as arena
from hexset.bots.heximax import heximax, NO_TRADE_WEIGHTS, TRADING_WEIGHTS
from hexset.bots.heximax.search import Heximax, HEXIMAX_TRADE_FLOOR
from hexset.bots.evaluate import Weights
from hexset.bots.stances import WIN_TEMPERATURE
from hexset.game import Game, to_move

SEED=717000000
N=400
PARAMS=dict(mode='honest',max_trades=None,depth=2,width=6,max_nodes=600,k=1,
            stance='win',temperature=WIN_TEMPERATURE,placement=True)
PROFILES={
    'new-notrade':dict(weights=asdict(NO_TRADE_WEIGHTS),expansion_value=.25),
    'old-notrade':dict(weights=asdict(replace(NO_TRADE_WEIGHTS,road=.1237)),expansion_value=0.0),
    'trading':dict(weights=asdict(TRADING_WEIGHTS),expansion_value=0.0),
}


def source_hash():
    root=Path(hexset.__file__).parent;h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');temp.replace(path)


def factory(entrant,board,rng):
    spec=PROFILES[entrant.name]
    bot=heximax(board,rng,**PARAMS,weights=Weights(**spec['weights']),expansion_value=spec['expansion_value'])
    assert {k:getattr(bot,k) for k in PARAMS}==PARAMS
    assert asdict(bot.evaluator.weights)==spec['weights'] and bot.expansion_value==spec['expansion_value']
    assert bot.trade_floor==HEXIMAX_TRADE_FLOOR==.0197
    return bot


def initialize():
    arena.register_entrant_kind('native-trading-profile',factory)


def play(job):
    identity,path=job;path=Path(path)
    if path.exists():
        row=json.loads(path.read_text())
        assert row['identity']==identity and row['complete']
        return row
    random.seed(SEED+identity['index'])
    entrant=lambda name:arena.Entrant(name,kind='native-trading-profile')
    lineup=(entrant('new-notrade'),)+(entrant(identity['opponent']),)*3
    trace=hashlib.sha256();actions=0;calls=[0]*4;finished=[]
    original_choose=Heximax.choose;original_gains=Heximax.gains_many;original_play=arena.play_game
    def choose(bot,game):
        nonlocal actions
        assert type(game) is Game and game.max_trades is None
        assert bot.max_trades is None
        action=original_choose(bot,game)
        trace.update(f'{to_move(game)}:{action!r}\n'.encode());actions+=1
        return action
    def gains(bot,view,received,counterparties):
        assert bot.max_trades is None
        calls[view.perspective]+=1
        return original_gains(bot,view,received,counterparties)
    def run_game(game,bots,**kwargs):
        assert all(b.max_trades is None for b in bots)
        result=original_play(game,bots,**kwargs);finished.append(result);return result
    Heximax.choose=choose;Heximax.gains_many=gains;arena.play_game=run_game
    started=time.perf_counter()
    try:
        outcome=arena._play_one((lineup,identity['index'],SEED,20000,False,False))
    finally:
        Heximax.choose=original_choose;Heximax.gains_many=original_gains;arena.play_game=original_play
    assert outcome.winner is not None and len(finished)==1
    game=finished[0];assert game.max_trades is None and all(c>0 for c in calls)
    trades=[t._asdict() for t in game.trades]
    candidate_seat=outcome.seating.index(0)
    row=dict(identity=identity,complete=True,winner=outcome.winner,seating=outcome.seating,
             candidate_seat=candidate_seat,turns=outcome.turns,points=outcome.points,
             actions=actions,action_sha256=trace.hexdigest(),seconds=time.perf_counter()-started,
             trade_gate_calls_by_seat=calls,domestic_trades=len(trades),trades=trades,
             candidate_trades=sum(t['a']==candidate_seat or t['b']==candidate_seat for t in trades))
    atomic(path,row);return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    sh=source_hash();assert sh=='0254eb3dbe0d14bea1b2d9eb42ec18a553d38c2f9100dd3477edb62347778c3e'
    identity=dict(protocol='native-trading-fair-share-v1',host='hexset',information_model='native-public-history',
                  source_sha256=sh,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime=dict(python=sys.version,numpy=np.__version__),seed=SEED,antithetic=False,
                  rule_set='standard',games_per_matchup=N,action_cap=20000,parameters=PARAMS,
                  profiles=PROFILES,trade_floor=HEXIMAX_TRADE_FLOOR,trading='enabled-all-seats')
    jobs=[({**identity,'opponent':opponent,'index':index},str(out/'games'/f'{opponent}-{index:05d}.json'))
          for opponent in ['old-notrade','trading'] for index in range(N)]
    atomic(out/'manifest.json',identity)
    initialize();started=time.perf_counter();start=datetime.now(timezone.utc).isoformat()
    # These are the first requested games, retained in the fixed 400/gate.
    for i in [0,N]:
        row=play(jobs[i]);assert row['domestic_trades']>0
        print(f"Trading preflight passed: {row['identity']['opponent']}, {row['domestic_trades']} exchanges",flush=True)
    rows=[]
    with get_context('spawn').Pool(30,initializer=initialize) as pool:
        for row in pool.imap_unordered(play,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%40==0:print(f'{len(rows)}/800 complete',flush=True)
    summary=[]
    for opponent in ['old-notrade','trading']:
        group=[r for r in rows if r['identity']['opponent']==opponent]
        assert len(group)==N and {r['identity']['index'] for r in group}==set(range(N))
        assert all(sum(r['candidate_seat']==s for r in group)==N//4 for s in range(4))
        wins=sum(r['winner']==0 for r in group)
        summary.append(dict(opponent=opponent,wins=wins,games=N,rate=wins/N,wilson95=arena.wilson(wins,N),
                            domestic_trades=sum(r['domestic_trades'] for r in group),
                            candidate_trades=sum(r['candidate_trades'] for r in group),
                            games_with_domestic_trades=sum(r['domestic_trades']>0 for r in group),
                            mean_turns=sum(r['turns'] for r in group)/N))
    result=dict(manifest=identity,workers=30,started_at=start,finished_at=datetime.now(timezone.utc).isoformat(),
                wall_seconds=time.perf_counter()-started,rows=summary)
    atomic(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
