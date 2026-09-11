"""Bounded robber confirmation through production GameSession, not arena play."""
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
import time
import sys
import hexset
import hexset.game as game_module
import hexset.trading as trading
from hexset.arena import deal_game
from hexset.bots.heximax import heximax, TRADING_WEIGHTS
from hexset.bots.heximax.adaptive import trading_profile
from hexset.record import Tape, recording, to_json, from_journal
from hexset.server.journal import Journal
from hexset.server.webplay import GameSession, action_to_wire
from hexset.victory import victory_points

SOURCE='2e7b8d58e7a020723e180374397f3235de213b11bd83aaf4ac799724ba802d85'
POLICY='36c62009d206f5a04c1db7102a643fb125ef8c96'
SEED=734000000
PREFLIGHT_SEED=734900000
GAMES=392


def fingerprint(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


class ObservedBot:
    def __init__(self,bot):
        self.bot=bot;self.observations=[];self.moves=0;self.gain_batches=0;self.estimate_batches=0
    def __getattr__(self,name):return getattr(self.bot,name)
    def observe_trade(self,**event):
        self.bot.observe_trade(**event);self.observations.append(dict(**event,alpha=self.bot.activity.mean))
    def choose(self,game):
        assert game.max_trades==0 and self.bot.max_trades is None
        action=self.bot.choose(game)
        weights,expansion=trading_profile(0)
        assert self.bot.evaluator.weights==weights and self.bot.expansion_value==expansion
        assert self.bot.activity.mean==0
        self.moves+=1;return action
    def gains_many(self,*args):
        self.gain_batches+=1;return self.bot.gains_many(*args)
    def estimate_many(self,*args):
        self.estimate_batches+=1;return self.bot.estimate_many(*args)


def play(job):
    index,seed,variant,out=job;out=Path(out)
    assert not out.exists(),'No retries or replacement games'
    started=time.monotonic();game=deal_game(seed,index,4,chance=recording)
    game.max_trades=0
    seating=[(e+index)%4 for e in range(4)]
    bots=[None]*4;profiles=[]
    for e,s in enumerate(seating):
        weight=.075 if variant=='candidate' and e<2 else -.30
        kwargs={} if variant=='stock' else {'trade_weights':replace(TRADING_WEIGHTS,robber_risk=weight)}
        b=heximax(game._state.board,random.Random(f'{seed}:{index}:{e}'),**kwargs)
        assert b.pin_weights is None and b.max_trades is None and b.trade_floor==0
        assert (b.depth,b.width,b.max_nodes,b.k)==(2,6,600,1)
        assert b.trade_evaluator.weights==replace(TRADING_WEIGHTS,robber_risk=weight)
        profiles.append(asdict(b.trade_evaluator.weights))
        bots[s]=b if variant=='stock' else ObservedBot(b)
    journal=Journal(str(out.parent/'journals'),out.stem)
    assert not journal.path.exists()
    session=GameSession(game=game,claimed_seats=set(range(4)),seed=f'{seed}:{index}:game',journal=journal,
        bot_names={s:'heximax' for s in range(4)})
    for s,b in enumerate(bots):session.set_trader(s,b)
    original_event=game_module.trade_event;original_best=trading._best_clearing
    counts=Counter()
    def forbidden(*args,**kwargs):raise RuntimeError('Automatic clearing is forbidden in evaluation')
    def disabled_event(g,gate):
        assert g.max_trades==0
        counts['disabled_automatic_events']+=1
        result=original_event(g,gate);assert not result;return result
    game_module.trade_event=disabled_event;trading._best_clearing=forbidden
    tape=Tape();notes=[]
    try:
        while not game_module.is_over(game):
            assert len(tape.actions)<20000,'Action cap reached; fail without replacement'
            actor=game_module.to_move(game);action=bots[actor].choose(game)
            start_events=len(session.events)
            session.submit(actor,action_to_wire(action))
            events=session.events[start_events:]
            ordinary=[ev for ev in events if ev.action is not None]
            assert len(ordinary)==1 and not ordinary[0].trades,'Automatic exchange reached action event'
            executed=[t for ev in events if ev.action is None for t in ev.trades]
            tape.step(action,executed)
            for ev in events:
                if ev.note is not None:notes.append(dict(turn=game.turns,round=ev.round_num,**ev.note._asdict()))
            counts['executions']+=len(executed)
            assert len(executed)<=1 and session.open_round is None and not session.trade_wait()
        assert game.won_by is not None and not journal._off
    finally:
        game_module.trade_event=original_event;trading._best_clearing=original_best
    rec=tape.sealed(game,seed=f'{seed}:{index}:game')
    converted=from_journal(journal.path)
    assert converted.actions==rec.actions and converted.trades==rec.trades
    assert converted.chance==rec.chance and converted.winner==rec.winner and converted.turns==rec.turns
    assert counts['executions']==len(rec.trades)
    assert not any(n=='catanatron' or n.startswith('catanatron.') for n in sys.modules)
    row=dict(index=index,seed=seed,variant=variant,policy_commit=POLICY,seating=seating,profiles=profiles,
        winner=seating.index(game.won_by),points=[victory_points(game.state(s,hidden=False),s) for s in seating],
        turns=game.turns,counts=dict(counts),notes=notes,record=json.loads(to_json(rec)),
        bots=[] if variant=='stock' else [dict(moves=b.moves,gain_batches=b.gain_batches,
            estimate_batches=b.estimate_batches,observations=b.observations) for b in bots],
        journal_sha256=hashlib.sha256(journal.path.read_bytes()).hexdigest(),seconds=time.monotonic()-started)
    atomic(out,row);return row


def stage(jobs,root,workers):
    started=last=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(play,j) for j in jobs}
        while pending:
            done,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for f in done:rows.append(f.result());last=time.monotonic()
            atomic(root/'heartbeat.json',dict(complete=len(rows),total=len(jobs),seconds=time.monotonic()-started))
            if time.monotonic()-last>180:raise RuntimeError('No completed game for 180 seconds')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=30)
    args=p.parse_args();root=args.out;assert not (root/'manifest.json').exists()
    assert fingerprint(Path(hexset.__file__).parent)==SOURCE
    started=time.monotonic()
    atomic(root/'manifest.json',dict(source_sha256=SOURCE,policy_commit=POLICY,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol='production GameSession served rounds',
        source_proposal='GameSession.begin_round/default_offer; engine referee coverable candidates',
        candidate_robber=.075,control_robber=-.30,spare_card=.15,trade_floor=0,
        game_max_trades=0,bot_max_trades=None,initial_card_cap=3,counter_card_cap=3,broadcasts_per_turn=1,
        move_slider=0,move_slider_reason='Existing served behavior: game-level automatic-off switch holds slider at zero',
        game_seed=SEED,preflight_seed=PREFLIGHT_SEED,confirmation_games=GAMES,previous_failed_attempts=6,antithetic=False,
        workers=args.workers,python=platform.python_version(),games_cap=408,seconds_cap=900))
    jobs=[(i,PREFLIGHT_SEED,v,str(root/'preflight'/f'{v}-{i}.json')) for i in range(2) for v in ('stock','wrapped')]
    jobs += [(i,PREFLIGHT_SEED+100,'candidate',str(root/'preflight'/f'candidate-{i}.json')) for i in range(2)]
    checks=stage(jobs,root/'preflight',6)
    for i in range(2):
        pair=[next(r for r in checks if r['variant']==v and r['index']==i) for v in ('stock','wrapped')]
        assert pair[0]['record']==pair[1]['record'] and pair[0]['notes']==pair[1]['notes']
    assert all(r['counts']['executions']>0 for r in checks)
    atomic(root/'preflight-summary.json',dict(passed=True,games=6,stock_instrumented_trace_pairs=2,
        candidate_games=2,checks='All records match independent journal conversion; zero automatic exchanges'))
    rows=stage([(i,SEED,'candidate',str(root/'games'/f'{i:04d}.json')) for i in range(GAMES)],root,args.workers)
    from hexset.arena import wilson
    wins=sum(r['winner']<2 for r in rows);interval=wilson(wins,len(rows))
    atomic(root/'summary.json',dict(candidate_wins=wins,control_wins=GAMES-wins,games=GAMES,win_rate=wins/GAMES,
        wilson95=interval,eligible_for_adoption=interval[0]>.5,independent_audit_required=True,
        total_games=GAMES+12,seconds=time.monotonic()-started,
        counts=dict(sum((Counter(r['counts']) for r in rows),Counter()))))


if __name__=='__main__':main()
