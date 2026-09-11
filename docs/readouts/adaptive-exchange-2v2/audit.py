"""Replay every game and reconstruct the public activity observations."""
from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import hexset
import hexset.record as record
from hexset.arena import deal_board, Entrant
from hexset.victory import victory_points
from run import SOURCE, POLICY, SEED, fingerprint

root = Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['source_sha256'] == fingerprint(Path(hexset.__file__).parent) == SOURCE
assert manifest['policy_commit'] == POLICY and manifest['seed'] == SEED
assert manifest['driver_sha256'] == hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert manifest['adaptive_move_both'] and manifest['exchange_expansion_both'] == manifest['floor_both'] == 0
assert len(list((root/'games').glob('[0-9]*.json'))) == 400
wins = Counter(); seats = [Counter() for _ in range(4)]
gate_alphas = [Counter(),Counter()]; move_alphas = [Counter(),Counter()]
exchanges = actions = events = 0
participant_counts = [0,0]; exchange_pairings = Counter(); seen_seeds = set()
expected_settings = [asdict(Entrant(k,kind=k)) for k in ['exchange-adaptive']*2+['exchange-fixed']*2]
original_advance = record.advance
for i in range(400):
    row = json.loads((root/'games'/f'{i:04d}.json').read_text())
    assert row['complete'] and row['identity'] == dict(index=i,seed=SEED,variant='mixed',policy_commit=POLICY)
    assert row['settings'] == expected_settings
    assert row['seating'] == [(e+i)%4 for e in range(4)]
    rec = record.from_json(json.dumps(row['record']))
    assert rec.seed == f'{SEED}:{i}:game' and rec.seed not in seen_seeds
    seen_seeds.add(rec.seed)
    assert rec.num_players == 4 and rec.first == 0
    assert all(getattr(rec,k) == v for k,v in record.board_fields(deal_board(SEED,i)).items())
    assert len(row['bots']) == 4
    assert all(b['observations'] == row['bots'][0]['observations'] for b in row['bots'])
    expected_observations = row['bots'][0]['observations']; observed = []
    history = deque([False]*8,maxlen=8)
    def advance(game, action, trades, seat=None):
        before = game.trade_event_turn
        record.apply(game,action,seat)
        if game.trade_event_turn != before:
            actor = game.current_player
            sizes = list(map(sum,game._state.hands))
            pairs = [[t.a,t.b] for t in trades]
            if sizes[actor] >= 2 and any(n>=2 for s,n in enumerate(sizes) if s!=actor):
                history.append(any(actor in pair for pair in pairs))
            observed.append(dict(turn=game.turns,actor=actor,hand_sizes=sizes,
                trade_participants=pairs,alpha=sum(history)/8))
        record.apply_trades(game,trades)
    record.advance = advance
    try: game = record.replay(rec)
    finally: record.advance = original_advance
    assert observed == expected_observations, i
    events += len(observed)
    winner = None if game.won_by is None else row['seating'].index(game.won_by)
    assert row['winner'] == winner and row['turns'] == game.turns
    assert row['points'] == [victory_points(game.state(s,hidden=False),s) for s in row['seating']]
    assert row['exchanges'] == len(rec.trades)
    exchanges += len(rec.trades); actions += len(rec.actions); wins[str(winner)] += 1
    for e,b in enumerate(row['bots']):
        side = int(e>=2)
        assert b['adaptive_exchange'] == (e<2)
        assert sum(b['move_alphas'].values())>0 and sum(b['gate_alphas'].values())>0
        assert all(float(a)*8 in range(9) for a in b['move_alphas'] | b['gate_alphas'])
        gate_alphas[side].update(b['gate_alphas']); move_alphas[side].update(b['move_alphas'])
        seats[e][row['seating'][e]] += 1
    for _,a,b,_ in rec.trades:
        sa,sb = [int(row['seating'].index(s)>=2) for s in (a,b)]
        participant_counts[sa]+=1; participant_counts[sb]+=1
        exchange_pairings['-'.join(map(str,sorted((sa,sb))))] += 1
assert all(s == Counter({i:100 for i in range(4)}) for s in seats)
count = wins['0']+wins['1']; total=400; p=count/total; z=1.959963984540054
center=(p+z*z/(2*total))/(1+z*z/total)
half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
summary=json.loads((root/'summary.json').read_text())
assert summary['adaptive_exchange_wins']==count and summary['fixed_exchange_wins']==wins['2']+wins['3']
assert summary['unfinished']==wins['None'] and summary['winners']==dict(wins)
assert summary['exchanges']==exchanges
assert all(abs(a-b)<1e-5 for a,b in zip(summary['wilson95'],[center-half,center+half]))
for i in range(4):
    stock=json.loads((root/'preflight'/f'{i}-stock.json').read_text())
    wrapped=json.loads((root/'preflight'/f'{i}-wrapped.json').read_text())
    checked=json.loads((root/'preflight'/f'{i}-mixed-check.json').read_text())
    assert stock['record']==wrapped['record'] and stock['winner']==wrapped['winner']
    assert all(b['checked_gains']>0 for b in checked['bots'])
report=dict(passed=True,games_replayed=400,legal_actions=actions,activity_events_reconstructed=events,
    independent_seeds=len(seen_seeds),adaptive_exchange_wins=count,fixed_exchange_wins=wins['2']+wins['3'],
    unfinished=wins['None'],win_rate=p,wilson95=[center-half,center+half],
    entrant_seats=[dict(s) for s in seats],exchanges=exchanges,
    exchange_participations_by_side=participant_counts,exchange_pairings=dict(exchange_pairings),
    gate_alpha_counts=[dict(c) for c in gate_alphas],move_alpha_counts=[dict(c) for c in move_alphas],
    preflight_games_excluded=12)
(root/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
