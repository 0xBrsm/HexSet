"""Independent full replay and selection audit for the bounded exchange sweep."""
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
from run import SOURCE, POLICY, SEED, CONFIRM_SEED, CANDIDATES, PROFILES, fingerprint

root = Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['source_sha256'] == fingerprint(Path(hexset.__file__).parent) == SOURCE
assert manifest['policy_commit'] == POLICY and manifest['screen_seed'] == SEED
assert manifest['confirmation_seed'] == CONFIRM_SEED != SEED
assert manifest['driver_sha256'] == hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert manifest['profiles'] == {k:asdict(v) for k,v in PROFILES.items()}
assert manifest['adaptive_move_both'] and manifest['exchange_expansion_both'] == manifest['floor_both'] == 0


def audit_cohort(folder, count, candidate, seed, variant, summary):
    assert len(list(folder.glob('[0-9]*.json'))) == count
    wins = Counter(); seats = [Counter() for _ in range(4)]
    gate_alphas = [Counter(),Counter()]; move_alphas = [Counter(),Counter()]
    exchanges = actions = events = 0
    participant_counts = [0,0]; exchange_pairings = Counter(); seen_seeds = set()
    expected_settings = [asdict(Entrant(k,kind='exchange-sweep')) for k in [candidate]*2+['control']*2]
    original_advance = record.advance
    for i in range(count):
        row = json.loads((folder/f'{i:04d}.json').read_text())
        assert row['complete'] and row['identity'] == dict(index=i,seed=seed,variant=variant,candidate=candidate,policy_commit=POLICY)
        assert row['settings'] == expected_settings
        assert row['seating'] == [(e+i)%4 for e in range(4)]
        rec = record.from_json(json.dumps(row['record']))
        assert rec.seed == f'{seed}:{i}:game' and rec.seed not in seen_seeds
        seen_seeds.add(rec.seed)
        assert rec.num_players == 4 and rec.first == 0
        assert all(getattr(rec,k) == v for k,v in record.board_fields(deal_board(seed,i)).items())
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
            profile = candidate if e<2 else 'control'
            assert b['profile'] == profile and b['exchange_weights'] == asdict(PROFILES[profile])
            assert sum(b['move_alphas'].values())>0 and sum(b['gate_alphas'].values())>0
            assert all(float(a)*8 in range(9) for a in b['move_alphas'] | b['gate_alphas'])
            gate_alphas[side].update(b['gate_alphas']); move_alphas[side].update(b['move_alphas'])
            seats[e][row['seating'][e]] += 1
        for _,a,b,_ in rec.trades:
            sa,sb = [int(row['seating'].index(s)>=2) for s in (a,b)]
            participant_counts[sa]+=1; participant_counts[sb]+=1
            exchange_pairings['-'.join(map(str,sorted((sa,sb))))] += 1
    assert all(s == Counter({i:count//4 for i in range(4)}) for s in seats)
    won = wins['0']+wins['1']; total=count; p=won/total; z=1.959963984540054
    center=(p+z*z/(2*total))/(1+z*z/total)
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    assert summary['candidate_wins']==won and summary['control_wins']==wins['2']+wins['3']
    assert summary['unfinished']==wins['None'] and summary['winners']==dict(wins)
    assert summary['games']==count and summary['exchanges']==exchanges
    assert all(abs(a-b)<1e-5 for a,b in zip(summary['wilson95'],[center-half,center+half]))
    return dict(candidate=candidate,games=count,candidate_wins=won,control_wins=wins['2']+wins['3'],
        unfinished=wins['None'],win_rate=p,wilson95=[center-half,center+half],
        legal_actions=actions,activity_events_reconstructed=events,exchanges=exchanges,
        entrant_seats=[dict(s) for s in seats],gate_alpha_counts=[dict(c) for c in gate_alphas])

screen = json.loads((root/'screen-summary.json').read_text())
selection = json.loads((root/'selection.json').read_text())
assert selection['screen_sha256'] == hashlib.sha256((root/'screen-summary.json').read_bytes()).hexdigest()
reports = [audit_cohort(root/'screen'/c,128,c,SEED,'screen',screen['results'][c]) for c in CANDIDATES]
best = max(reports,key=lambda r:r['candidate_wins'])
expected = best['candidate'] if best['candidate_wins']>64 else None
assert selection['candidate'] == expected
verdict = json.loads((root/'verdict.json').read_text())
if expected is not None:
    confirmation = json.loads((root/'confirmation-summary.json').read_text())
    assert confirmation['candidate'] == expected
    final = audit_cohort(root/'confirmation',800,expected,CONFIRM_SEED,'confirmation',confirmation)
    reports.append(final)
    assert verdict['eligible_for_adoption'] == (final['wilson95'][0]>.5)
    assert verdict['games'] == 1588
else:
    assert verdict['games'] == 788 and not verdict['eligible_for_adoption']
assert verdict['seconds'] < 900 and verdict['games'] <=1600
assert len(list((root/'preflight').glob('*.json'))) == 21  # includes heartbeat
for i in range(4):
    stock=json.loads((root/'preflight'/f'control-{i}-stock.json').read_text())
    wrapped=json.loads((root/'preflight'/f'control-{i}-wrapped.json').read_text())
    assert stock['record']==wrapped['record'] and stock['winner']==wrapped['winner']
for c in CANDIDATES:
    for i in range(2):
        checked=json.loads((root/'preflight'/f'{c}-{i}.json').read_text())
        assert all(b['checked_gains']>0 for b in checked['bots'])
report = dict(passed=True,strength_games_replayed=sum(r['games'] for r in reports),
    preflight_games_excluded=20,total_evaluation_games=verdict['games'],
    selection_verified=True,candidate=expected,eligible_for_adoption=verdict['eligible_for_adoption'],
    legal_actions=sum(r['legal_actions'] for r in reports),
    activity_events_reconstructed=sum(r['activity_events_reconstructed'] for r in reports),results=reports)
(root/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
