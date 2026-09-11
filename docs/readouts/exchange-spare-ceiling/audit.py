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
from run import SOURCE, POLICY, SEED, RUNOFF_SEED, CONFIRM_SEED, CANDIDATES, PROFILES, fingerprint

root = Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['source_sha256'] == fingerprint(Path(hexset.__file__).parent) == SOURCE
assert manifest['policy_commit'] == POLICY and manifest['screen_seed'] == SEED
assert manifest['confirmation_seed'] == CONFIRM_SEED != SEED
assert manifest['driver_sha256'] == hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert manifest['profiles'] == {k:asdict(v) for k,v in PROFILES.items()}
assert manifest['adaptive_move_both'] and manifest['exchange_expansion_both'] == manifest['floor_both'] == 0


def audit_cohort(folder, count, candidate, opponent, seed, variant, summary):
    assert len(list(folder.glob('[0-9]*.json'))) == count
    wins = Counter(); seats = [Counter() for _ in range(4)]
    gate_alphas = [Counter(),Counter()]; move_alphas = [Counter(),Counter()]
    exchanges = actions = events = 0
    participant_counts = [0,0]; exchange_pairings = Counter(); seen_seeds = set()
    cross_net = 0; cross_count = 0; threshold_crossings = [Counter(),Counter()]
    expected_settings = [asdict(Entrant(k,kind='exchange-sweep')) for k in [candidate]*2+[opponent]*2]
    original_advance = record.advance
    for i in range(count):
        row = json.loads((folder/f'{i:04d}.json').read_text())
        assert row['complete'] and row['identity'] == dict(index=i,seed=seed,variant=variant,candidate=candidate,opponent=opponent,policy_commit=POLICY)
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
            nonlocal cross_net, cross_count
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
            for trade in trades:
                sa,sb = [int(row['seating'].index(seat)>=2) for seat in (trade.a,trade.b)]
                if sa != sb:
                    delta = sum(trade.received)
                    cross_count += 1
                    cross_net += delta if sa==0 else -delta
                    for seat,side,change in ((trade.a,sa,delta),(trade.b,sb,-delta)):
                        before_hand = sum(game._state.hands[seat]); after_hand = before_hand+change
                        counts = threshold_crossings[side]
                        counts['participations'] += 1
                        counts['net_cards'] += change
                        counts['started_above_7'] += int(before_hand>7)
                        counts['crossed_above_7'] += int(before_hand<=7<after_hand)
                        counts['crossed_to_7_or_less'] += int(after_hand<=7<before_hand)
                record.apply_trades(game,[trade])
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
            profile = candidate if e<2 else opponent
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
    return dict(candidate=candidate,opponent=opponent,cross_policy_trades=cross_count,
        cross_policy_net_cards_to_candidate=cross_net,
        cross_policy_threshold_crossings=[dict(c) for c in threshold_crossings],games=count,candidate_wins=won,control_wins=wins['2']+wins['3'],
        unfinished=wins['None'],win_rate=p,wilson95=[center-half,center+half],
        legal_actions=actions,activity_events_reconstructed=events,exchanges=exchanges,
        entrant_seats=[dict(s) for s in seats],gate_alpha_counts=[dict(c) for c in gate_alphas])

screen = json.loads((root/'screen-summary.json').read_text())
selection = json.loads((root/'runoff-selection.json').read_text())
assert selection['screen_sha256'] == hashlib.sha256((root/'screen-summary.json').read_bytes()).hexdigest()
reports = [audit_cohort(root/'screen'/c,64,c,'control',SEED,'screen',screen['results'][c]) for c in CANDIDATES]
ranked = sorted(reports,key=lambda r:-r['candidate_wins'])
a,b = [r['candidate'] for r in ranked[:2]]
assert selection['runoff'] == [a,b]
verdict = json.loads((root/'verdict.json').read_text())
if ranked[0]['candidate_wins']>32:
    runoff = json.loads((root/'runoff-summary.json').read_text())
    assert runoff['candidate']==a and runoff['opponent']==b
    r = audit_cohort(root/'runoff',256,a,b,RUNOFF_SEED,'runoff',runoff);reports.append(r)
    expected = a if r['candidate_wins']>=r['control_wins'] else b
    final_selection = json.loads((root/'selection.json').read_text())
    assert final_selection['candidate']==expected
    assert final_selection['runoff_sha256']==hashlib.sha256((root/'runoff-summary.json').read_bytes()).hexdigest()
    confirmation = json.loads((root/'confirmation-summary.json').read_text())
    assert confirmation['candidate']==expected
    final = audit_cohort(root/'confirmation',800,expected,'control',CONFIRM_SEED,'confirmation',confirmation)
    reports.append(final)
    assert verdict['eligible_for_adoption'] == (final['wilson95'][0]>.5)
    assert verdict['games']==1592
else:
    expected=None
    assert verdict['games']==536 and not verdict['eligible_for_adoption']
assert verdict['seconds']<900 and verdict['games']<=1600
assert len(list((root/'preflight').glob('*.json')))==25
for i in range(4):
    stock=json.loads((root/'preflight'/f'control-{i}-stock.json').read_text())
    wrapped=json.loads((root/'preflight'/f'control-{i}-wrapped.json').read_text())
    assert stock['record']==wrapped['record'] and stock['winner']==wrapped['winner']
for c in CANDIDATES:
    for i in range(2):
        checked=json.loads((root/'preflight'/f'{c}-{i}.json').read_text())
        assert all(b['checked_gains']>0 for b in checked['bots'])
report=dict(passed=True,strength_games_replayed=sum(r['games'] for r in reports),
    preflight_games_excluded=24,total_evaluation_games=verdict['games'],
    selection_verified=True,candidate=expected,eligible_for_adoption=verdict['eligible_for_adoption'],
    legal_actions=sum(r['legal_actions'] for r in reports),
    activity_events_reconstructed=sum(r['activity_events_reconstructed'] for r in reports),results=reports)
(root/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
