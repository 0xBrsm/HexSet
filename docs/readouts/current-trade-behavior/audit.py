"""Replay current self-play records and measure executed-trade behavior."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import hexset
import hexset.record as record
from hexset.arena import deal_board, lineup_from_names
from hexset.bots.heximax import TRADING_WEIGHTS
from hexset.victory import victory_points
from run import SOURCE, POLICY, SEED, GAMES, fingerprint

root = Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
assert fingerprint(Path(hexset.__file__).parent) == manifest['source_sha256'] == SOURCE
assert manifest['policy_commit'] == POLICY and manifest['seed'] == SEED
assert manifest['exchange_weights'] == asdict(TRADING_WEIGHTS)
assert manifest['driver_sha256'] == hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert len(list((root/'games').glob('*.json'))) == GAMES
expected = [asdict(e) for e in lineup_from_names(['heximax']*4)]
reports=[]
original_advance = record.advance
for i in range(GAMES):
    row=json.loads((root/'games'/f'{i:04d}.json').read_text())
    assert (row['index'],row['seed'],row['policy_commit']) == (i,SEED,POLICY)
    assert row['settings'] == expected and row['seating'] == [(e+i)%4 for e in range(4)]
    rec=record.from_json(json.dumps(row['record']))
    assert rec.seed == f'{SEED}:{i}:game' and rec.num_players == 4 and rec.first == 0
    assert all(getattr(rec,k)==v for k,v in record.board_fields(deal_board(SEED,i)).items())
    events=[]; bundles=Counter(); participant_sizes=Counter(); thresholds=Counter()
    cards_given=[0]*4; participations=[0]*4
    def advance(game,action,trades,seat=None):
        before=game.trade_event_turn
        record.apply(game,action,seat)
        new_event=game.trade_event_turn != before
        assert not trades or new_event
        if new_event:
            actor=game.current_player
            sizes=list(map(sum,game._state.hands))
            eligible=sizes[actor]>=2 and any(n>=2 for s,n in enumerate(sizes) if s!=actor)
            events.append(dict(turn=game.turns,actor=actor,eligible=eligible,exchanges=len(trades)))
        for t in trades:
            assert t.a == game.current_player
            give=-sum(x for x in t.received if x<0)
            take=sum(x for x in t.received if x>0)
            assert 1<=give<=3 and 1<=take<=3
            bundles['x'.join(map(str,sorted((give,take))))]+=1
            participant_sizes[str(give)]+=1;participant_sizes[str(take)]+=1
            cards_given[t.a]+=give;cards_given[t.b]+=take
            participations[t.a]+=1;participations[t.b]+=1
            for who,delta in [(t.a,take-give),(t.b,give-take)]:
                n=sum(game._state.hands[who]);after=n+delta
                thresholds['participations']+=1
                thresholds['started_above_7']+=int(n>7)
                thresholds['crossed_above_7']+=int(n<=7<after)
                thresholds['crossed_to_7_or_less']+=int(after<=7<n)
            record.apply_trades(game,[t])
    record.advance=advance
    try: game=record.replay(rec)
    finally: record.advance=original_advance
    assert game.won_by is not None
    assert row['winner']==row['seating'].index(game.won_by)
    assert row['turns']==game.turns
    assert row['points']==[victory_points(game.state(s,hidden=False),s) for s in row['seating']]
    assert row['exchanges']==len(rec.trades)==sum(e['exchanges'] for e in events)==sum(bundles.values())
    assert len({e['turn'] for e in events})==len(events)
    assert sum(participations)==2*len(rec.trades)
    assert sum(int(k)*v for k,v in participant_sizes.items())==sum(cards_given)
    reports.append(dict(index=i,completed_player_turns=game.turns,player_turns_started=game.turns+1,
        exchanges=len(rec.trades),player_turns_with_event=len(events),
        eligible_events=sum(e['eligible'] for e in events),
        events_with_trade=sum(e['exchanges']>0 for e in events),
        events_with_multiple_trades=sum(e['exchanges']>1 for e in events),
        trades_per_event=dict(Counter(str(e['exchanges']) for e in events)),
        bundles=dict(bundles),participant_sizes=dict(participant_sizes),cards_moved=sum(cards_given),
        participations_by_seat=participations,cards_given_by_seat=cards_given,
        hand_thresholds=dict(thresholds),legal_actions=len(rec.actions)))


def summed(key):
    return sum(r[key] for r in reports)


def counts(key):
    result=Counter()
    for r in reports: result.update(r[key])
    return dict(sorted(result.items()))


def distribution(values):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values))
    values=sorted(values)
    return dict(mean=mean,mean_ci95_normal=[mean-1.959963984540054*se,mean+1.959963984540054*se],
        median=statistics.median(values),p90=values[math.ceil(.9*len(values))-1],
        p95=values[math.ceil(.95*len(values))-1],p99=values[math.ceil(.99*len(values))-1],max=max(values))

completion=json.loads((root/'completion.json').read_text())
assert completion['games']==GAMES and completion['unfinished']==0
assert completion['exchanges']==summed('exchanges') and completion['seconds']<300
exchanges=summed('exchanges');turns=summed('player_turns_with_event');bundles=counts('bundles')
report=dict(passed=True,games=GAMES,source_sha256=SOURCE,protocol=manifest['protocol'],
    legal_actions=summed('legal_actions'),evaluation_seconds=completion['seconds'],
    exchanges=exchanges,player_turns_with_event=turns,eligible_events=summed('eligible_events'),
    completed_player_turns_per_game=distribution([r['completed_player_turns'] for r in reports]),
    player_turns_started_per_game=distribution([r['player_turns_started'] for r in reports]),
    events_with_trade=summed('events_with_trade'),events_with_multiple_trades=summed('events_with_multiple_trades'),
    exchanges_per_player_turn=exchanges/turns,
    fraction_turns_with_trade=summed('events_with_trade')/turns,
    fraction_turns_with_multiple_trades=summed('events_with_multiple_trades')/turns,
    exchanges_per_trading_turn=exchanges/summed('events_with_trade'),
    exchanges_per_game=distribution([r['exchanges'] for r in reports]),
    trade_participations_per_player_game=2*exchanges/(4*GAMES),
    cards_moved=summed('cards_moved'),cards_moved_per_exchange=summed('cards_moved')/exchanges,
    cards_given_per_player_game=summed('cards_moved')/(4*GAMES),
    cards_moved_per_game=distribution([r['cards_moved'] for r in reports]),
    bundle_counts=bundles,bundle_fractions={k:v/exchanges for k,v in bundles.items()},
    fraction_exchanges_with_three_card_side=sum(v for k,v in bundles.items() if '3' in k)/exchanges,
    participant_cards_given_counts=counts('participant_sizes'),
    trades_per_event_counts=counts('trades_per_event'),hand_thresholds=counts('hand_thresholds'))
(root/'per-game-metrics.json').write_text(json.dumps(reports,indent=2,sort_keys=True)+'\n')
(root/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
