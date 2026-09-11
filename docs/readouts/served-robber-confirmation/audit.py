"""Independent replay, journal and protocol audit of served confirmation."""
from collections import Counter
from dataclasses import asdict
import hashlib,json,math
from pathlib import Path
import sys
import hexset
import hexset.record as record
from hexset.arena import deal_board
from hexset.victory import victory_points
from run import SOURCE,POLICY,SEED,PREFLIGHT_SEED,GAMES,fingerprint
root=Path(sys.argv[1])
m=json.loads((root/'manifest.json').read_text())
assert m['source_sha256']==fingerprint(Path(hexset.__file__).parent)==SOURCE
assert m['policy_commit']==POLICY and m['game_seed']==SEED and m['confirmation_games']==GAMES
assert m['protocol']=='production GameSession served rounds' and m['game_max_trades']==0
assert m['bot_max_trades'] is None and m['move_slider']==0
assert m['driver_sha256']==hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert m['candidate_robber']==.075 and m['control_robber']==-.30 and m['trade_floor']==0
assert len(list((root/'games').glob('*.json')))==GAMES
wins=Counter();counts=Counter();bundles=Counter();observations=actions=turns=0
seats=[Counter() for _ in range(4)]
for i in range(GAMES):
 p=root/'games'/f'{i:04d}.json';r=json.loads(p.read_text())
 assert (r['index'],r['seed'],r['variant'],r['policy_commit'])==(i,SEED,'candidate',POLICY)
 assert r['seating']==[(e+i)%4 for e in range(4)]
 for e,w in enumerate(r['profiles']):
  assert w['robber_risk']==(.075 if e<2 else -.30) and w['spare_card']==.15
  base=dict(r['profiles'][0]);base['robber_risk']=w['robber_risk'];assert w==base
  seats[e][r['seating'][e]]+=1
 rec=record.from_json(json.dumps(r['record']))
 assert rec.seed==f'{SEED}:{i}:game' and rec.first==0 and rec.num_players==4
 assert all(getattr(rec,k)==v for k,v in record.board_fields(deal_board(SEED,i)).items())
 journal=p.parent/'journals'/f'{i:04d}.jsonl'
 assert hashlib.sha256(journal.read_bytes()).hexdigest()==r['journal_sha256']
 converted=record.from_journal(journal)
 assert (converted.actions,converted.trades,converted.chance)==(rec.actions,rec.trades,rec.chance)
 journal_rows=[json.loads(line) for line in journal.read_text().splitlines()]
 assert all(not j.get('trades') for j in journal_rows if j['kind']=='action')
 assert sum(j['kind']=='trade' for j in journal_rows)==len(rec.trades)
 observed=[];old=record.advance
 def advance(game,action,trades,seat=None):
  before=game.trade_event_turn;record.apply(game,action,seat)
  if before!=game.trade_event_turn:
   observed.append(dict(turn=game.turns,actor=game.current_player,hand_sizes=list(map(sum,game._state.hands)),trade_participants=[],alpha=0.0))
  assert len(trades)<=1
  record.apply_trades(game,trades)
 record.advance=advance
 try:game=record.replay(rec)
 finally:record.advance=old
 assert game.won_by is not None and r['winner']==r['seating'].index(game.won_by)
 assert r['turns']==game.turns and r['points']==[victory_points(game.state(s,hidden=False),s) for s in r['seating']]
 for b in r['bots']:
  assert b['observations']==observed and b['moves']>0 and b['gain_batches']>0
 assert len(observed)==r['counts']['disabled_automatic_events']
 offers={};responses=Counter()
 for note in r['notes']:
  key=(note['turn'],note['actor'])
  counts[note['kind']]+=1
  if note['kind']=='offer':
   assert key not in offers;offers[key]=note
  if note['kind'] in ('accept','counter','pass'):
   assert key in offers and note['seat']!=note['actor']
   response_key=key+(note['seat'],);responses[response_key]+=1;assert responses[response_key]==1
 assert len(responses)==3*len(offers)
 for _,a,b,received in rec.trades:
  give=-sum(x for x in received if x<0);take=sum(x for x in received if x>0)
  assert a!=b and 1<=give<=3 and 1<=take<=3
  bundles['x'.join(map(str,sorted((give,take))))]+=1
 assert r['counts']['executions']==len(rec.trades)<=len(offers)
 counts['executions']+=len(rec.trades);observations+=len(observed);actions+=len(rec.actions);turns+=game.turns+1
 wins[str(r['winner'])]+=1
assert all(s==Counter({i:GAMES//4 for i in range(4)}) for s in seats)
for i in range(2):
 a=json.loads((root/'preflight'/f'stock-{i}.json').read_text());b=json.loads((root/'preflight'/f'wrapped-{i}.json').read_text())
 assert a['record']==b['record'] and a['notes']==b['notes'] and a['seed']==PREFLIGHT_SEED
summary=json.loads((root/'summary.json').read_text());won=wins['0']+wins['1'];p=won/GAMES;z=1.959963984540054
center=(p+z*z/(2*GAMES))/(1+z*z/GAMES);half=z*math.sqrt(p*(1-p)/GAMES+z*z/(4*GAMES*GAMES))/(1+z*z/GAMES)
ci=[center-half,center+half]
assert summary['candidate_wins']==won and summary['control_wins']==GAMES-won
assert all(abs(a-b)<1e-7 for a,b in zip(ci,summary['wilson95']))
assert summary['eligible_for_adoption']==(ci[0]>.5) and summary['total_games']==GAMES+12
failed=json.loads(Path(__file__).with_name('failed-preflight-receipt.json').read_text())
seconds=summary['seconds']+failed['seconds'];assert seconds<900 and GAMES+12<=408
report=dict(passed=True,games=GAMES,candidate_wins=won,control_wins=GAMES-won,winners=dict(wins),wilson95=ci,
 eligible_for_adoption=ci[0]>.5,automatic_exchanges=0,legal_actions=actions,public_observations=observations,
 moves_slider_zero=True,counts=dict(counts),bundle_counts=dict(bundles),mean_turns=turns/GAMES,
 total_attempts=GAMES+12,total_evaluation_seconds=seconds,
 protocol='Production GameSession served rounds; current served move slider zero; no custom Clio proposer')
(root/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
