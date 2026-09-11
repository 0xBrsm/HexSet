import json,random
import hexset.catanatron.public_ledger
from hexset.catanatron.duel import parse_cli_string,play_batch
from hexset.catanatron.public_ledger import PublicLedgerPlayer
from hexset.bots.heximax.search import Heximax
from catanatron.state_functions import player_num_resource_cards

def one(spec):
 random.seed(1200000 if 'AB:2' in spec else 1300000)
 players=parse_cli_string(spec); cand=players[0]; checks=0; nonzero=0; maxknown=0
 orig=cand.decide
 def wrapped(game,actions):
  nonlocal checks,nonzero,maxknown
  out=orig(game,actions); book=cand._public_book; candidate_seat=game.state.colors.index(cand.color)
  for seat,color in enumerate(game.state.colors):
   size=player_num_resource_cards(game.state,color); row=book.ledger.seats[seat]
   assert all(0<=row.known[r] for r in range(5))
   assert sum(row.known)+row.unknown==size,(seat,row.known,row.unknown,size)
   if seat != candidate_seat:
    assert all(row.known[r] <= size for r in range(5))
    if sum(row.known): nonzero+=1
    maxknown=max(maxknown,sum(row.known))
  checks+=1; return out
 cand.decide=wrapped; wins,pts,games=play_batch(1,players,quiet=True)
 assert isinstance(cand,PublicLedgerPlayer) and cand.fallbacks==0
 return {'spec':spec,'candidate_class':type(cand).__name__,'bot_class':type(cand._bot).__name__,'candidate_color':cand.color.value,'winner':games[0].winning_color().value,'decisions_checked':checks,'nonzero_opponent_decisions':nonzero,'max_typed_opponent_cards':maxknown,'fallbacks':cand.fallbacks}
with open('result.json','w') as f: json.dump([one('DCL:heximax-notrade,AB:2,AB:2,AB:2'),one('DCL:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade')],f,indent=2)
