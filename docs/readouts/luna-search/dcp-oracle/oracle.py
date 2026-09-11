import json,random
from hexset.catanatron.duel import parse_cli_string,play_batch
from catanatron.state_functions import player_num_resource_cards
from hexset.catanatron.public_ledger import PublicProductionLedgerPlayer
R=('WOOD','BRICK','SHEEP','WHEAT','ORE')
def one(spec,seed):
 random.seed(seed); ps=parse_cli_string(spec); c=ps[0]; checks=nonzero=mx=0; orig=c.decide
 def wrapped(g,a):
  nonlocal checks,nonzero,mx
  out=orig(g,a); book=c._public_book; cs=g.state.colors.index(c.color)
  for seat,col in enumerate(g.state.colors):
   row=book.ledger.seats[seat]; total=player_num_resource_cards(g.state,col); assert sum(row.known)+row.unknown==total
   pre='P'+str(g.state.color_to_index[col]); actual=[g.state.player_state[pre+'_'+x+'_IN_HAND'] for x in R]
   assert all(row.known[i]<=actual[i] for i in range(5)),(row,actual)
   if seat!=cs and sum(row.known): nonzero+=1
   mx=max(mx,sum(row.known))
  checks+=1; return out
 c.decide=wrapped; _,_,games=play_batch(1,ps,quiet=True); assert isinstance(c,PublicProductionLedgerPlayer) and c.fallbacks==0
 return {'seed':seed,'winner':games[0].winning_color().value,'decisions':checks,'nonzero_opponent_decisions':nonzero,'max_typed_opponent_cards':mx,'fallbacks':c.fallbacks}
specs=['DCP:heximax-notrade,AB:2,AB:2,AB:2','DCP:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade']
print(json.dumps([{'spec':s,**one(s,24000+i*1000+j)} for i,s in enumerate(specs) for j in range(15)]))
