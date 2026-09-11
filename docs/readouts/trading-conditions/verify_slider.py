"""Small native behavior checks; not a new strength sample."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from hexset import arena
from hexset.bots.heximax import Heximax, NO_TRADE_WEIGHTS, TRADING_WEIGHTS, heximax
from hexset.bots.heximax.adaptive import AdaptiveHeximax
from hexset.game import Game


def fixed_no_trade(entrant, board, rng):
    return heximax(board, rng, weights=NO_TRADE_WEIGHTS, expansion_value=.25,
                   trade_weights=TRADING_WEIGHTS, trade_floor=0)


def play(candidate, opponents, seed, index):
    trace=hashlib.sha256(); calls=0; samples=[]
    original_choose=Heximax.choose; original_play=arena.play_game
    original_observe=AdaptiveHeximax.observe_trade
    def choose(bot,game):
        nonlocal calls
        assert type(game) is Game
        action=original_choose(bot,game)
        trace.update(f'{game.current_player}:{action!r}\n'.encode());calls+=1
        return action
    def observe(bot,**public):
        original_observe(bot,**public)
        samples.append(bot.activity.mean)
    def seated(game,bots,**kwargs):
        assert all(bot.trade_floor==0 for bot in bots)
        return original_play(game,bots,**kwargs)
    Heximax.choose=choose;arena.play_game=seated;AdaptiveHeximax.observe_trade=observe
    try:
        outcome=arena._play_one(((candidate,*opponents),index,seed,20000,False,False))
    finally:
        Heximax.choose=original_choose;arena.play_game=original_play;AdaptiveHeximax.observe_trade=original_observe
    assert outcome.winner is not None
    return dict(winner=outcome.winner,points=outcome.points,seating=outcome.seating,
                turns=outcome.turns,actions=calls,action_sha256=trace.hexdigest(),
                observations=len(samples),activities=sorted(set(samples)))


def main():
    arena.register_entrant_kind('slider-fixed-no-trade',fixed_no_trade)
    candidate=arena.PRESETS['heximax-adaptive']
    fixed=arena.Entrant('fixed-N/T',kind='slider-fixed-no-trade')
    quiet=replace(arena.PRESETS['heximax-balanced'],max_trades=0)
    rows=[]
    for index in range(4):
        a=play(candidate,(quiet,)*3,723000000,index)
        b=play(fixed,(quiet,)*3,723000000,index)
        for key in ('winner','points','seating','turns','actions','action_sha256'):
            assert a[key]==b[key],(index,key)
        assert a['observations']>0 and a['activities']==[0]
        rows.append(dict(regime='none',index=index,adaptive=a,fixed=b))
    for index in range(4):
        a=play(candidate,(arena.PRESETS['heximax-balanced'],)*3,723010000,index)
        assert a['observations']>0 and max(a['activities'])>0
        rows.append(dict(regime='trading-smoke',index=index,adaptive=a))
    import hexset
    root=Path(hexset.__file__).parent;h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    result=dict(source_sha256=h.hexdigest(),games=12,exact_no_trade_pairs=4,
                complete_trading_smoke_games=4,rows=rows)
    Path(__file__).with_name('slider-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Four full no-trade pairs match exactly; four trading games complete with a moving slider.')


if __name__=='__main__':main()
