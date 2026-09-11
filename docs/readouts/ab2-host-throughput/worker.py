"""Four AB2 players, current supported reference pin, selectable host engine."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from catanatron.game import Game as NativeGame
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import player_key
from hexset.catanatron.speedups import catanatron_speedups


def play(host, mode, seed):
    # Identical initial board/deck/seating from the pinned native constructor.
    # Hosts can subsequently diverge in RNG consumption and turn semantics.
    random.seed(seed)
    native = NativeGame([AlphaBetaPlayer(c, '2') for c in Color], seed=seed)
    if host == 'hexset':
        from hexset.catanatron.board import translate_board
        from hexset.catanatron.state import translate
        from hexset.catanatron.bot import CatanatronBot
        from hexset.arena import play_game
        from hexset.victory import victory_points
        mapping = translate_board(native.state.board.map)
        game, seats = translate(native, mapping, random.Random(seed))
        bots = [CatanatronBot() for _ in range(4)]
        actions = []
        class TraceBot:
            def __init__(self, bot): self.bot = bot
            def choose(self, game):
                action = self.bot.choose(game)
                actions.append(repr(action))
                return action
        bots = [TraceBot(bot) for bot in bots]
    with catanatron_speedups(mode):
        wall, cpu = time.perf_counter(), time.process_time()
        if host == 'hexset':
            game = play_game(game, bots)
            seconds, cpu_seconds = time.perf_counter()-wall, time.process_time()-cpu
            winner = game.won_by
            points = [victory_points(game.state(0, hidden=False), s) for s in range(4)]
            turns = game.turns
        else:
            game = native
            win = game.play()
            seconds, cpu_seconds = time.perf_counter()-wall, time.process_time()-cpu
            winner = game.state.colors.index(win) if win is not None else None
            points = [game.state.player_state[player_key(game.state,c)+'_ACTUAL_VICTORY_POINTS'] for c in game.state.colors]
            turns = game.state.num_turns
            actions = [repr(record.action) for record in game.state.action_records]
    if winner is None:
        raise RuntimeError(f'{host} {mode} seed {seed} did not reach a winner')
    return {'host':host, 'mode':mode, 'seed':seed,'seconds':seconds,'cpu_seconds':cpu_seconds,
            'winner':winner,'points':points,'turns':turns,'actions':len(actions),
            'action_sha256':hashlib.sha256(json.dumps(actions).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['hexset','catanatron'],required=True)
    p.add_argument('--mode',choices=['off','fast'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    from hexset.catanatron.speedups import verify_runtime
    verify_runtime()
    import hexset, catanatron
    r=play(a.host,a.mode,a.seed)
    for package in (hexset, catanatron):
        root = Path(package.__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
        r[package.__name__+'_source_sha256'] = digest.hexdigest()
    r['python'] = sys.version
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
