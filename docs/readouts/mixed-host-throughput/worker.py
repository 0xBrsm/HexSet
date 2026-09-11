"""One Heximax-notrade plus three AB2, with CPU attribution by bot and bridge."""
import hashlib
import json
import random
import time

from catanatron.game import Game as NativeGame
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import player_key
import hexset.bots
from hexset.bots.heximax.search import Heximax
from hexset.arena import entrant_from_name, spawn, play_game
from hexset.catanatron.board import translate_board
from hexset.catanatron.bot import CatanatronBot
from hexset.catanatron.player import DevCatanPlayer
from hexset.catanatron.speedups import catanatron_speedups
from hexset.catanatron.state import translate
from hexset.victory import victory_points


def play(host, seed, *, instrument=True):
    counters = {f'{bot}_{part}_cpu': 0.0 for bot in ('ab2','heximax') for part in ('search','total')}
    counts = {'ab2': 0, 'heximax': 0}
    original_choose = Heximax.choose

    def heximax_choose(self, game):
        started = time.process_time()
        try:
            return original_choose(self, game)
        finally:
            counters['heximax_search_cpu'] += time.process_time()-started

    class TimedAB(AlphaBetaPlayer):
        def decide(self, game, actions):
            started = time.process_time()
            try:
                return super().decide(game, actions)
            finally:
                counters['ab2_search_cpu'] += time.process_time()-started

    class TraceBot:
        def __init__(self, bot, kind):
            self.bot, self.kind = bot, kind
        def choose(self, game):
            started = time.process_time()
            action = self.bot.choose(game)
            counters[self.kind+'_total_cpu'] += time.process_time()-started
            counts[self.kind] += 1
            actions.append(repr(action))
            return action

    random.seed(seed)
    ab_type = TimedAB if instrument else AlphaBetaPlayer
    native = NativeGame([ab_type(c, '2') for c in Color], seed=seed)
    hseat = (seed-690200000) % 4
    hcolor = native.state.colors[hseat]
    heximax = DevCatanPlayer(hcolor, 'heximax-notrade')
    native.state.players[hseat] = heximax
    actions = []
    if host == 'hexset':
        mapping = translate_board(native.state.board.map)
        game, seats = translate(native, mapping, random.Random(seed))
        derived = (seed*4 + list(Color).index(hcolor)) & 0xFFFFFFFF
        bots = [CatanatronBot(lambda c: ab_type(c, '2')) for _ in Color]
        bots[hseat] = spawn(entrant_from_name('heximax-notrade'), mapping.board, random.Random(derived))
        bots = [TraceBot(bot, 'heximax' if i==hseat else 'ab2') for i,bot in enumerate(bots)]
    else:
        game = native

    def decide(player, game, offered):
        kind = 'heximax' if player.color == hcolor else 'ab2'
        started = time.process_time()
        action = player.decide(game, offered)
        counters[kind+'_total_cpu'] += time.process_time()-started
        counts[kind] += 1
        actions.append(repr(action))
        return action

    if instrument:
        Heximax.choose = heximax_choose
    try:
        with catanatron_speedups('fast'):
            wall, cpu = time.perf_counter(), time.process_time()
            if host == 'hexset':
                game = play_game(game, bots)
                winner = game.won_by
            else:
                win = game.play(decide_fn=decide)
                winner = game.state.colors.index(win) if win is not None else None
            elapsed, cpu_seconds = time.perf_counter()-wall, time.process_time()-cpu
    finally:
        Heximax.choose = original_choose
    if winner is None:
        raise RuntimeError(f'{host} seed {seed} did not finish')
    if host == 'hexset':
        points = [victory_points(game.state(0, hidden=False), s) for s in range(4)]
        turns = game.turns
    else:
        points = [game.state.player_state[player_key(game.state,c)+'_ACTUAL_VICTORY_POINTS'] for c in game.state.colors]
        turns = game.state.num_turns
    return {'host':host,'seed':seed,'heximax_seat':hseat,'seconds':elapsed,'cpu_seconds':cpu_seconds,
            'winner':winner,'points':points,'turns':turns,'actions':len(actions),
            'action_sha256':hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            'decisions':counts,'cpu':counters,'heximax_fallbacks':heximax.fallbacks if host=='catanatron' else 0}
