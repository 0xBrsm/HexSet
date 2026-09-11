"""Independently replay and recount the fixed current-configuration benchmark."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

from hexset.arena import deal_board, lineup_from_names
from hexset.record import board_fields, from_json, replay
from hexset.victory import victory_points
from run import runtime

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
manifest = json.loads((root / 'manifest.json').read_text())
assert summary['runtime'] == manifest
local = runtime()
for key in ('policy_commit', 'catanatron_commit', 'hexset_source_sha256', 'catanatron_source_sha256'):
    assert local[key] == manifest[key], key
assert manifest['policy_commit'] == '38ac537e44944a892b881b5c0152d4de6828b2a4'
assert manifest['catanatron_commit'] == 'ecf931181b9a65bb4116a2153fb78c16f1438e00'
assert manifest['driver_sha256'] == hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
assert manifest['host'] == 'HexSet' and manifest['antithetic'] is False
assert len(list((root / 'games').glob('*.json'))) == 801  # includes heartbeat
results = []
seen_seeds = set()
for n, seed in ((2, 727300000), (4, 727400000)):
    wins = Counter()
    seats = Counter()
    alphas = set()
    exchanges = deadlines = actions = 0
    expected_settings = [asdict(e) for e in lineup_from_names(['heximax'] + ['catanatron'] * (n - 1))]
    for index in range(400):
        row = json.loads((root / 'games' / f'{n}-{index:04d}.json').read_text())
        assert row['complete'] and row['identity'] == dict(seats=n, index=index, seed=seed,
            mode=summary['mode'], policy_commit=manifest['policy_commit'],
            catanatron_commit=manifest['catanatron_commit'])
        assert row['settings'] == expected_settings
        rec = from_json(json.dumps(row['record']))
        assert rec.num_players == n and rec.first == 0
        assert rec.seed == f'{seed}:{index}:game' and rec.seed not in seen_seeds
        seen_seeds.add(rec.seed)
        fields = board_fields(deal_board(seed, index))
        assert all(getattr(rec, key) == value for key, value in fields.items())
        assert sorted(row['seating']) == list(range(n))
        assert row['seating'] == [(e + index) % n for e in range(n)]
        game = replay(rec)
        expected_winner = None if game.won_by is None else row['seating'].index(game.won_by)
        assert row['winner'] == expected_winner and row['turns'] == game.turns
        assert row['points'] == [victory_points(game.state(s, hidden=False), s) for s in row['seating']]
        assert row['exchanges'] == len(rec.trades)
        assert row['heximax_decisions'] > 0
        wins[str(expected_winner)] += 1
        seats[row['seating'][0]] += 1
        alphas.update(row['observed_alphas'])
        exchanges += row['exchanges']
        deadlines += row['deadline_hits']
        actions += len(rec.actions)
    count = wins['0']
    p, z, total = count / 400, 1.959963984540054, 400
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    original = next(r for r in summary['results'] if r['seats'] == n)
    assert original['heximax_wins'] == count and original['games'] == total
    assert original['winners'] == dict(wins)
    assert original['focal_seats'] == {str(s): total//n for s in range(n)}
    assert seats == Counter({s: total//n for s in range(n)})
    assert original['unfinished'] == wins['None']
    assert original['exchanges'] == exchanges and original['deadline_hits'] == deadlines
    assert original['observed_alphas'] == sorted(alphas)
    assert all(abs(a-b) < 1e-5 for a,b in zip(original['wilson95'], (center-half, center+half)))
    results.append(dict(players=n, replayed=400, legal_actions=actions, wins=dict(wins),
        heximax_win_rate=p, wilson95=[center-half, center+half], focal_seats=dict(seats),
        exchanges=exchanges, observed_alphas=sorted(alphas), deadline_hits=deadlines))
preflight = json.loads((root/'preflight-summary.json').read_text())
assert preflight['games'] == 12
for n in (2,4):
    for i in range(n):
        pair = [json.loads((root/'preflight'/f'{n}-{i}-{mode}.json').read_text()) for mode in ('off','fast')]
        if summary['mode'] == 'fast':
            assert pair[0]['record'] == pair[1]['record']
            assert pair[0]['winner'] == pair[1]['winner']
            assert not any(r['deadline_hits'] for r in pair)
report = dict(passed=True, games_replayed=800, distinct_game_seeds=len(seen_seeds),
              preflight_games_excluded=12, results=results)
(root/'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2))
