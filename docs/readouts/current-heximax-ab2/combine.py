"""Verify both preserved batches and pool the current-config headline counts."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

root = Path(__file__).parent
batches = [root, root / 'extension']
rows = []
seen = set()
manifests = []
for batch, seeds in zip(batches, ((727100000, 727200000), (727300000, 727400000))):
    for line in (batch / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((batch / name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((batch / 'manifest.json').read_text())
    manifests.append(manifest)
    audit = json.loads((batch / 'audit.json').read_text())
    assert audit['passed'] and audit['games_replayed'] == 800
    summary = json.loads((batch / 'summary.json').read_text())
    assert summary['runtime'] == manifest
    cohort = []
    with tarfile.open(batch / 'records.tar.gz') as archive:
        for member in archive.getmembers():
            if not re.fullmatch(r'games/[24]-\d{4}.json', member.name):
                continue
            row = json.load(archive.extractfile(member))
            ident = row['identity']
            n, i = ident['seats'], ident['index']
            assert n in (2, 4) and 0 <= i < 400
            assert ident['seed'] == seeds[(n == 4)]
            assert ident['policy_commit'] == manifest['policy_commit']
            assert ident['catanatron_commit'] == manifest['catanatron_commit']
            assert ident['mode'] == summary['mode']
            key = row['record']['seed']
            assert key == f"{ident['seed']}:{i}:game" and key not in seen
            seen.add(key)
            assert row['complete'] and row['settings'][0]['pin_weights'] is None
            cohort.append(row)
    assert len(cohort) == 800
    for n in (2, 4):
        group = [r for r in cohort if r['identity']['seats'] == n]
        assert len(group) == 400
        wins = Counter(str(r['winner']) for r in group)
        recorded = next(r for r in summary['results'] if r['seats'] == n)
        audited = next(r for r in audit['results'] if r['players'] == n)
        assert dict(wins) == recorded['winners'] == audited['wins']
        assert wins['0'] == recorded['heximax_wins']
    rows.extend(cohort)
for key in ('policy_commit', 'catanatron_commit', 'hexset_source_sha256',
            'catanatron_source_sha256', 'python', 'pythonhashseed', 'threads',
            'workers', 'host', 'antithetic'):
    assert manifests[0][key] == manifests[1][key], key
results = []
for n in (2, 4):
    group = [r for r in rows if r['identity']['seats'] == n]
    assert len(group) == 800
    assert all(r['settings'] == group[0]['settings'] for r in group)
    seats = Counter(r['seating'][0] for r in group)
    assert seats == Counter({s: 800 // n for s in range(n)})
    wins = Counter(str(r['winner']) for r in group)
    p, z, total = wins['0']/800, 1.959963984540054, 800
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    results.append(dict(players=n, games=total, heximax_wins=wins['0'], win_rate=p,
        wilson95=[center-half, center+half], winners=dict(wins),
        batch_heximax_wins=[sum(r['winner']==0 for r in group[:400]),
                            sum(r['winner']==0 for r in group[400:])],
        focal_seats=dict(seats), unfinished=wins['None'],
        exchanges=sum(r['exchanges'] for r in group),
        observed_alphas=sorted({a for r in group for a in r['observed_alphas']}),
        deadline_hits=sum(r['deadline_hits'] for r in group)))
report = dict(passed=True, games=1600, distinct_game_seeds=len(seen),
    preflight_games_excluded=24, source_hashes_match=True,
    interval='descriptive 95% Wilson; fixed user-requested second batch after first results',
    results=results)
(root / 'combined-summary.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2))
