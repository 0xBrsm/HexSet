"""Summarize a completed native checkpoint archive; no games are launched.

Example: python analyze.py artifacts/stage4.tar.gz --root screen-v1 --seed 711000000
Selection reports are exploratory. A holdout verdict requires its registered
fixed sample size and attempt-specific alpha, supplied explicitly below.
"""
import argparse
from collections import defaultdict
import json
from math import sqrt
from statistics import NormalDist
import tarfile


def wilson(wins, n, alpha=.05):
    z = NormalDist().inv_cdf(1-alpha/2)
    rate = wins/n
    center = rate + z*z/(2*n)
    radius = z*sqrt(rate*(1-rate)/n + z*z/(4*n*n))
    scale = 1+z*z/n
    return [(center-radius)/scale, (center+radius)/scale]


def analyze(path, root, seed, limit=None, holdout_games=None, attempt=None):
    groups = defaultdict(dict)
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.lstrip('./').startswith(root+'/games/') or not member.name.endswith('.json'):
                continue
            row = json.load(archive.extractfile(member))
            ident = row['identity']
            if ident['seed'] != seed or (limit is not None and ident['index'] >= limit):
                continue
            assert row['complete'] and row['winner'] is not None
            assert ident['host'] == 'hexset' and ident['information_model'] == 'native-public-history'
            assert ident['antithetic'] is False
            group = groups[ident['candidate'], ident['gate']]
            assert ident['index'] not in group, 'duplicate checkpoint'
            group[ident['index']] = row
    result = []
    for (candidate, gate), rows in sorted(groups.items()):
        n = len(rows)
        assert n % 4 == 0 and set(rows) == set(range(n)), 'partial or noncontiguous block'
        identities = {json.dumps({k:v for k,v in r['identity'].items() if k!='index'},sort_keys=True) for r in rows.values()}
        assert len(identities)==1, 'mixed provenance'
        wins = sum(r['winner']==0 for r in rows.values())
        item = dict(candidate=candidate, gate=gate, games=n, wins=wins, rate=wins/n,
                    wilson95=wilson(wins,n), threshold=.5 if gate=='ab2' else .25)
        if holdout_games is not None:
            assert n==holdout_games and attempt and attempt>0, 'holdout must be complete and identify its registered attempt'
            alpha=.05/(attempt*(attempt+1))
            item.update(attempt=attempt, alpha=alpha, adjusted_interval=wilson(wins,n,alpha))
            item['gate_passed']=item['adjusted_interval'][0]>item['threshold']
        result.append(item)
    if not result:
        raise ValueError('no matching records')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive');p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int)
    p.add_argument('--holdout-games',type=int);p.add_argument('--attempt',type=int)
    args=p.parse_args()
    print(json.dumps(analyze(args.archive,args.root,args.seed,args.limit,args.holdout_games,args.attempt),indent=2))
