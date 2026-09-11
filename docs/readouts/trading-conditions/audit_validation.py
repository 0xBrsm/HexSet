"""Independent completed-record checks; no simulation or result selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, variance
import tarfile
from activity import TradeActivity
from analyze import records

PRIMARY = ('quiet','moderate','active','mixed','rising','falling')
REGIMES = (*PRIMARY, 'none', 'full')


def audit(archive):
    groups = records(archive, 'validation')
    assert len(groups) == 26
    summary = []
    corrections = 0
    for ri, regime in enumerate(REGIMES):
        candidates = ['M:T:0.0', 'A:T:0.0', 'T:T:0.0']
        n = 256 if regime in PRIMARY else 128
        if regime not in PRIMARY:
            candidates.append('N:T:0.0')
        for candidate in candidates:
            g = groups[regime, candidate]
            assert len(g) == n
            for row in g.values():
                identity = row['identity']
                assert identity['seed'] == 722000000 + ri*10000
                assert identity['parameters']['max_trades'] is None
                assert not identity['antithetic']
                corrections += row['reported_candidate_seat'] != row['candidate_seat']
                events = row['public_trade_events']
                assert [(e['turn'], a, b) for e in events for a,b in e['trade_participants']] == [
                    (t['turn'],t['a'],t['b']) for t in row['trades']]
                observer = TradeActivity()
                i = 0
                for state in row['activity_trace']:
                    while i < len(events) and events[i]['turn'] < state['turn']:
                        observer.observe(**events[i]); i += 1
                    assert observer.mean == state['activity']
            summary.append(dict(regime=regime,candidate=candidate,games=n,
                wins=sum(r['winner']==0 for r in g.values()),
                domestic_trades=sum(r['domestic_trades'] for r in g.values()),
                candidate_trades=sum(r['candidate_trades'] for r in g.values()),
                mean_final_activity=mean(r['activity_trace'][-1]['activity'] for r in g.values())))
    contrasts=[]
    for a,b in [('M:T:0.0','T:T:0.0'),('A:T:0.0','M:T:0.0')]:
        means=[]; variances=[]
        for regime in PRIMARY:
            ga=groups[regime,a]; gb=groups[regime,b]
            assert all(ga[i]['seating']==gb[i]['seating'] for i in ga)
            d=[int(ga[i]['winner']==0)-int(gb[i]['winner']==0) for i in sorted(ga)]
            means.append(mean(d)); variances.append(variance(d)/len(d))
        estimate=mean(means)
        half=NormalDist().inv_cdf(.9875)*math.sqrt(sum(variances))/len(PRIMARY)
        contrasts.append(dict(a=a,b=b,difference=estimate,interval=[estimate-half,estimate+half]))
    with tarfile.open(archive) as t:
        original=json.load(t.extractfile('validation/summary.json'))
    for corrected, old in zip(contrasts,original['primary']):
        assert corrected['difference']==old['difference'] and corrected['interval']==old['interval']
    for row in summary:
        old=next(r for r in original['rows'] if (r['regime'],r['candidate'])==(row['regime'],row['candidate']))
        assert all(row[k]==old[k] for k in ('wins','games','domestic_trades'))
    return dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        records=sum(len(g) for g in groups.values()),seat_metadata_corrections=corrections,
        public_activity_replay='every logged value reproduced exactly from public events',
        primary=contrasts,rows=summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=audit(args.archive)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
