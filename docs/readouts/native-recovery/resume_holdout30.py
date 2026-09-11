"""Same registered holdout, with 30 workers after the user's resource correction."""
import json
from pathlib import Path
from hexset.bench.native_search import run
from run_local_followon import ROOT, GATES, indexed, lower, state


def main():
    prior=json.loads((ROOT/'local-followon-state.json').read_text())
    assert prior['stage']=='holdout'
    candidate=prior['candidate']
    choices=[]
    for path in (ROOT/'confirmation-local').glob('summary-715000000-512-*.json'):
        result=json.loads(path.read_text())
        if {r['candidate'] for r in result['rows']}=={candidate}:
            choices.append(result)
    assert len(choices)==1
    rows=indexed(choices[0])
    assert all(rows[candidate,g]['games']==512 for g in GATES)
    assert rows[candidate,'ab2']['wins']>=269 and rows[candidate,'shipped']['wins']>=144
    config=json.loads(Path('/configs/manifest-v10.json').read_text())['candidates']
    state('holdout',candidate=candidate,games_per_gate=2048,seed=716000000,
          attempt=1,alpha=.025,workers=30,worker_change='User prioritized using available 30-worker capacity; completed checkpoints retained.')
    holdout=run(config,[candidate],GATES,716000000,2048,30,ROOT/'holdout-local-attempt1')
    verdict=[]
    for row in holdout['rows']:
        lo95=lower(row['wins'],row['games'],.05)
        adjusted=lower(row['wins'],row['games'],.025)
        threshold=.5 if row['gate']=='ab2' else .25
        verdict.append(dict(gate=row['gate'],wins=row['wins'],games=row['games'],
                            lower95=lo95,adjusted_lower=adjusted,threshold=threshold,
                            passed=lo95>threshold and adjusted>threshold))
    state('holdout-complete',candidate=candidate,attempt=1,workers=30,verdict=verdict,
          endpoints_passed=all(v['passed'] for v in verdict))


if __name__=='__main__':
    main()
