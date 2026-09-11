"""Registered conditional continuation; never changes candidates or sample sizes."""
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from hexset.bench.native_search import run, atomic

ROOT=Path('/out')
SCREEN=ROOT/'screen-expansion-local'
PARENT='p10-exp025'
CANDIDATES=[PARENT,'p10-exp0125','p10-exp0375','road-zero-exp025','p10-exp025-prod2785']
GATES=['ab2','shipped']


def indexed(result):
    return {(r['candidate'],r['gate']):r for r in result['rows']}


def margin(rows, candidate):
    return min(rows[candidate,g]['rate']-(.5 if g=='ab2' else .25) for g in GATES)


def state(stage, **kwargs):
    value=dict(stage=stage,**kwargs)
    atomic(ROOT/'local-followon-state.json',value)
    print(json.dumps(value),flush=True)


def lower(wins,n,alpha):
    z=NormalDist().inv_cdf(1-alpha/2)
    p=wins/n
    return (p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def main():
    config=json.loads(Path('/configs/manifest-v10.json').read_text())['candidates']
    choices=[]
    for path in SCREEN.glob('summary-714000000-64-*.json'):
        result=json.loads(path.read_text())
        if {r['candidate'] for r in result['rows']}==set(CANDIDATES):
            choices.append(result)
    assert len(choices)==1,'expected one complete initial screen'
    screen=choices[0]
    assert screen['source_sha256']=='efaecd276604bc23c7d16a23a996cc3815064353dd2d453e52f146fc21d347b1'
    rows=indexed(screen)
    assert all(rows[c,g]['games']==64 for c in CANDIDATES for g in GATES)
    selected=[c for c in CANDIDATES if c!=PARENT and rows[c,'ab2']['wins']>=32
              and rows[c,'shipped']['wins']>=18 and margin(rows,c)>margin(rows,PARENT)]
    selected=sorted(selected,key=lambda c:(-margin(rows,c),c))[:2]
    if not selected:
        state('stopped',reason='no local arm met the registered extension rule')
        return
    state('extension',candidates=[PARENT,*selected],games_per_gate=256,seed=714000000)
    extended=run(config,[PARENT,*selected],GATES,714000000,256,16,SCREEN)
    rows=indexed(extended)
    viable=[c for c in selected if rows[c,'ab2']['wins']>=131 and rows[c,'shipped']['wins']>=72
            and margin(rows,c)>margin(rows,PARENT)]
    if not viable:
        state('stopped',reason='no changed arm justified fresh confirmation',extension=extended)
        return
    candidate=sorted(viable,key=lambda c:(-margin(rows,c),c))[0]
    state('confirmation',candidate=candidate,games_per_gate=512,seed=715000000)
    confirmation=run(config,[candidate],GATES,715000000,512,16,ROOT/'confirmation-local')
    rows=indexed(confirmation)
    if rows[candidate,'ab2']['wins']<269 or rows[candidate,'shipped']['wins']<144:
        state('stopped',reason='registered confirmation advancement rule failed',confirmation=confirmation)
        return
    state('holdout',candidate=candidate,games_per_gate=2048,seed=716000000,attempt=1,alpha=.025)
    holdout=run(config,[candidate],GATES,716000000,2048,16,ROOT/'holdout-local-attempt1')
    verdict=[]
    for row in holdout['rows']:
        lo95=lower(row['wins'],row['games'],.05)
        adjusted=lower(row['wins'],row['games'],.025)
        threshold=.5 if row['gate']=='ab2' else .25
        verdict.append(dict(gate=row['gate'],wins=row['wins'],games=row['games'],
                            lower95=lo95,adjusted_lower=adjusted,threshold=threshold,
                            passed=lo95>threshold and adjusted>threshold))
    state('holdout-complete',candidate=candidate,attempt=1,verdict=verdict,
          endpoints_passed=all(v['passed'] for v in verdict))


if __name__=='__main__':
    main()
