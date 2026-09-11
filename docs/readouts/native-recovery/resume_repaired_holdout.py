"""Narrow, audited compatibility continuation of the original fixed holdout."""
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import numpy
import networkx
from hexset.bench.native_search import initialize, play_job, atomic, source_hash, FROZEN
from hexset.catanatron.speedups import verify_runtime
from run_local_followon import lower

ROOT=Path('/out')
OLD=ROOT/'holdout-local-attempt1/games'
NEW=ROOT/'holdout-local-attempt1-repaired/games'
OLD_SHA='efaecd276604bc23c7d16a23a996cc3815064353dd2d453e52f146fc21d347b1'
NEW_SHA='8c677b8f7475ec45fe02ecd5e1ba1d9e18415956b0f594666d7da4a3de8ddafe'
CANDIDATE='road-zero-exp025'
REPAIR='stranded-road-building-empty-offer-v1'


def name(gate,index):
    return f'{CANDIDATE}-{gate}-716000000-{index:05d}.json'


def patched(identity):
    return {**identity,'source_sha256':NEW_SHA,'adapter_repair':REPAIR}


def job(identity,path):
    return dict(identity=identity,path=str(path))


def execute(item):
    try:
        return play_job(item)
    except Exception as exc:
        i=item['identity']
        raise RuntimeError(f"holdout failed: {i['gate']} index {i['index']}") from exc


def main():
    verify_runtime()
    assert source_hash()==NEW_SHA
    config={**json.loads(Path('/configs/manifest-v10.json').read_text())['candidates'],
            'frozen-shipped':FROZEN}
    initialize(config)
    template=json.loads((OLD/name('ab2',0)).read_text())['identity']
    assert template['source_sha256']==OLD_SHA
    assert template['candidate']==CANDIDATE and template['seed']==716000000
    assert template['phenotype']==config[CANDIDATE] and template['frozen_shipped']==FROZEN
    assert template['runtime']==dict(python=sys.version,numpy=numpy.__version__,networkx=networkx.__version__)
    assert template['host']=='hexset' and template['information_model']=='native-public-history'
    assert template['antithetic'] is False and template['audit'] is False and template['speedups']=='fast'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD.glob('*.json')}
    hashfile=ROOT/'holdout-legacy-record-hashes.json'
    if hashfile.exists():
        assert json.loads(hashfile.read_text())==hashes
    else:
        atomic(hashfile,hashes)
    proof=[]
    keys=('winner','seat','turns','points','seating','actions','action_sha256','informed_decisions')
    for index in [0,511,1023,1535]:
        old=json.loads((OLD/name('ab2',index)).read_text())
        replay=execute(job(patched(old['identity']),ROOT/'repair-preflight'/name('ab2',index)))
        assert {k:old[k] for k in keys}=={k:replay[k] for k in keys},index
        proof.append(index)
    atomic(ROOT/'holdout-repair-preflight.json',dict(original_source=OLD_SHA,patched_source=NEW_SHA,exact_trace_replays=proof,legacy_records=len(hashes)))
    print('Four legacy traces matched exactly.',flush=True)
    # Complete the failing fixed-seed game first; this is a real holdout
    # record, not an extra sample or a substitute for the failed game.
    failing={**template,'gate':'ab2','index':1935}
    execute(job(patched(failing),NEW/name('ab2',1935)))
    print('Original failing index 1935 completed.',flush=True)
    rows=[];jobs=[]
    for gate in ['ab2','shipped']:
        for index in range(2048):
            original={**template,'gate':gate,'index':index}
            old_path=OLD/name(gate,index)
            if old_path.exists():
                row=json.loads(old_path.read_text())
                assert row['identity']==original and row['complete'] and row['winner'] is not None
                assert not (NEW/name(gate,index)).exists(),'duplicate holdout game'
                rows.append(row)
            else:
                jobs.append(job(patched(original),NEW/name(gate,index)))
    with get_context('spawn').Pool(30,initializer=initialize,initargs=(config,)) as pool:
        for row in pool.imap_unordered(execute,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%32==0:print(f'{len(rows)}/4096 fixed holdout games ready',flush=True)
    assert len(rows)==4096
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD.glob('*.json')}
    verdict=[]
    for gate,threshold in [('ab2',.5),('shipped',.25)]:
        group=[r for r in rows if r['identity']['gate']==gate]
        assert len(group)==2048 and {r['identity']['index'] for r in group}==set(range(2048))
        wins=sum(r['winner']==0 for r in group)
        lo=lower(wins,2048,.05);adjusted=lower(wins,2048,.025)
        verdict.append(dict(gate=gate,wins=wins,games=2048,rate=wins/2048,lower95=lo,
                            adjusted_lower=adjusted,threshold=threshold,passed=lo>threshold and adjusted>threshold))
    result=dict(stage='holdout-complete',candidate=CANDIDATE,seed=716000000,attempt=1,alpha=.025,
                original_source_sha256=OLD_SHA,patched_source_sha256=NEW_SHA,adapter_repair=REPAIR,
                legacy_records=len(hashes),new_records=4096-len(hashes),verdict=verdict,
                endpoints_passed=all(r['passed'] for r in verdict))
    atomic(ROOT/'local-followon-state.json',result)
    atomic(ROOT/'holdout-repaired-verdict.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
