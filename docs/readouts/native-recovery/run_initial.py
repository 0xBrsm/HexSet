"""Verified preflight followed by the bounded initial native screen."""
import json
from pathlib import Path
import subprocess
import sys

MANIFEST='/study/docs/readouts/native-recovery/manifest-v1.json'

def run(out,workers,mode):
    subprocess.run([sys.executable,'-m','hexset.bench.native_search','--manifest',MANIFEST,
        '--candidates','control','--seed','710000000','--games','4','--workers',str(workers),
        '--audit','--speedups',mode,'--out','/out/'+out],check=True)

if __name__=='__main__':
    run('preflight-one',1,'fast')
    run('preflight-many',4,'fast')
    run('preflight-off',4,'off')
    fields=('winner','seat','turns','points','seating','actions','action_sha256','informed_decisions')
    files=sorted(Path('/out/preflight-one/games').glob('*.json'))
    assert len(files)==8
    for p in files:
        expected=json.loads(p.read_text())
        for directory in ('preflight-many','preflight-off'):
            actual=json.loads((Path('/out')/directory/'games'/p.name).read_text())
            assert all(expected[k]==actual[k] for k in fields),(directory,p.name)
    Path('/out/preflight-verified.json').write_text(json.dumps({'games':24,
        'matched_repeat_comparisons':16,'fields':fields,'public_ledger_audit':True},indent=2)+'\n')
    print('Native ledger, worker-count and AB2 acceleration preflight passed.',flush=True)
    subprocess.run([sys.executable,'-m','hexset.bench.native_search','--manifest',MANIFEST,
        '--candidates','control','p10','road-zero','--seed','711000000','--games','32',
        '--workers','16','--out','/out/screen-v1'],check=True)
