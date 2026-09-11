"""Verify frozen default traces, then run the registered expansion screen."""
import json
from pathlib import Path
from hexset.bench.native_search import run, atomic

def main():
    root=Path('/out')
    config=json.loads(Path('/study/docs/readouts/native-recovery/manifest-v8.json').read_text())['candidates']
    result=run(config,['control'],['ab2','shipped'],710000000,4,4,root/'expansion-preflight',audit=True)
    keys=('winner','seat','turns','points','seating','actions','action_sha256','informed_decisions')
    comparisons=[]
    for new_path in sorted((root/'expansion-preflight/games').glob('*.json')):
        new=json.loads(new_path.read_text())
        old=json.loads((root/'preflight-many/games'/new_path.name).read_text())
        assert {k:new[k] for k in keys}=={k:old[k] for k in keys},new_path.name
        assert new['identity']['phenotype']==old['identity']['phenotype']
        assert new['identity']['frozen_shipped']==old['identity']['frozen_shipped']
        comparisons.append(new_path.name)
    atomic(root/'expansion-preflight-verified.json',dict(comparisons=comparisons,source_sha256=result['source_sha256'],old_source_sha256=old['identity']['source_sha256']))
    run(config,['p10-exp025','p10-exp05'],['ab2','shipped'],711000000,64,16,root/'screen-expansion')


if __name__ == '__main__':
    main()
