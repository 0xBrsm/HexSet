"""At most 12 reserved full-game checks after a successful audited confirmation."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import json
import multiprocessing
from pathlib import Path
import time
import hexset
import hexset.arena as arena
from hexset.bots.heximax import heximax, TRADING_WEIGHTS
from hexset.bots.evaluate import Weights
from hexset.record import to_json
from run import atomic, fingerprint


def check(job):
    family,index,study,ab2 = job
    manifest=json.loads((Path(study)/'manifest.json').read_text())
    baseline=Weights(**manifest['profiles']['control'])
    selection=json.loads((Path(study)/'selection.json').read_text())['candidate']
    assert asdict(TRADING_WEIGHTS)==manifest['profiles'][selection]
    if family=='confirmation':
        reference=json.loads((Path(study)/'confirmation'/f'{index:04d}.json').read_text())
        def factory(entrant,board,rng):
            return heximax(board,rng,trade_weights=baseline if entrant.name=='control' else None)
        arena.register_entrant_kind('adopted-exchange',factory)
        entrants=[arena.Entrant(p,kind='adopted-exchange') for p in [selection]*2+['control']*2]
        result=arena._play_one((tuple(entrants),index,manifest['confirmation_seed'],arena.MAX_ACTIONS,False,True))
    else:
        n=int(family)
        reference=json.loads((Path(ab2)/'games'/f'{n}-{index:04d}.json').read_text())
        from hexset.catanatron.speedups import catanatron_speedups, verify_runtime
        from importlib.metadata import distribution
        direct=json.loads(distribution('catanatron').read_text('direct_url.json'))
        assert direct['vcs_info']['commit_id']==reference['identity']['catanatron_commit']
        verify_runtime()
        entrants=arena.lineup_from_names(['heximax']+['catanatron']*(n-1))
        with catanatron_speedups(reference['identity']['mode']):
            result=arena._play_one((tuple(entrants),index,reference['identity']['seed'],arena.MAX_ACTIONS,False,True))
    assert json.loads(to_json(result.record))==reference['record']
    assert result.winner==reference['winner'] and list(result.points)==reference['points']
    return dict(family=family,index=index,full_record_match=True,winner=result.winner)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--ab2',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();study=Path(args.study)
    audit=json.loads((study/'audit.json').read_text())
    verdict=json.loads((study/'verdict.json').read_text())
    assert audit['passed'] and audit['eligible_for_adoption'] and verdict['eligible_for_adoption']
    assert verdict['games']+12<=1600
    started=time.monotonic()
    jobs=[(family,i,args.study,args.ab2) for family in ('confirmation','2','4') for i in range(4)]
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        checks=list(pool.map(check,jobs))
    seconds=time.monotonic()-started
    assert verdict['seconds']+seconds<900
    atomic(args.out,dict(passed=True,games=12,total_evaluation_games=verdict['games']+12,
        seconds=seconds,total_evaluation_seconds=verdict['seconds']+seconds,
        adopted_source_sha256=fingerprint(Path(hexset.__file__).parent),checks=checks))


if __name__=='__main__':main()
