"""Small opt-in screen for the audited public-ledger Catanatron player."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import hexset.catanatron.public_ledger as _ledger  # register DCL/DCP explicitly
from .duel import run_duel

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--candidate', choices=('dcl','dcp'), required=True)
    p.add_argument('--gate', choices=('ab2','shipped'), required=True)
    p.add_argument('--games', type=int, required=True); p.add_argument('--workers',type=int,required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(argv)
    lineup = ('DCL' if a.candidate=='dcl' else 'DCP')+',AB:2,AB:2,AB:2' if a.gate=='ab2' else ('DCL' if a.candidate=='dcl' else 'DCP')+',DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade'
    r=run_duel(lineup,a.games,a.workers,seed=a.seed)
    doc={'candidate':a.candidate,'gate':'vs-'+a.gate,'games':a.games,'workers':a.workers,'seed':a.seed,'players':lineup,'wins':{str(k):v for k,v in r.wins.items()},'points':{str(k):v for k,v in r.points.items()},'report':r.report(),'source_sha256':hashlib.sha256(Path(_ledger.__file__).read_bytes()).hexdigest(),'protocol':'public-ledger-v1'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(doc,indent=2)+'\n'); print(json.dumps(doc,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
