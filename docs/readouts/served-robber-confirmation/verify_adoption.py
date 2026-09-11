"""Two reserved served traces using the promoted default candidate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import hexset
from hexset.bots.heximax import TRADING_WEIGHTS
import run
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
audit=json.loads((args.evidence/'audit.json').read_text())
assert audit['passed'] and audit['eligible_for_adoption'] and TRADING_WEIGHTS.robber_risk==.075
assert audit['total_attempts']+2<=408
started=time.monotonic();original=run.heximax;defaults=0
prior_path=args.out/'first-check-receipt.json'
prior=json.loads(prior_path.read_text()) if prior_path.exists() else {'seconds':0}
if prior_path.exists():
 assert (args.out/'0000.json').exists() and not (args.out/'0001.json').exists()


def candidate_default(*a,**kw):
 global defaults
 if kw['trade_weights'].robber_risk==.075:
  del kw['trade_weights'];defaults+=1
 return original(*a,**kw)

run.heximax=candidate_default
for i in range(2):
 destination=args.out/f'{i:04d}.json'
 if destination.exists():
  assert i==0 and prior_path.exists();defaults+=2
 else:run.play((i,run.SEED,'candidate',str(destination)))
 got=json.loads(destination.read_text())
 expected=json.loads((args.evidence/'games'/f'{i:04d}.json').read_text())
 assert got['record']==expected['record'] and got['notes']==expected['notes']
 assert got['profiles']==expected['profiles'] and got['winner']==expected['winner']
assert defaults==4
seconds=time.monotonic()-started+prior['seconds']
assert audit['total_evaluation_seconds']+seconds<900
run.atomic(args.out/'verification.json',dict(passed=True,games=2,default_candidate_instances=defaults,
 total_attempts=audit['total_attempts']+2,total_evaluation_seconds=audit['total_evaluation_seconds']+seconds,
 source_sha256=run.fingerprint(Path(hexset.__file__).parent),exchange_weights=asdict(TRADING_WEIGHTS),
 check='Promoted default versus frozen served confirmation full action/chance and round-note traces'))
