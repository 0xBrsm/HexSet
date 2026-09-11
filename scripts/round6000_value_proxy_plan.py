#!/usr/bin/env python3
"""Plan, without executing, a four-candidate ValueFunction proxy calibration."""
import argparse, hashlib, json, shlex
from pathlib import Path

SOURCE_HASH="11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
IMAGE="58c092875440"
IMAGE_SHA="58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be"
CONTROLLER_SHA="9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c"
CANDIDATES=("g6000-p00","g6000-p04","g6000-p05","g6000-p10")
BASE_SEED=610000000
GAMES=240

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--output-root",default="/home/bsm/tmp/luna-round6000-value-proxy"); a=p.parse_args()
 m=json.loads(a.manifest.read_text()); pop={c["id"]:c for c in m["population"]}
 if list(pop)!=[f"g6000-p{i:02d}" for i in range(12)]: raise ValueError("manifest population/order mismatch")
 if m.get("controller_source_hash")!=SOURCE_HASH: raise ValueError("source hash mismatch")
 files=sorted((a.raw_root/"games"/"generation-6000").glob("*.json")); by={}
 for f in files:
  d=json.loads(f.read_text()); key=(d.get("candidate"),d.get("gate"),d.get("game_index"))
  if d.get("candidate") in CANDIDATES and d.get("gate")=="vs-ab2" and d.get("stage")=="discovery": by[key]=d
 for cid in CANDIDATES:
  if len([k for k in by if k[0]==cid])!=GAMES: raise ValueError("missing archived AB2 records for "+cid)
 jobs=[]
 for cid in CANDIDATES:
  for i in range(GAMES):
   seed=BASE_SEED+i; out=f"{cid}-vs-value-discovery-6000-{i:04d}.json"
   lineup=f"DC:heximax-evolve-{cid},F,F,F"
   jobs.append({"candidate":cid,"index":i,"seed":seed,"lineup":lineup,"out":f"{a.output_root}/{out}","archived_ab2_record":str(next(f for f in files if f.name==f"{cid}-vs-ab2-discovery-6000-{i:04d}.json"))})
 plan={"schema":1,"status":"PLANNED_NO_EXECUTION","purpose":"calibrate native ValueFunction F as a selection proxy; no equivalence or strength claim","source":{"source_hash":SOURCE_HASH,"controller_file_sha256":CONTROLLER_SHA,"image":IMAGE,"image_sha256":IMAGE_SHA,"manifest_sha256":sha256(a.manifest)},"native_identity":{"value_cli":"F","value_class":"catanatron.players.value.ValueFunctionPlayer","value_builder":"base_fn","value_params":"DEFAULT_WEIGHTS","ab2_cli":"AB:2","ab2_depth":2,"ab2_params":"DEFAULT_WEIGHTS","equivalence":"unproven"},"population":CANDIDATES,"games_per_candidate":GAMES,"seed_range":[BASE_SEED,BASE_SEED+GAMES-1],"workers":30,"lineup":"DC:heximax-evolve-{candidate},F,F,F","jobs":jobs,"comparison":{"archived_gate":"vs-ab2","archived_seed_range":[610000000,610000239],"metrics":["per-game same-seed outcome difference (descriptive; lineups differ)","candidate win-rate rank agreement","candidate win-rate difference","walltime per game"]},"decision_rule":"Use as a speed-selection proxy only if rank agreement and paired differences are preregistered and acceptable; rerun final AB2/self gates unchanged."}
 a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_suffix(a.out.suffix+".tmp"); tmp.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n"); tmp.replace(a.out); print(json.dumps({k:plan[k] for k in ('status','population','games_per_candidate','seed_range','workers','native_identity','comparison')},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
