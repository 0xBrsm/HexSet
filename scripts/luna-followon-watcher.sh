#!/bin/bash
set -euo pipefail
STUDY=${STUDY:-/home/bsm/tmp/hexset-validation}; POST=${POST:-luna-heximax-post-stance}
FOLLOWON=${FOLLOWON:-/home/bsm/tmp/hexset-luna-followon-certified}; OUT=${OUT:-/home/bsm/tmp/hexset-followon-results}
CLI=${CLI:-followon_pipeline}
while :; do
  state=$(docker inspect -f '{{.State.Status}}' "$POST" 2>/dev/null || true)
  if [ "$state" = running ]; then sleep 30; continue; fi
  [ -s "$FOLLOWON/readiness.json" ] || { sleep 30; continue; }
  status="$STUDY/validation/status.json"; [ -s "$status" ] || status="$STUDY/validation/verdict.json"
  [ -s "$status" ] || { sleep 30; continue; }
  rel=${status#"$STUDY/"}
  verdict=$(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$STUDY:/study" --entrypoint python 58c092875440 -c 'import json,sys; print(json.load(open("/study/"+sys.argv[1])).get("status", ""))' "$rel")
  case "$verdict" in
    PASS) exit 0;;
    FAIL) [ -s "$STUDY/validation/confirmation.json" ] || { sleep 30; continue; }; prior=validation/handoff-prior.json; docker run --rm --network none --user "$(id -u):$(id -g)" -v "$STUDY:/study" --entrypoint python 58c092875440 -c 'import json; d=json.load(open("/study/validation/confirmation.json")); d["status"]="HOLDOUT_REJECTED"; d["rows"]=[dict(r,candidate_wins=int(r["wins"].get("Color.RED",r["wins"].get("0",-1)))) for r in d["rows"]]; json.dump(d,open("/study/validation/handoff-prior.json","w"))';;
    HOLDOUT_REJECTED|UNRESOLVED_CAP|REJECTED_AT_LOOK) prior=${rel};;
    *) sleep 30; continue;;
  esac
  if docker ps --format '{{.Names}}' | grep -Eq 'luna-heximax-(validation|stance|post-stance)'; then sleep 30; continue; fi
  if [ -s "$STUDY/evolve-checkpoint.json" ]; then checkpoint_hash=$(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$STUDY:/study" --entrypoint python 58c092875440 -c 'import json; print(json.load(open("/study/evolve-checkpoint.json"))["source_hash"])'); else checkpoint_hash=$(docker run --rm --network none -v "$FOLLOWON:/post" --entrypoint python 58c092875440 -c 'import json; print(json.load(open("/post/readiness.json"))["source_tree_sha256"])'); fi
  mkdir -p "$OUT"
  exec docker run --rm --network none --cpus 30 --user "$(id -u):$(id -g)" -v "$FOLLOWON:/followon" -v "$STUDY:/study" -v "$OUT:/out" -w /followon -e PYTHONPATH=/followon/src --entrypoint python 58c092875440 -m hexset.catanatron."$CLI" --prior "/study/$prior" --checkpoint /study/evolve-checkpoint.json --root /out --source-hash "$checkpoint_hash"
done
