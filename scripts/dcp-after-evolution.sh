#!/bin/bash
set -euo pipefail
C=${C:-kind_raman}; F=${F:-/home/bsm/tmp/hexset-luna-followon-certified}; O=${O:-/home/bsm/tmp/hexset-followon-results/dcp};
while [ "$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || true)" = running ]; do sleep 30; done
[ -s "${PIPELINE_OUT:-/home/bsm/tmp/hexset-followon-results}/result.json" ] || exit 0
while docker ps --format '{{.Names}}' | grep -Eq 'kind_raman|luna-heximax'; do sleep 30; done
mkdir -p "$O"
for spec in 'dcp ab2 3300000' 'dcp shipped 3301000' 'control ab2 3400000' 'control shipped 3401000'; do set -- $spec; out="$O/$1-$2-120-$3.json"; [ -e "$out" ] && continue; mod=dclcfg; [ "$1" = control ] && mod=bankcfg; docker run --rm --network none --cpus 30 --user "$(id -u):$(id -g)" -v "$F:/followon" -v "$O:/out" -w /followon -e PYTHONPATH=/followon/src --entrypoint python 58c092875440 -m hexset.catanatron."$mod" --candidate "$1" --gate "$2" --games 120 --workers 30 --seed "$3" --out "/out/$(basename "$out")"; done
