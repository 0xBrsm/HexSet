#!/bin/sh
# Deferred queue only: parent review must approve activation.
set -eu
P06=${P06:-luna-discovery-followup-p06}
IMAGE=${IMAGE:-58c092875440}
SRC=${SRC:-/home/bsm/tmp/hexset-luna-followon-certified}
OUT=${OUT:-/home/bsm/tmp/luna-20260910-discovery-3000/followup}
DCP_OUT=${DCP_OUT:-/home/bsm/tmp/luna-20260910-discovery-3000/dcp-followup}
SIDE="$OUT/g3000-selection-sidecar.json"
IDENTITY=${IDENTITY:-/home/bsm/tmp/luna-20260910-discovery-3000/dcp-followup/dcp-1000-followup-identity.json}
LOCK=${LOCK:-/home/bsm/tmp/luna-20260910-discovery-3000/.dcp-followup.lock}
SOURCE_HASH=11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb
LEDGER_SHA=f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea
DCLCFG_SHA=bc8d23876693434009c0ec9847da25bde7dbd4257473bab05518e40bb8f4ac0d
IMAGE_SHA=58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be
SIDECAR_SHA=e04e4f2345e1bd73a0326d243f45fd32c81dabd0cc07f1180ca068e3e8c3f57e
IDENTITY_SHA=cde5daf6e6d5ec0b42bb5d4fe12f53a225572c9054d56ed8ff1fe102c9b6c027

if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null || true)
  echo "ERROR DCP queue lock held at $LOCK by ${owner:-unknown}" >&2
  exit 26
fi
printf '%s\n' "$$" > "$LOCK/pid"
cleanup_lock() {
  if [ -f "$LOCK/pid" ] && [ "$(cat "$LOCK/pid")" = "$$" ]; then
    rm -f "$LOCK/pid"
    rmdir "$LOCK" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while :; do
  status=$(docker inspect "$P06" --format '{{.State.Status}}' 2>/dev/null || true)
  [ "$status" = running ] || break
  sleep 30
done
status=$(docker inspect "$P06" --format '{{.State.Status}}' 2>/dev/null || true)
exit_code=$(docker inspect "$P06" --format '{{.State.ExitCode}}' 2>/dev/null || true)
[ "$status" = exited ] && [ "$exit_code" = 0 ] || { echo "ERROR p06 terminal status=$status exit=$exit_code" >&2; exit 20; }
for f in g3000-p06-vs-ab2-1000.json g3000-p06-vs-shipped-1000.json; do
  [ -s "$OUT/$f" ] || { echo "ERROR missing p06 output $f" >&2; exit 21; }
done
[ -s "$SIDE" ] || { echo "ERROR missing immutable selection sidecar" >&2; exit 22; }
[ -s "$IDENTITY" ] || { echo "ERROR missing immutable DCP identity sidecar" >&2; exit 31; }
actual_identity=$(sha256sum "$IDENTITY" | awk '{print $1}')
[ "$actual_identity" = "$IDENTITY_SHA" ] || { echo "ERROR DCP identity digest $actual_identity" >&2; exit 32; }
actual_sidecar=$(sha256sum "$SIDE" | awk '{print $1}')
[ "$actual_sidecar" = "$SIDECAR_SHA" ] || { echo "ERROR sidecar digest $actual_sidecar" >&2; exit 23; }
actual_image=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[ "$actual_image" = "sha256:$IMAGE_SHA" ] || { echo "ERROR image digest $actual_image" >&2; exit 24; }
actual_source=$(docker run --rm --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/study/src \
  -v "$SRC:/study:ro" -w /study --entrypoint python "$IMAGE" -c \
  'import hashlib; from pathlib import Path; from hexset.catanatron.evolve import _source_hash; import hexset.catanatron.dclcfg as d; print(_source_hash()); print(hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest())')
set -- $actual_source
[ "$1" = "$SOURCE_HASH" ] || { echo "ERROR evolution source hash $1" >&2; exit 27; }
[ "$2" = "$DCLCFG_SHA" ] || { echo "ERROR dclcfg hash $2" >&2; exit 28; }
actual_ledger=$(docker run --rm --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/study/src \
  -v "$SRC:/study:ro" -w /study --entrypoint python "$IMAGE" -c \
  'import hashlib; import hexset.catanatron.public_ledger as m; print(hashlib.sha256(open(m.__file__, "rb").read()).hexdigest())')
[ "$actual_ledger" = "$LEDGER_SHA" ] || { echo "ERROR public ledger hash $actual_ledger" >&2; exit 25; }
# Validate p06 output identity in the pinned image before any DCP process.
docker run --rm --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/study/src \
  -v "$SRC:/study:ro" -v "$OUT:/out:ro" -w /study \
  --entrypoint python "$IMAGE" -c '
import json
from pathlib import Path
side=json.loads(Path("/out/g3000-selection-sidecar.json").read_text())
assert side["selected"]["candidate"] == "g3000-p06"
expected=side["weights_effective"]
for gate,seed,name,lineup in (
 ("vs-ab2",500000000,"g3000-p06-vs-ab2-1000.json","DC:heximax-evolve-g3000-p06,AB:2,AB:2,AB:2"),
 ("vs-shipped",500100000,"g3000-p06-vs-shipped-1000.json","DC:heximax-evolve-g3000-p06,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"),):
 d=json.loads(Path("/out/"+name).read_text())
 assert d["candidate"] == "g3000-p06"
 assert d["weights"] == expected
 assert d["stance"] == side["stance"] and d["temperature"] == side["temperature"]
 assert d["gate"] == gate and d["games"] == 1000 and d["workers"] == 30 and d["seed"] == seed
 assert len(d["rows"]) == 1 and d["rows"][0]["gate"] == gate
 row=d["rows"][0]
 assert row["games"] == 1000 and row["workers"] == 30 and row["seed"] == seed
 assert row["players"] == lineup
 wins=row["wins"]
 assert all(type(v) is int and 0 <= v <= 1000 for v in wins.values())
 completed=sum(wins.values())
 assert completed <= 1000
 points=row["points"]
 assert points and all(isinstance(v,list) and len(v)==completed for v in points.values())
 assert all(type(x) in (int,float) and not isinstance(x,bool) and x >= 0 and __import__("math").isfinite(x) for v in points.values() for x in v)
 draws=1000-completed
 print(gate,"p06 identity PASS; wins/points PASS; completed=",completed,"draws=",draws)
'
mkdir -p "$DCP_OUT"
for gate_seed_name in 'ab2 410000000 dcp-vs-ab2-1000-410000000.json luna-dcp-ab2-410000' 'shipped 410100000 dcp-vs-shipped-1000-410100000.json luna-dcp-shipped-410100'; do
  set -- $gate_seed_name
  gate=$1; seed=$2; name=$3; cname=$4
  [ ! -e "$DCP_OUT/$name" ] || { echo "ERROR existing DCP output $name" >&2; exit 29; }
  if docker inspect "$cname" >/dev/null 2>&1; then echo "ERROR existing DCP container $cname" >&2; exit 30; fi
  docker run --name "$cname" --network none --cpus 30 --user 1000:1000 \
    -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e PYTHONPATH=/followon/src -v "$SRC:/followon:ro" -v "$DCP_OUT:/out:rw" \
    -w /followon --entrypoint python "$IMAGE" -m hexset.catanatron.dclcfg \
    --candidate dcp --gate "$gate" --games 1000 --workers 30 --seed "$seed" \
    --out "/out/$name"
done
# Bind the retained container and output identities after both serial jobs.
ab_sha=$(sha256sum "$DCP_OUT/dcp-vs-ab2-1000-410000000.json" | awk '{print $1}')
self_sha=$(sha256sum "$DCP_OUT/dcp-vs-shipped-1000-410100000.json" | awk '{print $1}')
ab_id=$(docker inspect luna-dcp-ab2-410000 --format '{{.Id}}')
self_id=$(docker inspect luna-dcp-shipped-410100 --format '{{.Id}}')
ab_started=$(docker inspect luna-dcp-ab2-410000 --format '{{.State.StartedAt}}')
self_started=$(docker inspect luna-dcp-shipped-410100 --format '{{.State.StartedAt}}')
ab_finished=$(docker inspect luna-dcp-ab2-410000 --format '{{.State.FinishedAt}}')
self_finished=$(docker inspect luna-dcp-shipped-410100 --format '{{.State.FinishedAt}}')
tmp="$DCP_OUT/.dcp-1000-execution-manifest.tmp"
cat > "$tmp" <<JSON
{
  "schema": 1,
  "status": "COMPLETE",
  "candidate": "dcp",
  "protocol": "public-ledger-v1",
  "source_root": "$SRC",
  "source_hash": "$SOURCE_HASH",
  "dclcfg_sha256": "$DCLCFG_SHA",
  "public_ledger_sha256": "$LEDGER_SHA",
  "identity_sidecar": {"path": "dcp-1000-followup-identity.json", "sha256": "$IDENTITY_SHA"},
  "image": "$IMAGE_SHA",
  "pythonhashseed": "0",
  "thread_env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
  "workers": 30,
  "serial": true,
  "containers": {
    "vs-ab2": {"name": "luna-dcp-ab2-410000", "id": "$ab_id", "started": "$ab_started", "finished": "$ab_finished"},
    "vs-shipped": {"name": "luna-dcp-shipped-410100", "id": "$self_id", "started": "$self_started", "finished": "$self_finished"}
  },
  "outputs": {
    "vs-ab2": {"path": "dcp-vs-ab2-1000-410000000.json", "seed": 410000000, "sha256": "$ab_sha"},
    "vs-shipped": {"path": "dcp-vs-shipped-1000-410100000.json", "seed": 410100000, "sha256": "$self_sha"}
  }
}
JSON
mv "$tmp" "$DCP_OUT/dcp-1000-execution-manifest.json"
chmod 0444 "$DCP_OUT/dcp-1000-execution-manifest.json"
