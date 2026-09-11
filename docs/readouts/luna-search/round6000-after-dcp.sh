#!/bin/sh
# Inactive reviewed wrapper for the authorized round-6000 existing-controller run.
# Deliberately not invoked during review. It waits for the running DCP self gate,
# performs no-game identity probes, then launches one isolated generation.
set -eu

SOURCE_ROOT=${SOURCE_ROOT:-/home/bsm/tmp/hexset-luna-followon-certified}
INPUT_ROOT=${INPUT_ROOT:-/home/bsm/tmp/luna-round6000-input}
OUT=${OUT:-/home/bsm/tmp/luna-round6000-execution}
IMAGE=${IMAGE:-58c092875440}
CONTAINER=${CONTAINER:-luna-evolve-round6000}
QUEUE_ROOT=${QUEUE_ROOT:-/home/bsm/tmp/luna-20260910-discovery-3000}
DCP_OUT=${DCP_OUT:-$QUEUE_ROOT/dcp-followup}
DCP_AB_CONTAINER=${DCP_AB_CONTAINER:-luna-dcp-ab2-410000}
DCP_SELF_CONTAINER=${DCP_SELF_CONTAINER:-luna-dcp-shipped-410100}
LOCK=${LOCK:-$OUT/.round6000.lock}

SOURCE_HASH=11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb
CONTROLLER_SHA=9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c
INPUT_SHA=9e3fe29f39f079272b1fbfc772e881a7b6dc2e3504fd6b5fd2df924315d60596
MANIFEST_SHA=dd99767f08ac115da9251345113b308ca7f4593bd1b6f6343d32252f1bebc2b3
IMAGE_SHA=58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be
DCP_LEDGER_SHA=f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea
DCP_AB_SHA=6135cdd72efa2a6fec82b663b9e80ed3c0aa4ee5290de152aa090c34df44771c
DCP_SELF_SHA=1280be9ed930ef5b87c0a00c083ec7864989b093a9f84cefb731269805b2ae66
DCP_EXECUTION_MANIFEST_SHA=27a595de2e2a19533d123d2badf85bbc41f4d419ab37ceb48fa3c143c3d8f6d2
DCP_AB_WINS=419
DCP_AB_SEED=410000000
DCP_SELF_SEED=410100000
DCP_IDENTITY_SHA=cde5daf6e6d5ec0b42bb5d4fe12f53a225572c9054d56ed8ff1fe102c9b6c027

MANIFEST=$INPUT_ROOT/round6000-input-manifest.json
PACKAGED_INPUT=$INPUT_ROOT/round6000-input-pristine-rev2.json
PACKAGED_DIR=$INPUT_ROOT/round6000-input-pristine-rev2
AB_FILE=$DCP_OUT/dcp-vs-ab2-1000-410000000.json
SELF_FILE=$DCP_OUT/dcp-vs-shipped-1000-410100000.json
DCP_EXECUTION_MANIFEST=$DCP_OUT/dcp-1000-execution-manifest.json

fail() { echo "round6000: ERROR: $*" >&2; exit 1; }
log() { mkdir -p "$OUT"; echo "$(date -u +%FT%TZ) $*" >> "$OUT/round6000.log"; }

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ -f "$LOCK/pid" ] && [ "$(cat "$LOCK/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$LOCK/pid"
    rmdir "$LOCK" 2>/dev/null || true
  fi
  exit "$rc"
}
write_state() {
  state=$1; code=$2; reason=$3
  tmp="$OUT/.round6000-state.tmp.$$"
  cat > "$tmp" <<JSON
{"schema":1,"status":"$state","exit_code":$code,"reason":"$reason","container":"$CONTAINER","output_root":"$OUT","source_root":"$SOURCE_ROOT","source_hash":"$SOURCE_HASH","controller_file_sha256":"$CONTROLLER_SHA","image_sha256":"$IMAGE_SHA","input_manifest_sha256":"$MANIFEST_SHA","input_checkpoint_sha256":"$INPUT_SHA","generation_start":6000,"generation_end_exclusive":6001,"count":12,"games":240,"total_games":240,"promote_top":4,"seed":600000000,"workers":30,"stage":"discovery","start_method":"fork","pythonhashseed":"0","thread_env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},"luna_shared_cache":{"namespace":"luna_shared_cache","reuse":false,"candidate_game_cache":"disabled","raw_input":"pristine-rev2-only"},"updated_utc":"$(date -u +%FT%TZ)"}
JSON
  mv "$tmp" "$OUT/round6000-state.json"
}
on_int() { write_state INTERRUPTED 130 signal_int 2>/dev/null || true; exit 130; }
on_term() { write_state INTERRUPTED 143 signal_term 2>/dev/null || true; exit 143; }
trap cleanup EXIT
trap on_int INT
trap on_term TERM

mkdir -p "$OUT"
if find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  fail "round6000 output root is not fresh"
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null || true)
  fail "lock held at $LOCK by ${owner:-unknown}"
fi
printf '%s\n' "$$" > "$LOCK/pid"
write_state WAITING_FOR_DCP 0 dcp_dependency
actual_image=$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)
[ "$actual_image" = "sha256:$IMAGE_SHA" ] || fail "pinned image mismatch before preflight: $actual_image"

wait_terminal_zero() {
  name=$1
  while :; do
    state=$(docker inspect "$name" --format '{{.State.Status}}' 2>/dev/null || true)
    case "$state" in
      running|created|restarting|paused) sleep 30 ;;
      exited)
        code=$(docker inspect "$name" --format '{{.State.ExitCode}}' 2>/dev/null || true)
        [ "$code" = 0 ] || fail "$name exited $code"
        return 0
        ;;
      *) fail "$name is absent or has unexpected state '$state'" ;;
    esac
  done
}
wait_terminal_zero "$DCP_AB_CONTAINER"
[ -s "$AB_FILE" ] || fail "missing DCP AB artifact $AB_FILE"
[ "$(sha256sum "$AB_FILE" | awk '{print $1}')" = "$DCP_AB_SHA" ] || fail "DCP AB artifact SHA mismatch"
wait_terminal_zero "$DCP_SELF_CONTAINER"
[ -s "$SELF_FILE" ] || fail "missing DCP self artifact $SELF_FILE"
[ "$(sha256sum "$SELF_FILE" | awk '{print $1}')" = "$DCP_SELF_SHA" ] || fail "DCP self artifact SHA mismatch"
[ -s "$DCP_EXECUTION_MANIFEST" ] || fail "missing DCP execution manifest"
[ "$(sha256sum "$DCP_EXECUTION_MANIFEST" | awk '{print $1}')" = "$DCP_EXECUTION_MANIFEST_SHA" ] || fail "DCP execution manifest SHA mismatch"

set +e
docker run --rm -i --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/followon/src \
  -v "$SOURCE_ROOT:/followon:ro" -v "$DCP_OUT:/dcp:ro" -w /followon \
  --entrypoint python "$IMAGE" - /dcp/dcp-vs-ab2-1000-410000000.json /dcp/dcp-vs-shipped-1000-410100000.json "$DCP_AB_SHA" "$DCP_SELF_SHA" "$DCP_AB_WINS" "$DCP_LEDGER_SHA" "$DCP_AB_SEED" "$DCP_SELF_SEED" <<'PY' > "$OUT/.dcp-identity-probe.$$" 2>&1
import hashlib, json, math, sys
from pathlib import Path
ab, self_path, ab_sha, self_sha, ab_wins, ledger_sha, ab_seed, self_seed = sys.argv[1:]
for path, gate, seed, expected_sha, expected_wins in (
    (ab, "vs-ab2", int(ab_seed), ab_sha, int(ab_wins)),
    (self_path, "vs-shipped", int(self_seed), self_sha, None),
):
    p = Path(path)
    if expected_sha is not None:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha
    d = json.loads(p.read_text())
    assert d["candidate"] == "dcp"
    assert d["gate"] == gate and d["games"] == 1000 and d["workers"] == 30
    assert d["seed"] == seed and d["source_sha256"] == ledger_sha
    assert d["protocol"] == "public-ledger-v1"
    expected_players = ("DCP,AB:2,AB:2,AB:2" if gate == "vs-ab2" else
                        "DCP,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
    assert d["players"] == expected_players
    wins = d["wins"]
    assert isinstance(wins, dict) and wins
    assert all(type(v) is int and v >= 0 for v in wins.values())
    assert sum(wins.values()) == 1000
    reds = [v for k, v in wins.items() if str(k).split(".")[-1].upper() == "RED"]
    assert len(reds) == 1
    if expected_wins is not None:
        assert reds[0] == expected_wins
    points = d["points"]
    assert isinstance(points, dict)
    assert all(isinstance(v, list) and len(v) == 1000 for v in points.values())
    assert all(type(x) in (int, float) and not isinstance(x, bool) and
               math.isfinite(x) and x >= 0 for v in points.values() for x in v)
print("DCP_AB_AND_SELF_IDENTITY_PASS")
PY
probe_status=$?
set -e
[ "$probe_status" -eq 0 ] || fail "DCP identity probe failed; retained $OUT/.dcp-identity-probe.$$"
grep -Fq DCP_AB_AND_SELF_IDENTITY_PASS "$OUT/.dcp-identity-probe.$$" || fail "DCP identity probe sentinel missing"

identity="$DCP_OUT/dcp-1000-followup-identity.json"
[ -s "$identity" ] || fail "missing DCP identity sidecar"
[ "$(sha256sum "$identity" | awk '{print $1}')" = "$DCP_IDENTITY_SHA" ] || fail "DCP identity sidecar SHA mismatch"
actual_dcp_self_sha=$(sha256sum "$SELF_FILE" | awk '{print $1}')
[ "$actual_dcp_self_sha" = "$DCP_SELF_SHA" ] || fail "DCP self artifact SHA mismatch after identity probe"
[ -s "$MANIFEST" ] || fail "missing round6000 input manifest"
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA" ] || fail "input manifest SHA mismatch"

if [ -f "$PACKAGED_INPUT" ] && [ "$(sha256sum "$PACKAGED_INPUT" | awk '{print $1}')" = "$INPUT_SHA" ]; then
  CHECKPOINT=$PACKAGED_INPUT
elif [ -f "$PACKAGED_DIR/checkpoint.json" ]; then
  CHECKPOINT=$PACKAGED_DIR/checkpoint.json
else
  fail "round6000 pristine checkpoint is absent or has no approved SHA"
fi
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$INPUT_SHA" ] || fail "input checkpoint SHA mismatch"

SOURCE_PROBE=$(docker run --rm --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/study/src -v "$SOURCE_ROOT:/study:ro" \
  -w /study --entrypoint python "$IMAGE" -c \
  'import hashlib; from pathlib import Path; from hexset.catanatron.evolve import _source_hash; print(_source_hash()); print(hashlib.sha256(Path("/study/src/hexset/catanatron/evolve.py").read_bytes()).hexdigest())')
set -- $SOURCE_PROBE
[ "${1:-}" = "$SOURCE_HASH" ] || fail "computed controller source hash mismatch: ${1:-missing}"
[ "${2:-}" = "$CONTROLLER_SHA" ] || fail "controller file SHA mismatch: ${2:-missing}"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  fail "round6000 container name already exists: $CONTAINER"
fi
for cid in $(docker ps -q); do
  name=$(docker inspect "$cid" --format '{{.Name}}' 2>/dev/null || true)
  inspected=$(docker inspect "$cid" --format '{{.HostConfig.NanoCpus}} {{json .Config.Cmd}} {{json .Config.Entrypoint}}' 2>/dev/null || true)
  nano=${inspected%% *}
  cmd=${inspected#* }
  case "$name" in
    /luna-dcp-ab2-410000|/luna-dcp-shipped-410100) continue ;;
  esac
  if [ -n "$nano" ] && awk "BEGIN { exit !($nano >= 30000000000) }" 2>/dev/null; then
    fail "another active container has >=30 CPU: $name"
  fi
  if printf '%s\n' "$cmd" | grep -Eiq 'hexset[^[:space:]]*catanatron[^[:space:]]*evolve|catanatron[.]evolve'; then
    fail "another active evolve controller: $name"
  fi
done

CHECKPOINT_PARENT=$(dirname "$CHECKPOINT")
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
set +e
docker run --rm -i --network none --cpus 1 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e PYTHONPATH=/study/src \
  -v "$SOURCE_ROOT:/study:ro" -v "$CHECKPOINT_PARENT:/input:ro" -w /study \
  --entrypoint python "$IMAGE" - "$CHECKPOINT_NAME" "$SOURCE_HASH" "$CONTROLLER_SHA" <<'PY' > "$OUT/.native-preflight.$$" 2>&1
import ast, hashlib, inspect, json, multiprocessing as mp, sys
from dataclasses import asdict, replace
from pathlib import Path
name, expected_source, expected_controller = sys.argv[1:]
assert sys.version_info[:3] == (3, 12, 14)
assert mp.get_start_method() == "fork"
state = json.loads((Path("/input") / name).read_text())
assert state["source_hash"] == expected_source and state["protocol"] == 1
assert state["config"] == {"count": 12, "games": 240, "generations": 6001,
                            "promote_top": 4, "seed": 600000000, "total_games": 240}
assert len(state.get("generations", [])) == 1
record = state["generations"][0]
assert record["generation"] == 3000 and record["complete"] is True
candidates = record["next_candidates"]
assert isinstance(candidates, list) and len(candidates) == 12
assert [c["id"] for c in candidates] == [f"g6000-p{i:02d}" for i in range(12)]
fields = {"buy_progress", "diversity", "knight", "port", "production", "road",
          "robber_risk", "scarce", "spare_card", "victory_point"}
for c in candidates:
    assert (c["depth"], c["width"], c["max_nodes"], c["k"]) == (2, 6, 600, 1)
    assert (c["mode"], c["max_trades"], c["stance"]) == ("notrade", 0, "win")
    assert c["temperature"] == 2.476644394795811 and set(c["weights"]) == fields

import hexset.bots
from hexset.arena import Entrant, entrant_from_name, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.bots.heximax.search import heximax
from hexset.bots.heximax import presets
assert hashlib.sha256(Path("/study/src/hexset/catanatron/evolve.py").read_bytes()).hexdigest() == expected_controller
assert inspect.signature(heximax).parameters["placement"].default is True
tree = ast.parse(inspect.getsource(presets._spawn))
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
         getattr(getattr(n, "func", None), "id", None) == "heximax"]
assert len(calls) == 1 and all(k.arg != "placement" for k in calls[0].keywords)

names = []
expected = {}
for c in candidates:
    pname = "heximax-evolve-" + c["id"]
    effective = replace(NO_TRADE_WEIGHTS,
                        **{k: float(v) for k, v in c["weights"].items() if k != "scarce"})
    assert asdict(effective) == c["weights"]
    register_preset(pname, Entrant(
        pname, kind="heximax", weights=effective, depth=c["depth"], width=c["width"],
        max_nodes=c["max_nodes"], k=c["k"], mode=c["mode"],
        max_trades=c["max_trades"], stance=c["stance"], temperature=c["temperature"]))
    names.append(pname); expected[pname] = c

def resolve(pname):
    e = entrant_from_name(pname)
    return {"name": e.name, "weights": asdict(e.weights), "kind": e.kind,
            "depth": e.depth, "width": e.width, "max_nodes": e.max_nodes,
            "k": e.k, "mode": e.mode, "max_trades": e.max_trades,
            "stance": e.stance, "temperature": e.temperature,
            "entrant_placement": e.placement,
            "internal_placement_default": inspect.signature(heximax).parameters["placement"].default}
ctx = mp.get_context("fork")
with ctx.Pool(1) as pool:
    resolved = pool.map(resolve, names)
assert len(resolved) == 12
for row in resolved:
    c = expected[row["name"]]
    assert row["weights"] == c["weights"]
    for key in ("kind", "depth", "width", "max_nodes", "k", "mode", "max_trades", "stance", "temperature"):
        assert row[key] == ("heximax" if key == "kind" else c[key])
    assert row["entrant_placement"] is False and row["internal_placement_default"] is True
    assert row["weights"]["scarce"] == c["weights"]["scarce"]
print(json.dumps({"status":"ROUND6000_NO_GAME_PREFLIGHT_PASS","registered":len(resolved),
                  "start_method":ctx.get_start_method(),"internal_placement_default":True,
                  "scarce":"baseline_no_trade","candidate_ids":names}, sort_keys=True))
PY
native_status=$?
set -e
[ "$native_status" -eq 0 ] || fail "native no-game preflight failed; retained $OUT/.native-preflight.$$"
grep -Fq ROUND6000_NO_GAME_PREFLIGHT_PASS "$OUT/.native-preflight.$$" || fail "native preflight sentinel missing"

cp "$CHECKPOINT" "$OUT/checkpoint.json"
[ "$(sha256sum "$OUT/checkpoint.json" | awk '{print $1}')" = "$INPUT_SHA" ] || fail "working checkpoint copy changed"
chmod 0644 "$OUT/checkpoint.json"
write_binding() {
  tmp="$OUT/.run-binding.tmp.$$"
  cat > "$tmp" <<JSON
{"schema":1,"status":"PREPARED","source_root":"$SOURCE_ROOT","source_hash":"$SOURCE_HASH","controller_file_sha256":"$CONTROLLER_SHA","image":"$IMAGE","image_sha256":"$IMAGE_SHA","workers":30,"generation":6000,"generation_end_exclusive":6001,"stage":"discovery","count":12,"games_per_candidate_per_gate":240,"total_games":240,"promote_top":4,"seed":600000000,"gates":["vs-ab2","vs-shipped"],"manifest_path":"$MANIFEST","manifest_sha256":"$MANIFEST_SHA","input_checkpoint_path":"$CHECKPOINT","input_checkpoint_sha256":"$INPUT_SHA","start_method":"fork","pythonhashseed":"0","thread_env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},"effective_phenotype":{"kind":"heximax","depth":2,"width":6,"max_nodes":600,"k":1,"mode":"notrade","max_trades":0,"stance":"win","temperature":2.476644394795811,"internal_placement_default":true,"scarce":"baseline_no_trade","registered_candidates":12},"luna_shared_cache":{"namespace":"luna_shared_cache","reuse":false,"candidate_game_cache":"disabled","raw_input":"pristine-rev2-only"},"dcp_dependency":{"ab2_artifact":"dcp-vs-ab2-1000-410000000.json","ab2_sha256":"$DCP_AB_SHA","ab2_wins":419,"shipped_artifact":"dcp-vs-shipped-1000-410100000.json","shipped_sha256":"$DCP_SELF_SHA","identity_sidecar_sha256":"$DCP_IDENTITY_SHA"},"created_utc":"$(date -u +%FT%TZ)"}
JSON
  mv "$tmp" "$OUT/run-binding.json"
  chmod 0444 "$OUT/run-binding.json"
}
write_binding
write_state PREPARED 0 preflight_passed
log ROUND6000_PREFLIGHT_PASS registered=12 source_hash=$SOURCE_HASH
write_state RUNNING 0 controller_started
set +e
docker run --name "$CONTAINER" --network none --cpus 30 --user 1000:1000 \
  -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e PYTHONPATH=/study/src -v "$SOURCE_ROOT:/study:ro" -v "$OUT:/out:rw" -w /study \
  --entrypoint python "$IMAGE" -m hexset.catanatron.evolve --checkpoint /out/checkpoint.json \
  --generation 6000 --generations 6001 --count 12 --games 240 --total-games 240 \
  --promote-top 4 --seed 600000000 --source-hash "$SOURCE_HASH" --run \
  >> "$OUT/round6000.log" 2>&1
status=$?
set -e
container_id=$(docker inspect "$CONTAINER" --format '{{.Id}}' 2>/dev/null || true)
container_started=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}' 2>/dev/null || true)
container_finished=$(docker inspect "$CONTAINER" --format '{{.State.FinishedAt}}' 2>/dev/null || true)
container_exit=$(docker inspect "$CONTAINER" --format '{{.State.ExitCode}}' 2>/dev/null || true)
checkpoint_sha=$(sha256sum "$OUT/checkpoint.json" 2>/dev/null | awk '{print $1}' || true)
binding_sha=$(sha256sum "$OUT/run-binding.json" 2>/dev/null | awk '{print $1}' || true)
execution_status=ERROR
[ "$status" -eq 0 ] && execution_status=COMPLETE
execution_tmp="$OUT/.round6000-execution-manifest.tmp.$$"
cat > "$execution_tmp" <<JSON
{"schema":1,"status":"$execution_status","exit_code":$status,"container":{"name":"$CONTAINER","id":"$container_id","started":"$container_started","finished":"$container_finished","reported_exit_code":"$container_exit"},"source_root":"$SOURCE_ROOT","source_hash":"$SOURCE_HASH","controller_file_sha256":"$CONTROLLER_SHA","image":"$IMAGE","image_sha256":"$IMAGE_SHA","output_root":"$OUT","checkpoint_path":"$OUT/checkpoint.json","resulting_checkpoint_sha256":"$checkpoint_sha","input_checkpoint_sha256":"$INPUT_SHA","manifest_sha256":"$MANIFEST_SHA","run_binding_sha256":"$binding_sha","generation_start":6000,"generation_end_exclusive":6001,"count":12,"games":240,"total_games":240,"promote_top":4,"seed":600000000,"workers":30,"stage":"discovery","start_method":"fork","python_version":"3.12.14","pythonhashseed":"0","thread_env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},"luna_shared_cache":{"namespace":"luna_shared_cache","reuse":false,"candidate_game_cache":"disabled","raw_input":"pristine-rev2-only"},"dcp_dependency":{"ab2_sha256":"$DCP_AB_SHA","shipped_sha256":"$DCP_SELF_SHA","execution_manifest_sha256":"$DCP_EXECUTION_MANIFEST_SHA"},"finished_utc":"$(date -u +%FT%TZ)"}
JSON
mv "$execution_tmp" "$OUT/round6000-execution-manifest.json"
chmod 0444 "$OUT/round6000-execution-manifest.json"
if [ "$status" -eq 0 ]; then write_state COMPLETE 0 controller_exit_0; else write_state ERROR "$status" controller_nonzero; fi
log ROUND6000_EXIT status=$status container_id=$container_id checkpoint_sha=$checkpoint_sha
exit "$status"
