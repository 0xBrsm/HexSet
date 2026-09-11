#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC="$ROOT/src"
OUT=${1:?usage: port_liquidity_queue.sh OUTPUT_DIR CONFIRMATION_STATE}
CONFIRMATION_STATE=${2:?usage: port_liquidity_queue.sh OUTPUT_DIR CONFIRMATION_STATE}
STATE="$OUT/queue-state"
LOCK="$OUT/.port-screen.lock"
mkdir -p "$OUT"
if ! mkdir "$LOCK" 2>/dev/null; then
  printf '%s\n' "queue lock exists: $LOCK" >&2
  exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
write_state() { tmp="$STATE.tmp"; printf '%s\n' "$1" > "$tmp"; mv "$tmp" "$STATE"; }
fail() { write_state "ERROR:$1"; exit 1; }
trap 'cleanup; fail interrupted' HUP INT TERM
trap cleanup EXIT
[ "${PYTHONHASHSEED:-}" = 0 ] || fail "PYTHONHASHSEED must be 0"
PYTHONPATH="$SRC" python -m hexset.catanatron.port_liquiditycfg --plan > "$OUT/plan.json" || fail plan
if ! PYTHONPATH="$SRC" python - "$OUT/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["screen"]["games"] == 240
assert p["screen"]["workers"] == 30
assert len(p["screen"]["jobs"]) == 12
PY
then
  fail plan-validation
fi
if [ ! -f "$CONFIRMATION_STATE" ] || [ "$(sed -n "1p" "$CONFIRMATION_STATE")" != "COMPLETE" ]; then
  fail "confirmation not complete"
fi
write_state RUNNING
for gate in ab2 shipped; do
  for arm in control drop-flat-port flat-port-4x liquidity-1.3925 liquidity-2.785 liquidity-5.57; do
    marker="$OUT/summaries/$gate-$arm.json"
    PYTHONHASHSEED=0 PYTHONPATH="$SRC" python -m hexset.catanatron.port_liquidity_runner \
      --arm "$arm" --gate "$gate" --games 240 --workers 30 --out "$OUT" \
      > "$OUT/$gate-$arm.stdout" || fail "$gate/$arm"
    [ -f "$marker" ] || fail "$gate/$arm missing summary"
    printf '%s %s\n' "$gate" "$arm" >> "$OUT/completed.log"
  done
done
write_state COMPLETE
