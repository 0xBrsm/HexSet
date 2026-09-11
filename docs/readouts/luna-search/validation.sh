#!/bin/bash
set -euo pipefail
cd /study
mkdir -p validation
python - <<'PY'
import glob,json
from pathlib import Path
from hexset.catanatron.validation import rank_screen
paths=[Path(p) for p in glob.glob('/study/results/*.json')]
if len(paths)!=9: raise SystemExit("screen requires exactly 9 files")
top=rank_screen(paths)
Path("/study/validation/top3.json").write_text(json.dumps({"candidates":top},indent=2))
print("TOP3",*top)
PY
read -ra TOP <<< "$(python -c "import json; print(*json.load(open('/study/validation/top3.json'))['candidates'])")"
for i in "${!TOP[@]}"; do
 c="${TOP[$i]}"; seed=$((1200000+i*1000)); out="/study/validation/confirm-$(printf '%02d' "$i")-$c.json"
 test ! -e "$out"
 python -m hexset.catanatron.searchcfg --candidate "$c" --games 1024 --workers 30 --seed "$seed" --out "$out"
done
python - <<'PY'
import glob,json
from pathlib import Path
from hexset.catanatron.validation import GATES,THRESHOLDS
ds=[]
for f in glob.glob("/study/validation/confirm-*.json"):
 d=json.load(open(f)); assert d["games"]==1024 and len(d["rows"])==2
 r={x["gate"]:x for x in d["rows"]}; w={g:int(r[g]["wins"].get("Color.RED",r[g]["wins"].get("0",0))) for g in GATES}
 ds.append((min(w[g]/1024-THRESHOLDS[g] for g in GATES),d["candidate"]))
if len(ds)!=3: raise SystemExit("confirmation incomplete")
ds.sort(key=lambda x:(-x[0],x[1]))
Path("/study/validation/finalist.json").write_text(json.dumps({"candidate":ds[0][1],"ranking":ds},indent=2))
PY
F=$(python -c "import json; print(json.load(open('/study/validation/finalist.json'))['candidate'])")
for spec in "ab2 1300000" "shipped 1400000"; do
 set -- $spec; out="/study/validation/holdout-$1-$F.json"; test ! -e "$out"
 python -m hexset.catanatron.searchcfg --candidate "$F" --games 2048 --workers 30 --seed "$2" --gate "vs-$1" --out "$out"
done
python - <<'PY'
import glob,json
from hexset.catanatron.validation import decide_holdout,GATES
fs=glob.glob("/study/validation/holdout-*.json"); assert len(fs)==2
rows={}
for f in fs:
 d=json.load(open(f)); assert d["games"]==2048 and len(d["rows"])==2
 for r in d["rows"]:
  if (("ab2" in f and r["gate"] != "vs-ab2") or ("shipped" in f and r["gate"] != "vs-shipped")): continue
  w=r["wins"]; k="Color.RED" if "Color.RED" in w else "0"; assert sum(map(int,w.values()))==2048
  rows[r["gate"]]=(int(w[k]),2048)
assert set(rows)==set(GATES)
json.dump({"z":2.5758293035489004,"rows":rows,"pass":decide_holdout(rows),"attempts":1},open("/study/validation/verdict.json","w"),indent=2)
PY

