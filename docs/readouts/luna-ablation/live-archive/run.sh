#!/usr/bin/env bash
set -u
cd /study
candidates=(baseline drop-production drop-buy_progress drop-robber_risk drop-spare_card drop-road drop-diversity drop-port drop-knight prod-half prod-141 progress-half progress-141 risk-half risk-141 road-half road-zero spare-half spare-141)
for i in "${!candidates[@]}"; do
  c=${candidates[$i]}
  out=/study/results/$(printf '%02d-%s.json' "$i" "$c")
  if [ -s "$out" ]; then continue; fi
  echo "START $c $(date -u +%FT%TZ)" | tee -a /study/progress.log
  python -u -m hexset.catanatron.ablation --candidate "$c" --games 120 --workers 30 --seed $((610000+i*100)) --out "$out" >> /study/progress.log 2>&1
  rc=$?
  echo "DONE $c rc=$rc $(date -u +%FT%TZ)" | tee -a /study/progress.log
  [ "$rc" -eq 0 ] || exit "$rc"
done
# Selection is intentionally a separate, fresh-seed step.  The helper below
# reads completed screen JSONs and confirms the four best point estimates.
python -u /study/select_and_confirm.py
