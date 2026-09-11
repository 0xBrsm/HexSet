# Native trading-enabled fair-share comparison

September 11, 2026. One seat uses the new no-trade profile (road=0,
expansion=.25), with trading enabled. Every opponent also has trading enabled.

| Three opponents | Candidate wins | Win rate | 95% Wilson interval | Domestic trades |
| --- | ---: | ---: | ---: | ---: |
| Old no-trade weights, expansion=0 | 138 / 400 | 34.50% | 30.01–39.29% | 0 |
| Trading weights, expansion=0 | 129 / 400 | 32.25% | 27.86–36.98% | 0 |

Both intervals exceed 25% fair share. However, **no player-to-player trade
cleared in any of the 800 games**. Trading was enabled in the real HexSet game
and every bot: max_trades=None, mode=honest, standard trade floor .0197.
All four seats evaluated offers in every game. Bank and port exchanges
remained available. These games do not establish which profile is best in
games with actual domestic exchanges, or justify consolidating the two weight
sets on that basis. No default profiles were changed by this experiment.

## Trade diagnostics

| Opponent profile | Batched gate calls | Offer evaluations | Evaluations above own floor | Largest observed gain |
| --- | ---: | ---: | ---: | ---: |
| Old no-trade | 39,251 | 7,680,278 | 26,691 | .043516 |
| Trading | 39,839 | 8,367,779 | 25,220 | .042704 |

Counts include repeated offers across game states and each side's evaluations.
Some individual gains exceeded .0197, but no offer cleared both sides' gates.
This identifies the acceptance condition behind zero exchanges; it does not
establish that lowering the threshold would improve play.

## Protocol and verification

Native HexSet public-history ledger, source 396e227 from PR141, standard rules,
independent boards, seed 717000000, indices 0–399 per matchup,
antithetic=False, 100 games in each candidate seat. Both matchups use the same
board schedule. Search is depth 2, width 6, 600 leaves, k=1, win temperature
2.476644394795811 and the standard opening prior. Explicit weights and
expansion coefficients separate the policy profile from permission to trade.
No Catanatron code or opponents participate in this comparison.

Thirty workers completed the fixed 800-game v2 batch in 116.36 seconds,
about 6.88 games/second. This includes its two retained initial verification
games and pool startup, but excludes the original 1.69-second preflight and
preparation. It is throughput for this workload, not a matched speedup.

Verification checked all 800 complete unique records, exact manifest identities,
balanced candidate seating, active trade gates, zero recorded exchanges,
and independently recomputed the win intervals (allowing for arena's rounded
normal quantile). The original preflight stopped because an overly strict
runner assertion required one trade in its first game. Version 2 removed that
assertion and added gain counters; its replay matched the original full action
trace and outcome. The original record is preserved separately and excluded
from the 800-game sample. No game rules, policies, thresholds, seeds or sample
sizes changed.

Reproduce in the pinned campaign image with exported source and one
OpenBLAS/OMP/MKL thread per worker, PYTHONHASHSEED=0:

```
PYTHONPATH=/study/src python /study/docs/readouts/native-trading-fair-share/run_v2.py --out /out/v2
```

`manifest.json` records exact source/runner hashes, runtime and profiles.
`games.tar.gz` contains all 800 v2 game records, manifest and summary, plus the
separately identified original preflight. SHA256 is in `archive.sha256`.
Runtime receipts and `verification.json` are alongside this readout.
