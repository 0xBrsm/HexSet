# Adaptive versus fixed exchange weights: native 2v2

> **Protocol correction (2026-09-11): research-only automatic clearing.**
> This run does not validate served trading. Its adoption, fitting and served
> behavior conclusions are withdrawn; frozen results below record what ran.
> For mixed studies this applies to trading-enabled cells; no-trade cells
> remain separately scoped. See [the correction](../trading-protocol-correction.md).

**Keep the current fixed exchange evaluator.** With adaptive move search on
both sides, the two fixed-exchange seats won 265/400 games against two seats
using the adaptive coefficient vector for exchanges. This tests the exchange
weights directly; neither side uses fixed move weights.

| Exchange weights | Wins / 400 | Side win rate | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| Adaptive: current move coefficients | 135 | 33.75% | 29.3–38.5% |
| Fixed: current TRADING_WEIGHTS | 265 | **66.25%** | **61.5–70.7%** |

Each bot plays to win individually. A side's wins count either of its two
seats; its equal-share baseline is 50%. The adaptive side was 16.25 percentage
points below that baseline (interval -20.7 to -11.5 points). This supports
separate exchange weights in this matchup, without establishing optimal
coefficients or superiority against every opponent population.

The fixed vector measured here used exchange robber risk −0.30. A later
[bounded sweep](../exchange-weight-sweep/README.md) adopted −0.225 after fresh
confirmation. This archived 400-game comparison has not been relabeled as a
measurement of that later vector.

## What changed

The candidate updates its exchange coefficients from the current adaptive
profile at each valuation batch, using its latest public eight-eligible-turn
activity. The control retains TRADING_WEIGHTS. Both sides use the same stock
unpinned adaptive move search. Exchange expansion credit remains zero on both
sides; move expansion credit follows the unchanged adaptive policy. Thus this
compares exchange coefficient vectors, not search or expansion settings.

- Policy source: `38ac537e44944a892b881b5c0152d4de6828b2a4`, source SHA256
  `c5b426a446f78e5686bca51c66f053bef44f58e924911bbb38a53eb1f743d9ef`.
- Frozen [plan](PLAN.md) and [runner](run.py): committed as `102f634` before launch.
- Host: **native HexSet**. No Catanatron modules were imported.
- Standard 10-VP rules; trading enabled for all seats at floor zero.
- Depth 2, width 6, 600 leaves, k=1, default stance/temperature and opening prior.
- Exactly 400 fresh independent boards: seed 728000000, indices 0–399,
  antithetic=False. A,A,F,F rotates through all four seats: each entrant plays
  100 games in each seat. This uses adjacent same-policy seats; alternating
  placement and other opponent mixtures are outside this test.
- No tuning, exclusions, replacement seeds or outcome-dependent extension.
  Production defaults remain unchanged.

## Trading and validation

All 400 games finished. The audit replayed **97,272 legal actions** and
independently reconstructed **24,667 public activity observations** from the
recorded games, matching every bot's activity history exactly. It also checked
boards, seeds, entrant settings, seating, final scores and the pooled win count.

The games cleared **23,382 exchanges** (58.46 per game): 15,828 across policies,
3,411 between adaptive-exchange seats, and 4,143 between fixed-exchange seats.
All nine slider values from zero through one were exercised. Mean activity
at exchange-valuation calls was 0.608 for the adaptive side and
0.608 for the fixed side; the latter observes activity for move search
but retains fixed exchange weights. These call-weighted means describe the
realized trading environment, not independent observations for inference.

Preflight excluded 12 games: four complete stock-versus-wrapped control pairs
matched exactly, and four additional mixed games checked **153,206 candidate
gains** against independently constructed evaluators. This verifies both the
unchanged control and the candidate's intended exchange valuation.

The strength sample took **62.57 seconds**, plus a 2.99-second preflight and
subsequent independent audit, on Wintermute with 30 processes, one BLAS/OpenMP
thread each and networking disabled. Runtime image and limits are recorded.

## Artifacts and reproduction

[Summary](summary.json), [independent audit](audit.json),
[source/configuration manifest](manifest.json), [runtime receipt](runtime-receipt.json),
[preflight verdict](preflight-summary.json), [all records](records.tar.gz),
and [checksums](SHA256SUMS).

With the recorded HexSet source and runtime:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python docs/readouts/adaptive-exchange-2v2/run.py --out /tmp/exchange-2v2 --workers 30
PYTHONPATH=src python docs/readouts/adaptive-exchange-2v2/audit.py /tmp/exchange-2v2
```

The [audit](audit.py) needs the base HexSet dependency only; it performs no bot
search when replaying strength records. A resume requires the same driver and
runtime manifest. The archived games retain complete action/chance tapes,
trade observations and per-bot move/exchange activity histograms.
