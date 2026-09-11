# Exchange robber weight: positive values and a bounded ceiling search

**Adopt +0.075 for exchange valuation only.** It won **608/800 fresh games
(76.0%, 95% Wilson 72.9–78.8%)** against the preceding −0.225 exchange weight
in native 2v2. Both move endpoints retain their −0.30 robber penalty. All
other weights, adaptive move search, floor zero and exchange expansion zero
remain unchanged.

This experiment supports a small positive exchange coefficient against the
measured incumbent. It does not identify a unique optimum: +0.075 versus
+0.15 was inconclusive in their direct head-to-head, and the broad screen is
selection data. Higher positive values declined in the screen, with +2.40
at equal share; there is no evidence here for increasing the weight indefinitely.

A subsequent [spare-card sweep](../exchange-spare-ceiling/README.md) retained
spare_card = 0.15: no alternative passed its initial screen with +0.075 robber
pricing held fixed. That follow-up does not add to this study’s sample.

## Screen, runoff and fresh confirmation

All games use adjacent A,A,B,B seating with cyclic rotation. Each bot plays
to win individually; a side's win rate counts either of its two seats, so
50% is equal share. The screen uses the same 64 independent boards for every
candidate against −0.225; each entrant occupies each seat 16 times.

| Exchange robber weight | Wins / 64 | Screen win rate |
| --- | ---: | ---: |
| −0.15 | 50 | 78.1% |
| 0 | 50 | 78.1% |
| **+0.075** | **57** | **89.1%** |
| +0.15 | 51 | 79.7% |
| +0.30 | 50 | 78.1% |
| +0.60 | 47 | 73.4% |
| +1.20 | 44 | 68.8% |
| +2.40 | 32 | 50.0% |

The two leading candidates played 256 fresh head-to-head games: +0.075 won
**133/256 (52.0%, 95% Wilson 45.8–58.0%)** against +0.15. The frozen rule
selected +0.075 by win count. The interval includes 50%; this is not evidence
that +0.075 is superior to +0.15. Runoff games remain selection data.

The sole finalist then faced the −0.225 incumbent on 800 fresh boards:

| Exchange profile | Wins | Side win rate | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| +0.075 | **608/800** | **76.0%** | **72.9–78.8%** |
| −0.225 incumbent | 192/800 | 24.0% | 21.2–27.1% |

The two finalist bots won 325 and 283 games; the incumbents won 88 and 104.
All games finished. Confirmation cleared 42,540 exchanges. Only these 800
games enter confirmation inference; neither earlier selection stage is pooled.

## What the exchange records suggest

The term measures hand exposure above the discard limit. A negative exchange
coefficient rewards reducing that exposure when pricing a deal; a positive
one reverses that local preference. Actual discard rules and move-search risk
valuation are unchanged.

Across confirmation's 31,082 cross-policy exchanges, the +0.075 side received
**20,549 net resource cards** from the −0.225 side: 25.7 cards per game across
the candidate pair, or 0.661 net cards per cross-policy exchange.

| Cross-policy exchange diagnostic | +0.075 side | −0.225 side |
| --- | ---: | ---: |
| Net cards received | +20,549 | −20,549 |
| Hand crossed above seven | 6,678 | 0 |
| Hand crossed down to seven or fewer | 35 | 9,443 |

These replay-derived counts support the hypothesis that the negative evaluator
is too willing to give up cards to reduce exposure, and the positive evaluator
benefits from that willingness. Card counts do not capture resource quality
or the counterfactual value of a trade, so they do not prove the entire causal
mechanism or establish strength against other exchange evaluators or humans.

The positive-versus-positive runoff was very different: +0.15 received 632
net cards from +0.075 across 6,744 cross-policy exchanges, but did not win more
games. Accumulating more cards alone therefore does not explain every ranking.
This is also a different matchup and seed set, not a paired causal comparison.

## Budget, provenance and verification

The [plan](PLAN.md) and [runner](run.py) were frozen in `e3e9388` before launch.
Baseline policy source is `b328d2de769ddaafc7e867525a740e955d79d201`, SHA256
`6b60259813b77ecfca12316ab46de7838b6a93b335f7b79c80955ee385fb2d49`.
The exact incumbent float is the preceding study's `-0.30 * 0.75`.

- Native HexSet, standard 10-VP rules, stock adaptive search depth 2,
  width 6, 600 leaves, k=1, default stance/temperature and opening prior.
- Trading enabled, floor zero and exchange expansion zero on every seat.
- Screen seed 730000000, indices 0–63 per candidate. Runoff seed 730100000,
  indices 0–255. Confirmation seed 730200000, indices 0–799. No antithetic
  repeats, seed replacements, additional candidates or outcome-driven extension.
- Wintermute Python 3.12.14, 30 workers, one BLAS/OpenMP thread each,
  networking disabled. The study imports no Catanatron; only the separately
  budgeted AB2 adoption checks load the pinned external reference.

Total: **1,600 evaluation games and 250.38 seconds (4 minutes 10 seconds)**:
24 preflight, 512 screen, 256 runoff, 800 confirmation, eight adoption checks.
Process-group timeouts enforced the 900-second cumulative evaluation cap.
Independent audit and ordinary software testing are separate verification work.
The study is complete and stops here.

All **1,568 strength games** passed independent replay: **397,203 legal
actions** and **99,010 reconstructed public activity observations**, plus
coefficient, source, board, seating, final-score, selection and interval checks.
Preflight matched four stock/wrapper trace pairs and independently checked
679,397 candidate exchange valuations in 16 additional games.

The eight reserved adoption games matched four frozen confirmation records
and four original AB2 records exactly (two per format). These are behavior
checks, not new strength observations. AB2 declines exchanges, and the update
does not change move search. Original reference headline counts remain their
recorded benchmark rather than being relabeled as a fresh post-change sample.

## Artifacts and reproduction

[Screen](screen-summary.json), [runoff selection](runoff-selection.json),
[runoff](runoff-summary.json), [finalist selection](selection.json),
[confirmation](confirmation-summary.json), [verdict](verdict.json),
[independent audit](audit.json), [manifest](manifest.json),
[preflight](preflight-summary.json), [runtime](runtime-receipt.json),
[adoption checks](adoption-verification.json), [adoption runtime](adoption-runtime-receipt.json),
[raw records](records.tar.gz), and [checksums](SHA256SUMS).

Use the frozen baseline source; current production uses the promoted
coefficient and deliberately fails the study's baseline fingerprint guard:

```sh
git worktree add /tmp/robber-baseline e3e9388
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/robber-baseline/src timeout --signal=KILL 900s \
  python docs/readouts/exchange-robber-ceiling/run.py --out /tmp/robber-study --workers 30
PYTHONPATH=/tmp/robber-baseline/src \
  python docs/readouts/exchange-robber-ceiling/audit.py /tmp/robber-study
```

The runner refuses an existing output manifest. [verify_adoption.py](verify_adoption.py)
checks the promoted source using a successful audit and the original AB2 records.
