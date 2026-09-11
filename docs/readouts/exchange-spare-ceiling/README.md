# Exchange spare-card sweep after the robber update

**Retain spare_card = 0.15 and robber_risk = +0.075 for exchange valuation.**
None of the eight alternatives exceeded 50% in the registered screen, so the
experiment stopped after 536 games (80.70 seconds). There was no runoff,
confirmation or production change. This is a failed screen, not evidence that
0.15 is globally optimal or that nearby values are distinguishable.

The earlier exchange-only screen favored spare_card = 0.1875 against 0.15
(83/128 wins), with robber_risk = −0.30. Here both sides use the subsequently
selected +0.075 robber coefficient. That changes the context of the spare
comparison: the two features both influence willingness to hold resource
cards. Their effects may interact; this sequential study is not a factorial
interaction test, and its 64-board screen has limited resolution.

## Registered screen

Each candidate faced the incumbent on the same 64 independent boards in
native HexSet. Two candidate seats and two incumbent seats rotate cyclically
from A,A,B,B; each bot plays individually. Either candidate seat winning counts
as a candidate win, so equal share is 50%. Adaptive move weights and search,
other exchange weights, floor zero and exchange expansion zero stayed fixed.
The native automatic exchange rules still allow three cards per side.

| Exchange spare_card | Wins / 64 | Win rate |
| --- | ---: | ---: |
| 0 | 4 | 6.25% |
| 0.075 | 22 | 34.38% |
| 0.1875 | 28 | 43.75% |
| 0.225 | 21 | 32.81% |
| 0.30 | 11 | 17.19% |
| 0.45 | 2 | 3.13% |
| 0.60 | 0 | 0% |
| 1.20 | 0 | 0% |

The best screen candidate, 0.1875, has a descriptive 95% Wilson interval of
32.3–55.9%; it does not establish a loss against the incumbent either.
`runoff-selection.json` records the ranking only: the subsequent stopping
gate prevented launching those games. No finalist was selected or adopted.
Selection data are not pooled into a confirmation claim.

Replay diagnostics show lower spare weights gave away net cards (−2,882 for
zero; −1,343 for 0.075), whereas all higher values acquired net cards. Yet
high spare values lost heavily. Card quantity alone is insufficient to explain
strength; these counts omit resource quality and counterfactual trade value.

## Budget and validation

The [plan](PLAN.md), runner and audit were frozen in `68027d1` before launch.
Baseline source commit: `270b9270ea7684c9db60dd346b1f3444e1b9fd0c`.
Source SHA256: `d9cd0c7c833130e401a8d0fbf468a7b6113824e6cb3a88310b47df90410411e2`.

- 24 preflight games matched four stock/wrapper full-trace pairs and checked
  504,447 exchange gains independently in 16 candidate games.
- All 512 screen games finished and passed independent replay, including
  129,496 legal actions and 32,750 reconstructed public activity events.
- The audit verified source, coefficients, boards, seating, final scores,
  ranking, intervals and the early-stop rule.
- Wintermute, Python 3.12.14, 30 workers, one BLAS/OpenMP thread each,
  networking disabled; no Catanatron module imported.
- Screen seed 731000000, indices 0–63 for each candidate; preflight seeds
  follow the plan. No antithetic repeats, retries or replacement seeds.
- Total 536 evaluation games, 80.70 seconds against the 1,600-game / 900-second
  cap. Unused budget is not reassigned. No adoption trace games were needed
  because source and production behavior did not change.

## Artifacts and reproduction

[Screen](screen-summary.json), [ranking](runoff-selection.json),
[verdict](verdict.json), [audit](audit.json), [manifest](manifest.json),
[preflight](preflight-summary.json), [runtime](runtime-receipt.json),
[raw records](records.tar.gz), [checksums](SHA256SUMS).

```sh
git worktree add /tmp/spare-baseline 68027d1
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/spare-baseline/src timeout --signal=KILL 900s \
  python docs/readouts/exchange-spare-ceiling/run.py --out /tmp/spare-study --workers 30
PYTHONPATH=/tmp/spare-baseline/src \
  python docs/readouts/exchange-spare-ceiling/audit.py /tmp/spare-study
```

The runner refuses an existing output manifest. The reserved adoption verifier
was frozen with the plan but was not executed because the screen stopped.
