# Bounded exchange-weight sweep: adopted result

**Adopt exchange robber-risk weight −0.225 instead of −0.30.** In fresh
confirmation, two candidate seats won **600/800 games (75.0%, 95% Wilson
71.9–77.9%)** against two incumbent seats. Both sides retained the same
current adaptive move policy, search settings and trading floor zero.
The registered adoption criterion was a lower bound above 50%; it passed.

The implementation preserves the tested arithmetic `-0.30 * 0.75` exactly.
Only the exchange coefficient changes. Both move endpoints retain their
robber-risk weight of −0.30. Exchange expansion credit stays zero; move
expansion continues to adapt. No other candidate changes were combined.

A later [sign/range experiment](../exchange-robber-ceiling/README.md) selected
+0.075 against this study's −0.225 result. The measurements here remain the
original −0.225-versus-−0.30 study, with their original source and records.

## Fixed budget and result

The [plan](PLAN.md) and [runner](run.py) were frozen in `a7c6777` before launch.
Each candidate faced the incumbent in adjacent 2v2 seating, rotated through
all seats. Bots play individually; the candidate side wins when either of its
two seats wins, giving an equal-share baseline of 50%.

| Screen: exchange coefficient changed | Candidate wins / 128 | Side win rate |
| --- | ---: | ---: |
| Purchase progress ×0.75 | 43 | 33.6% |
| Purchase progress ×1.25 | 73 | 57.0% |
| Spare-card value ×0.75 | 46 | 35.9% |
| Spare-card value ×1.25 | 83 | 64.8% |
| Robber risk ×0.75 | **106** | **82.8%** |
| Robber risk ×1.25 | 46 | 35.9% |

Screening used the same 128 independent boards for each candidate. These
selection results are exploratory; they are excluded from confirmation.
The top candidate was recorded in [selection.json](selection.json) before
playing 800 fresh confirmation boards. It won 600; the incumbent won 200.
All games finished. Confirmation alone cleared 43,613 exchanges.

| Evaluation stage | Games |
| --- | ---: |
| Preflight | 20 |
| Six-candidate screen | 768 |
| One fresh confirmation | 800 |
| Reserved adoption behavior checks | 12 |
| **Total** | **1,600** |

Evaluation wall time totaled **241.18 seconds (4 minutes 1 second)**, within
the 900-second cap. Process-group timeouts enforced the execution limits.
The independent replay audit and ordinary software tests are separate
verification work. The sweep is complete: no expanded ranges, second
finalists, candidate combinations, replacement seeds or further evaluations.

## Configuration and evidence

- Baseline HexSet source: `38ac537e44944a892b881b5c0152d4de6828b2a4`;
  fingerprint in [manifest.json](manifest.json). The runner export at
  `a7c6777` has identical production source.
- Host: native HexSet, standard 10-VP rules. The sweep imports no Catanatron.
- Unpinned adaptive move search on every seat; depth 2, width 6, 600 leaves,
  k=1, default stance/temperature and opening prior.
- Every exchange vector fixed for its game. Only one of purchase progress,
  spare-card value or robber risk changes by ×0.75 or ×1.25. All other
  coefficients, zero floor and zero exchange expansion credit are fixed.
- Screen seed 729000000, indices 0–127 per candidate; confirmation seed
  729100000, indices 0–799. Independent boards, antithetic=False. Each
  entrant occupies each seat 32 times in screening and 200 in confirmation.
- Wintermute: Python 3.12.14, 30 workers, one BLAS/OpenMP thread per process,
  networking disabled, same runtime image as preceding studies.

The independent [audit](audit.json) replayed all **1,568 strength games**,
validated **374,166 legal actions**, reconstructed **95,012 public activity
observations**, and checked coefficients, boards, seating, final scores,
screening selection and the fresh confirmation interval. Every check passed.
The 20 preflight games included four exact stock/wrapper trace pairs and
12 candidate games independently checking 515,673 exchange valuations.

The [adoption checks](adoption-verification.json) matched four confirmation
games through the promoted default and eight original AB2 games (four 1v1,
four four-player) in full action/chance records and outcomes. These 12 are
verification replays, not additional strength observations. The AB2 reference
still declines all exchanges; move search is unchanged, and exchange pricing
has separate caches and does not consume the move RNG. The adoption changes
no decisions in those eight checked reference games. The original 1,600-game
AB2 study remains separately recorded; it was not rerun as a new sample.

This supports the adopted change in the tested native adjacent-2v2 matchup.
It is not a global optimum claim or validation against every opponent mix.
The rest of the inherited exchange vector has not received a dedicated fit.

## Artifacts and reproduction

[Screen summary](screen-summary.json), [selection](selection.json),
[confirmation summary](confirmation-summary.json), [verdict](verdict.json),
[audit](audit.json), [manifest](manifest.json), [preflight](preflight-summary.json),
[runtime receipt](runtime-receipt.json), [adoption runtime](adoption-runtime-receipt.json),
[raw records](records.tar.gz), and [checksums](SHA256SUMS).

Use the frozen baseline source, because the adopted source changes the
incumbent and deliberately fails this study's fingerprint guard:

```sh
git worktree add /tmp/exchange-baseline a7c6777
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/exchange-baseline/src timeout --signal=KILL 900s \
  python docs/readouts/exchange-weight-sweep/run.py --out /tmp/exchange-sweep --workers 30
PYTHONPATH=/tmp/exchange-baseline/src \
  python docs/readouts/exchange-weight-sweep/audit.py /tmp/exchange-sweep
```

The runner refuses an existing output manifest. Archived records allow replay
without rerunning either policy. [verify_adoption.py](verify_adoption.py) is
the separately budgeted check of the adopted source against frozen candidate
and AB2 records; it requires the successful audit and original AB2 archive.
