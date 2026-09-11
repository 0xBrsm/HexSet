# Current Heximax versus Catanatron AB2

This fixed benchmark measures the unified **unpinned adaptive `heximax`** from
PR #147 in the **native HexSet engine**, against Catanatron's depth-two
alpha-beta player as an external reference. Both formats use standard 10-VP
rules, fresh boards and balanced focal seats.

| Matchup | Heximax wins | Win rate | 95% Wilson interval | Equal-share baseline |
| --- | --- | --- | --- | --- |
| One Heximax vs one AB2 | 292 / 400 | **73.0%** | 68.4–77.1% | 50% |
| One Heximax vs three AB2 | 224 / 400 | **56.0%** | 51.1–60.8% | 25% |

All 800 games finished. The independent audit replayed **198,435 legal
actions**, verified 800 distinct game seeds and every recorded final score,
and found zero reference deadline hits. There were zero domestic exchanges;
Heximax remained unpinned and its observed slider position was zero throughout.
The fixed sample completed in 194.17 seconds, excluding the 20.00-second
preflight and subsequent replay audit. This is workload throughput, not an
isolated comparison between engines.

Trading is enabled with Heximax's normal zero gain floor. The AB2 adapter
declines domestic exchanges; the observed activity and slider position are
reported in the audit. This matchup measures how the current adaptive bot
plays against AB2; it does not measure adaptation against trading opponents.

## Frozen configuration

- HexSet policy source: `38ac537e44944a892b881b5c0152d4de6828b2a4`.
- Catanatron dependency: `ecf931181b9a65bb4116a2153fb78c16f1438e00`.
- Heximax: `pin_weights=None`, depth 2, width 6, 600 leaves, k=1,
  default objective, temperature and opening prior. The runner checks the
  adaptive evaluator and expansion credit at every Heximax decision.
- AB2: stock `catanatron` entrant, depth 2, default parameters, worlds=0.
- 1v1: one Heximax versus one AB2, seed 727100000, indices 0–399.
- Four players: one Heximax versus three AB2, seed 727200000, indices 0–399.
- One independent board per game, no antithetic repeats; 200 games per focal
  seat in 1v1 and 100 per focal seat in four-player games.
- The sample size and seeds were committed before launch in [PLAN.md](PLAN.md).
  Every scheduled game counts in the denominator; no seed replacement,
  parameter selection or outcome-dependent sample extension.

## Runtime and verification

The study ran on Wintermute under Python 3.12 with 30 processes and one
BLAS/OpenMP thread per process, in a container with networking disabled.
The exact Catanatron source and distribution metadata were mounted over the
older image dependency; runtime guards checked the commit and source layouts.
Source fingerprints identify the code actually imported.

Six separate preflight boards were each played with speedups off and fast.
All six pairs matched their complete action/chance records and outcomes,
with zero reference deadline hits, so the fixed sample used fast mode.
The 12 preflight games are excluded from the reported 800 games.

The [audit script](audit.py) independently replays every action and chance
stream, checks board seeds, entrant settings, seating and final scores,
and recomputes win counts and Wilson intervals. These are descriptive 95%
intervals for these matchups, without comparison to earlier configurations.

Artifacts:

- [Summary](summary.json), [audit](audit.json), [source manifest](manifest.json),
  [preflight verdict](preflight-summary.json), [runtime receipt](runtime-receipt.json).
- [Raw game records and preflight](records.tar.gz), with checksums in
  [SHA256SUMS](SHA256SUMS).
- Frozen [runner](run.py) and [plan](PLAN.md).

To reproduce, install this source revision with its pinned `catanatron` extra,
then run:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python docs/readouts/current-heximax-ab2/run.py --out /tmp/current-ab2 --workers 30
PYTHONPATH=src python docs/readouts/current-heximax-ab2/audit.py /tmp/current-ab2
```

A resume requires the identical runtime manifest and driver. The raw records
also replay without running either bot's search.
