# Current Heximax versus Catanatron AB2

This two-batch benchmark measures the unified **unpinned adaptive `heximax`** from
PR #147 in the **native HexSet engine**, against Catanatron's depth-two
alpha-beta player as an external reference. Both formats use standard 10-VP
rules, fresh boards and balanced focal seats.

| Matchup | Heximax wins | Win rate | 95% Wilson interval | Equal-share baseline |
| --- | --- | --- | --- | --- |
| One Heximax vs one AB2 | 590 / 800 | **73.75%** | 70.6–76.7% | 50% |
| One Heximax vs three AB2 | 450 / 800 | **56.25%** | 52.8–59.6% | 25% |

The first batch won 292/400 and 224/400 respectively; the additional batch
won 298/400 and 226/400. All **1,600 games** finished. Independent audits
replayed **397,520 legal actions**, verified 1,600 distinct game seeds and
every recorded final score, and found zero reference deadline hits. There
were zero domestic exchanges; Heximax remained unpinned and its observed
slider position was zero throughout.

The two batches completed in 194.17 and 201.31 seconds, excluding
their 20.00- and 18.15-second preflights and subsequent replay audits. These
are workload timings, not an isolated comparison between engines.

Trading is enabled with Heximax's normal zero gain floor. The AB2 adapter
declines domestic exchanges; the observed activity and slider position are
reported in the audit. This matchup measures how the current adaptive bot
plays against AB2; it does not measure adaptation against trading opponents.

## Exchange-only update after this benchmark

The [bounded exchange-weight sweep](../exchange-weight-sweep/README.md)
subsequently changed only exchange robber risk from −0.30 to −0.225. Both move
profiles, adaptation and search remain unchanged. AB2 declines all domestic
exchanges. Eight reserved adoption replays matched the original full game
records exactly, including four games in each format. These are behavior
checks; the 1,600 headline observations remain the original benchmark, not a
new sample under the updated source. The exact adopted source hash and checks
are in [the adoption receipt](../exchange-weight-sweep/adoption-verification.json).

## Frozen configuration

- HexSet policy source: `38ac537e44944a892b881b5c0152d4de6828b2a4`.
- Catanatron dependency: `ecf931181b9a65bb4116a2153fb78c16f1438e00`.
- Heximax: `pin_weights=None`, depth 2, width 6, 600 leaves, k=1,
  default objective, temperature and opening prior. The runner checks the
  adaptive evaluator and expansion credit at every Heximax decision.
- AB2: stock `catanatron` entrant, depth 2, default parameters, worlds=0.
- 1v1: one Heximax versus one AB2; seeds 727100000 and 727300000,
  indices 0–399 per seed.
- Four players: one Heximax versus three AB2; seeds 727200000 and 727400000,
  indices 0–399 per seed.
- One independent board per game, no antithetic repeats; pooled totals are
  400 games per focal seat in 1v1 and 200 per focal seat in four-player games.
- The original 400-per-format [plan](PLAN.md) was committed before launch.
  After seeing its results, the user requested another 400 per format to
  tighten the headline intervals. The [extension plan](extension/PLAN.md)
  fixed the additional sample before its launch. These are descriptive
  Wilson intervals for the pooled sample, not a sequential stopping test.
  Every scheduled game counts; no exclusions, replacement seeds or tuning.

## Runtime and verification

The study ran on Wintermute under Python 3.12 with 30 processes and one
BLAS/OpenMP thread per process, in a container with networking disabled.
The exact Catanatron source and distribution metadata were mounted over the
older image dependency; runtime guards checked the commit and source layouts.
Source fingerprints identify the code actually imported.

Each batch played six separate preflight boards with speedups off and fast.
All 12 pairs matched their complete action/chance records and outcomes,
with zero reference deadline hits, so both batches used fast mode.
The 24 preflight games are excluded from the reported 1,600 games.

The [audit script](audit.py) independently replays every action and chance
stream, checks board seeds, entrant settings, seating and final scores,
and recomputes win counts and Wilson intervals. These are descriptive 95%
intervals for these matchups, without comparison to earlier configurations.

Artifacts:

- [Combined summary](combined-summary.json) and [pooling verifier](combine.py):
  check both archives, equal source fingerprints, disjoint seeds and all counts.
- [Second batch](extension/README.md), including its separate records and audit.

- Original batch: [summary](summary.json), [audit](audit.json), [source manifest](manifest.json),
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

For the second batch, use the commands in its [readout](extension/README.md).
Run `python docs/readouts/current-heximax-ab2/combine.py` to verify and pool
the preserved archives.

A resume requires the identical runtime manifest and driver. The raw records
also replay without running either bot's search.
