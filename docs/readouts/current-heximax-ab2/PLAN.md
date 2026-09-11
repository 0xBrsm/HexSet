# Current Heximax versus AB2: fixed benchmark

Purpose: quote the current unified, unpinned adaptive Heximax against the current
Catanatron AB2 reference in the README. No historical configurations or hosts are
substituted, no parameter selection, and no outcome-driven sample extension.

- Policy source: `38ac537e44944a892b881b5c0152d4de6828b2a4` (PR #147).
- Dependency: Catanatron `ecf931181b9a65bb4116a2153fb78c16f1438e00`.
- Real-game host: native HexSet, standard 10-VP / discard-over-7 rules.
- Heximax: stock `heximax`, pin_weights=None, trading enabled, floor 0,
  depth 2, width 6, 600 leaves, k=1, default win stance/temperature and opening prior.
- Reference: stock `catanatron` entrant, AB depth 2, default parameters,
  worlds=0. It declines domestic exchanges, so zero observed exchanges are
  expected, with adaptive Heximax staying at slider position 0.
- 1v1: 400 independent boards, seed 727100000, indices 0..399,
  200 games in each focal seat; one Heximax versus one AB2.
- Four players: 400 independent boards, seed 727200000, indices 0..399,
  100 games in each focal seat; one Heximax versus three AB2.
- Native arena `_play_one`, antithetic=False, action cap MAX_ACTIONS.
  Every game records the native action/chance tape, entrant/seat mapping,
  outcome, reference deadline observations, adaptive activity and settings.
- One 30-worker CPU pool, BLAS/OpenMP threads 1, PYTHONHASHSEED=0,
  network disabled during evaluations. Atomic per-game checkpoints and a
  15-second heartbeat, with last-progress time and explicit failure status.
- Preflight: two independent 1v1 boards and four four-player boards at seeds
  727900000 and 727910000, each played with speedups off and fast. Compare full
  records, outcomes and zero reference deadline hits. Use fast only if all
  match; otherwise use off. These games never enter the quoted sample.
- Report Heximax wins / all 400 games, descriptive 95% Wilson intervals,
  opponent wins, unfinished games, seat census and exchange counts.
  No failed game may be discarded or assigned a replacement seed. An error
  stops the stage; a checkpoint resume retains every completed original index.
- Independently audit all records before publishing; preserve raw records,
  manifests, source/dependency hashes and runtime receipts. These are strength
  measurements for the recorded stack, not a controlled engine-speed comparison.
