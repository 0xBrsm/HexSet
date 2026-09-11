# AB2 adapter map sharing and board reconstruction

September 11, 2026 UTC, Wintermute. Source commit
`f87fb605ecae8ce83fdd6597c73a474c5cdd8c76`; product fingerprint
`0cc6d8a2d6d65da7bacd801062655f22fc4c2c0ca903ced93ced11a9ad561b44`.
The baseline is HexSet `d450a4e46c121d8ead7682f9a48a7cdbfca953ee`.
Both use the supported Catanatron pin
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`, fast mode, four AB2 players,
standard rules and the unchanged search deadline. The native arm uses the same
pinned search and speedups. AB2 hypothetical search remains in Catanatron in
all arms; the HexSet arms host the real game through HexSet's arena.

## Change

Seats using the same immutable HexSet board and seating size share one native
map. A weak registry releases the mirror when its bots are discarded. Reusing
a bot with a different board or seating size refreshes its mirror and players.

The mirror caches one native board template keyed by vertex owners, building
types, road owners and longest-road holder. Each decision copies the template
and refreshes the robber position. Occupancy changes reconstruct the template
through the existing native road-network code. Hands, player features, turn
flags and legal actions are still rebuilt each decision. Search receives
independent mutable board structures, preserving isolation from the template.

## Repeated throughput measurement

Two rounds of 120 games per arm, with 30 workers and a 30-CPU container:

1. baseline HexSet, native Catanatron, fixed HexSet;
2. fixed HexSet, native Catanatron, baseline HexSet.

Each arm starts a fresh interpreter and pool. Tasks use the same predetermined
seeds 680200000..680200119, one game per dynamic task. Timings include pool
startup/shutdown and game construction. No competing benchmark runs overlap.
The second round reverses order to reduce order effects; two rounds do not
establish statistical significance or remove machine-load variability.

| Arm | Round 1 seconds | Round 2 seconds | Combined games/s | Summed game CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| Baseline HexSet | 73.891 | 71.353 | 1.652 | 3596.15 |
| Fixed HexSet | 71.251 | 70.878 | 1.689 | 3538.57 |
| Native Catanatron | 65.998 | 65.393 | 1.827 | 3390.37 |

Combined throughput is total games divided by total batch time, not an average
of rates. The fix improves observed HexSet throughput **2.19%** and reduces
summed game CPU time **1.60%**. The individual throughput improvements are
3.71% and 0.67%; the baseline varied more between rounds than the fixed arm.
The fixed host is still **7.55% below native throughput**, with **4.37% more
CPU work** and lower effective CPU concurrency (24.90 versus 25.80).
**This patch does not establish throughput parity.** The hosts' game
trajectories differ, so the residual is not an isolated rules-engine cost.

All **720 game plays** reached a winner and matched the original corresponding
host's complete chosen-action digest, action count, winner, all-seat VP and
turn count. Thus every fixed game also matches its baseline, in both rounds.
The verification receipt is [verified.json](verified.json). Full per-game data
and source fingerprints are retained in the six round/arm JSON files.

## Diagnostic confirmation

After the throughput runs, a separate one-CPU profile repeats HexSet seed
680200000 using the unchanged [profile script](../profile_host.py). Its full
action trace and outcome match the prior profile and clean runs.

| Diagnostic | Before | Fixed |
| --- | ---: | ---: |
| Decisions | 191 | 191 |
| Feature-cache resets | 45 | 1 |
| Feature-cache misses | 2488 | 2486 |
| Native board reconstructions | 191 | 49 |
| State conversion cumulative profile seconds | 0.1833 | 0.0699 |

Map sharing eliminates the redundant resets, but only avoids **two misses in
this game**: a reset count alone overstates its performance impact because
cache keys include the evaluating seat and changing board state. Reusing the
board reconstruction removes most of the measured conversion cost. The
profiles were taken at different times and are diagnostic, not a throughput
comparison; the controlled batch results above are the performance evidence.

## Validation and reproduction

Local default suite: 852 passed, 14 skipped, one slow test deselected. Adapter
suite: 64 passed, one slow test deselected. Tests cover map sharing/lifetime,
changing board/seating, fresh-versus-cached board equivalence, robber/occupancy/
holder invalidation and mutation isolation with speedups both off and fast.
GitHub Actions passed core Python 3.11/3.13 and the optional-dependency/full-game
integration job for the measured source commit.

Run [run_compare.py](run_compare.py) in the immutable Python 3.12.14 image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`.
Mount baseline source as `/baseline/src`, fixed source as `/study/src`, the
unchanged [worker.py](../worker.py) as `/study/worker.py`, this driver as
`/study/run_compare.py`, and the [original pool data](../pool.json) as
`/study/prior-pool.json`. Mount a fresh writable output directory as `/out`.
Set PYTHONHASHSEED=0 and BLAS/OpenMP thread limits to one, use 30 CPUs and no
network, then run `PYTHONPATH=/study/src python /study/run_compare.py`.

The driver rejects existing outputs and checks all traces against the original
pool data. [runtime.json](runtime.json) records the exact container, mounts,
limits, environment and successful exit. The additional profile uses one CPU;
its receipt is [profile-runtime.json](profile-runtime.json). The binary profile
and extracted function data are [fixed-profile.prof](fixed-profile.prof) and
[fixed-profile.json](fixed-profile.json).
