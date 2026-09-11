# Pure HexSet engine profile

Measured September 11, 2026 UTC on Wintermute using HexSet main commit
`d450a4e46c121d8ead7682f9a48a7cdbfca953ee` (0.48.2), Python 3.12.14 and one CPU.
The source comes from a clean git archive, excluding the unrelated uncommitted
research changes in the original workspace. This is HexSet's own arena, game
engine and Heximax search throughout. The harness asserts that **no Catanatron
module is loaded**. The container image contains Catanatron, but none is used.

## Workload and clean timings

Four-seat self-play, default depth 2 / width 6 / k=1 / win objective, under
standard rules. Both `heximax` (trading) and `heximax-notrade` play three games
with arena seed 660000000 and game indices 0, 1, 2. These use the arena's
separate board, game and per-seat bot random streams. Every game reached a
winner. Clean and profiled arms each start a fresh interpreter, and match the
ordered chosen-action digest, decision count, winner, all-seat VP and turns.

| Preset | Clean total seconds | Seconds/game | Decisions | Mean decision ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heximax | 3.358 | 1.119 | 856 | 3.557 | 0.739 | 16.289 |
| heximax-notrade | 4.393 | 1.464 | 1050 | 4.044 | 1.011 | 17.519 |

These timings include game setup and a small chosen-action hashing wrapper,
but exclude interpreter startup. The presets play different games and game
lengths, so their wall-time difference is not the isolated cost of trading.
Three games per preset establish a profiling sample, not a strength verdict
or a measured 30-worker throughput figure.

## Where time goes

Cumulative cProfile percentages below use total attributed profile time as the
denominator. Rows overlap through callers and callees and must not be added.
The profiler roughly doubles wall time in this workload; the clean timings
above are the performance measurements.

| Function | Trading | No-trade |
| --- | ---: | ---: |
| Heximax._search | 88.66% | 94.07% |
| HonestEvaluator.evaluate_game | 46.88% | 52.50% |
| hand_terms (scalar arithmetic) | 12.66% | 13.76% |
| longest_road | 12.99% | 12.95% |
| imagine (whole hypothetical game) | 13.25% | 12.83% |
| HonestEvaluator._walk | 11.69% | 11.42% |
| Evaluator.survey | 8.14% | 7.72% |
| HonestEvaluator.belief_for | 5.64% | 6.99% |
| trade_event | 8.81% | 3.60% |
| copy_state | 2.10% | 2.19% |

The no-trade sample has 44,000 leaf evaluations and 153,700 per-seat hand
calculations. Longest-road computation runs 39,820 times; its caller is
`road_lengths`, which recomputes every seat after a hypothetical road/building
change. The dominant caller of `update_longest_road` is `build_road` (7,066
calls). `imagine` runs 49,254 times; 48,302 calls shuffle a copied development
deck. Those shuffles alone account for 0.492 profiled seconds, versus 0.194
seconds for raw state copies.

## Next optimization candidates

1. Cache or specialize repeated hand scoring using every consumed input:
   hand, purchase availability, port ratios, deck availability, player count
   and discard rules. Preserve arithmetic and strict tie behavior.
2. Reuse longest-road results for unchanged road networks and enemy blockers;
   a road placement need not recompute unaffected opponents' networks. A
   settlement can split an opponent's network, so that case needs separate
   invalidation and award/tie checks.
3. Measure redundant evaluation/belief work across sibling states. Existing
   board, belief and evaluation caches already operate within a decision;
   any new cache should first demonstrate useful hits and cheap keys.
4. Investigate deferred deck randomization only where future draw semantics
   allow it. Skipping a shuffle changes RNG consumption, so this is not an
   automatically behavior-preserving optimization.

Raw state copying accounts for about 2% here. Further copy work alone has
limited upside; the previous Catanatron results should not determine HexSet's
optimization priorities. No engine or bot behavior was changed in this study.

## Reproduce

In a clean checkout of the measured revision, copy `profile_run.py` alongside
it, then run each arm in a fresh process with one CPU and single-threaded BLAS:

```sh
PYTHONPATH=src PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python profile_run.py --preset heximax-notrade --games 3 --seed 660000000 \
  --out /tmp/heximax-notrade-clean.json
# Repeat with --profile and a fresh output path, then both arms for heximax.
```

`profile_run.py` delegates game creation and play to the existing
`hexset.bench.profile_heximax` / `hexset.arena` path. JSON files include source
and harness fingerprints, decision latency, game outcomes and full function
statistics. `.prof` files retain caller relationships for `pstats` inspection.

- [heximax-clean.json](heximax-clean.json) — SHA-256 `ba9b8e63d35d9f46faff438cace568c928676cd39a5bec29f82b696d6b809d6e`
- [heximax-notrade-clean.json](heximax-notrade-clean.json) — SHA-256 `97e08b7410e8b8f0a4ba28eb953a562bb73e4c114f0bf5365f2feda1148c621e`
- [heximax-notrade-profile.json](heximax-notrade-profile.json) — SHA-256 `b84bfc7d1936b1d696d2a68826b455b55a211d38998602ba454d6650d84794c3`
- [heximax-notrade-profile.prof](heximax-notrade-profile.prof) — SHA-256 `e61e1d966f8321783dc637bf852c3c891402a7ac40fdcfcd529abd76a931bd99`
- [heximax-profile.json](heximax-profile.json) — SHA-256 `a63fced752cbf75f208b0b352c7e7b24d53d96b6b02ad14c60da8534f1fa5139`
- [heximax-profile.prof](heximax-profile.prof) — SHA-256 `b40befd46e127cbd691716d5571ddc12d877db90abe35013dc555f4671cb52a5`
- [profile_run.py](profile_run.py) — SHA-256 `f216109e6628eef80af12e77ce68860da854ac4905e1a44f3b0662905ea52e5e`
- [runtime.json](runtime.json) — SHA-256 `4042fe651947a52fcd45948f4c99d743f611ab9131c1fc68763dd5ff78cb31b7`
