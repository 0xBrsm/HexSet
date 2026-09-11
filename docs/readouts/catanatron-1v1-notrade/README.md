# Heximax-notrade against Catanatron: 300-game 1v1 baseline

On September 10, 2026, the unchanged `heximax-notrade` preset won
**194/300 games (64.7%; Wilson 95% interval 59.1–69.9%)** against
Catanatron `AB:2` in Catanatron's engine. Catanatron won the other 106 games.
Mean final victory points were 8.65 for Heximax and 7.00 for Catanatron.

This is a direct two-player baseline: fair share is 50%, making this result
1.29× fair share. It does not measure the four-player goal of exceeding 50%
against three Catanatron opponents.

## Configuration and execution

- Wintermute, Debian WSL, Docker; **30 workers × 10 games**, plus one coordinator.
- Matchup: `DC:heximax-notrade,AB:2`. Heximax depth 2, width 6, k=1,
  no-trade weights; Catanatron AlphaBeta depth 2, base value function.
- Player-to-player trading disabled; ordinary bank/port trades remain available.
  No weight changes, trade interpolation, or separate trade evaluator.
- Shard seeds 20260910–20260939; `PYTHONHASHSEED=0`.
- Catanatron 3.3.0, commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.
- HexSet HEAD `d5ec5eb99adc3012478e5075bb5c94e7f0fd6f79`, with the exact
  source snapshot archived below. Existing uncommitted trade-curve benchmark
  modules were included in the snapshot but are not used by this runner.
- Standard Catanatron `play_batch`, with its JSON output accumulator enabled.
  The wrapper retains returned game metadata and adapter counters.
- Elapsed simulation time: 42.8 seconds. Container exited successfully.

## Validation

All 300 games finished, with 300 distinct game seeds and IDs. All 300 raw
records were read and checked against summary winners, seating, action counts,
and final victory points. All 30 shards contain exactly 10 completed games.

Catanatron randomized seating. Heximax won 92/146 games (63.0%) when first
in the initial seating and 102/154 (66.2%) when second.

The unchanged adapter used its random legal-move fallback on **11/32,678
decisions (0.034%)**. These moves remain part of the measured shipped baseline;
the result is not a zero-fallback run. Mean game length was 71.05 turns.

## Artifacts

- [Report](report.txt), [summary and per-game metadata](summary.json),
  [validation](validation.json), and [execution log](run.log).
- [Runner](run_baseline.py) and [launch provenance](launch.json).
- [All 300 raw game records and shard results](raw-results.tar.gz).
- [Exact source snapshot](source.tar.gz).

To repeat in the recorded image, extract the source archive into an empty
working directory and run with `PYTHONPATH=src PYTHONHASHSEED=0`:

```sh
python -u docs/readouts/catanatron-1v1-notrade/run_baseline.py
```

The runner refuses to overwrite an existing `results` directory. The launch
metadata records the image digest and source archive hash; validation records
the raw results archive hash. Future independent baselines should use a fresh,
nonoverlapping shard seed range.
