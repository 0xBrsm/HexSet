# Catanatron acceleration

The duel runner supports `--catanatron-speedups cached` for faster native AB
and ValueFunction opponents. The default is `off`, preserving the pinned
reference implementation for historical comparisons. The selected mode is
printed in every report and passed explicitly to each worker, including when
Python uses spawn or forkserver.

```sh
python -m hexset.catanatron.duel \
  --players=DC:heximax-notrade,AB:2,AB:2,AB:2 \
  --num=1000 --workers=30 --seed=640000000 \
  --catanatron-speedups=cached
```

`basic` combines structural board copies, elimination of duplicate production
extraction, and calculation of only the reachability features the native base
evaluator consumes. `cached` adds a bounded board-feature cache shared by the
native evaluators in each worker. Entries include player perspective, seating,
ordered buildings, roads, connected components, buildable nodes, and robber
location. Changing maps clears the cache; it retains at most 4096 entries.
Hands, victory points, development cards, army and longest-road lengths are
read from each leaf rather than cached. Native board maps are immutable.

The implementation verifies the relevant pinned Catanatron source hashes before
activation. It uses a process-global context intended for single-threaded
workers, restores the original functions on exit (including errors), and rejects
nested activation or conflicting board/evaluator patches. It is also available
as `hexset.catanatron.speedups.catanatron_speedups` for other experiment runners.

This preserves the native evaluator's arithmetic, including its original
seven-card discard penalty and relative P1 enemy-production term. Native search
and its 20-second deadline are unchanged. A faster implementation can complete
more search before that deadline, so decision equivalence requires measurement;
it is not guaranteed for deadline-limited decisions.

## Measured gain

On eight paired four-player AB2 games, `basic` reduced total wall time from
103.04 to 55.90 seconds (1.84x throughput); `cached` reduced it to 52.19 seconds
(1.97x). Every ordered action trace and outcome matched. These are one-CPU
measurements; see the [full readout and raw records](readouts/ab-speedups/README.md).

## Verification and measurement

The tests compare exact scalar values on real positions from all four
perspectives with both default and nondefault weights, check invalidation and
copy isolation, and verify worker configuration and context restoration.

The paired benchmark launches a fresh interpreter per game/arm and rotates arm
order. It compares actual game seeds, ordered actions, winners, and VP. Audit
mode additionally checks every scalar leaf against the original evaluator,
compares leaf counts and ordered score digests, and records deadline observations.
Use separate seed sets for audits and uninstrumented timings:

```sh
python -m hexset.catanatron.speed_benchmark --audit --games=2 \
  --seed=630000000 --out=/tmp/ab-speed-audit.json
python -m hexset.catanatron.speed_benchmark --games=8 \
  --seed=630100000 --out=/tmp/ab-speed-clean.json
```

The benchmark refuses existing summary paths. Every completed child has its
own JSON record, and the summary is updated atomically after each child. An
observed action/score mismatch stops the run. Clean timings include complete
games and exclude leaf instrumentation. They measure one serial worker;
30-worker throughput needs a separate assessment.

## Why not share the entire search tree?

The opponents use different player perspectives and act from different states.
A reusable minimax entry needs full dynamic state, remaining depth, perspective,
and an exact-value/lower-bound/upper-bound designation. Reusing a cutoff as an
exact score would change decisions. This change shares board-feature work while
leaving those search semantics intact. Catanatron also already caches several
lower-level geometric helpers, so their cost cannot be saved twice.
