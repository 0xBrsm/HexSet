# External-reference acceleration

HexSet is the primary engine and evaluation framework; Catanatron is an
external reference opponent. These patches accelerate Catanatron's AB2 and
ValueFunction implementations. The examples on this page use the separate
**Catanatron-hosted** diagnostic runner, not the HexSet ablation framework.
Use [the native evaluation protocol](evaluation.md) for Heximax policy work.
Stock `DC:` Heximax in this external runner receives a memoryless public
ledger; its strength results do not establish native HexSet policy rankings.


HexSet's optional dependency remains pinned to `d3f4ad0`, before upstream's
per-game RNG change (`3d9b462`, #380) and bot registry/lifecycle change
(`ecf9311`, #386). The adapters and hash-guarded patches documented here target
that pin. The separate upstream performance PR #389 incorporates both newer
commits and preserves their RNG sharing and player APIs; it does not upgrade
HexSet's dependency integration. Migrating that integration still requires
adapter changes and reproducibility checks against the new APIs.

The duel runner supports `--catanatron-speedups fast` for faster native AB
and ValueFunction opponents. The default is `off`, preserving the pinned
reference implementation for historical comparisons. The selected mode is
printed in every report and passed explicitly to each worker, including when
Python uses spawn or forkserver.

```sh
python -m hexset.catanatron.duel \
  --players=DC:heximax-notrade,AB:2,AB:2,AB:2 \
  --num=1000 --workers=30 --seed=640000000 \
  --catanatron-speedups=fast
```

`basic` combines structural board copies, elimination of duplicate production
extraction, and calculation of only the reachability features the native base
evaluator consumes. `cached` adds a bounded board-feature cache shared by the
native evaluators in each worker. Entries include player perspective, seating,
ordered buildings, roads, connected components, buildable nodes, and robber
location. Changing maps clears the cache; it retains at most 4096 entries.
Hands, victory points, development cards, army and longest-road lengths are
read from each leaf rather than cached. Native board maps are immutable.

`fast` adds structural state copies (removing the remaining pickle round trip),
reads just the perspective player's hand, and uses a smaller cache key containing
the actual enemy node/edge blockers. It preserves ordered per-seat production
inputs and component iteration order. `cached` remains available as the previous
acceleration baseline.

The implementation verifies the relevant pinned Catanatron source hashes before
activation. It uses a process-global context intended for single-threaded
workers, restores the original functions on exit (including errors), and rejects
nested activation or conflicting board/evaluator patches. It is also available
as `hexset.catanatron.speedups.catanatron_speedups` for other experiment runners.
The native `hexset.bench.duel` CLI currently has no `--catanatron-speedups`
option; accelerated native-host studies explicitly enter the context inside
each worker. The recorded speedups are not automatically enabled by selecting
the arena's `catanatron` entrant.

This preserves the native evaluator's arithmetic, including its original
seven-card discard penalty and relative P1 enemy-production term. Native search
and its 20-second deadline are unchanged. A faster implementation can complete
more search before that deadline, so decision equivalence requires measurement;
it is not guaranteed for deadline-limited decisions.

## Measured gain

On eight paired four-player AB2 games, `basic` reduced total wall time from
103.04 to 55.90 seconds (1.84x throughput); `cached` reduced it to 52.19 seconds
(1.97x). Every ordered action trace and outcome matched. These are one-CPU
measurements; see the [initial readout and raw records](readouts/ab-speedups/README.md).

On a new eight-seed sample, `fast` reduced wall time from 111.00 seconds native
and 54.98 seconds cached to 35.37 seconds: 3.14x native throughput and 55.46%
more throughput than cached. All full action traces and outcomes matched. See
the [follow-up readout and records](readouts/ab-throughput/README.md).

In two 240-game trials per profile on 30 workers, mean batch time fell from
184.91 seconds cached/static to 121.72 fast/static and 109.15 fast/dynamic.
Combined throughput increased 69.41%, with every full trace and outcome matching
across all profiles and repetitions.

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

## Keep workers busy

Duel games are assigned dynamically by default (`--scheduling=dynamic`), one game
per task. A worker picks up another game as soon as it finishes. Results are
aggregated in game-index order, so completion order cannot reorder per-seat VP.
`--scheduling=static` retains the previous fixed-shard assignment for comparisons.
The parent reports completed games on stderr; each game still starts from
`random.seed(seed + game_index)` under either scheduler.

The pool benchmark compares cached/static, fast/static, and fast/dynamic with
a fresh interpreter and pool per trial. Two repetitions reverse trial order;
all trials use the same 240 seeds and check actual seeds, ordered actions,
winners and VP for every game. Only one 30-worker pool runs at a time:

```sh
python -m hexset.catanatron.throughput_benchmark --games=240 --workers=30 \
  --repeats=2 --seed=650200000 --out=/tmp/ab-throughput.json
```
