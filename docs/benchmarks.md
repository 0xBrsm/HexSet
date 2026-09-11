# Recorded benchmarks

These results provide context for the bots' playing strength. They describe
the revisions and configurations measured, rather than a performance
guarantee for the current code.

## Primary evaluation and profiling framework

**HexSet is the primary engine and evaluation framework. Catanatron is an
external reference opponent.** Use `hexset.arena` / `hexset.bench.duel` for
self-play, ablations, candidate validation and performance profiling. Matches
against the `catanatron` entrant also run in HexSet. See the
[evaluation protocol and recovery steps](evaluation.md). Catanatron-hosted
runs are separate external compatibility experiments and must be labeled
with their host and information model.

The [September 11 native-engine profile](readouts/native-engine/README.md)
measures trading and no-trade Heximax self-play without importing Catanatron.
It identifies repeated leaf/hand evaluation and longest-road calculations as
larger targets than raw state copying, and preserves clean timings, decision
trace checks, source fingerprints and raw cProfile data.

The [four-AB2 host comparison](readouts/ab2-host-throughput/README.md) measures
AB2 games/second on both hosts using the same supported native pin. With fast
mode and 30 workers, the initial 120-game batches measured 1.63 games/s through
HexSet's adapter and 1.85 directly in Catanatron. The subsequent
[adapter fix and repeated comparison](readouts/ab2-host-throughput/adapter-fix/README.md)
measured 1.689 versus 1.827 games/s. AB2's hypothetical search uses Catanatron
in both configurations. These workload-specific timings do not establish
information-model equivalence or select the host for HexSet policy research.

## Native Heximax improvement (September 11, 2026)

The [fixed native holdout](readouts/native-expansion/README.md) passed the
registered gates with the adopted no-trade configuration: 1,086/2,048 wins
against three pinned AB2 external references (53.03%, 95% lower bound 50.86%)
and 714/2,048 against three frozen previous Heximax-notrade bots (34.86%,
lower bound 32.83%). Stricter first-attempt bounds also pass. All games use
HexSet's public-history ledger and balanced candidate seats. The readout
preserves complete game records and the narrowly scoped adapter repair that
was needed to finish one originally failing seed.

## Information-model correction (September 11, 2026)

Heximax results below using the stock `DC:` Catanatron adapter were measured
with a **memoryless public hand ledger**. The adapter reconstructs that ledger
at every decision; native HexSet retains public resource history. Heximax uses
those beliefs in search, so equal weights and search settings do not make the
hosted policies equivalent.

These historical results describe the restricted adapter protocol. They do
not establish rankings, candidate rejection, or optimal weights/hyperparameters
for native HexSet's normal public-history policy. Native-policy conclusions
require revalidation in HexSet. A gate against other `DC:heximax` players also
uses the restricted protocol, even when described as self-play.

This particular defect does not affect pure native HexSet runs or Catanatron
AB2-versus-ValueFunction games without Heximax. Copy/evaluator equivalence and
throughput measurements retain their stated implementation/workload scope.
The raw results below are preserved; their scope is corrected here.

## Heximax against Catanatron: four players

The September 7, 2026 readout records `heximax-notrade` against three
Catanatron `AB:2` players. Each `AB:2` is Catanatron's depth-two alpha-beta
player. The games ran in Catanatron through HexSet's adapter, with
player-to-player trading disabled. Heximax used its no-trade weight profile
and the adapter's memoryless subset of its seat's information set.

| Run seed | Heximax wins | Win rate | Mean Heximax victory points |
| --- | --- | --- | --- |
| 7 | 238 / 500 | 47.6% | 7.93 |
| 8 | 235 / 500 | 47.0% | 7.94 |
| Combined recorded results | 473 / 1,000 | 47.3% | — |

Both runs used 30 workers, `PYTHONHASHSEED=0`, and Catanatron 3.3.0 at
commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.

The raw reports describe shard seeds 7–36 and 8–37. Those ranges overlap,
so the combined count should not be read as 1,000 independent trials.
The individual run counts are retained here; no combined confidence interval
is asserted. The equal-share reference for this four-player lineup is 25%.

Sources are preserved at HexSet commit
`acd057555c7b2db6675cc0827eedfd41685450a9`:

- [Benchmark readout](https://github.com/0xBrsm/HexSet/blob/acd057555c7b2db6675cc0827eedfd41685450a9/docs/readouts/heximax-fit/README.md)
- [Seed 7 raw report](https://github.com/0xBrsm/HexSet/blob/acd057555c7b2db6675cc0827eedfd41685450a9/docs/readouts/heximax-fit/heximax-notrade-s7.txt)
- [Seed 8 raw report](https://github.com/0xBrsm/HexSet/blob/acd057555c7b2db6675cc0827eedfd41685450a9/docs/readouts/heximax-fit/heximax-notrade-s8.txt)

The archived reports have not been rerun for this cleanup.

## Heximax against Catanatron: two players, colonist rules

These September 10 results are historical aggregate reports; raw per-game
records and an exact HexSet run revision are not included here. They were
not rerun during PR review. The original rules PR used shard-based seeding;
PR #134 changes the game sequence to per-game seeding, so seed 0 alone is
not sufficient to reproduce an older run with the current runner.

Three 400-game duels run September 10, 2026. `DC:heximax` is Heximax
played through HexSet's Catanatron adapter; `AB:2` is Catanatron's
depth-two alpha-beta player and `F` its value-function player. The games
use colonist-style rules -- 15 victory points to win, discard over 9 cards
-- implemented as `Rules` carried on `GameState`
(`hexset.rules.COLONIST_1V1`) and selected in the duel CLI with
`--game-type=colonist-1v1`. The adapter forces `max_trades=0` because
Catanatron never generates a trade offer as a playable action, so the
`heximax` arm below used its trading weight profile with trading switched
off, while `heximax-notrade` used the no-trade profile.

| Heximax arm | Opponent | Heximax wins | Win rate, Wilson 95% | Mean Heximax VP |
| --- | --- | --- | --- | --- |
| heximax | AB:2 | 258 / 400 | 64.5% [59.7%, 69.0%] | 13.21 |
| heximax-notrade | AB:2 | 265 / 400 | 66.2% [61.5%, 70.7%] | 13.20 |
| heximax-notrade | F | 272 / 400 | 68.0% [63.3%, 72.4%] | 13.30 |

All three runs used seed 0, 2 workers, `PYTHONHASHSEED=0`, and Catanatron
3.3.0 at commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`. The marginal intervals overlap substantially. Establishing a difference
between weight profiles or opponents requires a direct comparison, ideally
using paired per-game outcomes; interval overlap alone is not such a test.

## Heximax against Catanatron: four players, standard rules

September 10, 2026: `heximax-notrade` against three Catanatron `F`
(value-function) players, 400 games under standard rules (10 VP, discard
over 7), seed 0, 2 workers, `PYTHONHASHSEED=0`, Catanatron 3.3.0 at
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.

| Seat | Wins | Win rate, Wilson 95% | Mean VP |
| --- | --- | --- | --- |
| heximax-notrade | 230 / 400 | 57.5% [52.6%, 62.3%] | 8.52 |
| F | 51 / 400 | 12.8% [9.8%, 16.4%] | 6.21 |
| F | 58 / 400 | 14.5% [11.4%, 18.3%] | 6.21 |
| F | 61 / 400 | 15.2% [12.1%, 19.1%] | 6.31 |

The equal-share reference for this four-player lineup is 25%. The three
opponent seats split the remaining wins about evenly, as expected.

## Catanatron AB:2 against Catanatron ValueFunction, two players

September 10, 2026: 800 games, standard rules (10 VP, discard over 7),
seed 0, 2 workers, `PYTHONHASHSEED=0`, Catanatron 3.3.0 at
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.

| Player | Wins | Win rate, Wilson 95% | Mean VP |
| --- | --- | --- | --- |
| AB:2 | 413 / 800 | 51.6% [48.2%, 55.1%] | 7.88 |
| F | 387 / 800 | 48.4% [44.9%, 51.8%] | 7.92 |

This does not reproduce the 97-53 result for the value-function player
reported in [bcollazo/catanatron#381](https://github.com/bcollazo/catanatron/issues/381)
at n=150. This sample does not establish an advantage for either bot; it does not
prove equivalence. The cause of the discrepancy (catanatron revision, value-function build,
or harness differences) was not determined.
