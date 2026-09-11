# Recorded benchmarks

These results provide context for the bots' playing strength. They describe
the revisions and configurations measured, rather than a performance
guarantee for the current code.

## Heximax against Catanatron: four players

The September 7, 2026 readout records `heximax-notrade` against three
Catanatron `AB:2` players. Each `AB:2` is Catanatron's depth-two alpha-beta
player. The games ran in Catanatron through HexSet's adapter, with
player-to-player trading disabled. Heximax used its no-trade weight profile
and its seat's information set.

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
3.3.0 at commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`. The two weight
profiles and the two opponents are all within noise of each other here:
the intervals overlap almost entirely, so neither comparison separates at
n=400.

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
at n=150. On the build measured here the two bots are evenly matched;
the cause of the discrepancy (catanatron revision, value-function build,
or harness differences) was not determined.
