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

The archived reports have not been rerun for this cleanup. A comparable
two-player result has not been verified.
