# Heximax trade-frequency experiment

All results in this readout use a **zero trade gain threshold for every
player**. A trade still requires both players to expect a strictly positive
gain. This is an exploratory test of whether move-evaluation weights should
depend on opponents' trade participation.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m hexset.bench.trade_curve --games 32 --workers 4 --trade-floor 0 --seed 410000 --out docs/readouts/trade-curve/positive-gain.json
```

Each game seats one focal player against three Heximax opponents. The focal
player's move weights are `(1 - alpha) * NO_TRADE_WEIGHTS + alpha * TRADING_WEIGHTS`,
with `alpha` in `{0, 0.5, 1}`. Opponent move weights and every player's trade
valuation stay at `TRADING_WEIGHTS`; search and the opening placement prior
are held fixed. The focal player is always willing to consider a trade.

Opponents consider trading on their own turns with probability `p` and on
others' turns with probability `p`, for `p` in `{0, 0.5, 1}`. Separate random
draws are stable throughout a turn and independent of the search RNG, hands,
and number of candidate bundles. This avoids retrying random acceptance over
hundreds of bundles until one happens to pass. The experiment uses automatic
arena clearing; `p` is willingness to participate, not an observed acceptance
rate. Initiation and response probabilities can be separated in later runs
using `--regimes`.

There are 32 games per cell, with balanced seat rotations and two games per
board. All nine cells reuse the same 16 boards and initial random streams.
The full grid contains 288 games, but those are **not 288 independent board
samples**. Paired comparisons average adjacent games before estimating
uncertainty. The JSON retains every game's winner, points, seating, and
completed-trade counts.

The benchmark's full-game regression checks that full willingness exactly
reproduces ordinary zero-threshold Heximax, including actions, chance draws,
and executed trades. Other checks cover weight/gate isolation, turn-stable
willingness, unchanged search RNG, and board pairing.

This tests best responses to a fixed opponent population along the line
between two existing weight sets. It does not search arbitrary weights,
establish a continuous optimum, or learn trade frequency from a ledger.
Selection on this grid requires confirmation on fresh boards before changing
the playing policy.

## Initial results

[Raw results and per-game outcomes](positive-gain.json). All 288 games finished.

Entries below are focal wins out of 32 games. Trading activity is averaged
across the three blends at each willingness setting.

| Opponent willingness | No-trade weights (0) | Midpoint (0.5) | Trading weights (1) | Table trades/game | Focal trades/game |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 7/32 | 7/32 | 7/32 | 0.0 | 0.0 |
| 50% | 6/32 | 6/32 | 4/32 | 31.3 | 21.0 |
| 100% | 5/32 | 5/32 | 5/32 | 55.7 | 28.1 |

The manipulation produced distinct trading environments: roughly 0, 31, and
56 completed table trades per game. It did **not establish a preferred weight
blend or a trade-dependent curve**. All paired win-rate differences against
the trading weights have exploratory 95% intervals that include zero.

At 50% willingness, both alternatives gained 6.25 percentage points in
observed win rate. The paired interval is [-13.5, +26.0] points for the
no-trade weights and [-6.0, +18.5] for the midpoint. These intervals are broad;
this pilot does not demonstrate equivalence or rule out useful improvements.

The next substantive measurement is a larger run of this same zero-threshold
grid on fresh boards. It should test whether relative performance changes
with willingness before selecting or fitting an adaptive mapping. A shared
win count alone does not mean the policies took identical actions.
