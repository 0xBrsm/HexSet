# Heximax road weight sweep

Challenger heximax (`Weights.road`/`Weights.card` varied) vs baseline heximax
(unmodified `TRADING_WEIGHTS`), depth 2, width 6, honest mode, trading on --
the shipped `heximax` preset in both seats. 384 games per cell, `--seed
90000`, boards identical across cells. Raw per-game data (winner, terminal
VP, roads/settlements/cities per seat, turns) is in `sweep.json`. Script:
`src/hexset/bench/road_sweep.py`.

## Seat-bias correction

The six-cell run below used an interleaved `[challenger, baseline,
challenger, baseline]` seating. `hexset.arena`'s antithetic pairing swaps
seats by `seats // 2` between the two games it plays on each board, which
cancels the seat term for a *grouped* lineup (`[a, a, b, b]`) but not an
interleaved one: shifting an interleaved lineup by two maps each side's
seat-pair onto itself, so the challenger held the same two seats on both
games of every board. The control cell (challenger weights byte-identical to
the baseline) read 44.3% instead of the expected ~50% for exactly this
reason -- confirmed by rerunning the control alone, 96 games, with the
lineup regrouped to `[challenger, challenger, baseline, baseline]`: **49.0%
[39.2%, 58.8%]**, consistent with no effect (`control-rerun.json`).
`road_sweep.py` now uses the
grouped lineup; the fix was not worth re-running the full sweep for, since
every cell shares the same board sequence and the same seat mechanics, so
the bias is common to all six and the "vs control" column below cancels it
out to first order.

## Results

| cell (road / card) | games | win% [95% CI] | vs control | roads/game (c vs b) | mean VP (c vs b) | mean turns |
|---|---|---|---|---|---|---|
| 0.1209 / 0.005406 (control) | 384 | 44.3% [39.4%, 49.3%] | -- | 9.80 vs 9.86 | 6.52 vs 6.94 | 89.1 |
| 0.08 / 0.005406 | 384 | 44.5% [39.6%, 49.5%] | +0.3 pt | 9.67 vs 9.82 | 6.56 vs 6.90 | 89.4 |
| 0.04 / 0.005406 | 384 | 44.3% [39.4%, 49.3%] | +0.0 pt | 9.66 vs 9.75 | 6.60 vs 6.90 | 89.4 |
| 0.0 / 0.005406 | 384 | 41.7% [36.8%, 46.7%] | -2.6 pt | 7.55 vs 10.13 | 6.62 vs 6.93 | 87.9 |
| 0.04 / 0.02 | 384 | 49.0% [44.0%, 53.9%] | +4.7 pt | 8.85 vs 9.72 | 6.85 vs 6.96 | 88.5 |
| 0.0 / 0.02 | 384 | 44.8% [39.9%, 49.8%] | +0.5 pt | 8.00 vs 9.98 | 6.79 vs 7.10 | 90.2 |

All 384 games decided in every cell (no timeouts).

## Reading

Lowering the road weight from 0.1209 to 0.04 barely moves the road count
(9.80 to 9.66 roads/game) and does not move strength either -- three cells
that read essentially the same ~44%. Only zeroing `road` outright cuts the
challenger's own build meaningfully, to 7.55 roads/game against the
baseline's unchanged ~10, and it costs a little rather than gaining: -2.6
points versus control, well inside the interval's noise. Nothing here is a
clear win -- no cell's interval excludes 50% once the seat bias is accounted
for, including the +4.7-point cell (road=0.04, card=0.02), whose edge over
control is the largest observed but still overlaps no-effect; that cell also
builds the fewest settlements and the most cities (1.28 vs 1.04), suggesting
a richer hand pushes heximax toward upgrading rather than away from roads
specifically. Read together: the road weight is not free -- zeroing it
trades a few points of strength for markedly fewer roads -- but nothing
tested here is a clean upgrade over the shipped weights. Next: rerun
road=0.04/card=0.02 at around 1000 games with the corrected pairing to see
if its edge survives, and ablate `surplus_card` and `progress` instead of
`road`, since those look like more plausible levers on why the search buys
roads over settlements at depth 2.
