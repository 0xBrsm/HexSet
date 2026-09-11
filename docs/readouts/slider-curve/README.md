# Validating the direct adaptive curve

Fresh confirmation is running under the frozen [plan](PLAN.md). No policy,
window, margin or sample-extension decision depends on confirmation outcomes.

The current slider runs from N (the current no-trade move profile) to M (the
best previously validated trading policy). Thus fixed .5 here is halfway
between N and M; it is not M itself. Every seat uses the T exchange evaluator
with floor zero. Candidates differ only in their fixed position or the
production adaptive rule. Real games and information histories are native
HexSet throughout; Catanatron is neither host nor opponent.

## Completed screen

64 games per policy and condition; five fixed positions plus adaptive A.
Seven conditions, 2,688 games, completed in 417.84 seconds with 30 workers.
Eight extra control games matched full endpoint action/trade histories exactly.

| Condition | Fixed0 | .25 | .5 | .75 | 1 | Adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 21 | 23 | 24 | 22 | 16 | 21 |
| Moderate | 15 | 19 | 26 | 27 | 31 | 20 |
| Heavy | 16 | 18 | 18 | 17 | 11 | 18 |
| Full | 11 | 11 | 15 | 15 | 13 | 13 |
| Mixed | 17 | 21 | 18 | 17 | 17 | 15 |
| Rising | 21 | 19 | 15 | 15 | 11 | 18 |
| Falling | 18 | 14 | 12 | 15 | 17 | 14 |

The screen selects fixed .5 as the strongest global control (128/448 wins;
adaptive 119/448). Selected condition controls are .5,1,.5,.5,.25,0,0 in table
order (148/448 combined). The conditional control's apparent advantage is
selection-biased in these data; it must be assessed on fresh boards. These
64-game cells do not establish a continuous optimum or a monotonic curve.

The independent audit checked all 2,688 games: source identities, complete and
balanced samples, correct entrant-to-seat mapping, positive-gain whole-game
trade records, public event participants, every logged activity and applied
alpha, and the selected controls and sample-size calculation.

## Fresh confirmation

The registered variance formula selected 512 games per policy per condition.
The uncapped target was 457.36; the 1024 cap did not apply. Planning targets 80%
power per primary contrast under equal true strength, not 80% joint power.
Three policies (A, global .5, selected condition control) would require 10,752
games. Where the two controls are identical, they share one cohort, reducing
the actual run to 9,216 games without reducing either comparison's sample.

Each primary comparison uses 3,584 matched boards across seven equally weighted
conditions. Both one-sided 97.5% lower bounds must exceed -2 percentage points
for the registered non-inferiority claim. This controls family-wise error at 5%
for the two aggregate contrasts. Per-condition 95% intervals are descriptive;
passing aggregate non-inferiority would not certify the margin in every
condition, nor prove a globally optimal curve. A result that does not pass is
not automatically evidence of inferiority. No outcome-driven sample extension.

## What the diagnostics measure

Activity and applied alpha are logged at the first focal decision of each
global turn. The bot may update again after that turn's trade event; these
summaries are not an average over every within-turn action. Every logged value
can be reconstructed from public observations preceding that decision.

Phase bins average within game first, then across games reaching the bin.
Late-bin means therefore condition on continued play. The rising/falling
pre/post statistic pairs bins 20–39 and 60–79 within games that reach both.
Absolute step sizes and sign reversals describe motion, not proof of harmful
oscillation. A responsive signal need not select a strong weight position.

The screen's mean activity was 0 without trading, .21 moderate, .40 heavy,
.57 full, .30 mixed, .23 rising and .29 falling. Rising activity increased
alpha by about .45 and falling activity reduced it by about .39 in the paired
pre/post comparison. Those changes verify direction; strength is a separate
fresh-data question. At full participation not every eligible turn produces
a mutually beneficial exchange, so an activity estimate below 1 is expected.

## Reproduction

Production source bcb9b92; plan/driver committed 3c1e1cc before launch.
Source fingerprint:
`d2df60dd0abeb72f1bd0ae4222716ce1b4246ac249c64cd2d76aeb0aa611b001`.
Python 3.12.14 / NumPy 2.5.2, Docker image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`.
One 30-worker, 30-CPU pool, evaluation network disabled, BLAS/OpenMP threads 1,
PYTHONHASHSEED=0. The host supplied about 29.2 CPUs during the measured screen
snapshot; other concurrent application load was small.

From that frozen source export with PYTHONPATH=/study/src:

```sh
python docs/readouts/slider-curve/run.py --stage screen --out /out/screen
python docs/readouts/slider-curve/run.py --stage confirmation \
  --selection /out/screen/summary.json --out /out/confirmation
```

Archive each completed output directory and run `analyze.py` independently.
`plot.py` generates PNG and SVG figures from audited analysis JSON; plotting
dependencies were installed separately and never enter the evaluation runtime.
Raw records, original summaries, audited analyses and SHA256 sidecars remain
separate. Screen seeds 724000000; fresh seeds 725000000, plus 10000 per condition;
antithetic=False. No verification replay is counted as a strength sample.
