# Heximax trade-frequency study

> **Scope correction:** This is a split-evaluator ablation, not a validation of
> a complete Heximax weight configuration. It varied move weights while
> retaining trading-profile trade valuations. Its results must not be used
> to recommend swapping Heximax's shared evaluator weights or to establish
> optimal weights for either trading regime. The earlier recommendation was
> broader than this experiment supports.

**The independent validation favors fixed no-trade move weights.** They
improved win rate by **3.45 percentage points** over trading move weights,
with an adjusted interval of **+0.60 to +6.30 points**. The selected adaptive
mapping scored 1.63 points below the fixed candidate; its interval is
**-3.84 to +0.58 points**. The upper bound rules out a five-point adaptive
gain in this tested population, and even a one-point gain for this mapping.

This supports pursuing a fixed move-weight change before adding a temporal
feature or adaptive ledger. **Trade valuations were held at TRADING_WEIGHTS**
for every player. These results do not validate changing both move and trade
evaluators together, nor do they rule out other adaptive functions.

## Independent validation

Selection used the first 256 boards. Validation used 256 fresh boards,
with two seat assignments and three equally weighted trading environments
per board: 1,536 game outcomes per evaluated policy. The two comparisons
use individual 97.5% intervals, providing Bonferroni family-wise 95% coverage.

| Move-weight policy | Holdout wins | Win rate |
| --- | ---: | ---: |
| Trading weights, fixed | 375/1536 | 24.41% |
| No-trade weights, fixed | 428/1536 | 27.86% |
| Selected adaptive mapping | 403/1536 | 26.24% |

The selected mapping used trading weights at willingness 0 and 0.5, and
no-trade weights at willingness 1. It was selected exclusively from the
first half of the data, under the previously recorded
[holdout plan](holdout-plan.json). The
[frozen selection](wintermute-study/selection.json) matches the selection
recomputed by the validation program.

## Does weight advantage depend on trading?

The prespecified interaction was **-4.30 percentage points** for trading
weights relative to no-trade weights as willingness rose from 0 to 1; its
95% interval was **[-8.56, -0.04] points**. Equivalently, the no-trade profile
gained 4.30 points of relative advantage as trading became more available.
The interval only narrowly excludes zero, so this is limited evidence of
an interaction, not a measured continuous optimal curve.

The complete grid below is descriptive; it includes both selection and
validation data and is not used to reselect the validated mapping.

| Opponent willingness | No-trade weights | Midpoint | Trading weights | Table trades/game |
| --- | ---: | ---: | ---: | ---: |
| 0% | 243/1024 | 233/1024 | 244/1024 | 0.0 |
| 50% | 317/1024 | 311/1024 | 301/1024 | 32.8 |
| 100% | 289/1024 | 257/1024 | 246/1024 | 57.9 |

## Execution and records

Ran on **Wintermute with 30 workers**, with every gate threshold fixed
at **zero**: 9,216 games on 512 independent board pairs, shared across
all nine cells. Runtime was 20 minutes 45 seconds. All games finished.
The [earlier pilot](pilot.md) supplied a planning variance estimate and is
excluded from these results. Interrupted local checkpoints are also excluded.

- [Study summary](wintermute-study/summary.json)
- [Independent validation](wintermute-study/holdout.json)
- [Run plan](wintermute-study/plan.json)
- [Complete raw checkpoints](wintermute-raw.tar.gz)
- [Launch and source provenance](wintermute-launch.json)

To recompute the analysis without rerunning games:

```sh
study_tmp=$(mktemp -d)
tar -xzf docs/readouts/trade-curve/wintermute-raw.tar.gz -C "$study_tmp"
PYTHONPATH=src python -m hexset.bench.trade_curve_holdout "$study_tmp"
```

## Design and reproduction

The focal move weights blend the no-trade and trading profiles at alpha 0,
0.5, or 1. Opponents' willingness to initiate and respond is 0, 0.5, or 1.
Opponent move weights and every trade evaluator remain fixed at the trading
profile. A zero threshold still requires strictly positive expected gains
for both participants. Willingness draws remain stable throughout each turn.

The prespecified primary contrast is:

```
(trading weights - no-trade weights) at willingness 1
  minus
(trading weights - no-trade weights) at willingness 0
```

Positive values mean the trading weights gain relative to the no-trade
weights as trading becomes more available. The study has no early stopping
for significance. Secondary within-regime comparisons are exploratory.

The [holdout plan](holdout-plan.json), recorded before any validation boards
ran, selects one static blend and one blend per willingness level using the
first eight blocks. The last eight blocks test the selected static blend
against the trading-weight baseline, and the selected adaptive mapping
against that static blend. These two secondary validation comparisons use
97.5% individual intervals, controlling their family-wise error at 5% by
Bonferroni correction. All three trading environments receive equal weight.

[Launch record](wintermute-launch.json) identifies the host, container, source
archive, worker count, and zero threshold. The interrupted local run is not
included. The Docker source snapshot is isolated from Wintermute's other
checkouts. Its full-game regression reproduced ordinary zero-threshold
Heximax exactly, including 62 executed trades in the checked game.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m hexset.bench.trade_curve_study --games 1024 --chunk-games 64 --workers 30 --seed 420000 --out study
PYTHONPATH=src python -m hexset.bench.trade_curve_holdout study
```

Completed cells are checkpointed. Repeating the study command resumes them;
the runner rejects changes to the plan or source. The game count is fixed
before observing the study outcomes. All 9,216 games completed, with no unfinished games. All 144 checkpoint
files were verified to use 30 workers and zero gate thresholds. The complete
raw archive was checked against every checkpoint before packing the readout.
