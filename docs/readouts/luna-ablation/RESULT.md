> Evaluation-scope correction — 2026-09-11
>
> Both recorded gates use the stock `DC:` Catanatron adapter, including the gate against shipped Heximax. That adapter uses a memoryless public hand ledger. The historical FAIL is a result for that restricted protocol; it does not reject road-zero or establish the shipped weights for native HexSet with public history.
> Raw results and historical verdicts are preserved. See [scope audit](../ledger-scope-audit/README.md).

# Final result

The campaign completed 19 screen candidates, four confirmation candidates,
and one finalist. The finalist was `road-zero` (road weight 0). Its designated
fresh holdouts were:

| gate | wins | rate | Wilson 95% CI | threshold | result |
|---|---:|---:|---:|---:|---|
| candidate vs 3× Catanatron AB2 | 964/2048 | 47.1% | 44.9–49.2% | >50% | FAIL |
| candidate vs 3× shipped Heximax-notrade | 537/2048 | 26.2% | 24.4–28.2% | >25% | FAIL |

The machine verdict is `FAIL`. The screen leader was `progress-141`; it was
not adopted because the fresh confirmation selected `road-zero`, and only the
final holdout is confirmatory. The port ablation was retained in the screen:
`drop-port` scored 53/120 vs AB2 and 33/120 vs shipped Heximax, exploratory
only.

The launcher executed 14,800 games. The two finalist holdout invocations each
ran both gates; the verdict uses one designated fresh 2,048-game row per gate,
with the other 4,096 games retained as ancillary results.

## Accounting and controls

The valid screen was 19 candidates × 120 games × 2 gates = 4,560 games,
including the shipped-vector `baseline` control (60/120 vs AB2; 26/120 vs
shipped Heximax). The four confirmations were 256 games per gate: `progress-141`
113/256 and 57/256; `road-half` 112/256 and 63/256; `road-zero` 113/256 and
84/256; `prod-half` 101/256 and 70/256 (AB2 first, shipped second). The
confirmation finalist was `road-zero`.

The planned budget was 10,704 games. The launcher’s holdout command runs both
gates on every invocation. It was invoked once with seed 810000 and once with
seed 820000, producing two 2,048-game files, each containing both gate rows.
Thus it executed 4,096 designated holdout games plus 4,096 additional
cross-gate games, for 14,800 games executed. The verdict uses only the
preassigned `vs-ab2` row from seed 810000 and `vs-shipped` row from seed 820000;
the extra rows are retained as ancillary data. There was no rerun, no second
finalist, and no reuse of screen or confirmation seeds.

The initial 64-game attempt was invalid for the 30-worker requirement and is
preserved under `results-screen64-excluded/`; it contributed no selection or
verdict data.
