# Round6000 recovery and continuation

Recorded 2026-09-10 UTC after the certified Wintermute generation-6000 run
exited with status 0. The source, controller, image, 30-worker configuration,
and no-trade Heximax phenotype remain pinned; no engine or policy source was
changed.

Strict post-validation passed all 5,760 records; selection was derived from validated raw records and native promotion metadata was excluded.
The local validation artifact is `/data/data/com.termux/files/usr/tmp/luna-round6000-recovery/round6000-validation.json`.

| Candidate | AB2 | Shipped | Joint margin |
|---|---:|---:|---:|
| g6000-p00 | 111/240 (46.25%) | 70/240 (29.17%) | -0.037500 |
| g6000-p01 | 108/240 (45.00%) | 70/240 (29.17%) | -0.050000 |
| g6000-p02 | 109/240 (45.42%) | 72/240 (30.00%) | -0.045833 |
| g6000-p03 | 100/240 (41.67%) | 69/240 (28.75%) | -0.083333 |
| g6000-p04 | 111/240 (46.25%) | 77/240 (32.08%) | -0.037500 |
| g6000-p05 | 91/240 (37.92%) | 61/240 (25.42%) | -0.120833 |
| g6000-p06 | 102/240 (42.50%) | 59/240 (24.58%) | -0.075000 |
| g6000-p07 | 109/240 (45.42%) | 68/240 (28.33%) | -0.045833 |
| g6000-p08 | 97/240 (40.42%) | 51/240 (21.25%) | -0.095833 |
| g6000-p09 | 108/240 (45.00%) | 75/240 (31.25%) | -0.050000 |
| g6000-p10 | 113/240 (47.08%) | 68/240 (28.33%) | -0.029167 |
| g6000-p11 | 106/240 (44.17%) | 61/240 (25.42%) | -0.058333 |

The strict joint-margin selection is `g6000-p10`. Its effective phenotype is
depth 2, width 6, max_nodes 600, k=1, `notrade`, max_trades 0, win stance,
temperature `2.476644394795811`, internal placement enabled, and weights:

```text
production=3.8803981040675644 diversity=0.360074083309872
scarce=0.07039861111111112 buy_progress=0.4606045314909384
road=0.00019147194463070284 knight=0.16043960472833257
spare_card=0.11421297701105623 robber_risk=-0.31245697996217825
port=0.019305197991678444 victory_point=1.0
```

The planner sidecar is at
`/data/data/com.termux/files/usr/tmp/luna-round6000-recovery/round6000-confirmation-sidecar.json`.
The AB2 continuation is running on Wintermute as
`luna-round6000-confirm-vs-ab2`, 1,000 games, 30 workers, seed 620000000.
The shipped gate remains queued for seed 620100000 and must start only after
the AB2 container exits successfully.
