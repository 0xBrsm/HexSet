# Heximax and endpoint pinning

`heximax` is the single handcrafted bot. It uses the validated linear adaptive
policy by default, in the arena, server, Gym opponents and Python factory.
Recent public exchanges determine its move weights over an eight-eligible-turn
window. Its exchange evaluator stays fixed, with a zero gain floor.

For controlled tests, pin the slider to an endpoint:

| Setting | Move profile | Expansion credit | Trading |
| --- | --- | --- | --- |
| No pin | Adapts to public exchange activity | Interpolates | Enabled |
| `pin_weights=0` | Zero-activity endpoint | .25 VP | Enabled |
| `pin_weights=1` | Full-activity endpoint | .125 VP | Enabled |

Zero uses `NO_TRADE_WEIGHTS`; one uses `BALANCED_WEIGHTS`, the best validated
trading move profile. Both use `TRADING_WEIGHTS` to price exchanges. The numeric
pin describes a slider position, not whether the bot participates in trading.

```python
from hexset.bots.heximax import heximax

normal = heximax(board, rng)
fixed_zero = heximax(board, rng, pin_weights=0)
fixed_one = heximax(board, rng, pin_weights=1)
```

A pin holds weights and expansion credit throughout the game, regardless of
observed activity or trading restrictions. To decline trades independently,
pass `max_trades=0`. An unpinned bot immediately uses the zero endpoint when
its own or the game's trading switch is off. A pinned bot retains its pin.

## Evaluation flags

Compare normal Heximax against three copies pinned at one:

```sh
python -m hexset.bench.duel heximax heximax \
  --pin-weights-b 1 --geometry abbb --games 400 --workers 4
```

Use `--pin-weights-a` to pin side A. Omit both flags to let both sides adapt.
Flags apply to every seat on that side and are stored in each entrant's
experiment settings. They are rejected for other bot kinds or custom vectors.

Other arena entrypoints accept the same settings in a single entrant spec:

```text
heximax
heximax:pin-weights=0
heximax:pin-weights=1
heximax:trading=off
heximax:pin-weights=1:trading=off
```

Python entrants carry `Entrant.pin_weights`, so settings survive worker
serialization. `profile_heximax --pin-weights 0` supports the same experiment
without creating another bot preset. `ablate` and `weight_sweep` also take
`--pin-weights 0|1`, with a separate `--no-trading` switch. The normal server picker stays `heximax`;
pins are testing controls, not additional opponent names.

## Migration and recorded experiments

The separate `heximax-adaptive`, `heximax-balanced` and `heximax-notrade`
presets and kinds are removed. Use `heximax`, `heximax:pin-weights=1`, and
`heximax:pin-weights=0:trading=off`, respectively. The Python `adaptive` and
`mode` arguments are replaced by this default and `pin_weights`; `BY_MODE`
and `MODES` are removed.

Explicit `weights=` still builds a fixed custom experiment, with a separately
specified `expansion_value` (zero if omitted). Such a vector cannot be combined
with a pin. The standard factory always uses the separate trading evaluator;
custom `trade_weights`, `trade_expansion_value` and `trade_floor` remain available
for controlled research and replay. Fitting/ablation tools now construct
endpoint controls explicitly, including their expansion credit.

The old bare `heximax` meant the legacy fixed trading profile and .0197 floor.
It now means the validated adaptive policy. Historical readouts keep their
original labels and source identities; use their recorded revisions to rerun
frozen scripts. This API consolidation does not turn their old match results
into measurements of today's default or the current Catanatron dependency.

The [curve confirmation](readouts/slider-curve/README.md) and
[shape screen](readouts/slider-shape/README.md) describe the adopted policy.
