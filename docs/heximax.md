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

## Why exchange weights stay fixed

A native 400-game 2v2 comparison kept adaptive move search identical on both
sides and varied only the exchange coefficients. The then-current fixed
exchange weights (robber risk −0.30) won **265/400 games (66.25%, 95% interval 61.5–70.7%)** against using
the adaptive move coefficients for exchanges. All 400 games passed replay and
public-activity audits, with 23,382 completed exchanges. This supports the
split for that matchup; it does not establish optimal weights across all
opponents. See the [exchange-weight readout](readouts/adaptive-exchange-2v2/README.md).

The subsequent [bounded exchange-only sweep](readouts/exchange-weight-sweep/README.md)
adopted a smaller exchange robber-risk penalty: **−0.225 instead of −0.30**.
With move search held fixed, it won **600/800 fresh 2v2 games (75.0%, 95%
interval 71.9–77.9%)** against the previous exchange vector. All other exchange
coefficients and both move profiles are unchanged. The full sweep and adoption
checks stopped at 1,600 games and 4 minutes 1 second of evaluation time.

The [sign/range follow-up](readouts/exchange-robber-ceiling/README.md) now adopts
**+0.075 for exchange robber risk**, with both move penalties still −0.30.
It won **608/800 fresh games (76.0%, 95% interval 72.9–78.8%)** against −0.225.
A direct +0.075-versus-+0.15 runoff was inconclusive, so this is a supported
candidate rather than a precisely identified optimum. Exchange records show
large net card transfers from the negative-weight incumbent to the positive
candidate; this motivates further behavioral analysis without claiming that
actual robber exposure is beneficial. The bounded follow-up is complete.

## Current reference results

With its default unpinned adaptive configuration, Heximax won **590/800
1v1 games (73.75%)** and **450/800 four-player games (56.25%)** against one
and three Catanatron AB2 references respectively, all hosted in HexSet.
Trading was enabled; AB2 declined exchanges and the slider remained at zero.
The [benchmark readout](readouts/current-heximax-ab2/README.md) records exact
revisions, confidence intervals, settings and replay validation of all 1,600 games.

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
