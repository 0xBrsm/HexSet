# Port feature scale audit

This is a read-only strength review. No source or strength run was changed.

The exact current formula is in `src/hexset/bots/evaluate.py:263-355`.
`Evaluator.__init__` precomputes per-vertex `(hex, resource, pips)` tuples at
lines 237-244 and indexed port `(resource, ratio)` tuples at lines 249-256.
`survey` walks owned vertices, accumulates `total += built * p` (lines 293-307),
updates the cheapest generic and resource-specific ratios (308-313), then
returns `rate=total / ROLLS` and

```python
ratios = tuple(generic if generic < best else best for best in specific)
port_gain = sum(BASE_TRADE_RATIO - ratio for ratio in ratios)
```

at lines 340-349. `BASE_TRADE_RATIO=4`, `GENERIC_RATIO=3`, and
`SPECIFIC_RATIO=2` are fixed in `src/hexset/board/ports.py:10-12`. The economy
API confirms the same per-resource cheapest-rate semantics at
`src/hexset/economy.py:53-68`.

Therefore a single owned specific 2:1 port gives `port_gain=2`, regardless of
whether the seat produces that resource; a single generic 3:1 gives
`port_gain=5` for every resource. The feature is an access count, not liquidity.

## Numeric scale

The shipped no-trade port coefficient is `0.03063` in
`src/hexset/bots/heximax/evaluate.py:108-128`. The selected p06 vector has
`port=0.019305197991678444`, `production=3.8803981040675644`, and
`diversity=0.360074083309872` in
`/data/data/com.termux/files/usr/tmp/luna-p06-1000/*.json` and the immutable
selection manifests.

| reachable port state | `port_gain` | shipped port | shipped ×2 | shipped ×4 | p06 port | p06 ×2 | p06 ×4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| one specific 2:1 | 2 | 0.061260 | 0.122520 | 0.245040 | 0.038610 | 0.077221 | 0.154441 |
| one generic 3:1 | 5 | 0.153150 | 0.306300 | 0.612600 | 0.096526 | 0.193052 | 0.386104 |
| five settlements, observed gain range | 6–10 | 0.183780–0.306300 | 0.367560–0.612600 | 0.735120–1.225200 | 0.115831–0.193052 | 0.231661–0.386104 | 0.463325–0.772208 |

For a concrete reachability scale check, I generated 300 seeded public boards,
placed one through five legal settlements on port vertices, and measured the
existing survey. This was local feature arithmetic only, not game play. The
mean weighted terms were:

| owned settlements | port at shipped `0.03063` | port at p06 `0.019305` | production at shipped `2.785` | production at p06 `3.8804` | diversity at shipped `.358` | diversity at p06 `.3601` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1051 | 0.0662 | 0.3396 | 0.4732 | 0.4391 | 0.4417 |
| 2 | 0.1617 | 0.1019 | 0.6645 | 0.9259 | 0.7828 | 0.7874 |
| 3 | 0.1973 | 0.1243 | 0.9706 | 1.3524 | 1.0442 | 1.0502 |
| 4 | 0.2200 | 0.1387 | 1.2989 | 1.8098 | 1.2673 | 1.2747 |
| 5 | 0.2380 | 0.1500 | 1.6163 | 2.2521 | 1.4272 | 1.4355 |

The count term is therefore smaller than production/diversity on typical
single-opening states, but it can be a material discrete preference when two
candidate openings differ by a port. The current port feature also rewards a
zero-income port exactly as much as an income-producing port.

## What has already been tested

The historical one-term sweep is precisely `src/hexset/bench/weight_sweep.py`.
It used `games=1024`, `workers=30`, factor rings `0, .5, 2` and `.71, 1.41`.
`run_cell` passes `[challenger, challenger, baseline, baseline]` to
`hexset.arena.compete`; these are paired self-play comparisons, not AB2 gates.
The no-trade port step (`git show acd0575:docs/readouts/heximax-fit/sweep-notrade.json`)
used seed 94000 and values `0`, `.015315`, `.06126`, then seed 102000 and
`.0217473`, `.0431883`; wins were respectively `516`, `517`, `527`, then
`508`, `491`. None was accepted. The trading sweep tested `0`, `.0036865`,
`.014746`, then `.00523483`, `.01039593`, with wins `522`, `527`, `528`,
then `518`, `516`; none was accepted.

The opening-placement prior is a separate fixed-pips model. Its docstring in
`src/hexset/placement.py:2-15` says port access was offered but null at fixed
pips; `score` at lines 43-58 uses only total pips, resource diversity, and
scarcity. That result does not test the later `Survey.port_gain` feature during
post-opening search.

The archived `results/08-port-aware.json` is a different experimental feature
(conversion-aware purchase progress), reporting 41/120 versus its recorded AB2
lineup and 16/120 versus shipped. `docs/readouts/luna-search/port-aware-correctness-20260910.md`
shows that artifact has no source fingerprint and cannot certify the corrected
implementation. It should not be used as evidence for a port-weight scale.
The `drop-port` ablation (`docs/readouts/luna-ablation/results/results/07-drop-port.json`)
reports 53/120 vs AB2 and 33/120 vs shipped, but is exploratory and does not
isolate a production-weighted port feature.

The p06 1,000-game artifacts are identity-complete for their candidate and
lineups: p06 vs AB2 is `447/1000`, with
`DC:heximax-evolve-g3000-p06,AB:2,AB:2,AB:2`; p06 vs shipped is `269/1000`,
with `DC:heximax-evolve-g3000-p06,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade`.
They are candidate outcome evidence, not an isolated port ablation.

## Justified next feature experiment

A better state feature should measure usable conversion of the cards the seat
actually produces, while retaining the existing `ratios` and maritime rules:

```python
liquidity = sum(
    rate_r * (1.0 / ratio_r - 1.0 / BASE_TRADE_RATIO)
    for rate_r, ratio_r in zip(resource_rates, ratios)
)
```

Here `resource_rates[r]` is the existing survey's per-resource expected cards
per roll, computed from the same `self.yields` loop and divided by `ROLLS`.
A 2:1 port then contributes `0.25 * rate_r`; a generic 3:1 contributes
`sum(rate_r) / 12`; an off-resource specific port contributes zero. The
candidate can replace the flat `port_gain` feature in an isolated evaluator,
leaving all other terms and placement policy unchanged.

At a typical total production rate around `0.66`, coefficient `2.785` makes a
generic-port liquidity contribution `2.785 * .66 / 12 = .153175`, matching the
shipped flat generic-port contribution `5 * .03063 = .15315`. This is a scale
anchor, not a fitted result. A preregistered local ring should compare
`c = 1.3925`, `2.785`, and `5.57` against the unchanged flat-port baseline,
with production/scarcity, diversity, setup prior, depth/width, and opponent
lineups held fixed. The same candidate should be tested on states containing
specific ports, generic ports, and no port, because the key hypothesis is
resource alignment rather than port count.

The test must be a fresh-source, paired strength experiment after an exact
scalar/vectorized score-equivalence check. It should report per-game port
occupancy, `resource_rates`, `ratios`, liquidity, production, diversity, and
opening placement separately. The historical placement null and the old port
sweep cannot answer this interaction. A port-aware opening prior should remain
a separate arm: the placement prior currently ignores ports by design, so it
must not be folded into the post-opening liquidity result.
