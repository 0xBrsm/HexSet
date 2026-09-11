# Heximax under varying trading conditions

## Current policy

The unified policy is now `heximax-adaptive`: directly slide between the current
no-trade profile N and the best validated trading policy M/T. See
[SLIDER.md](SLIDER.md). Its finite activity window reaches both endpoints; the
old adaptive screen used N to legacy T with a persistent prior, so its reported
win rate is not a measurement of this revised slider. The original study and
its preregistered decision are retained below as historical evidence.

## Original decision and fresh evidence

The original study recommended the selectable **`heximax-balanced`** preset for native floor-zero games
where trading is permitted but opponent participation is uncertain. It retained
`heximax-notrade` for explicitly non-trading play. Do not infer that trading
is unavailable from a short quiet stretch. Its public-activity adaptation remained
experimental: it tracked conditions, but did not improve on the fixed midpoint.

| Policy | Wins in six primary conditions | Win rate |
| --- | ---: | ---: |
| Fixed midpoint M/T | 523 / 1,536 | 34.05% |
| Public-activity adaptive A/T | 520 / 1,536 | 33.85% |
| Trading baseline T/T | 396 / 1,536 | 25.78% |

The prespecified M minus T gain is **8.27 percentage points**, adjusted interval
**4.93 to 11.61**. A minus M is **-0.20 points**, interval **-2.87 to +2.48**.
These are the two registered 97.5% intervals, together controlling family-wise
error at 5%. The adaptive result is inconclusive, not proof of equivalence.
Following the frozen rule, select M; do not spend another sweep trying to make
adaptation win. This is a supported compromise, not an optimal frontier claim.

| Condition | M wins | A wins | T wins | Games per policy |
| --- | ---: | ---: | ---: | ---: |
| Quiet | 97 | 101 | 71 | 256 |
| Moderate | 103 | 95 | 68 | 256 |
| Active | 90 | 103 | 67 | 256 |
| Mixed | 63 | 65 | 63 | 256 |
| Rising | 81 | 72 | 73 | 256 |
| Falling | 89 | 84 | 54 | 256 |
| None | 34 | 34 | 34 | 128 |
| Full | 39 | 43 | 30 | 128 |

The aggregate benefit is not universal: M tied T in mixed opponents and had a
small descriptive advantage in rising activity. At the non-trading boundary,
N won 43/128 versus M's 34/128; at full participation, N won 33/128 versus M's
39/128. Boundary samples are descriptive and do not prove a condition-specific
switching rule. Existing no-trade support also has its separate native holdout.

All 5,632 games completed in 726.65 seconds (7.75 games/second with 30 workers).
The two screens plus fresh validation consumed 9,664 strength games and about
20.7 minutes of pool wall time. Replays are separately identified verification.
The independent audit reproduces both primary intervals and every logged
activity value from public events across all 5,632 validation games.

The productive next adaptive hypothesis is to estimate the exchange value of
specific surplus resources and unmet needs, then test that signal against
this stronger fixed baseline and a stage-only control. A table-wide trade
frequency is not automatically the correct interpolation coefficient for every
feature. The present data do not identify an optimal mapping or establish
performance under human negotiation or other opponents' exchange evaluators.

## What the policies mean

Move valuation and exchange valuation are separate decisions. A policy written
M/T uses midpoint move weights and the trading profile to price exchanges.
All candidates and opponents permit trades with floor zero. Each completed
exchange must still improve both participants' own evaluations. This is the
native engine's automatic exchange mechanism; bank and port exchanges remain
ordinary actions. Catanatron is an external reference and participates in none
of these experiments, as host or player.

N is the new no-trade move profile, including bounded useful expansion credit
of .25 VP. T is the existing trading profile, with expansion credit zero.
M is the arithmetic midpoint of every N/T move coefficient, including expansion
credit .125. Its exchange evaluator is T, without expansion credit. M is a
candidate compromise, not a claim of optimal coefficients or universal play.

| Feature | Balanced move weight | Exchange weight |
| --- | ---: | ---: |
| buy_progress | 0.45 | 0.45 |
| diversity | 0.358 | 0.358 |
| knight | 0.10335 | 0.1041 |
| port | 0.0190015 | 0.007373 |
| production | 4.926 | 7.067 |
| road | 0.06045 | 0.1209 |
| robber_risk | -0.3 | -0.3 |
| scarce | 0.124499305556 | 0.1786 |
| spare_card | 0.12825 | 0.15 |
| victory_point | 1 | 1 |
| useful expansion cap | .125 | 0 |

A interpolates N to T with alpha equal to a public exchange-activity estimate.
A Beta(1,3) prior shrinks sparse history; excess counts decay with an eight
eligible-turn half-life. An eligible turn requires public hand size at least
two for the actor and some partner. Each turn contributes one success if an
exchange involving the actor occurred, otherwise one failure. Turns without
this coarse opportunity contribute neither. Several deals on one turn still
count once. The weights change at the next move decision and remain fixed
through that search; exchange valuation remains T.

This estimates realized exchange availability, not hidden willingness. No-trade
observations can reflect incompatible hands, already-satisfied needs, or
strategic refusal. The focal bot itself affects the history. The observer reads
only turn, actor, public hand sizes and completed-trade participants, never
opponents' private hands, gains, participation probabilities or scenario names.

## Conditions and inference

The two screens used independent seeds and seven regimes. The initial fixed
screen had 1,344 games; the adaptive screen had 2,688. Their strongest fixed
and adaptive candidates were selected before fresh validation. The adaptive
screen's five-interior mixture gave M 130/320, A 117/320 and T 88/320. Those are
selection data, not the confirmation result.

Fresh validation has 5,632 games. Each of six primary conditions has 256
independent boards for each of M/A/T. Conditions receive equal weight:

| Condition | Opponent participation probabilities | Opponent move profiles |
| --- | --- | --- |
| Quiet | .10, .10, .10 | T, T, T |
| Moderate | .30, .30, .30 | T, T, T |
| Active | .65, .65, .65 | T, T, T |
| Mixed | .05, .45, .85 | N, M, T |
| Rising | .10 to .75 at global turn 40 | N, T, M |
| Falling | .75 to .10 at global turn 40 | M, N, T |

Participation controls use private, reproducible per-turn draws separately
for acting and responding roles, then the ordinary positive-gain gate. They
model heterogeneous participation, not a realistic population of human offer
strategies. Every opponent's exchange evaluator is T. Boundary checks add 128
games each for M/A/T/N against never-participating old no-trade move opponents
and fully participating trading-profile opponents. Boundaries do not select
the primary result. All candidates use the identical native engine, search budget and information
model; only their policies differ. All candidate seats balance exactly; boards are independent,
not antithetic. Comparisons are paired on board, seat and random-stream seeds.

Two prespecified contrasts, M minus T and A minus M, use two-sided 97.5%
stratified normal intervals, controlling family-wise error at 5% by Bonferroni.
The six regime-specific paired means receive equal weights. No optional
stopping, validation-driven reselection, or treating repeated traces as new
samples. Individual-condition intervals are descriptive, not simultaneous.

If an adaptive policy beats the best fixed alternative, the registered plan
requires a fresh stage-only control before adoption: trade activity could be
acting as a proxy for time in the game. An adaptive rule's responsiveness by
itself is not evidence that adapting improves strength.

## Implementation and verification

Select through the arena registry:

```python
import random
from hexset.arena import PRESETS, spawn
import hexset.bots.heximax  # registers presets

bot = spawn(PRESETS["heximax-balanced"], board, random.Random(1))
```

Its move weights are frozen as `BALANCED_WEIGHTS`, expansion credit is .125,
and exchange weights are `TRADING_WEIGHTS` with zero expansion credit and
floor zero. Search remains depth 2, width 6, 600 leaves, one determinization and
the existing win temperature. The historical `heximax` preset remains a
reference with its existing .0197 floor; the validation baseline explicitly
used floor zero on every seat. Comparing the two presets without matching
floors does not reproduce this study.

`heximax(..., weights=move_weights, expansion_value=move_expansion,
trade_weights=exchange_weights, trade_expansion_value=0, trade_floor=0)`
uses distinct evaluators with the same stance and temperature. Search and
exchange pricing have independent caches; the dedicated exchange cache lasts
one valuation batch. Exchange pricing does not consume the move RNG. Omitting
`trade_weights` preserves the shared evaluator and the existing defaults.

The selectable preset and public API exactly reproduced **40 frozen full-game
traces**, covering all eight validation conditions and all four candidate seats
for M, plus unchanged T controls. Actions, winners, points, every completed
trade and every public activity observation match. The integration source is
commit 9b9d091. The first replay attempt stopped on instrumentation checking the
internal exchange helper's floor metadata; forwarding the configured floor
fixed that assertion. No strength data were changed or rerun.

The production API is tested against a standalone exchange evaluator across
state changes, win and relative stances, custom temperature, move/RNG parity,
and the trade-off switch. The default suite passed 860 tests (14 skipped,
one deselected). The observer's five focused tests pass. Every logged adaptive
screen activity value was reproduced exactly from its public events, and the
public event participants match the whole-game trade census in all 2,688 games.

[REPORTING-CORRECTION.md](REPORTING-CORRECTION.md) explains the runner's inverted
candidate-seat metadata. Derived counts use `Outcome.seating[0]`; original
archives remain immutable. This correction affects candidate participation
counts only. It changes no actions, winners, total trades or primary inference.

## Reproduction and artifacts

Validation was launched from commit 54ac458, source SHA256
`0254eb3dbe0d14bea1b2d9eb42ec18a553d38c2f9100dd3477edb62347778c3e`.
Runtime: Python 3.12.14, NumPy 2.5.2, Docker image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`.
One 30-worker pool, 30 CPUs, network disabled; BLAS/OpenMP threads capped at one.
Fixed seeds begin 720000000; adaptive 721000000; validation 722000000, with
10,000 separating regimes. Driver files and each game identity specify the
complete search configuration, profiles, seeds, rules and script hashes.

Run `validate.py --out /out/validation` from that frozen source export with
`PYTHONPATH=/study/src`, `PYTHONHASHSEED=0`, and the stated runtime. The source
fingerprint assertion deliberately rejects a different production engine.
Changing production source does not retroactively change the tested policy.
`verify_integration.py` replays selected validation traces through the new API;
those are behavior checks, not additional strength observations.

The fixed and adaptive screens have raw archives, SHA256 files, summaries,
and a separate adaptive analysis here. Fresh validation's archived summary
preserves the runner's original report; `audit_validation.py` provides the
corrected, independently checked analysis. Archives include all live trade
events, public observations, action digests and per-game provenance.
