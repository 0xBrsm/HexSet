# heximax fit: weights and temperature by conditional logit

Registration: dev-HexNet `agents/reference/heximax.md`, "Registration
2026-09-07: fitting heximax by conditional logit". Everything below was
pre-stated there except the numbers.

## Why the climb went

Two overnight hill climbs (`hexset.tuning`, since deleted) spent 80,000 games
and five box-hours on this branch's nine free weights. Climb 1 confirmed null
(51.8% [50.0, 53.5] over 3,072 games); climb 2 confirmed 54.1% [52.3, 55.8]
but moved `production` 7.07 → 3.73, `port` 0.007 → 0.35 and flipped
`robber_risk` positive, and the alternating temperature refit oscillated
3.35 → 2.57 → 2.92. The planted control of 2026-08-09 had already shown the
climb cannot resolve a three-point error. Nothing from those climbs was
adopted.

## The estimator

The bot reads a position as `softmax(v / T)[seat]`, `v_i = w · f_i`,
`victory_point` pinned at 1. Label every recorded position with the seat that
eventually won and that reading is a conditional logit over the four seats
with one shared coefficient vector `β = w / T`. The log-likelihood is concave,
so it is one Newton solve, and `T = 1 / β_vp`, `w = β / β_vp` come out of the
same solve. `scarce` stays derived (`0.91 · production / ROLLS`), folded into
the production column and re-derived. Every position of a game shares one
label, so the intervals are cluster-robust by game and block-bootstrapped by
game. Rows are read honestly -- the knower's row exact, every other row
through the ledger -- which is how the deployed evaluator reads them.

Two evaluator changes landed first (`77da4cc`): an opponent's development
cards are worth their expected victory points (held count × VP share of the
unseen pool), so the anchor term means the same thing in every row; and a
candidate carries its own temperature, so fitted and incumbent can sit at
one table.

## Data

5,000 `heximax`×4 self-play games at the branch's shipped pair
(`TRADING_WEIGHTS`, hand terms 0.45 / 0.15 / −0.15, `T = 2.4766`);
`hexset.bench.generate --bot heximax --games 5000 --seed 300000`,
`PYTHONHASHSEED=0`, 5,000/5,000 decided, 87.1 turns and 349 actions a game,
700 s on 30 workers. Stride 8, main-phase positions, one choice set per
knower per position: **411,700 choice sets**, split 4,000 / 1,000 games. The
records (25 MB) are not in git; `fit-trading-a.json` carries every number.

## The fit (`fit-trading-a.json`)

Held-out log loss against the eventual winner, 1,000 unseen games:

| reading | test loss |
|---|---|
| uniform | 1.3863 |
| shipped pair (`TRADING_WEIGHTS`, T = 2.4766) | 1.2223 |
| **`standard` (all terms)** | **1.1699** |
| `no_risk` (`robber_risk` pinned at 0) | 1.1701 |
| `leader_gap` | 1.1699 (unidentified, see below) |
| `remaining` (production × distance to 10) | 1.1699 |
| `both` | 1.1699 |

**Fit bar met**: the played variant predicts winners better than what ships.
Played variant by the pre-stated rule: `standard` (`no_risk` does not beat it).

The fitted pair, cluster-robust SE, and the 40-replicate block bootstrap
(`fit-trading-a-bootstrap.json`):

| term | shipped | fitted | ± SE | bootstrap 95% |
|---|---|---|---|---|
| T | 2.4766 | **1.7485** | 0.040 | [1.69, 1.82] |
| production | 7.067 | **2.174** | 0.14 | [1.85, 2.40] |
| diversity | 0.358 | 0.428 | 0.041 | [0.35, 0.50] |
| scarce (derived) | 0.1786 | 0.0550 | — | [0.047, 0.061] |
| buy_progress | 0.45 | **1.654** | 0.059 | [1.54, 1.76] |
| road | 0.1209 | **−0.105** | 0.015 | [−0.14, −0.07] |
| knight | 0.1041 | 0.148 | 0.026 | [0.10, 0.18] |
| spare_card | 0.15 | 0.092 | 0.026 | [0.05, 0.13] |
| robber_risk | −0.15 | **+0.252** | 0.061 | [0.13, 0.32] |
| port | 0.0074 | 0.077 | 0.015 | [0.06, 0.11] |

Reading the coefficients, with the caveat that a likelihood over positions
ranks positions across games and the search only compares siblings:

- **Production falls to a third, progress rises almost fourfold.** The
  shipped 7.07 was a one-ply trading-era fit; at the positions a depth-2
  bot faces, what predicts the winner is a hand that buys something now, not
  a rate. `buy_progress` at 1.65 says a hand covering a settlement is worth
  about a point and a half more than an empty one.
- **Roads are negative.** −0.10 a road, five SEs from zero: the seat with
  more roads is behind, other things equal. This is the road-spam finding
  from the other end again.
- **`robber_risk` comes out positive, +0.25 ± 0.06**, and dropping it costs
  nothing in held-out loss (1.1701 vs 1.1699). This is the collinearity the
  review named: the term is a scaled copy of bank value over seven cards,
  and a seat holding a big hand is a seat that is winning. As a leaf term in
  a search whose tree already expands the next roll's 7, a positive weight
  tells the bot to sit on big hands. The pre-stated rule plays `standard`
  anyway; the duel decides, and the sign is on record.
- **`leader_gap` is unidentifiable by construction.** The leader's points are
  constant across the seats of one position, so the gap differs from the
  anchor only by a per-position constant, which a softmax cannot see (SE
  1,607 on T). The IIA fix the review wanted needs a nonlinear leader
  attribute (an indicator, or a gap that is zero for the leader), not this.
- **`remaining_production` is real but does not help.** −0.106 ± 0.04: the
  same production is worth more the closer a seat is to 10 (`v = prod ·
  (1.78 + 0.106 · VP)`), the review's intuition confirmed in sign -- but it
  lowers held-out loss by nothing at the fourth decimal, so it does not earn
  an evaluator term on this data.

**Attenuation, honest vs omniscient rows** (`fit-trading-a-omniscient.json`):
read with every row exact, the same design scores 1.1458 (incumbent 1.1994)
and fits `production` 2.28, `buy_progress` 1.62, `spare_card` 0.096,
`robber_risk` 0.24, T 1.70 -- the hand coefficients move by well under one
SE. The honest reading costs prediction (it knows less) but does not bend
the vector.

## The duel (`duel-trading-a.json`) -- the bar is not met

`hexset.bench.fit_duel`, fitted `standard` pair vs the shipped heximax,
3,072 games, blocked, duel seed 42000, 30 workers, 423 s:

| | |
|---|---|
| fitted wins | **655 / 3,072 = 21.3% [19.9%, 22.8%]** |
| paired VP | **−1.363 [−1.433, −1.294]** |
| unfinished | 0 |

Pre-stated bar: Wilson lower bound above 50%. **Not adopted.** The vector
that predicts the winner best on held-out games loses three games in four
to the vector it out-predicts.

### Reading

This is the mismatch the fitting module's own docstring warns about, and it
is not a small one. A likelihood over positions learns *which positions
belong to winners*; the search asks *which of these sibling positions, one
move apart, should I move to*. Those are different questions, and the fitted
coefficients answer the first in a way that wrecks the second:

- A hand that covers a settlement is worth 1.65 in the fit because seats
  holding such hands go on to win. But the search prices *building* it as
  `+1 VP + 2.17 × Δrate − 1.65`, roughly zero: the fit has already credited
  the hand with the settlement, so realising it earns nothing and holding it
  costs nothing. Under the shipped pair the same build is worth about +2.3.
  A depth-2 search cannot see the compounding a held card forgoes, so the
  evaluator has to be paid to act, and a calibrated state value is not.
- Roads at −0.10 make the bot refuse the roads it needs to reach a spot, for
  the same reason: seats that build many roads lose, so roads predict
  losing, and the fit cannot tell a road that opens a spot from a road that
  is spam.
- `robber_risk` at +0.25 tells it to sit on a big hand.

Each of these is a *correlational* coefficient standing in for a *causal*
one, and the search consumes them as causal. The honest conclusion is that
an outcome likelihood over positions is the right instrument for `T` and for
the calibration of state terms (`victory_point`, production, diversity,
port, knight), and the wrong instrument for the instrumental terms (hand
progress, roads, risk) at depth 2. That is exactly where the shipped pair
came from: its one-ply climb was a play test, not a prediction.

What the fit does establish, and which stands: the shipped pair is
mis-calibrated as a *probability* (1.222 vs 1.170 held-out; T = 2.48 vs
1.75 with everything else free), the anchor-term fix is right, and roads,
risk and hand progress behave as instruments rather than as state.

## The no-trade profile (`fit-notrade-a.json`, `duel-notrade-a.json`) -- the same null, larger

5,000 `heximax-notrade`×4 games, seed 310000, 106 turns and 403 actions a
game, 1,725 s; **530,368 choice sets**. Incumbent `NO_TRADE_WEIGHTS` at
T = 2.4766.

| reading | test loss |
|---|---|
| uniform | 1.3863 |
| shipped no-trade pair | 1.1697 |
| **`standard`** | **1.1330** |
| `no_risk` | 1.1338 |

Fitted: T 2.00 ± 0.05, production 3.72 ± 0.19 (shipped 2.785), diversity
0.29, buy_progress **2.34** ± 0.08, road **−0.125** ± 0.016, knight 0.075,
spare_card **−0.23** ± 0.03, robber_risk **+0.72** ± 0.07, port 0.108. The
fit bar is met again, and the sign pattern is the trading fit's, sharper:
spare cards negative, big hands positive, roads negative.

Duel, fitted vs `heximax-notrade`, 3,072 blocked, seed 42000: **526 / 3,072
= 17.1% [15.8%, 18.5%]**, paired VP **−1.904 [−1.972, −1.837]**. Not
adopted. With no trade gate to soften it, the "do not act" reading of the
coefficients bites harder: the bot that best predicts the winner from a
position loses five games in six.

## Referents at the branch tip (`referent-*.json`)

800 games each, blocked, `PYTHONHASHSEED=0`, the gate driver's seeds. The
"branch base" column is the same duel on the same seeds at `dc26938`, the
commit before the anchor-term change, so the two columns differ only by it.

| duel | branch tip | branch base (`dc26938`) | previous baseline (HexSet main, 2026-09-04) |
|---|---|---|---|
| heximax vs search2, trading, seed 42000 | 441/800 = 55.1% [51.7, 58.5], VP −0.19 | 441/800 = 55.1%, VP −0.17 | 66.4% [63.0, 69.6], VP +0.97 |
| heximax-notrade vs search2-notrade, seed 59300 | 551/800 = 68.9% [65.6, 72.0], VP +0.93 | — | 62.6% [59.2, 65.9], VP +0.61 |
| heximax-omni vs heximax, seed 42000 | 364/800 = 45.5% [42.1, 49.0], VP −0.20 | 366/800 = 45.8%, VP −0.18 | 49.6% [46.2, 53.1] |

Two readings. **The anchor-term change is play-neutral at this resolution**:
441 wins either side of it on identical boards, 364 against 366 for the
omniscient read. It is correct and it costs nothing; it ships.

**The eleven-point drop against `search2` on the trading table predates it
and belongs to the hand-terms redesign this branch inherited.** Both bots
share `evaluate.Weights`, so the redesign that closed heximax's exploitable
gate closed search2's too; the redesign's own gates played the new heximax
against a *shipped-hand* search2 (384/384) and never re-read this referent.
Whether heximax itself got stronger or weaker in absolute terms this pair
cannot say -- the no-trade referent rose six points -- and the fixed
opponent that can (Catanatron, previous 32.5%) is owed after the bridge
image is rebuilt on this engine. Recorded, not adjudicated here.

## The play sweep (`sweep-trading.json`, `sweep-notrade.json`) -- what actually moved

Registration: "tune the weights by play, then new-vs-old and Catanatron"
(dev-HexNet `heximax.md`, 2026-09-07 evening). `hexset.bench.weight_sweep`:
one free term at a time, rung at ×0, ×0.5, ×2 of the incumbent and then
×0.71, ×1.41, each cell 1,024 paired games against the incumbent on
identical boards (grouped `[c, c, b, b]`, antithetic swaps), a move only
when the best cell's Wilson lower bound clears 50%. `scarce` moves with
production. 96 minutes for the trading profile and 128 for the no-trade one (the latter sharing the box) on 30 workers.

**Trading profile** (start `TRADING_WEIGHTS`, seed 98000): sixteen rings,
one move.

| term | shipped | ×0 | ×0.5 | ×2 | verdict |
|---|---|---|---|---|---|
| production | 7.067 | 44.8% | 50.7% | 49.8% | holds |
| buy_progress | 0.45 | **23.6%** | 40.8% | 46.4% | holds (a peak) |
| robber_risk | −0.15 | 36.8% | 43.8% | **54.2% [51.1, 57.2]** | **→ −0.30** |
| spare_card | 0.15 | 46.7% | 49.0% | **21.3%** | holds |
| road | 0.1209 | 45.0% | 50.1% | 40.4% | holds |
| diversity | 0.358 | 47.9% | 49.5% | 48.0% | holds |
| port | 0.0074 | 51.0% | 51.5% | 51.6% | holds (inert) |
| knight | 0.1041 | 49.0% | 49.6% | 48.2% | holds (inert) |

Second ring (×0.71, ×1.41): nothing moved; robber_risk is flat from −0.21 to
−0.42. **Confirm, swept vs start, 3,072 fresh boards (seed 598000): 1,613 =
52.5% [50.7%, 54.3%]. Adopted** into `TRADING_WEIGHTS`; `search2` keeps the
bare default as the frozen referent.

**No-trade profile** (start `NO_TRADE_WEIGHTS`, seed 88000): two moves.
robber_risk −0.15 → −0.30 (53.2% [50.2, 56.3]; zero 37.8%), and on the
second ring spare_card 0.15 → 0.1065 (53.1% [50.1, 56.2]; ×1.41 reads
39.6%). The six other terms held. **Confirm: 1,625 / 3,072 = 52.9% [51.1%,
54.7%]. Adopted** into `NO_TRADE_WEIGHTS`.

Reading. The play sweep and the likelihood fit disagree on exactly the
terms the duel said they would: play wants *more* robber fear (the fit said
+0.25, play says −0.30, and zeroing the term costs 13 points either table),
and play keeps `buy_progress` at 0.45 where the fit wanted 1.65 (doubling
it to 0.9 already loses 3.6 points). Four terms -- production, diversity,
port, knight -- are inert within a point and a half of 50% at half and
double their value: at depth 2 the search barely leans on them, which is the
review's collinearity point seen from the play side. "Optimal" here means
settled at the resolution 1,024 paired games buy, about a point and a half.

## New vs old heximax (`new-vs-old-heximax.json`)

New = this branch at the adopted trading weights; old = the pre-redesign
bot frozen in `hexset.bench.shipped_hand` (old three hand terms, old
weights). 1,536 games, grouped, seed 99000.

| pairing | result |
|---|---|
| trading off both sides | **965 / 1,536 = 62.8% [60.4%, 65.2%]**, roads 8.83 vs 10.06 a game |
| trading on both sides | **cannot be played**: the engine's trade-event invariant fires ("a trade event revisited a position: a gate is not strictly increasing in the acting seat's own value") |

The trading-on failure is the old gate's hole expressed as an engine
assertion rather than a score: the pre-redesign gate priced a card at
0.005 VP and being over seven at −0.39, so at a table with the new bot the
clearing cycles. The 384/384 the hand-valuation readout recorded for this
pairing predates the trade-event revisit check (HexSet #61). The
position-ranking read stands at 62.8%, with power, and the old bot's road
spam (10.1 a game) is gone from the new one (8.8).

## New vs Catanatron (`heximax-omni-s*.txt`, `heximax-notrade-s*.txt`)

Three `AB:2`, 500 games a seed, seeds 7 and 8, `PYTHONHASHSEED=0`,
catanatron `3.3.0 @ d3f4ad05bb78`, run in `hexset:local` against this
branch's `src`. The bridge forces trading off, so `heximax-notrade` plays
`NO_TRADE_WEIGHTS` and `heximax-omni` plays `TRADING_WEIGHTS` with every
hand visible.

| arm | seed 7 | seed 8 | pooled n = 1,000 | previous (2026-09-04) |
|---|---|---|---|---|
| `heximax-omni` | 220/500 = 44.0% | 214/500 = 42.8% | **434/1,000 = 43.4% [40.4%, 46.5%]** | 35.00% [32.11, 38.01] |
| `heximax-notrade` | 238/500 = 47.6% | 235/500 = 47.0% | **473/1,000 = 47.3% [44.2%, 50.4%]** | 32.50% [29.67, 35.47] |

Fair-share bar (lower bound > 25%) cleared by both arms with room, and both
are well above the previous intervals: **+14.8 points for the honest
no-trade bot, +8.4 for the omniscient one**, with the two seeds agreeing to
within a point in each arm. Against a fixed opponent that shares none of our
weights, the branch's heximax is the strongest handcrafted bot this project
has fielded: nearly half the games at a four-seat table against three
`AB:2`, where a fair share is a quarter. Honest now beats omniscient here
(47.3% vs 43.4%) -- the omniscient arm plays `TRADING_WEIGHTS`, fitted for a
table with trading, on a bridge that forces trading off, so it is the
no-trade profile's sweep that the honest arm is showing.

## What ships

`TRADING_WEIGHTS`: the shipped vector with `robber_risk = -0.30`.
`NO_TRADE_WEIGHTS`: `robber_risk = -0.30`, `spare_card = 0.1065`. The
anchor-term change (`expected_card_points`). The temperature is unchanged at
2.4766: not load-bearing for play, mis-calibrated as a probability by about a
fifth (the likelihood puts it at 1.75 free, 3.04 with the weights held), and
that belongs to whatever consumes probabilities as magnitudes.

## Amendment: the temperature alone (`fit-trading-a-calibrated.json`, `duel-trading-a-calibrated.json`)

Stated after the duel above and before it ran: hold the shipped weights and
fit `T` alone on the same choice sets -- the part of the likelihood that is
sound for a search evaluator, since it rescales every sibling alike. One
candidate, one duel, same bar.

| | |
|---|---|
| T with the shipped weights held | **3.0414** (shipped 2.4766) |
| held-out loss, calibrated / shipped | 1.2123 / 1.2223 |
| duel vs incumbent, 3,072 blocked, seed 42000 | **1,535 / 3,072 = 50.0% [48.2%, 51.7%]**, paired VP +0.009 [−0.075, +0.092] |

**Not adopted**: an exact tie. At depth 2 the bot's decisions do not turn on
`T` anywhere in the range the likelihood can distinguish -- which is also
why the three alternating refits (3.35, 2.57, 2.92) never mattered. `T`
matters where win *probabilities* are consumed as magnitudes: the trade
gate's floor and the acceptance model. That is where a calibrated `T`
belongs, and the shipped 2.4766 is too sharp by about a fifth against these
records.
