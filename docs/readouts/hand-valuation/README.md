# Hand valuation: a card is worth what it can buy

`heximax` and `search2` share `hexset.bots.evaluate.Weights`. Its three hand
terms were `card = 0.005406` per card held, `surplus_card = -0.3891` per card
over seven and `progress = 0.01843`. At the g4 table heximax gave three wood
and three ore for one wheat, then two wheat for one wood: both cleared its
own private gate, because dumping six cards to get under seven was worth
about two victory points to it and the cards themselves were worth almost
nothing. The same arithmetic drives the road spam the road sweep could not
explain from the road weight (PR #39), and it is the counterparty half of
every lopsided trade in the trade census (PR #42).

## The design, in ten lines

1. `buy_progress` — for each of the four purchases, the fraction of its cost
   the hand already covers, times what that purchase is worth to *this seat*;
   the term is the best of the four.
2. A purchase counts only where the board leaves it open: a settlement where
   a legal spot exists and the supply holds, a city over a settlement the
   seat owns, a dev card while the deck has one, a road only where no
   settlement spot is open — a road is priced as the thing that opens a spot,
   not as a place to put two cards.
3. `PURCHASE_VALUE` is 1.0 for a settlement and a city (each is a victory
   point), 0.4 for a dev card, 0.35 for a road. Shape, not fit: the one
   fitted number is the scale over all four.
4. `spare_card` — every card the best purchase does not need, at the seat's
   own rate: a quarter of a card at the bank, a half behind a port. Never
   zero, which is what made a card dumpable.
5. `robber_risk` — the expected loss to a 7, not a cliff: the chance of a 7
   before this seat plays again (`1 - (5/6)^(n-1)`), times the half-hand it
   would discard, times what those cards are worth at the same bank/port
   rate.
6. The discard is made continuous in hand size — zero at seven cards and
   below, half the hand at eight and above, linear between — so an *expected*
   hand crossing the threshold does not jump.
7. `progress` did not survive as itself: `buy_progress` is what it became,
   with the board gate and the purchase weighting it lacked, and the flat
   card count it used to sit beside is gone.
8. Everything is a function of the hand vector and this seat's own board
   facts, which `Survey` now carries (settlements, cities, roads, legal
   spots, per-resource trade ratio) from the walk it already did.
9. One `hand_terms` function serves both evaluators, so the incremental trade
   gate that recomputes hand terms for every seat from the post-trade pool
   reads bit-identically what `evaluate` would have read.
10. The baseline every gate below plays is not a weight vector but the old
    term set, frozen in `hexset.bench.shipped_hand` — and checked: it
    reproduces all 20 recorded `heximax` census hashes exactly.

## The fit

`python -m hexset.bench.hand_valuation fit --games 128 --seed 95000`.
Challenger heximax at the candidate vector against the frozen shipped hand
valuation, grouped `[c, c, b, b]` seating, antithetic-paired boards, depth 2
width 6, trading on — `road_sweep`'s harness, generalised to take entrants
rather than only weight vectors. Every cell shares the same board sequence.

**Against the shipped hand valuation the grid saturates.** The first cell
run — the centre of the grid, `buy_progress = 0.30`, `spare_card = 0.15`,
`robber_risk = -0.30` — read **128/128 = 100%** [97.1%, 100.0%] against
`heximax-shipped-hand` at 128 games, with a control (shipped against
shipped, same boards) at 47.7% [39.2%, 56.3%]. A margin that cannot go up
cannot rank cells, so the fit was re-run against the **centre of the grid**
instead: each candidate plays a heximax at the centre vector, control
centre-against-centre. `fit-vs-shipped.json` keeps the saturated reading;
`fit-stage1.json` and `fit-stage2.json` are the grid, `gates.json` the four
gate cells, `census-*.json` the three censuses (rolled up; the per-trade
records are reproducible from the command in each file's `settings`), and `marginal_scale.py` the
re-derivation of `MARGINAL_SCALE`.

| cell (`buy_progress` / `spare_card` / `robber_risk`) | games | win% vs centre [95% CI] | roads/game (cell vs centre) |
|---|---|---|---|
| centre vs centre (control) | 192 | 44.3% [37.4%, 51.3%] | 8.48 vs 8.37 |
| 0.20 / 0.15 / −0.30 | 192 | 15.1% [10.7%, 20.9%] | 8.52 vs 8.17 |
| 0.30 / 0.15 / −0.05 | 192 | 62.0% [54.9%, 68.5%] | 8.09 vs 7.53 |
| 0.30 / 0.15 / −0.15 | 192 | 67.2% [60.3%, 73.4%] | 8.69 vs 7.74 |
| 0.30 / 0.15 / −0.60 | 192 | 24.0% [18.5%, 30.5%] | 7.59 vs 8.79 |
| **0.45 / 0.15 / −0.15** | 192 | **83.9% [78.0%, 88.4%]** | 7.37 vs 7.89 |
| 0.45 / 0.15 / −0.30 | 192 | 75.0% [68.4%, 80.6%] | 7.20 vs 8.58 |
| 0.60 / 0.15 / −0.15 | 192 | 80.2% [74.0%, 85.2%] | 6.75 vs 8.02 |
| 0.60 / 0.15 / −0.30 | 192 | 75.5% [69.0%, 81.1%] | 6.64 vs 8.56 |

Both scales are strongly load-bearing and both point the same way — value
the hand more, fear the robber less. `buy_progress` peaks around 0.45 (0.20
is catastrophic at 15%, 0.60 is a little worse than 0.45); `robber_risk`
peaks at −0.15, with −0.05 and −0.60 both worse, so the term earns its keep
but not at anything like the old cliff's price. `spare_card` is not swept:
it adds the same quarter-card to every hand at the table and the search only
ever reads differences. **The fit is `buy_progress = 0.45`, `spare_card =
0.15`, `robber_risk = -0.15`**, and `MARGINAL_SCALE` — the shared unit every
seat's published vector is squashed onto, defined as the mean absolute
one-card marginal of the shipped profile — is re-derived from those weights
over the same trade-free census games its comment names.

One caveat on the fit's own terms: nine cells at 192 games is a coarse
instrument with several comparisons in it, so the winning cell's margin over
its neighbours is not separately resolved. The gates below are on a
different seed at 384 games, which is where the choice is actually checked.

## The gates

All at `--seed 96000`, boards identical across cells, grouped seating, and
all against the shipped hand valuation rather than a weight vector.

### (i) strength, heximax(new) vs heximax(old)

Bar, pre-stated: ≥ 384 games, point ≥ 50%, Wilson lower bound > 45%.

| arm | games | wins | win% [95% CI] | verdict |
|---|---|---|---|---|
| heximax(new) vs heximax(shipped hand) | 384 | 384 | **100.0% [99.0%, 100.0%]** | **met** |
| the same, both sides' trading switched off | 384 | 236 | **61.5% [56.5%, 66.2%]** | (not a stated bar) |

Every game, all 384. The trading-off arm is not one of the pre-stated gates;
it was added because a shutout against a bot whose trade gate you exploit
says nothing about whether you play better. With both sides refusing every
exchange, the redesigned hand terms still win by 11.5 points, so the gain is
not only the gate.

### (ii) strength, heximax(new) vs search2(old)

Bar, pre-stated: ≥ 384 games, must not fall below the standing 65.25% by
more than its interval.

| arm | games | wins | win% [95% CI] |
|---|---|---|---|
| heximax(new) vs search2(shipped hand) | 384 | 384 | **100.0% [99.0%, 100.0%]** |
| heximax(shipped hand) vs search2(shipped hand) — this instrument's own control | 384 | 211 | 54.9% [49.9%, 59.9%] |

**Met**, and worth the caveat: the standing 65.25% was read on another
instrument (`hexset.bench.duel`, 800 games, seed 42000), and the shipped bot
reads **54.9%** on this one — 384 games, seed 96000, `road_sweep`'s grouped
harness — an interval that excludes 65.25%. Whatever the difference between
the two harnesses is, it is not this change: the reading that is actually
paired is the second row against the first, on identical boards.

### (iii) roads per game

Report, not a bar. Read on the self-play cells, where both sides play games
of the same length (a shutout suppresses the loser's build outright — the
shipped bot builds 4.78 roads a game in gate (i) simply because the games
are short).

| lineup | roads/game |
|---|---|
| heximax(shipped hand) self-play (fit control, 192 games) | 10.09 / 10.13 |
| heximax(new) self-play (fit control, 192 games) | 8.48 / 8.37 |
| gate (i) with trading off, new vs shipped, 384 games | **8.17** vs **9.12** |

Down from about 10.1 to about 8.4 in self-play — fewer, as expected, and by
more than any cell of the road sweep moved them (9.80 → 9.66 at `road =
0.04`, and only `road = 0` reached 7.55, at a cost in strength).

### (iv) trade lopsidedness — **the pre-stated bar is not met**

`hexset.bench.trade_census`, 96 games a lineup, identical boards, grouped
and rotated seating, seed 97000. No network entrant in this venv, so the
heuristic tables only, as the registration allows.

| lineup | bot | trades/turn | given | received | imbalance | bulk | 3+:1 trades | held 8+ | value swing |
|---|---|---|---|---|---|---|---|---|---|
| shipped ×4 | heximax(shipped hand) | 0.355 | 2.35 | 2.35 | 0.10 | 53.5% | **0** | 9.5% | **+0.000** |
| new ×4 | heximax(new) | 0.712 | 2.67 | 2.67 | 0.19 | 63.5% | 0 | 38.5% | **+0.000** |
| new ×1 + shipped ×3 | heximax(new) | 0.475 | 1.97 | 3.51 | 0.30 | 73.4% | **667** | 76.4% | **+0.384** |
| new ×1 + shipped ×3 | heximax(shipped hand) | 0.590 | 3.23 | 1.99 | 0.26 | 66.3% | 667 | 5.5% | **−0.309** |

The bar was "value swing per trade should move toward zero and 'gave 3+ for
1' should fall", at a mixed table. **Both move the other way**: at the mixed
table the new bot's swing is **+0.384** a trade against the shipped bot's
**−0.309**, and 667 of the table's trades are 3-or-more-for-one where the
shipped self-play census has none at all. This is recorded as a miss, not
re-read against a different bar and not tuned.

What the numbers say is that the lopsidedness *reversed*: the +0.18 to +0.30
a trade the network was taking off heximax, and the −0.15 to −0.25 heximax
was paying (trade census, PR #42), is now +0.384 / −0.309 with the new
heximax in the network's chair. A mixed table of two bots that price cards
differently cannot show a swing near zero for either of them — the swing is
measured at one flat 4:1 yardstick, and one of the two valuations is the one
the redesign exists to replace. The table that can answer "does it trade
sanely" is self-play, and there the swing is **exactly zero** and cards given
equal cards received, as it is for the shipped bot.

Two behaviours in that table are worth the owner's attention rather than a
verdict here. The new bot trades **twice as often** (0.712 vs 0.355
trades/turn) and in **bulkier** bundles (63.5% vs 53.5%, imbalance 0.19 vs
0.10) than the shipped one, and **38.5% of its trades are made holding eight
or more cards** against the shipped bot's 9.5% — a direct consequence of
pricing the robber at its expected loss instead of −0.39 a card, and the same
"inhuman bulk trading" the owner flagged in the policy.

### (v) the choice census is re-recorded

Byte-identical choices were never on offer — this is a behaviour change by
design. All 30 fixture games change: 20 `heximax`, 5 `heximax-notrade`, 5
`heximax-omni`. Re-recorded with `--write-census` in its own commit, labelled
as such, so the diff that changes behaviour and the diff that re-baselines
the record are separable. `tests/test_seating.py`'s five `greedy` traces
(`BYTE_IDENTITY_TRACES`) are in the same commit for the same reason: `greedy`
scores with the same `Weights`, so replacing three of its terms necessarily
moves what it plays.

`MARGINAL_SCALE` moved with the weights: 0.10231140469178995 → **0.0513359464196004**
(8110 marginals, was 9280), re-derived from the fitted vector over the same
trade-free games its comment names, after the fit and before the gates. The
first run of these gates read the shipped baseline at 57.0% instead of 54.9%
because the frozen term set was still publishing on the *new* scale;
`shipped_hand.SHIPPED_MARGINAL_SCALE` freezes it, and the frozen bot then
reproduces all 20 recorded `heximax` census hashes exactly.

## Reading

The shipped hand valuation was not merely miscalibrated, it was a hole:
heximax with the redesigned terms beats heximax with the old ones **384 out
of 384** games, and search2 with the old ones the same 384 out of 384, which
is the same shutout a network seat was posting against heximax at the g4
table. Most but not all of that is the trade gate — with both sides' trading
switched off the new terms still win 61.5% [56.5%, 66.2%], so the redesign
ranks positions better as well as pricing exchanges better. The fit says both
new scales are strongly load-bearing and both point the same way: value the
hand more (`buy_progress` 0.45, where 0.20 collapses to 15% against the
centre) and fear the robber far less (`robber_risk` −0.15, where the old
cliff's equivalent, −0.60, reads 24%). Roads fall from about 10.1 a game to
about 8.4 without touching the road weight, which is the road sweep's
unresolved question answered from the other end: heximax bought roads because
holding the cards was worth nothing, not because a road was worth 0.12.
Trading is where the reading is uncomfortable: with its own kind the new bot
trades exactly value-fair, but it trades twice as often, in bulkier bundles,
and 38.5% of the time while holding eight or more cards — pricing the robber
honestly removed the reason not to sit on a big hand. Gate (iv)'s pre-stated
bar is not met and is recorded as a miss: at a mixed table the swing moved
from zero to +0.384 a trade in the new bot's favour, which is the old
lopsidedness reversed rather than removed, and a mixed table of two
differently-priced valuations cannot show otherwise. The honest summary for
the PI is that heximax is no longer the exploitable gate that trade-obs-c5 is
training against, but it is now an exploiter of the bot it replaces, and
whether its bulk trading is load-bearing or inhuman is the same open question
the policy's is. Two things follow that this readout does not settle: the
network checkpoints were selected against the old gate and their strength
readouts should be re-read against this one, and `MARGINAL_SCALE` is now a
derived quantity that any future refit has to re-derive rather than inherit.
