# One-event trading — the readouts

The gates of the trading design's registration ("Registration — the one-event
trade mechanic", `agents/reference/trading-design.md`), run inside the PR that
lands the mechanic. Every number below is in the JSON beside this file; the
adjudication is the PI's.

## (iii) Strength — a fresh comparison

`heximax` vs `search2`, 800 blocked games, duel seed 42000, 26 workers.
Both bots changed, so this is read on its own terms, not against the
retiring protocol's 65.25%.

| | |
|---|---|
| wins | **488 / 800 = 61.0%** |
| Wilson 95% | [57.6, 64.3] |
| paired VP | **+0.841** [+0.665, +1.017] |
| bar | point ≥ 50%, Wilson lower bound > 45% |

`heximax-vs-search2.json`.

## (iii) Paired non-regression — and where the registration was wrong

`heximax-notrade` vs `search2-notrade`, 800 blocked games, duel seed 59300.
The registration required this to reproduce **485/800 = 60.625%**
bit-for-bit, on the reasoning that this pair never trades under either
mechanic.

It reads **499 / 800 = 62.375%**, Wilson [59.0, 65.7], paired VP +0.767.

**Why, established rather than guessed.** Both no-trade presets suppressed
*proposing* and *accepting*, but neither filtered the engine's offer sample
out of the interior of its own search tree: `SearchBot._value` and
`Heximax._options_in` both took `legal_actions` whole. The no-trade referents
were never trade-free games in their own search's model.

Re-running the same duel on `origin/main` (839dcb6) with one change —
`actions._offer_actions` stubbed to `[]`, everything else about the offer
protocol untouched — gives **499/800, paired VP +0.766875: identical to the
digit**. So the offer sample's removal is the *entire* difference, and
nothing else in the mechanic moves a no-trade game. The same attribution
holds at the census level: that stub reproduces every re-baselined no-trade
hash exactly (20/20 `search2-notrade`, 5/5 `heximax-notrade`, 5/5 the
unlocked-`greedy` traces in `test_seating`).

`heximax-notrade-vs-search2-notrade.json`, `no-trade-attribution.json`.

## (v) Cost

The mirror protocol from `hexset.bots.heximax.search`'s module docstring:
three four-seat games an arm, board seeds 0/1/2, every seat the same preset,
`search2` as the control (it was `search2-offers3`, and there is no budget
any more), arms interleaved seed by seed, one process.

| | heximax | search2 | ratio |
|---|---|---|---|
| ms / move | 4.193 | 2.750 | **1.53x** |
| function calls / move | 19,447 | 16,860 | **1.15x** |
| moves (3 games) | 950 | 1216 | |

Comfortably inside the design's "≤2x `search2` per move" ceiling; the last
paired reading under the offer protocol was 2.357x. Read per *move*: heximax's
games are shorter, so a per-game total flatters it. The phase-neutral
re-weighting the docstring also asks for is moot — it existed because
`search2` booked ~20x more `TRADE_RESPOND` decisions and those cheap
decisions sat in the denominator, and there is no such phase any more.

**Trades per turn**, over the lineup (iii) actually plays
(`[heximax, heximax, search2, search2]`, blocked, 6 games, 590 events):

| trades in the event | events |
|---|---|
| 0 | 506 |
| 1 | 79 |
| 2 | 5 |

Mean **0.151**, max 2. In the mirror games: heximax 0.210/turn, search2
0.126/turn — heximax's `tanh(marginal / MARGINAL_SCALE)` vector advertises
more than search2's single `+1`/`-1` pair, and clears more. The engine's one
assertion (trades in an event ≤ cards on the table) never fired.

`cost.json`.

## Re-run after `trade_event`/`Game.publish` split (fix/publish-not-call)

The engine correction registered in `agents/reference/trading-design.md`'s
post-data note "HexNet lands contract 5": `trade_event(game, gate)` reads
`game.valuations` instead of calling each seat's `valuation` at event time;
drivers publish once, right after the acting seat's own decision
(`Game.publish`). Both gates (iii)/(v) are re-run rather than reused, since
the timing change moves what a seat's vector reads as at anyone else's
trade event and therefore re-baselines every trading game (the two censuses
confirm this: `heximax`/`heximax-omni`/`search2` changed, `heximax-notrade`/
`search2-notrade` stayed byte-identical).

### (iii) Strength

`heximax` vs `search2`, 800 blocked games, duel seed 42000, 26 workers.

| | |
|---|---|
| wins | **485 / 800 = 60.6%** |
| Wilson 95% | [57.2, 64.0] |
| paired VP | **+0.929** [+0.759, +1.100] |
| bar | point ≥ 50%, Wilson lower bound > 45% |
| met | yes |

`publish-heximax-vs-search2.json`.

### (v) Cost

Same mirror protocol, re-timed in one process (`sys.setprofile` call
counting active throughout, which inflates every absolute ms/move figure
above the un-instrumented `cost.json` reading but is common to both arms,
so the *ratio* is the number to read).

| | heximax | search2 | ratio |
|---|---|---|---|
| ms / move (profiled) | 6.963 | 4.427 | **1.57x** |
| function calls / move | 10,904 | 7,954 | **1.37x** |
| moves (3 games) | 896 | 1051 | |

Comfortably inside the design's "≤2x `search2` per move" ceiling, and in
the same range as the pre-fix 1.53x — this readout is a non-regression
check on heximax's own search cost, not a demonstration of the collector
speedup the fix targets: that 5.8x-slower-per-step finding was in HexNet's
batched collector (`agents/reference/trading-design.md`'s post-data note),
a different repo's driver, not `arena.play`'s per-move cost measured here.

**Trades per turn**, over the lineup (iii) plays
(`[heximax, heximax, search2, search2]`, blocked, duel seed 42000, the
first 6 games):

| trades in the event | events |
|---|---|
| 0 | 393 |
| 1 | 57 |

Mean **0.127**, max 1, over 450 events — same order as the pre-fix 0.151
mean, no cycle and no runaway trading; the engine's one assertion never
fired.

`publish-cost.json`.

## Re-run at the bundle engine (fix/bundle-candidates)

Three corrections land together in this PR, each registered in
`agents/reference/trading-design.md` before code: **candidates are bundles**
(owner review and PI correction, 2026-09-03) — `_candidates` enumerates
signed multi-card bundles on disjoint resources instead of only coverable
one-for-one swaps, gated at `GATE_BUDGET` (8) candidates per clearing
attempt by public surplus; **no bundle-size cap** (owner review against the
rulebook, 2026-09-03) — a bundle is bounded only by what each hand holds,
not a fixed 1..3-cards-a-side limit; **trade and build interleave** (the
same review) — the event runs at the start of `Phase.MAIN` and again after
every MAIN action the current player takes (build, buy, a bank/port trade,
a development card), not once before the first build; and the **tie-break**
(owner review, 2026-09-03) is now the acting seat's own surplus among
equally fair deals, then the total, then a canonical order for determinism,
replacing fewer-cards/canonical/lower-seat as the whole rule. `heximax`'s
ranking arithmetic is vectorised with `numpy` above ~32 candidates
(`_rank_candidates_vectorized`), exact and not an approximation of the
loop version — measured 2-3x faster than the loop at the candidate counts
an uncapped hand actually produces late in a game.

**Candidate counts, one concrete position** (turn 30, a real `heximax` vs
`heximax` game, hands `[2, 3, 6, 6]` cards): the old one-for-one enumeration
found **17** candidates for the mover; the bundle engine finds **297**.
Across five games, mid-game (turns 20-40) candidate counts ranged
85-1491 (mean 85-236 depending on the game); whole-game maxima, driven by
large late-game hands, ranged roughly 1,300-16,500.

### (iii) Strength — a fresh comparison

`heximax` vs `search2`, 800 blocked games, duel seed 42000, 26 workers. Both
arms changed again (bundle candidates, no cap, interleaving, the new
tie-break), so this is read on its own terms, not against the retiring
488/800.

| | |
|---|---|
| wins | **495 / 800 = 61.9%** |
| Wilson 95% | [58.5, 65.2] |
| paired VP | **+0.648** [+0.467, +0.829] |
| bar | point ≥ 50%, Wilson lower bound > 45% |
| met | yes |

`bundles-heximax-vs-search2.json`.

### (v) Cost, trades/turn, bundle sizes, and the gate budget

Same mirror protocol (three four-seat games an arm, board seeds 0/1/2,
`search2` the control, arms interleaved seed by seed, one process, idle
box), extended with the bundle-size distribution per executed trade and
how often `GATE_BUDGET` binds.

| | heximax | search2 | ratio |
|---|---|---|---|
| ms / move | 3.816 | 2.080 | **1.83x** |

Comfortably inside the design's "≤2x `search2` per move" ceiling (repeated
readings on a shared, noisy box ranged 1.83-2.18x; the clean reading above
was taken with the box otherwise idle). Higher than the pre-bundle 1.53x,
as expected: candidates are no longer capped at three cards a side, and the
event now runs several times a turn instead of once.

**Bundle-size distribution** (cards given-for-received) of executed trades,
pooled over the three mirror games each arm:

| bundle | heximax | search2 |
|---|---|---|
| 1-for-1 | 22 | 4 |
| 1-for-2 | 7 | 2 |
| 1-for-3 | 1 | 0 |
| 2-for-1 | 6 | 4 |
| 2-for-2 | 12 | 6 |
| 2-for-3 | 1 | 1 |
| 3-for-1 | 0 | 1 |
| **total trades** | **49** | **18** |

Most executed trades are still one- or two-card a side even with no cap;
larger bundles (up to 3-for-2 seen here) do clear, confirming the mechanism
this PR exists for actually fires in ordinary play, not only in the
synthetic tests.

**Trades per turn** (mean, over the same games): heximax 0.161, search2
0.057. Higher than the pre-bundle mirror reading (0.21/turn for heximax
under the old protocol) is not comparable directly — this is per-turn, that
was per-event, and the interleaving means several event attempts can share
one turn.

**The gate budget binds often** in this sample: heximax's clearing attempts
hit the `GATE_BUDGET` (8) ceiling without a clear on 61.4% of turns
(183/298), search2 on 89.2% (280/314) — expected once candidates are
uncapped, since most attempts (especially search2's, whose vectors put many
resources at the same `+1`/`-1` value and so tie broadly) advertise well
above 8 candidates and only a few clear. Binding is not itself a failure:
`trades_per_turn_max` stayed at 2 in this sample and the engine's one
assertion (trades per event ≤ cards on the table) never fired. This is the
number the owner's correction asked to be measured, not adjudicated.

`bundles-cost.json`.

### Census

`heximax`/`heximax-omni`/`search2` re-baseline by construction (the
candidate set, the ranking, and the interleaving all changed);
`heximax-notrade`/`search2-notrade` are checked byte-identical against
`origin/main` before writing the new fixtures, and are — `max_trades=0`
still short-circuits `trade_event` before any of this PR's code runs.
