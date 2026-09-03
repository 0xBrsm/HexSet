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

## Registered ablation — the gate budget and the ranking (2026-09-03)

The "bundles land" post-data note's registered ablation
(`agents/reference/trading-design.md`), run before the registered first run
opens: five arms, each a 400-game blocked duel `heximax` vs `search2` at
duel seed 42000, 26 workers. `gate_budget` and `order` are now keyword
parameters of `trade_event` (defaults unchanged: `GATE_BUDGET` (8),
`"maximin"`) rather than constants edited per run, so the five arms are one
code path with different arguments (`hexset.arena.play`/`compete` and
`hexset.bench.duel --gate-budget`/`--order` thread them through). No-trade
and default-arm census fixtures verified byte-identical.

| arm | wins/400 (Wilson 95%) | paired VP | trades/turn (heximax) | budget-bind rate (heximax) | ms/move (x search2) | max candidates asked/event |
|---|---|---|---|---|---|---|
| gate-budget-8 (today) | 240/400 = 60.0% [55.1, 64.7] | +0.594 [+0.346, +0.842] | 0.163 | 61.3% | 1.89x (1.79–2.03x, 3 runs) | 22 |
| gate-budget-16 | 228/400 = 57.0% [52.1, 61.8] | +0.627 [+0.371, +0.884] | 0.205 | 42.3% | 1.55x (1.54–1.57x) | 32 |
| gate-budget-32 | 236/400 = 59.0% [54.1, 63.7] | +0.755 [+0.499, +1.011] | 0.250 | 30.6% | 1.94x (1.86–2.05x) | 63 |
| gate-budget-unbounded | 250/400 = 62.5% [57.7, 67.1] | +0.762 [+0.517, +1.008] | 0.254 | 0.0% | 1.97x (1.94–2.00x) | 6798 |
| gate-budget-minimal (`order=minimal_bundle`, budget 8) | 229/400 = 57.2% [52.4, 62.0] | +0.425 [+0.172, +0.678] | 0.196 | 59.1% | 1.48x (1.42–1.54x) | 20 |

`gate-budget-{8,16,32,unbounded,minimal}.json`, each carrying its own
`duel` (the 400-game strength read) and `cost` (the mirror-table read:
three four-seat games a side, board seeds 0/1/2, `search2` the control,
extended with trades/turn, the executed-bundle-size distribution, the
gate-budget bind rate, and the max candidates put to a private gate in one
trade event) sections. `search2`'s own trades/turn ran 0.057/0.109/0.124/
0.186/0.077 over the same arms — lower than heximax's throughout, and its
bind rate stays above 65% even unbounded's own vectors (search2's simple
`+1`/`-1` valuations tie broadly, so it hits the same maximin key on far
more candidates than heximax's continuous `tanh` vector does).

**Reading, pre-stated in the registration; reported here, not adjudicated:**
cheapest arm whose trades/turn is within 10% of the unbounded arm's and
whose cost stays ≤2x; if unbounded is itself ≤2x, the budget goes away.

- **Trades/turn within 10% of unbounded's 0.254** (i.e. ≥ 0.229): only
  gate-budget-32 (0.250, −1.6%) and gate-budget-unbounded itself qualify.
  Gate-budget-8, -16 and -minimal fall short by 36%, 19% and 23%
  respectively.
- **Cost ≤2x**: on the 3-run mean, all five arms read under 2x (1.89x,
  1.55x, 1.94x, 1.97x, 1.48x) — but gate-budget-32 and gate-budget-unbounded
  each poked above 2x on at least one of the three individual runs (2.047x
  and 1.999x), so neither clears the ceiling comfortably on this box.
- Combining both conditions, only **gate-budget-32** and
  **gate-budget-unbounded** satisfy them among the five arms measured.
- **Unbounded's own cost (mean 1.97x, range 1.94–2.00x) reads at or under
  2x** — the pre-stated condition under which "the budget goes away" — on
  a measurement close enough to the ceiling that a cleaner box could move
  it either side.

**Not exact — the box was not idle.** An unrelated job (another agent's
`pytest` run, `hexset-publish-points`) shared the machine for the whole
measurement window, pushing load average above 25 while the 26-worker
duels ran. `hexset.bots.heximax.search`'s module docstring warns that
heximax's per-move cost inflates faster than search2's under contention, so
the ms/move ratios above are the mean of 3 back-to-back single-process
passes (all 5 arms interleaved seed-by-seed within each pass, per
`ratio_noise_check` in each JSON) rather than one reading. The deterministic
statistics — trades/turn, bind rate, bundle-size distribution, max
candidates asked — are identical across all 3 passes and are not affected
by the contention; only the ms/move ratios carry the noise shown above.

**Wall time.** The five 400-game duels: 68–129 s each (~7 min total, the
unbounded arm slowest as expected). The cost script: ~67 s per pass, 3
passes (~3.5 min). Verification (the parameterisation's own tests plus the
full byte-identity census, run once, not repeated per arm): ~23 min,
almost entirely contention-bound on the same shared box.

`gate-budget-8.json`, `gate-budget-16.json`, `gate-budget-32.json`,
`gate-budget-unbounded.json`, `gate-budget-minimal.json`.

## Re-run for "publish points and the event trigger" (fix/publish-points)

The correction registered in `agents/reference/trading-design.md`'s post-data
note "HexNet lands contract 5", PI amendment "publish points and the event
trigger": a seat publishes once a turn (`Game.publish_due`, the engine-defined
post-roll/robber point) instead of after every action, and the turn's first
trade event no longer runs inside `enter_main` — it runs lazily, the first
time the current player's own `legal_actions`, `Game.state(seat)` (at
`hidden=True` only — see below), or `Game.publish` is reached, whichever
comes first (`Game.event_pending`).

### (iii) Strength — a fresh comparison

`heximax` vs `search2`, 800 blocked games, duel seed 42000, 26 workers. Both
arms changed again (publish timing, event trigger), so this is read on its
own terms, not against the bundle engine's 61.9%.

| | |
|---|---|
| wins | **478 / 800 = 59.8%** |
| Wilson 95% | [56.3, 63.1] |
| paired VP | **+0.681** [+0.503, +0.859] |
| bar | point ≥ 50%, Wilson lower bound > 45% |
| met | yes |

`publish-points-heximax-vs-search2.json`.

### (v) Cost, publish calls per turn

Same mirror protocol (three four-seat games an arm, board seeds 0/1/2, every
seat the same preset, `search2` the control, arms interleaved seed by seed,
one process), both arms now seated as the game's own gates and publishing
through the same `Game.publish_due` gate the drivers use, extended with
publish-calls-per-turn.

| | heximax | search2 | ratio |
|---|---|---|---|
| ms / move | 7.708 | 4.559 | **1.69x** |
| function calls / move | — | — | **1.43x** |

Comfortably inside the design's "≤2x `search2` per move" ceiling on both
readings. The call-count ratio is load-independent; the ms/move ratio was
taken on a contended box (loadavg ~15–33 throughout this run, two other
pytest processes and an 800-game duel running concurrently), so read it as
"comfortably under 2x," not to the second decimal — every prior reading in
this file notes the same caveat when it applies.

**Publish calls per turn: 1.01 for both arms** (heximax 286/283, search2
324/321, and the lineup readout (iii) plays: 609/603) — matches the
design's expectation of one publish per seat per turn, down from "after
every action" (the collector-cost problem this correction exists to fix).

**Trades per turn, over the lineup readout (iii) actually plays**
(`[heximax, heximax, search2, search2]`, blocked, duel seed 42000, first 6
games): mean **0.0166**, max 2, 10 trades over 603 turns (594 turns clear
nothing, 8 clear one, 1 clears two) — an order of magnitude lower than the
bundle engine's 0.161/0.057 mean. This is a real, expected consequence of
the correction, not a regression: a seat's vector is now fixed for the
*entire* turn at whatever it published at the post-roll point (the
design's own "public vectors fixed within one event, published at the last
decision" — now literally the *only* decision each turn that publishes),
rather than refreshed after every build/trade as the previous ("publish
after every action") shape did. Fewer, staler vector updates mean fewer
opportunities for two seats' advertised wants to overlap.

`publish-points-cost.json`.

### Census

`heximax`/`heximax-omni`/`search2` re-baseline by construction (publish
timing changed, which changes which trades clear and when — verified
directly: `search2-notrade`/`heximax-notrade` reproduce their prior hashes
bit-for-bit, 20/20 and 5/5, confirming the *only* thing that moved a
no-trade game was gone; the trading presets do not reproduce and the
fixtures were regenerated).

### Two correctness bugs found and fixed while implementing this, not just measured

- **`Game.state(seat, hidden=False)` must not be an event trigger, only
  `hidden=True` is.** `hexset.bench.aivat.chance_outcomes` re-seats a
  *hypothetical* child position with the real seated bots' gates so it can
  score what a real trade event would do there; a value function reading
  `child.state(0, hidden=False)` to hash the position was, before this
  fix, enough to fire a live trade event using those bots' real judgement
  as a side effect of being asked to score a position, measurably
  diverging the real game (`test_aivat.py`'s exact-replay gate caught it:
  `_play_one` and `instrumented()` stopped reproducing the same game
  bit-for-bit). Fixed in the engine (`Game.state`), not by special-casing
  `aivat.py`.
- **`hexset.server.webplay.GameSession.state_view` read `trades`/
  `valuations` before the field that would have triggered the pending
  event (`legal_actions`, further down the same dict).** A poll of a fresh
  `MAIN` position could report `trades: []` even though the event had a
  clearing deal waiting, because the snapshot's own `legal_actions` field —
  the thing that would have fired it — was computed after the fields that
  needed its result. Fixed by observing the current player's own state
  (`hidden=True`, discarding the result) before those fields are built.

### A known gap, not fixed here

`hexset.server.webplay.GameSession._apply`'s own per-action `Event`/journal
`trades` field is captured as `game.trades` since the *previous* `_apply`
call, which undercounts a turn's first event when it fires lazily between
two `_apply` calls (e.g. via an embedded bot's pre-submit publish, or a
client's own `legal_actions`-fetching poll) rather than inside either one:
the trade is fully reflected in the live `state_view` (fixed above), but is
not attributed to any discrete step in `self.events`/the journal, so a
game resumed from the journal after a server restart would not replay it.
No test currently exercises this path; `hexset.record.record_game` has the
analogous fix (attributing a lazily-triggered first event to the *previous*
action's step, `len(actions) - 1`, matching what `hexset.record.advance`
already replays as `apply(that action); apply_trades(...)`), which the
session's own per-action bookkeeping in `webplay.py` would need too, and
does not yet have.
