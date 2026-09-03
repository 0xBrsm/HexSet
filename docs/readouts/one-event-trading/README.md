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
