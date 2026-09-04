# heximax profile — where the time goes

`hexset.bench.profile_heximax`, three games an arm, seed 100, single process
(the box carries a GPU run and an 8-worker sweep, so no worker pool), commit
`61ac38e`, x86_64, Python 3.12.14. Every seat the given preset; `heximax`
(depth 2, width 6, honest, trading on), `heximax-notrade` (`max_trades=0`,
`NO_TRADE_WEIGHTS`), `search2` for scale. Raw `.prof` files beside this
README; numbers below are `pstats` self-time (`tottime`) unless marked
cumulative. **cProfile inflates wall time roughly 2-4x** (instrumentation
overhead on every call), so the absolute ms/decision below reads higher on
its cheap end and the totals do not reproduce the unprofiled ~25ms/~19
CPU-second figures directly — the profiled comparison *between* the three
arms, run under the same overhead, is the reliable part.

## Cost

| preset | ms/decision mean | p50 | p95 | s/game | decisions/game |
|---|---|---|---|---|---|
| heximax | 8.891 | 1.041 | 39.766 | 6.265 | 312.3 |
| heximax-notrade | 7.256 | 0.442 | 36.038 | 2.727 | 337.0 |
| search2 | 4.958 | 1.076 | 22.682 | 4.920 | 339.7 |

## Buckets (% of total self-time)

| bucket | heximax | heximax-notrade |
|---|---|---|
| (a) move generation / legal actions | 4.4% | 8.8% |
| (b) state copy & apply (`copy_state`, `imagine`, `apply`) | 7.3% | 7.7% |
| (c) evaluation (`Evaluator`/`HonestEvaluator`, survey, progress) | 27.8% | 20.8% |
| (d) info-set/view (`View`, `belief_for`, `_move_hand`) | 15.4% | 10.7% |
| (e) trading's own code (`_candidates`, `_best_clearing`, `_delta`, gates) | 9.4% | 7.8% |
| (f) everything else (builtins, dict ops, RNG, tree-walk glue) | 35.7% | 44.2% |

## Top 10 functions, heximax, by cumulative time (3 games)

| cum s | tot s | calls | function |
|---|---|---|---|
| 10.241 | 0.009 | 28,915 | `game.py:706 run_trade_event` |
| 10.232 | 0.015 | 509 | `trading.py:271 trade_event` |
| 10.214 | 0.047 | 564 | `trading.py:461 _best_clearing` |
| 8.857 | 0.001 | 1,086 | `trading.py:520 ask` |
| 8.857 | 0.024 | 1,086 | `trading.py:148 judged_many` |
| 8.832 | 0.035 | 57,136 | `trading.py:142 judged` |
| 8.785 | 0.036 | 57,136 | `heximax/search.py:365 accepts` |
| 8.749 | 0.360 | 57,136 | `heximax/search.py:427 _delta` |
| 8.327 | 0.006 | 937 | `heximax/search.py:196 choose` |
| 6.984 | 0.005 | 399 | `heximax/search.py:276 _search` |

`run_trade_event`'s own chain (rows 1-8) costs *more* cumulative time than
`choose` (row 9, the bot's whole search) over the same three games —
the real per-turn trade clearing is the single largest cost center in a
heximax game, ahead of the tree search it was meant to sit beside.

## Reading

`trading.trade_event` does **not** run inside the search's lookahead: `imagine()`
builds every hypothetical child with `gates=None`, and `run_pending_event`
no-ops whenever `gates is None`, so a search node never re-runs a real trade
event — the "inside the search" cost the owner asked about is zero by
construction. What is expensive is the *real* game-level trade event, called
after every actual MAIN action in the outer loop: `_best_clearing` ranks
`_candidates`' bundle enumeration by public surplus and then asks each
candidate's private gate (`judged_many` → `accepts` → `_delta`) in that order
until one clears or the list runs out, with no cap on how many it asks —
`_delta` alone fires 57,136 times over three games, each one cloning the
whole `GameState`/`PublicLedger` and re-scoring both seats. That chain's
cumulative cost (10.24s) exceeds `choose`'s own (8.33s) over the same games.
The heximax-vs-notrade delta (10.45s of self-time over 18.4s vs 7.9s) is the
direct price of trading, but only 10.6% of that delta is trading's own code
(bucket e); 33% lands in evaluation and 19% in the info-set/view machinery,
because `_delta` re-invokes `evaluate()`/`belief_for()` per candidate and
because trading changes the games' own trajectories (more legal actions,
more distinct positions, lower evaluation-cache hit rates) — bucket (e) is
the narrow measurement, the full delta is the honest one. `_candidates`
enumeration itself is nearly as expensive in the no-trade arm (120,423 calls)
as in heximax (156,700), because it still walks every counterparty's hand
even when that counterparty's published vector is all zero and can never
clear — wasted work in both arms.

## Top 5 optimisation candidates (not implemented — gated behind the
choice-census guard, separate PRs)

1. **Cap candidates a private gate is asked about per trade event**, the way
   `NETWORK_GATE_ROWS` already bounds a *network* gate
   (`trading.py:461 _best_clearing`, `search.py:427 _delta`). `_delta`'s
   chain is the largest cost center in the game (10.24s cumulative of 18.4s
   total self-time over 3 games, exceeding the search itself). `trading.py`'s
   own comment on `NETWORK_GATE_ROWS` reports bounding it to 32 costs "at
   most ~5% of trades/turn" for a network gate, because clearing deals rank
   near the top of the public ranking — the same bound for a heuristic gate
   should be similarly cheap. **Payoff: the largest available, plausibly
   15-25% of per-decision time**, since the chain it shortens is bigger than
   the search loop it sits beside.
2. **Skip `_candidates` enumeration for a counterparty whose published
   valuation is all-zero** (`trading.py:240 _candidates`, its `:266`/`:268`
   generators — 156,700 calls / 0.552s self in heximax, 120,423 calls /
   0.321s in notrade, where it can *never* clear). A `v == NO_VALUATION`
   check before generating bundles is a one-line, free win. **Payoff:
   ~3-5%**, more in lineups with non-trading seats.
3. **Precompute or vectorise `progress_toward`'s inner sum**
   (`evaluate.py:186-202`: `sum(min(hand[r], n) for r, n in needed)`,
   1,118,184 calls plus a 4,456,611-call generator, ~1.4s combined, 7.6% of
   total). The module's own comment already flags this runs three times per
   seat per leaf. **Payoff: ~5-8%.**
4. **Lighten `View.__init__`/`belief_for`** (`view.py:61` 1.035s/93,919
   calls, `evaluate.py:102 belief_for` 0.789s/158,591 calls). `belief_for` is
   already content-keyed and memoized for the life of one `choose()`, but the
   key tuple itself (every hand, every ledger seat, the bank, the board) is
   expensive to build on every call, memoized or not, and a cache miss still
   pays full `View` construction. **Payoff: ~6-10%.**
5. **Cheaper cloning for the marginal/delta checks**
   (`state.py:78 copy_state` 0.671s/104,984 calls, `ledger.py copy`
   0.270s+0.156s/520,615 calls combined). `_marginal_gain`, `_marginal_loss`
   and `_delta` each deep-copy the whole `GameState` and `PublicLedger` to
   test moving one or a few cards; a narrow hand-only diff would skip
   cloning the board, deck and dev-card state, none of which changes.
   **Payoff: ~4-6%.**
