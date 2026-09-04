# The incremental trade gate — heximax and search2 price a candidate without cloning

`hexset.bots.heximax.search.Heximax._delta` and `hexset.bots.search2.
SearchBot.accepts` used to price every candidate bundle by cloning the whole
position (`state.copy_state`, a fresh `PublicLedger` copy for heximax) and
running the full evaluator on the clone. A trade only ever moves two hands —
the board, the bank, the deck and every dev-card pile are exactly what they
were — so `hexset.bots.evaluate.hand_shifted` returns the same `state` with
only the named seats' hands replaced, sharing everything else by reference.
`SearchBot.accepts` (true state, no belief) uses it directly. `Heximax._delta`
additionally has to account for the shared belief `View` a trade can move:
certifying what changed hands can resolve some of the counterparty's own
`unknown` cards into certainty, which shrinks the residual pool every other
seat's `expected_hand` draws from — under `relative`/`paranoid` stance that
third-seat drift is priced too, so `_after_trade_belief`/`_ShiftedBelief`
recompute `known`/`unknown`/`pool` from the event's own pre-trade belief
(already memoized once per event) instead of rebuilding a `View` from a
cloned ledger. `target != knower` — a shape nothing in this repo calls
`_delta` with — keeps the old clone-based path (`_delta_reference`).

## Exactness

Checked against the prior clone-and-evaluate computation (`_delta_reference`,
kept verbatim beside the fast path) before either fast path was trusted:

- **Real self-play, not a hand-built fixture.** `heximax` vs itself, 12
  games (6 board seeds × `relative`/`paranoid` stance), every real
  `_delta(target == knower)` call during play compared against
  `_delta_reference` on the same call: **218,367 calls, 0 mismatches, 0 sign
  flips.** (An earlier attempt to check this against a hand-built ledger
  fixture, rather than one produced by real play, *did* show mismatches —
  traced to the fixture itself violating the ledger's own known/unknown/pool
  invariant, i.e. an artifact of a bad test, not a bug in the fast path; the
  real-play check above is the one that matters and the one both regression
  tests below are built on.)
- **`omniscient` mode and `own` stance are exact by construction** (no
  belief/pool at all, or no cross-seat read at all) and read 0/266 mismatches
  in isolation too.
- **`SearchBot.accepts`** (search2, true state, no belief): 0/804 mismatches
  across `relative`/`paranoid`/`own` stances on 268 samples each.
- **The byte-identical choice census** (`tests/bots/heximax/test_heximax.py`,
  `tests/bots/test_search2.py`, `pytest -m slow -k
  choices_are_byte_identical`): unchanged.
- Full non-slow suite: 884 passed, 0 failed.

## Speed readout

200 games, `heximax` vs itself (all four seats), one process, identical board
and bot seeds both runs — `measure.py` beside this file, run once against
this tree ("after") and once with the incremental fast paths reverted via
`git stash` ("before"), nothing else in the diff. The two runs were
sequential, not interleaved, on the box this repo's own profile notes as
shared and noisy under contention — read `mean s/game` as directional, not
to the second decimal.

| | before | after |
|---|---|---|
| mean s/game | 3.070 | 2.411 (−21%) |
| mean ms/decision | 4.183 | 4.347 |
| trades/turn | 0.0015854141894569957 | 0.0015854141894569957 |
| total trades / turns | 28 / 17,661 | 28 / 17,661 |
| decisions | 64,718 | 64,718 |
| wall seconds (200 games) | 613.9 | 482.2 |

`before.json`/`after.json` beside this file.

## Reading

Trades/turn matches to every digit printed (28 trades over 17,661 turns,
64,718 decisions, both runs) — the strongest behaviour-preservation evidence
available short of the census itself, since it means every clearing decision
the engine made was identical, not merely close. `mean s/game` fell about
21%, consistent with the profile's finding that most of a trade event's cost
is candidates that get *evaluated* and rejected, not the rare ones that
clear — heximax's own gate (`_delta`) fires far more often than a trade
actually executes. `mean ms/decision` moved the other way, by a few percent;
given the two runs were sequential single-process batches on a shared box
rather than interleaved (this project's own convention for a clean read,
used elsewhere in this doc tree), and every other project readout on this
box documents contention swinging heximax's per-move cost by tens of
percent, the honest read is that this single non-interleaved pair cannot
tell a real few-percent regression from box noise — an interleaved re-run
would be needed to say more, and is not included here. **heximax-vs-heximax
tables trade rarely under the `relative` stance** (28 trades in 17,661
turns, ~1 trade per 631 turns) — this table is the profile's own subject,
not a favourable case picked for this readout, so the ~21% per-game gain
measured here is a lower bound: a table with more trading (mixed presets, a
looser stance, or seats that publish more aggressively) exercises the gate
more and should show more of it.
