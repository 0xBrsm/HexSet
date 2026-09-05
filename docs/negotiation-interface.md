# Negotiation interface for human and LLM seats (draft, PI to ratify)

**Superseded, 2026-09-05 — shipped.** This design is built (`agents/
reference/trading-final.md` item 5); `docs/bot-api.md` §3 is the endpoint
reference from here on. What follows is the current shape, in brief, then
the original draft below it kept as history rather than instructions —
several of its particulars (a live public vector, an opt-out `confirm`
flag) describe a design that was not, in the end, the one built.

## As shipped

**No public layer, so no gap to patch one into.** §1's premise — a manual
seat's gate is a rubber stamp over its own published vector — does not
apply: `Game.valuations`/`PostedValuation` and every vector are gone
(`trading-final.md` item 1). A manual seat's gate is `hexset.server.
webplay.PendingGate`, installed unconditionally the instant the seat is
claimed (`GameSession.confirm_mode`, called from `api.Tables.create`/
`Table.join`) — there is no `confirm` flag on the wire, for a person at the
page or an LLM's `new_game`/`join` alike, and no other gate a manual seat
can have. As counterparty, it never clears anything on its own; every
candidate the real actor's own gate already priced above zero is recorded,
unexecuted, to `Game.pending`, sorted by that actor's own gain and capped
at 5 (`GameSession.pending_for`) for whichever seat it names.

**Three routes**, all seat-token gated (exact shapes in `docs/bot-api.md`
§3):

- `POST /api/games/<code>/trade` — compose and submit a bundle against any
  counterparty, on the proposer's own turn against anyone or during
  another seat's turn against that seat only; the counterparty's gate must
  price it above zero, the proposer's own gate is never asked.
- `GET /api/games/<code>/trade/acceptable` — the actor's own read-only
  preview: every bundle a bot counterparty's gate already accepts right
  now, grouped by counterparty, no engine mutation. A manual counterparty
  never appears here — its answer comes back asynchronously through its
  own `pending`.
- `POST /api/games/<code>/trade/confirm` / `.../decline` — execute or drop
  one of this seat's own pending offers by index. Declining is final: the
  bot that offered it has already played on by the time this seat ever saw
  it.

A trade executed through either of the two routes above moves cards
outside the automatic event, with no board action to carry it, so it gets
its own entry in the sidebar log and the journal
(`GameSession.execute_manual_trade`, `Journal.manual_trade`) — without
that it would still move the cards live but a resumed game would silently
forget it, since resume rebuilds hands purely from recorded actions.

**The page** wires the trade modal to all three: a counterparty picker and
the give/want cards for "Offer to players," the acceptable-deals list
(its 1-for-1 entries, since that's what the single give/want cards can
hold) so a deal can be picked directly, and a pending-offers panel with
Confirm/Decline that surfaces on its own once `state.pending` is non-empty.
No advertisement panel, no vector chips, no live surplus indicator — there
is no vector to read one from.

**MCP** carries `propose_trade`, `trade_acceptable`, `confirm_trade` and
`decline_trade` as thin wrappers over the same three routes, plus
`get_table` for `pending`; `set_valuation` and the `confirm` argument on
`new_game`/`join` are gone with the vector and the flag.

---

Drafted 2026-09-03 against `HexSet` `main` @ `954b688` (PR #15
"one-event-trading" + three follow-ups through PR #19). Cites the shipped
one-event mechanic only — later dev-hexset registration notes on bundle
candidates beyond one-for-one and interleaved per-action events are not
assumed to exist in this engine, and are flagged where they matter.

**The gap.** `hexset.server.webplay.PostedValuation.accepts` — what a human
or LLM seat's gate is today — is unconditionally `True`
(`webplay.py:184-202`): the engine only asks about a bundle whose public
surplus already favours that seat, so today's "gate" just republishes the
advertisement vector. Nothing at submit time is actually consulted. This
design adds a real gate for those seats, with no change to the action space,
`trade_event`'s once-a-turn trigger, or how bots trade.

## 1. Engine surface

**`Game.pending`**, new field (`game.py`, beside `trades`/`trades_made`,
`game.py:84-86`): `list[Trade]`, reusing `Trade` (`trading.py:70-75`) since a
pending proposal has the same shape as an executed one. Cleared at the start
of every `trade_event` call and again by `end_turn` (`game.py:585-590`,
which already zeroes `trades`/`trades_made`). Not copied by `imagine`
(`game.py:214-236`), for the same reason `gates` is not (`game.py:95-105`):
a hypothetical must not leak a real seat's pending offers.

**`hexset.server.webplay.PendingGate`**, parallel to today's
`PostedValuation` (`webplay.py:180-202`): `accepts(view, received,
counterparty)` always returns `False` and, as its only side effect, appends
`Trade(seat, counterparty, received)` to `game.pending`. `valuation(view)`
is unchanged — a manual seat still advertises through the five numbers.
Because `_best_clearing` asks `gate(me, ...)` before `gate(them, ...)` and
short-circuits on `False` (`trading.py:288`), a proposal is only ever
recorded for a seat sitting as `them` once the *other* seat's own gate has
already said yes to that exact bundle — nothing pending is speculative. This
is server policy (who is asked to confirm), not an engine rule, so it lives
in `hexset.server` like `PostedValuation` does today.

**`Game.execute_trade(proposer, counterparty, bundle)`**, engine (`game.py`
wrapper over a new `trading.execute_trade(game, proposer, counterparty,
bundle)`). `bundle` is signed, positive towards `proposer`. Reuses, in
order: `holds` (`trading.py:124-126`) for both sides' coverage; the
dot-product public-surplus test `_best_clearing` already runs
(`trading.py:274-279`) against `game.valuations`, for **both** seats — a
manual bundle is only legal if the counterparty actually advertised wanting
it; `judged` (`trading.py:108-111`) against `game.gates[counterparty]` and
`game.state(counterparty)` (`game.py:127-148`) for the counterparty's
private gate — **the proposer's own gate is never asked.** Submitting is the
proposer's consent, human or LLM alike. On success it calls `exchange`
(`trading.py:129-140`) and `game.ledger.apply_hand_diff`, the same two calls
`trade_event` makes (`trading.py:227-229`), appends to `game.trades`, and
returns the `Trade`. On failure it raises `ValueError` naming the failed
check, which `ApiError`'s existing `ValueError`→400 path already handles
(`api.py:899-916`).

Accepted asymmetry: `_candidates` (`trading.py:151-171`) still enumerates
one-for-one only, so the *automatic* event never offers more than one card
each way. `execute_trade` bypasses `_candidates`/`_best_clearing` entirely —
it takes the bundle as composed — so a manual trade can already be any
bundle both sides can cover. Widening `_candidates` itself is separate,
out-of-scope engine work.

**Layering.** `Game.pending`/`execute_trade` are `hexset` (engine): any
driver, not only the server, can want a manual-confirm trade.
`PendingGate`/`PostedValuation` are `hexset.server`: policy about which gate
a seat gets, the layer that already owns that choice.

## 2. HTTP API

Already on the wire (`webplay.py:1118-1129`): `valuations` and `trades`,
both table-public. **New:** a `pending` block beside them, but filtered
**per viewer** — only the seat named `a` in a pending `Trade` sees that
entry (`state_view` already filters this way for `trade_ratios`,
`webplay.py:1113-1115`). No separate `GET`: folded into `GET /api/state`,
keeping the module's "one `state`" shape (`api.py:1-9`).

**`POST /api/games/<code>/trade`**, seat-token gated like every mutating
route. Body: `{"counterparty": <seat>, "give": {<resource>: <count>},
"receive": {<resource>: <count>}}` — named amounts, matching the wire's
resource-name convention (`RESOURCE_NAMES`, `webplay.py:62`). The handler
builds the signed bundle and calls `execute_trade(seat, counterparty,
bundle)` under `table.lock`. Errors (400 via `ApiError`, tagged by which
check failed): **not affordable**, **not clearing** (either public surplus
not strictly positive), **wrong turn** (neither seat is `current_player`),
**bot declined** (counterparty's private gate said no).

**Turn-timing rule**, enforced inside `execute_trade` itself so it holds for
every caller: a seat may propose **on its own turn**, to anyone, once
`Phase.MAIN` is open — this needs no `pending` lookup, since the human
composes directly from every seat's public vector. A seat may also propose
**during another seat's turn**, naming only that seat as counterparty —
mirrors the automatic event's "current player only" rule
(`trading.py:203-204`). In practice a human answers a bot's open vector this
way: the bot's one trade event already ran, at the start of its own `MAIN`
(`game.py:559-581`), and whatever it accepted sits in `game.pending` until
that bot's `end_turn`. **Today's engine gives exactly one such window per
turn** (the event fires once, at `enter_main`, not after every later
action); the not-yet-shipped interleaved-event design would widen this
window through the same `pending` field, with no interface change needed.

**Confirm-mode endpoints**: `POST .../trade/confirm` and `.../decline`, body
`{"index": <int into that seat's pending>}` — confirm calls `execute_trade`
with the pending entry's own `(a, b, received)`, decline just drops it, no
cards move. Shared by the UI's "accept" button (§3) and the LLM tools (§4).

## 3. Web UI

A **negotiation panel** sits below today's five-toggle advertisement panel
(`#trading`, `renderTrading()`, `index.html:1612-1655`) rather than
replacing it — `setValuation` (`index.html:1603-1610`) still publishes the
seat's own vector, but toggling it no longer implies acceptance (it never
did, engine-side; the UI now matches).

**Layout.** One block per counterparty with any nonzero vector entry: their
wants (positive) as green card chips, their gives (negative) as red chips,
read from `state.valuations[seat]` (already wired, `index.html:1618,1636`).
Clicking a chip adds/removes one card of that resource to the draft bundle
(give side from their wants, receive side from their gives). A running
total per side sits beside the chips.

**Live indicator**, client-side, no round trip: `mine = dot(valuations[my
seat], bundle)`, `theirs = dot(valuations[counterparty], -bundle)` — the
same formula `_best_clearing` runs server-side (`trading.py:274-279`).
Styled as "clears" only when both are strictly positive. **Affordability**,
also client-side: each `give` count `<= own_hand[resource]`; each `receive`
count bounded by the counterparty's known hand total (exact per-resource
counts aren't visible cross-seat).

**Submit** disabled until clears-and-affordable, then `POST .../trade`
(§2); a rejection shows the server's message via the same `showNotice` path
`setValuation` uses (`index.html:1607`), since hands can move between render
and submit.

**Pending proposals** (§1's `game.pending`) render as one-click cards above
the composer, Accept/Decline hitting the confirm/decline routes — the
common case on a bot's turn, since the human composed nothing.

**By whose turn it is.** On the human's own turn: every seat's panel is
open (proposer role). During a bot's turn: only that bot's panel and its
pending cards are active; other seats' panels stay visible, read-only, with
submit disabled ("not this seat's turn").

## 4. MCP

Four tools in `mcp.py`, thin wrappers exactly like the existing ones
(`_request_ok`, `mcp.py:104-146`):

- **`set_valuation(vector)`** → `PUT /api/games/<code>/valuation`
  (`api.py:807-820`, already implemented, just never exposed as a tool).
- **`get_table()`** → `GET /api/state`, returning `valuations`, `pending`
  and `trades` alongside everything `state()` already carries.
- **`propose_trade(counterparty, give, receive)`** → `POST .../trade`.
- **`confirm_trade(index)` / `decline_trade(index)`** → the confirm-mode
  routes (§2), meaningful only when the seat's gate is `PendingGate`.

**Default gate.** Not `PendingGate` — an LLM seat that never calls
`confirm_trade` shouldn't have to. Default: publish `v` and accept whatever
the engine's own public-surplus test already restricts it to, the same
`judged`-driven acceptance any bot gets — **not** `PostedValuation`'s
always-accept, since that is the gap this design closes. Confirm mode is
opt-in, a flag alongside `opponents`/`name` on `new_game`, swapping that
seat's gate to `PendingGate`.

## 5. Bot side

An embedded bot's `accepts` is already asked synchronously, once per
candidate, inside `_best_clearing` (`trading.py:288`) — one hypothetical
evaluation, not a search, per the trading design's own cost readout.
`execute_trade` asks it the identical way, through the same
`judged`/`game.state` pair, for a human- or LLM-composed bundle instead of
an engine-enumerated one. **No bot code changes**: from a checkpoint's or
`search2`/heximax's own gate, a manual trade and an automatic one look
identical, because the call is.

## 6. Tests, scope, open questions, cost

**Tests** (naming follows `tests/test_trading.py`,
`tests/server/test_webplay.py`, `tests/server/test_api.py`):
- `execute_trade`: clears on both-positive-surplus-plus-accepting-gate;
  rejects on non-coverage, either non-clearing surplus, a declining gate,
  and a seat that is neither proposer nor current player; never calls the
  proposer's own gate (a gate that raises if called, on the proposer's seat,
  still succeeds).
- `PendingGate`: always `False`; records exactly one `Trade` per
  other-side-already-accepted candidate; records nothing when the other
  side declined first (the short-circuit property).
- `game.pending` lifecycle: cleared at each `trade_event` and by `end_turn`;
  not copied by `imagine`.
- API: `POST .../trade` end to end against an embedded bot, both directions
  (human proposing; human responding during the bot's turn); `pending` in
  `GET /api/state` never shown to a seat it doesn't name; confirm/decline
  against a `PendingGate`-seated stub.
- UI: clears/affordability as a pure-function test against fixed vectors
  and hands; panel visibility by whose turn it is.

**Out of scope.** Free-text chat; counter-proposals beyond composing a
fresh bundle from the public vectors; any change to how bots trade, publish
or gate; widening `_candidates` past one-for-one for the automatic event;
human-to-human discovery (§1's short-circuit means a candidate between two
`PendingGate` seats is never auto-recorded — direct `POST .../trade` still
works between two humans, but nothing suggests it the way a bot's accepted
candidate does).

**Open questions for the PI.**
1. Does a submitted bundle also update the proposer's advertisement vector,
   or stay independent of the five toggles?
2. Should a pending proposal expire before the counterparty's turn ends —
   e.g. through that seat's *next* action — given today's one-window-per-
   turn engine behaviour?
3. Is confirm mode the default for LLM seats rather than opt-in?
4. Are `execute_trade`'s two public-surplus checks waivable (a human
   knowingly making a bad trade with another human), or is "both sides'
   own vectors must agree it helps them" a rule manual trades must not
   bypass?

**Cost.** Engine (`Game.pending`, `execute_trade`, tests): 1 day. Server
(`PendingGate`, the three routes, per-viewer filtering): 1–1.5 days. Web UI
(panel, client-side formula, pending cards): 2 days. MCP (four tools,
confirm-mode flag): 0.5 day. Total, one engineer: **~5 days**, excluding
whatever the PI's answers above add.

## PI ratification (2026-09-03, Fable)

Decisions on the open questions:

1. **A submitted bundle does not update the advertisement vector.** The
   advertisement is a standing statement for bots' candidate search; a
   submission is a one-off proposal. Keeping them independent means a human
   who never touches the five controls can still trade by composing.
2. **Pending proposals are a snapshot of the last event.** They are recomputed
   at every trade event and expire at the next event or at the end of the
   proposing bot's turn, whichever comes first. Nothing pending survives a
   change of hands it was computed against.
3. **Confirm mode is opt-in for LLM seats.** An LLM's vector is its statement,
   the same as a bot's; confirm mode exists for seats that want a per-trade
   veto and is enabled per seat at seat-up.
4. **The counterparty's public-surplus check is a hard rule; the proposer's is
   waived.** A submission is the proposer's consent, so its own vector (which
   may be zero or unpublished) is not consulted; the counterparty's published
   vector must say the exchange is good for it, then its private gate decides.
   Manual trades never bypass the counterparty's vector.

Sequencing: implement after HexSet's bundle and event-timing PRs land (same
engine files: `Game.pending`, `execute_trade`). The interleaved-event design
noted above as "not yet implemented here" lands in those PRs first; this
interface then sees several windows per bot turn as designed.

**2026-09-03, default correction.** Decision 3 above says confirm mode is
opt-in *for LLM seats*; the implementation had shipped it opt-in for every
seat, human included, so `POST /api/games`/`POST /api/join` — the web page's
own seat-up — left a human at `PostedValuation` unless it explicitly asked
for `PendingGate`. That let a bot's trade event clear against a human who
never confirmed anything, which decision 3's own reasoning does not license:
an LLM's vector is its statement because *it* chose confirm mode's absence,
but a human at the web page never sees a `confirm` toggle at all, so nothing
it did could be read as that choice. Fixed by splitting the default: `POST
/api/games`/`POST /api/join` now install `PendingGate` when a request omits
`confirm` — a human seat's gate is the explicit submit, not an advertised
vector that clears itself, and `confirm: false` still opts back out to
auto-accept — while `hexset.server.mcp`'s `new_game`/`join` tools send
`confirm` explicitly on every call, so an LLM seat's own default stays the
opt-in one decision 3 describes.

**2026-09-03, human trading surfaces withheld from the page for now (owner),
API retained.** "No trading for humans at this point. We need to build back
up gradually." The browser page carries none of §3: no advertisement
controls, no negotiation panel, no pending-offer cards, and it calls no
trading route. Everything in §1, §2, §4 and §5 stands — `PUT
/api/games/<code>/valuation`, `POST .../trade`, `.../trade/confirm`,
`.../trade/decline` and the MCP tools all answer as documented, and an LLM
seat is unaffected. A person's seat is now gated at seat-up rather than at
its first publish (`GameSession.confirm_mode`): a `PendingGate` over
`NO_VALUATION`, so the seat advertises nothing and clears nothing, and a
zero vector is dropped at `_best_clearing`'s ranking before any gate is
asked — a human is simply not a counterparty. Bots go on trading with each
other through the same engine event. §3 comes back when the page does.
