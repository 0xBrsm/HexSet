# Engine divergence audit — `hexset_ui` vs `hexset` (dev-hexset)

> **2026-09-02 (later), rename note:** this repo (`HexSet-UI`) was renamed
> `HexSet`, and the private training repo referenced throughout as
> `dev-hexset` was renamed `dev-HexNet`. Left unrewritten below — the names
> here are as they were at the time of the audit.

2026-09-02, written by the agent that collapsed the duplication on
`feat/engine-from-hexset` (branched from `pr-2` = `feat/external` @ `dbbfa62`).

`dev:` = `0xBrsm/dev-hexset` `src/hexset/` at branch `feat/heximax` @ `810dec7`
(package `hexset` 0.13.0), which is the tree every line reference below was
taken against and the tree the diffs were classified from. `ui:` = this repo at
the branch point `dbbfa62`.

dev moved while this branch was being written (`feat/heximax` is at `b1763e6`,
the catanatron-adapter merge, as of this commit); the suite here was re-run
green against that head, and none of its commits touch the engine modules this
document is about. The line numbers may nonetheless have shifted by a few —
read them as "the tree at `810dec7`".

This repo used to carry its own copy of the engine — `actions`, `board/`,
`cards`, `devcards`, `economy`, `game`, `ledger`, `mcts`, `roads`, `robber`,
`search2`, `state`, `trading`, `victory`, `encoding`, `record`. That copy is
deleted on this branch; `hexset_ui` now depends on the `hexset` distribution
and keeps only the interface layer. This document is the record of what was
in the copy that is not in `hexset`, so nothing is lost silently.

Every hunk is classified:

* **(a) pure rename/prune** — the UI copy dropped a helper it never used, or
  reworded a comment. Nothing behavioural. Resolved by deletion.
* **(b) rules difference** — the two engines would play differently. **Not
  settled here.** dev-hexset's behaviour is what this branch now runs (that is
  the mechanical consequence of consuming the package); each one is listed
  below with a recommendation for the PI, who owns the decision.
* **(c) interface-only addition** — the UI added something to an engine file
  that is really about the gym. Moved into the interface layer.

## The tension the PI has to settle

On 2026-08-29 the owner declared `hexset-ui`'s engine the **rules reference**
(it is what humans actually played against, and it had the piece supply that
dev-hexset lacked). dev-hexset's `arena`/`duel` record is the **training
referent** — every ladder number, every gate, every duel JSON under
`runs/eval/` was produced by `dev:src/hexset/`. Those two claims now point at
different code in four places (B1–B4 below).

This branch cannot honour both, and consuming the package resolves them all in
dev's favour by construction. **That is a mechanical consequence, not a
verdict.** If the PI prefers the UI reading on any of B1–B4, the fix is a
change in dev-hexset, not a re-fork here.

---

## (b) Rules-level findings — for the PI

### B1. The `offered` re-proposal filter — dev has it, the UI does not

* dev: `Game.offered: set[tuple[Bundle, Bundle]]` (`dev:game.py:86`), added on
  propose (`:450`), cleared on `end_turn` (`:501`), copied by `imagine`
  (`:140`), and acted on by the legal-action *sample*
  (`dev:actions.py:311-312`): a `(give, want)` pair already put to the table
  this turn is not offered again.
* ui: no `offered` field, no skip (`ui:actions.py:290`).
* Effect: on dev, a bot's `PROPOSE_TRADE` sample shrinks as a turn goes on and
  it cannot burn its offer budget re-asking a bundle the table just declined.
  On the UI it can. Re-proposal stays *legal* on both — this is a sample, not
  a rule — so no client is refused a move either way.
* **Recommendation: dev is right, keep it.** It is strictly a better sample,
  it costs no legality, and every checkpoint under `models/` that was trained
  in dev was trained with it. It is also invisible to humans and to external
  bots, both of which get `fair_legal_actions` rather than the engine sample.
* Risk of adopting it: none identified. `offered` is engine-private and never
  reaches the wire.

### B2. Who gets asked first — dev shuffles, the UI goes clockwise

* dev (`dev:game.py:461-463`): with no `ask`, the eligible responders are put
  in a **random permutation drawn from `game.rng`**. The commit that did this
  (2026-08-29) is documented at `dev:game.py:423-448` and in the trading design
  note: an offer stops at the first taker, so the first-asked has first
  refusal, and clockwise-from-proposer hands that permanently to the next seat
  — measured as a +0.35 VP "seat geometry" effect in a 2v2 duel where the two
  copies sit together.
* ui (`ui:game.py:479-485`): no shuffle; `trading.responders` order, which is
  clockwise from the proposer (`ui:trading.py:80-84`).
* Effect: different games from the same seed, and a real (measured) advantage
  redistribution.
* **Recommendation: dev is right.** The measurement exists, it is in dev's
  duel record, and the UI's docstring argument ("over a game the advantage
  rotates with the proposer") is exactly the argument dev's own comment says
  was tested and found false *within a lineup*.
* Consequence for this repo, and it is user-visible: **a seeded game replays
  differently.** Journals written before this branch replay through dev's
  `propose_trade`, which draws from `game.rng` — the dice sequence after the
  first offer diverges. Journal *replay* is unaffected (it replays recorded
  actions, not dice), but a "same seed, same game" expectation is broken. No
  API shape changes.

### B3. MCTS hidden-draw resampling — dev has it, the UI does not

* dev `mcts.py` has `draws_hidden` (`dev:mcts.py:130`), `sampled_children`
  (`:152`) and `_drawn` (`:194`): an action that draws a hidden card (a dev
  card off the deck, a steal) is expanded as a **chance node over the possible
  draws**, not as one deterministic child.
* ui `mcts.py` special-cases `ROLL` only; a dev-card buy or a steal is
  expanded as if its outcome were known.
* Effect: any `search=mcts` checkpoint served embedded searches a tree that
  believes it knows what it drew. dev's version is the one every `mcts:` duel
  in `runs/eval/` was measured with.
* **Recommendation: dev is right**, and this is the one (b) whose adoption
  *improves* the served product. dev's `mcts` also depends on
  `hexset.rewards.relative_points` and `hexset.bots.STANCES`, both of which
  come in with the package.

### B4. The trade mask: omniscient vs honest — the two repos disagree about the *engine*

* dev `actions._offer_actions` (`dev:actions.py:296`) computes
  `wanted_available[r] = any(opponent holds r)` and skips a `want` no opponent
  can cover. The UI's copy does the same (`ui:actions.py:277`). **The engines
  agree.**
* Where they diverge is who is allowed to see it. The UI added
  `webplay._proposable_options`/`fair_legal_actions`
  (`ui:webplay.py:210,248`) — the honest own-hand-only list — and serves it to
  humans, to MCP clients and to `GET /api/record`, while embedded bots kept
  calling the engine sample through `spawn_bot` → `NetworkBot.choose` →
  `actions.options_for`.
* This is PR #2 defect 4, and it is **fixed on this branch**: every seat now
  gets `fair_legal_actions`, embedded bots included (see "Defect 4" below).
* **What is left for the PI**, and it is a real training/serving asymmetry, not
  a bug in either repo: dev's *training* record
  (`dev:onnx_record.record_from_game`, which calls `legal_actions(game)`
  itself) is built over the omniscient sample. A checkpoint trained in dev and
  served here now sees `want` slots enabled that it never saw enabled in
  training. The honest direction is the right one — the alternative leaks a
  specific opponent's hand composition to a human — but the price has not been
  measured. **Recommendation: land PI review §1.3 in dev (drop
  `wanted_available` from `_offer_actions`) and duel before/after on
  `trade-obs-2400`.** Until then this branch is honest-and-unmeasured, which is
  strictly better than PR #2's honest-for-some.

### B5. `hexset_ui.encoding` is frozen at contract 1 and cannot be replaced by `hexset.encoding`

Measured, not inferred: same board, same seed, `encode(game, 0)` gives

| tensor | `hexset` | `hexset_ui` |
|---|---|---|
| `hexes` | (19, 11) | (19, 11) — identical |
| `vertices` | (54, 14) | (54, 14) — identical |
| `edges` | (72, 5) | (72, 5) — identical |
| `globals` | **(86,)** | **(50,)** |

dev widened the global block with a live-offer section and a per-seat ledger
section; the UI froze the narrow layout deliberately (`ui:encoding.py:15-24`)
because every `contract=1` file under `models/` was trained against it.

**This is not resolvable by deletion.** Swapping in `hexset.encoding` would
feed a 86-wide `globals` to a graph traced for 50 and either crash or, worse,
run on garbage. On this branch `encoding.py` is therefore **kept**, renamed to
`hexset_ui/encoding_v1.py`, and documented as what it is: a frozen serving
artifact for legacy contract-1 checkpoints, not a copy of the engine. It has
no reverse dependency — nothing in `hexset` reads it and nothing in
`hexset_ui` but `onnxbot`'s contract-1 path does.

**Recommendation to the PI:** decide whether contract-1 checkpoints are still
worth serving. If they are not, `encoding_v1.py` and `OnnxPolicy` both go and
the last engine-shaped file leaves this repo. If they are, it stays frozen
forever, which is what `contract` metadata is for.

**2026-09-02, post-recommendation note:** the owner dropped contract 1.
`encoding_v1.py` and `OnnxPolicy` are deleted; `onnxbot._load_cached` now
refuses `contract=1` (and the absent-key default that meant it) by the same
path as any other unsupported contract, naming what it found and that this
server serves `2, 3, 4`. `models/` in this working tree held nothing but
`.gitkeep`, so no checkpoint was lost; the g4 deployment share
(`//10.10.10.10/stacks/hexset-ui/models/`) was not reachable from this
environment and was not checked — if a contract-1-only `.onnx` is sitting
there, it will now be refused at load rather than served, and is a candidate
for removal from that share. `constants.LEGACY_CONTRACTS` is gone along with
it; `RECORD_CONTRACTS` (`2, 3, 4`) is the whole table now.

### B6. `hexset_ui.record` cannot yet be replaced by `hexset.onnx_record` — dev change required

`dev:onnx_record.py` is the canonical definition of the ONNX record
(`RECORD_FIELDS`, `record_from_game`) — but **it imports torch at module
scope** (`dev:onnx_record.py:48-50`, plus `from .policy import pair_mask`,
and `policy.py` imports torch and `model.HexNet`). So does `export_onnx.py`,
which owns `_shapes` and `_CONTRACT_VERSION`.

This repo ships an onnxruntime-only image. It cannot take a torch dependency
to reach a function that builds numpy arrays.

`ui:record.py` is therefore the **one remaining engine-adjacent duplicate on
this branch**, and it is marked as such in its own docstring. It is now
guarded by `tests/test_record_contract.py`, which asserts the UI's field names
and per-field shapes against `hexset.onnx_record.RECORD_FIELDS` and
`hexset.export_onnx._shapes` — skipped when torch is absent, but a real
cross-repo check on any box that has it.

**This is engine change request R1 (below).** Once dev splits the torch-free
half out, `ui:record.py` deletes.

---

## (c) Interface-only additions the UI had made to engine files

These were real code the UI needed; none of them is a rules statement. All are
now in the interface layer.

| what | was | now |
|---|---|---|
| `is_legal(game, action, options)` — `PROPOSE_TRADE` checked via `can_propose` rather than sample membership | `ui:actions.py:302-318` | `hexset_ui/rules.py` |
| `Stuck`, `options_for(game)` | `ui:actions.py:406-419` | `hexset.play.Stuck`, `hexset.bots.options_for` — dev has both, byte-equivalent; the UI copies are deleted |
| `Game.locked`, `start(first=)`, `_advance_setup`, `_next_unlocked`, `lock_seat` — the per-seat setup lock | `ui:game.py:71-75, 88-107, 161-206, 527` | `hexset_ui/seating.py` (see the caveat under R2) |

## (a) Pure prune/rename — resolved by deletion, nothing lost

Everything the UI copy dropped relative to dev, confirmed unused by any
`hexset_ui` module (grep over the interface layer, all absent):

| module | dropped from the UI copy |
|---|---|
| `state.py` | `gold_claims` (`dev:state.py:212-226`); `Terrain` import |
| `robber.py` | `random_discard` (`dev:robber.py:81-87`) |
| `economy.py` | `BANK_TRADE_RATIO`, `total_in_play`, `expected_total` (`dev:economy.py:115-120`) |
| `devcards.py` | `play_road_building` (`dev:devcards.py:73-89`) |
| `game.py` | `submit_discard` (`dev:game.py:271-280`) |
| `actions.py` | `space_for` (`dev:actions.py:184`), `legal_mask` (`:408`); `victim_of` made private as `_victim` |
| `board/__init__.py` | the whole re-export block; `board/maps.py` (`BASE_LAYOUT`/`MINI_LAYOUT` inlined into `coords.py` instead), `LAYOUTS`, `islands`, `ORIGIN`, `neighbors`, `translate` |
| `mcts.py` | `visit_policy` (`dev:mcts.py:606`) — and B3's resampling, which is *not* a prune |
| every file | the `# SPDX-License-Identifier: GPL-3.0-only` header |
| `state.py` `MAX_ROADS/SETTLEMENTS/CITIES` | identical values; only the provenance comment differs. Piece supply is **on** in both. |
| `ledger.py` | **byte-identical** to dev but for a four-line provenance paragraph (`ui:ledger.py:5-8`). The PR #2 port was faithful. |

One consequence worth naming: `ui:board/coords.py` defined `BASE_LAYOUT` and
`MINI_LAYOUT`; in `hexset` they live in `hexset.board.maps` and are re-exported
from `hexset.board`. `tests/test_webplay.py` imported them from
`hexset_ui.board.coords`; it now imports from `hexset.board`.

### Licensing

`hexset` is `GPL-3.0-only`; this repo is `AGPL-3.0`. AGPLv3 §13 permits the
combination, and the resulting work is distributed under AGPLv3 — which is
what the Docker image already does. Depending on the package rather than
carrying a header-stripped copy makes the provenance correct rather than
merely compatible.

> **2026-09-02 note:** superseded — HexSet relicensed to `GPL-3.0-only` as
> one distribution; the AGPLv3 §13 analysis above no longer applies. See
> `LICENSE` and `NOTICE.md`.

---

## Engine changes needed from dev-hexset

Filed here rather than made: this branch does not touch `/workspaces/dev-hexset`.

**R1 (blocking the last duplicate). Split the torch-free half out of
`onnx_record.py`.** `RECORD_FIELDS`, `record_from_game`, `_port_code`,
`action_mask`/`pair_index`/`pair_mask` (currently reached via `policy.py`) and
`export_onnx._shapes`/`_CONTRACT_VERSION` are all pure numpy. Moving them to,
say, `hexset/record_fields.py` (imported by `onnx_record.py` for the torch
encoder, and by `export_onnx.py` for the shapes) lets this repo delete
`hexset_ui/record.py` and consume the contract definition instead of mirroring
it. Note `hexset.record` is already taken by the arena's game record, so the
name needs choosing.

**R1b.** `record_from_game(game, perspective, space)` calls
`legal_actions(game)` itself. This repo must pass the *honest* option list
(B4). Give it an `options: Sequence[Action] | None = None` parameter that
defaults to today's behaviour.

**R2 (correctness, not just tidiness). Upstream the seat lock into
`hexset.game`.** `Game.locked: frozenset[int]`, `start(..., first=)`,
`_advance_setup`, `_next_unlocked`, `lock_seat`, and `imagine` copying
`locked`. dev's own PI review already asks for this (§E6a) and notes it is
behaviour-preserving at `first=0, locked=∅`.

Until it lands, `hexset_ui/seating.py` implements the lock as a *post-apply
correction* on the live game only: after every applied action the session
advances the setup snake or the turn rotation past a locked seat. This is
correct for the game actually being played, and it is what every wire surface
reports. **It is not correct inside a bot's search:** `hexset.game.imagine`
does not carry `locked`, so an embedded bot searching forward will simulate
turns for a retired seat as if it still played (it holds nothing and builds
nothing, so it is a wasted ply rather than a wrong one, but it is a
divergence). PR #2's own copy did not have this problem because `locked` lived
on the dataclass. **This is a small, real regression on this branch, and R2
removes it.** `tests/test_setup_lock.py` pins the live-game behaviour, which
is unchanged.

**R3 (already in the PI review, restated because this branch wants it).** A
public `dev_played[players][NUM_DEV_CARDS]` record field, and `first` /
`setup_queue` on the wire, so a peer client can reconstruct a `Game` without
privilege. Not needed for anything on this branch; needed for heximax as a
peer.

**R4.** `hexset.arena.PRESETS["search2"]` has `placement=False` and
`max_offers=None`, while this repo's `search2()` factory built the same bot
with `Config.max_offers` (default 1). This branch uses
`replace(PRESETS[...], max_offers=config.max_offers)`, which is exact — no
change needed in dev, recorded so the difference is not rediscovered.

---

## API-shape changes this branch makes, and why

The HTTP/MCP/Web shapes documented in `README.md` and `docs/bot-api.md` are
otherwise unchanged. Three responses change; all three are documented in
`docs/bot-api.md` as well.

1. **`GET /api/state` → `to_move` during `TRADE_RESPOND`.** Was the head of
   `pending_responders`, i.e. a seat that demonstrably holds the wanted
   resource; every poller including the token-free observer saw it. Now a seat
   that is not itself the current responder is told `to_move = <proposer>`
   (the proposer is public, the offer names them). The responder on the spot,
   and only them, still sees their own seat. `current_player` is filtered the
   same way. See "Defect 3".
2. **`GET /api/record` → `action_mask` / `pair_mask` / `options`.** Unchanged
   in shape; unchanged in content. What changed is that the *embedded* bot now
   computes its mask the same way (defect 4), so the commit-message claim that
   the record is what an in-process bot computes is now true, and pinned by
   `test_record_matches_the_embedded_bots_options`.
3. **`POST /api/undo`.** Now returns `409` with
   `"this action cannot be undone"` in the one case where the ledger cannot be
   rewound, instead of silently corrupting it. In practice the ledger is
   always snapshotted alongside the state, so the refusal path is defensive.
   See "Defect 2".

---

## The five PR #2 defects

Status as landed on this branch; the tests named are new.

| # | defect | status | tests |
|---|---|---|---|
| 1 | contract dispatch re-stamps 29 fields as `"2"`; no real dev export loads | fixed, except for one missing fixture (below) | `tests/test_contract_dispatch.py` (18), `tests/test_record_contract.py` (5, torch-gated) |
| 2 | undo restores state but not the ledger | fixed | `test_ledger.py::test_undo_restores_the_ledger_with_the_state`, `::test_undo_keeps_the_ledger_honest_across_a_played_game` — both verified to fail with the one restoring line removed |
| 3 | `to_move` during `TRADE_RESPOND` reveals who can cover | fixed | `test_webplay.py::test_to_move_does_not_reveal_who_can_cover_an_offer`, `::test_to_move_is_unfiltered_in_every_phase_but_trade_respond` — the first verified to fail with the filter removed |
| 4 | embedded bots get the omniscient mask, everyone else the honest one | fixed for ONNX bots and for the new default bot; **not** for `search2` (below) | `test_api.py::test_the_honest_and_omniscient_trade_samples_really_do_differ_here`, `::test_an_embedded_bot_is_offered_the_same_mask_the_wire_serves`, `::test_record_matches_the_embedded_bots_options` |
| 5 | bot polls refresh `last_seen`; tests leak runner threads | fixed | `test_api.py::test_a_bot_poll_does_not_keep_a_table_alive`, `::test_a_table_only_bots_are_watching_is_evicted`, `::test_the_code_being_looked_up_is_never_evicted_out_from_under_the_request`, `::test_closing_a_registry_stops_every_runner`, plus `conftest.stop_bot_runners` (autouse) |

Test counts: **143 passed / 3 skipped** at the branch point, **177 passed /
4 skipped** here. The extra skip is `test_record_contract.py`, which needs
torch. Live `bot-*` threads after the suite: **67 before, 0 after**.

### Defect 4, precisely — what is fixed and what is not

Fixed: `onnxbot`'s `options_for` is `rules.options_for`, so an embedded
checkpoint's `action_mask`/`pair_mask` are byte-identical to what
`GET /api/record` serves — asserted through `Tables.record` on a position
where the two samples provably differ. And the default embedded opponent is
now `heximax` in its `honest` mode, which does not use the engine's sample at
all: it reads its options off worlds determinized from the public ledger and
replaces `PROPOSE_TRADE` wholesale with its own belief-driven candidates
(`hexset.heximax.Heximax._root_options`). That is the "`can_propose`/
`can_accept` gating from the ledger, as dev does" the PI's recommendation
names.

**Not fixed: `search2`.** `hexset.bots.SearchBot.choose` calls
`hexset.bots.options_for` internally, which is the engine's omniscient
`legal_actions`. Nothing in this repo can intercept that — the call is inside
the bot — so a `search2` seat still searches over a trade sample filtered by
opponents' true hands. It cannot propose an offer nobody can cover, which no
human at the table could know to avoid. The fix is the dev-side one the PI
already recommends (PI review §E4: drop `wanted_available` from
`actions._offer_actions`), and it is filed here as part of B4 rather than
worked around.

### Defect 1, what could not be finished

`tests/fixtures/dev-contract2.onnx` is a genuine dev export and it now loads,
plays 300 plies, and survives a live `TRADE_RESPOND` position. It was also
seated as an embedded bot in a real game over HTTP as a manual check.

**There is still no genuine contract-3 or contract-4 export anywhere to test
against**, and none can be produced on this box: `hexset.export_onnx` requires
torch. The two stubs are real in shape and synthetic in weights. Producing one
real contract-4 export and adding it as a fixture is the one piece of defect 1
this branch could not finish, and it needs a machine with torch — i.e. it is a
dev-hexset-side job.

### Defect 1, in detail — what "a real export" could and could not be tested

`tests/fixtures/dev-contract2.onnx` is a **genuine dev-hexset export**:
`dev:tmp/export/linear2k.onnx`, contract `"2"`, 23 inputs, `search=mcts`,
`exporter_commit 36a8fa03`, exported 2026-08-31. It is the file PR #2 broke.

No genuine contract-`"3"` or `"4"` export exists anywhere on this box, and one
cannot be produced: `hexset.export_onnx` requires torch, which is not
installed and is a ~2 GB dependency this repo will not take. The contract-4
fixture is therefore the PR's own 29-input stub, **re-stamped `"4"`** (the
value dev actually writes) — real in shape, synthetic in weights. Its field
names, shapes and dtypes are pinned against dev's own `_shapes` table by
`tests/test_record_contract.py` wherever torch is available, which is the best
available substitute and is stated here so nobody mistakes it for the real
thing. **Producing one genuine contract-4 export and adding it as a fixture is
the one piece of defect 1 this branch could not finish.**

The fix itself does not depend on the fixtures:

* `onnxbot._load_cached` dispatches `contract in {"2", "3", "4"}` → `V2Policy`,
  absent/`"1"` → `OnnxPolicy`, anything else → a loud `ValueError` naming the
  contract, rather than silently falling through to `OnnxPolicy`.
* `V2Policy._run` builds the feed from `session.get_inputs()` names rather than
  from every key the record has, so a 23-input `"2"`, a 27-input `"3"` and a
  29-input `"4"` graph all load and play. A graph asking for a field the record
  does not have fails loudly with the field named.
* `botclient.RecordBrain.load` accepts `{"2", "3", "4"}` and its refusal
  message names the contract it actually found.


---

## The resulting `hexset_ui` module map

Nothing here is the engine. `hexset` is.

| module | what it is |
|---|---|
| `api.py` | tables, seats, join codes, seat tokens, the `/api/*` dispatch, the model picker, the embedded-bot runners and their lifecycle |
| `web.py` | the HTTP transport over `api.py`, the CLI, the static file route |
| `mcp.py` | a stdio MCP client of the same HTTP routes; standard library only |
| `webplay.py` | `GameSession`: what a seat may see (`state_view`), the human-readable log, undo, the wire encoding of an `Action`, board layout maths |
| `rules.py` | **new.** The one legality authority every seat shares: `fair_legal_actions`, `proposable_options`, `options_for`, `is_legal` |
| `seating.py` | **new.** The setup snake starting at the creator, and retiring a seat nobody claimed |
| `journal.py` | the full per-game JSON-lines record, and replay/resume from it |
| `botclient.py` | a bot as a peer client: `RecordBrain` (external, off `/api/record`), `LocalSearchBrain` (embedded, privileged), `BotRunner`, `LocalTransport`, `HttpTransport` |
| `onnxbot.py` | the model boundary: `OnnxPolicy` (contract 1), `V2Policy` (record contracts), the loader and its contract dispatch, `spawn`/`network_bot`/`searcher` |
| `record.py` | the ONNX record. **A duplicate of `hexset.onnx_record`'s torch-free half** — change request R1 |
| `encoding_v1.py` | the frozen contract-1 feature layout. **Not the engine's encoder** — B5 |
| `modelmeta.py` | reading and clamping a checkpoint's `search`/`simulations`/`wave` metadata |
| `constants.py` | the token header, and the contract-number table. Imports nothing |
| `static/index.html` | the entire frontend |

What it imports from `hexset`: `actions`, `arena`, `board.*`, `bots`, `cards`,
`devcards`, `economy`, `game`, `ledger`, `mcts`, `roads`, `state`, `trading`,
`victory` — and, through `arena.spawn`, `heximax`, `evaluate`, `placement`.
All torch-free; verified by the suite running in a venv with no torch in it.
