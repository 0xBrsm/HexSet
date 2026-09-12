# Changelog

Changes to the HexSet distribution. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Changed

- **Heximax's `k` is a cap on distinct worlds, not a quota of searches.**
  `Heximax.worlds` draws `k` samples from the belief through the new
  `hexset.bots.determinized.distinct_worlds` -- the one place that decides
  when two draws are the same world, which `determinized` now reads too --
  keyed on every other seat's hidden holdings (`holdings_signature`; the
  deck's order is a chance stream and does not distinguish a world), and
  weights each distinct world by its share of the draws in the root average
  (`world_weights`). A
  belief the ledger has pinned to one world -- every duel, and most
  four-seat positions -- is searched once however large `k` is. Before, `k`
  identical worlds shared one leaf budget, and at `k=100` the budget was gone
  after the first few root options: heximax then chose among those, and lost
  eleven of eleven 1v1 games against colonist's bot on 2026-09-12 building
  almost nothing.

- **Road Building is not offered to a seat with no road pieces left.** The
  card places roads; at fifteen on the board there is nothing to place.
  colonist refuses the play outright, and a live seat that took the engine's
  word for it was refused and abandoned three games on 2026-09-12.

### Added

- **Hidden hands and cards in `GameState`.** A state written by a seat rather
  than by the referee can now say "n cards, types unknown": `hands[seat]`,
  `dev_cards[seat]` and `new_dev_cards[seat]` accept a `HiddenHand`/
  `HiddenCards` count and `deck` a `HiddenDeck` length
  (`hexset.state`). Sizes, `len` and the `[:]` copy idiom work; indexing,
  iterating or summing one raises `HiddenRead` instead of returning a
  fabricated number. `observed_by(state, seat)` downgrades a true state to
  what one seat can see. A `View`, `View.sample`, `Heximax`, `HonestEvaluator`,
  `hexset.encoding` and `hexset.onnx_record` all read the same off an observed
  state as off the truth it was observed from; an identity read -- an
  opponent's `holdings`, `card_points`, `victory_points`, `is_over`, or
  `legal_actions` for a seat whose hand is hidden -- raises. A live adapter
  therefore needs no placeholder composition for the hands it cannot see.

- Opt-in sampled-world voting for model/search callbacks, with a decision-local
  cache and frequency-weighted votes (`hexset.bots.determinized`). The default
  key includes the sampled development deck; models that cannot observe it can
  explicitly use `holdings_signature` for greater reuse. `CatanatronBot` exposes
  the same optional vote while preserving its default reference behavior.
  See [the cache contract and examples](docs/determinized-worlds.md).

### Changed

- **Fewer MCP round-trips that decide nothing.** A seat's own trade round
  is settled inside the wait once everyone has answered and there is
  nothing to choose: every answer a pass closes it, exactly one accept as
  offered with no counter executes it. The `undo` tool and `can_undo` are
  gone from MCP (a browser convenience; a settling reply leaves no moment
  for it), as is `get_table` (an alias of `state` since the first pass).
  Replies drop `to_move`, `awaiting_confirm` and `legal_count`, and
  `locked`/`trades`/`winner`/`started` while they say nothing.

- **Uncoverable offers reach the MCP seat again.** The auto-pass added
  earlier today fired on any broadcast offer the hand could not cover;
  a seat can still counter such an offer (and did, successfully), so the
  server now passes only while the hand is empty. Each `pending` entry
  carries `can_accept` so the reader sees at once whether `accept` is
  possible or only `counter`/`pass`.

- **`summary.spots` says whether a port matches.** A spot on a port carries
  `port_matches`: true for a 3:1, or a 2:1 in a resource the vertex itself
  yields -- the join a reader had to make by hand to value the port.

- **MCP replies trim four more repeats.** Per-player `longest_road` and
  `largest_army` are gone (`summary.race` already names the holder and
  this seat's own count); each `summary.afford` build's `legal` key is
  omitted once `legal_actions` itself is empty, rather than reading
  `false` four times for the one reason; `discard_quota` is omitted when
  every seat's is zero; and `trade_ratios` is sent only when it differs
  from the last reply this session was sent, tracked in a new `Session`
  field -- a new or reclaimed seat always gets it once.

- **MCP replies carry `can_offer`.** True exactly when `offer_trade()`
  would be accepted: this seat's own turn, `MAIN`, a legal action to take,
  and no trade round of its own already open. Reading `phase`/`to_move`
  alone can't tell a caller the last two -- an empty seat can sit in MAIN
  with nothing legal, and the wire would silently replace an open round
  rather than refuse a second offer -- so this checks the same
  preconditions `TableApi.open_round` (`api.py`) does.

- **MCP `summary.roads` joins a legal road to the vertex it reaches.**
  Every legal BUILD_ROAD/SETUP_ROAD edge gets `to` -- the endpoint this
  seat's own network doesn't already touch, so the vertex the road
  actually opens up, not the stub it grows from -- that vertex's pips,
  resources and port, and `settle` (a settlement could go there). `board`
  drops its `edges` block: an edge id was opaque without the vertex pair
  anyway, and only the legal ones (which `summary.roads` now names) were
  ever worth resolving.

- **MCP `legal_actions` drops what `summary` already covers.** Once
  `summary.spots` is non-empty, its SETUP_SETTLEMENT/BUILD_SETTLEMENT/
  BUILD_CITY groups leave `legal_actions`; once `summary.robber` is,
  MOVE_ROBBER does too -- the same entries, the same `index`, joined to the
  board either way, so keeping both groups said nothing a reader used
  twice. `spots` also lists every legal placement now, not the best 15:
  the cap (and `spots_omitted`) existed only because the raw group next to
  it repeated the tail for nothing; `legal_count` is unaffected.

- **MCP `discard(cards)` plays a whole seven in one call.** A nine-card
  hand on a seven used to cost four `act(index)` round-trips, one DISCARD
  at a time; `discard({"Wood": 2, "Ore": 1, ...})` posts them all, checked
  up front against `discard_quota` and the hand, and matched one at a time
  against the freshest `legal_actions` after each lands. `act(index)` on a
  single `DISCARD` entry still works.

- **MCP tool text cut by half.** The `tools/list` descriptions and argument
  schemas an LLM seat re-reads on every call went from 13.5 KB to 8.9 KB
  (descriptions 7.2 KB to 3.7 KB). The state reply's shape is now documented
  once, on `state`; every other playing tool says only what differs and
  "reply as state()". `get_table` is described as what it always was, an
  alias of `state`, and its trade-field reference moved into `state`'s. The
  three `get_table()'s ...` index errors now say `state()'s ...`. Transport
  detail (SSE keepalives) and design history left the descriptions; nothing
  a caller acts on did.

- **MCP acting tools reply at the caller's next move.** `new_game`, `join`,
  `resume_game`, `act`, `undo`, `offer_trade`, `answer_trade` and
  `choose_trade` no longer answer the instant after the action: each blocks
  until `your_move` is something other than `wait` -- `act(END_TURN)` comes
  back when the table has played round to the caller, an offer needs its
  answer, or the game is over -- for at most `timeout` seconds (a new
  argument on each; default and cap 600, `0` for the state right now). The
  loop an LLM seat runs is act -> act -> act, with nothing to poll and no
  moment at which it has to decide to wait; `wait_for_turn` remains for a
  reply whose `timeout` ran out. Every `tools/call` is now answered as an
  SSE stream with keepalives, not only `wait_for_turn` (`web.py` has one
  response path instead of two). `state`, `get_table`, `board`, `bots` and
  `leave_game` still reply immediately.

- **MCP replies compacted server-side.** `board` is text: an incidence
  encoding (each hex with its vertices; each vertex with pips, resources,
  port and the vertices a road reaches; edge ids as `id:v0-v1`), 10 KB to
  2.7 KB. `legal_actions` groups and `summary.spots`/`robber` are
  `(keys):row|row` tables instead of one dict per entry, and JSON carries
  no spaces: a `new_game` reply halves, a robber-move state drops by half.
  This is the compaction first written client-side in the Terra bridge
  (hexset-terra) on 2026-09-11, moved into the server so every MCP client
  gets it. Also fixes the MCP layer annotating -- and, since this morning,
  stripping `x`/`y` from -- the table's own `layout` dict in place, which
  the browser draws from; it works on a copy now.

- **MCP forced moves are played inside the settle.** A lone `ROLL` (no
  Knight to choose over it) and a `pass` on every broadcast offer the seat's
  hand cannot cover are played by the server before a reply is handed back
  -- the first live game through the settling tools spent one round-trip
  per bot turn passing on unaffordable offers and one per own turn rolling.
  `board` drops `x`/`y`, `size` and the constant name tables (a quarter of
  the reply); `state`'s description is shortened to fit a client cap that
  truncated it.

- **MCP `models` tool is `bots`; the `model` argument is `identity`.** The
  HTTP API's `model` is a bot engine (`/api/models`, `POST /api/bot`), while
  the MCP argument was a free string naming the caller for `resume_game`;
  one word for both misread. `new_game(identity=...)`,
  `join(code, identity=...)`, `resume_game(code, identity)`; `bots()` lists
  the names `opponents` accepts. No compatibility shim: `model` is now a
  bad argument.

- **MCP replies pruned.** Every state-returning MCP reply drops the wire
  fields a reader never acts on: `version` (no MCP tool takes it),
  `claimed_seats`, `waiting_for` and `trade_wait` (all folded into
  `your_move`/`waiting_on` already), each player's `last_roll` (the table's
  own `last_roll` is the current one), `seats` (its `kind` moves onto each
  `players` entry) and `summary.race.winning_points` (the top-level field).
  `hand`, `known`, `dev_cards` and `bank` come sparse, a missing name
  meaning zero, as the trade dicts always have. A mid-game reply is about
  a sixth smaller. The HTTP API is untouched.

- Size-only readers go through `hexset.economy.hand_size` and
  `hexset.devcards.dev_count` rather than summing a hand or a
  development holding, so the robber, the discard rule, the encoding, the
  record, the arena census and the Catanatron bridge's ledger read on an
  observed state too. `HonestEvaluator`'s belief and evaluation caches key
  on the knower's own hand and cards plus every other seat's *sizes* --
  exactly what the evaluation reads -- instead of every seat's composition.

- **One Heximax policy.** `heximax` now uses the validated adaptive slider
  everywhere, including the server and Gym defaults. `pin_weights=0` or `1`
  holds an endpoint for testing without changing trading participation;
  native duel flags and entrant specs record those settings. The separate
  adaptive/balanced/notrade presets, `mode`/`adaptive` factory arguments and
  `BY_MODE`/`MODES` exports are removed. Fixed custom vectors remain supported.
  See [configuration and migration](docs/heximax.md).

- **Pinned Catanatron to `ecf931181b9a65bb4116a2153fb78c16f1438e00`.**
  The adapter follows two upstream reworks: players now construct as
  `(color, params)` and register through the shared player registry
  (`parse_cli_string`/`register_cli_player` are gone), and each game owns
  its `random.Random` instead of reseeding the global module. `duel.py`
  builds players via `REGISTRY` (`DC:` specs keep their whole
  colon-containing tail), `DevCatanPlayer` declares a `Params` model and
  resets per-game state on the `before` observer hook, and the
  `catanatron-play` bridge file targets `--bot` instead of the removed
  `--code`. The optional speedups follow the new source (updated
  verification digests; the structural `State` copy shares the per-game
  RNG like the native one) and the speed benchmark builds players through
  `build_players`. Reference bots hosted by HexSet use their seeded entrant
  RNG for search; mirrored games and copies share that stream, independently
  of live chance draws and other games.

## 0.50.0

This release changes the MCP tool contract in ways an existing MCP client
will notice (see **Changed** and **Removed**). Under 0.x semantic versioning
a minor bump is the signal for that, and 0.50.0 is already one over 0.49.1,
so the number stands.

### Changed

- **MCP replies group `legal_actions` by type and list the board sparsely.**
  The flat `{type, a, b}` list repeated the type string on every one of
  fifty entries and carried a dead `b: 0` on most; it is now
  `{type: [{index, <operand>}]}` with the operand named (`edge`, `vertex`,
  `hex` and `victim`, or the resource names for bank trades, discards,
  Monopoly and Year of Plenty) and `legal_count` the flat total. `index` is
  unchanged and is what `act` takes. The three dense occupancy arrays
  (`vertex_owner`, `vertex_building`, `edge_owner`, mostly `-1`) are
  replaced by `buildings` (vertex, seat, kind) and `roads` (edge ids per
  seat): roughly neutral in tokens late in a game, cheaper early, and
  readable throughout. **Breaking** for an MCP client reading either old
  shape; the HTTP API's `/api/state` is unchanged.

- **`act(index)` guards against a moved list with `expect`, not `version`.**
  Pass the `legal_actions` entry you chose, with its group key as `type`
  (e.g. `{"type": "BUILD_ROAD", "edge": 17}`; raw `a`/`b` work too), and
  `act` refuses if that index now names something else.
  The old `version` argument compared against the whole table's change
  counter, which bumps on every other seat's move, every trade answer, and
  even a read that fires a pending trade event -- so in a 4-seat game it
  was stale by the time the reply had been read. **Breaking:** `act` no
  longer accepts `version`; a call that passes it fails as bad arguments.

### Removed

- **`version` on `answer_trade` and `choose_trade`.** In a trade round the
  other seats' answers to the *same* offer bump the table version, so a
  seat that passed it could not answer at all -- the one LLM game played
  through these tools had to stop sending it to make trading work. The
  server already refuses an answer that does not match the exact open offer
  (`actor` + bundle) and a choice that does not match a recorded answer
  (`seat` + bundle), which is the only staleness that can hurt either call.
  The HTTP API's own `version` field on those routes is unchanged.

### Added

- **`summary` on every MCP reply that carries state: the derived facts a
  turn turns on, computed once server-side.** `board()` already joined hex
  tokens into per-vertex pips because that arithmetic is what an LLM does
  unreliably at the board's size; the first LLM game through these tools
  showed it redoing the rest of that work by hand every turn. `summary`
  now carries `afford` (per build: whether the hand covers it, what it is
  short, whether `legal_actions` offers it right now), `race` (points,
  `to_win`, the public leader, and for longest road and largest army your
  count, the holder's and the `need` that would take it), and -- only when
  such a move is legal -- `spots` (each settlement/city placement joined to
  its vertex's pips, resources and port, best first, capped at fifteen with
  `spots_omitted`) and `robber` (each hex the robber may go to, its pips,
  whose buildings it hits, and an `act` index per victim). Every entry
  names the `legal_actions` index to play. The state view also gains
  `winning_points`, the rule the game is actually played to, so `to_win`
  is right for a 15-point variant too.

- **`your_move` and `waiting_on` on every MCP reply that carries state.**
  `your_move` is `act`, `discard`, `answer_trade` or `choose_trade` -- the
  tool the table wants from this seat now -- or `wait`, with `waiting_on`
  naming the seats it is waiting for, or `game_over`. The same answer used
  to be spread across `legal_actions`, `pending`, `trade_round.awaiting`,
  `trade_wait` and `to_move`, and the first LLM game through these tools
  spent several calls working out how they relate. `wait_for_turn` returns
  exactly when `your_move` stops being `wait`; the underlying fields are
  unchanged. `new_game` and `join` now answer with the same translated
  shape as every other state reply (they used to come back raw, without
  the named trade dicts or the transcript cursor fields).

- **The transcript is sent incrementally, by default, on every MCP tool
  that answers with game state** (`state`, `act`, `wait_for_turn`,
  `get_table`, and the three trade tools). `log` was otherwise re-rendered
  and resent whole on every single call, so what an LLM seat paid to read
  the table grew with the length of the game and was by a wide margin the
  largest part of each reply. Each MCP session now remembers how many lines
  it has been sent and every reply carries only what is new; `log_from`
  names the index the slice starts at and `log_total` the whole length.
  Nothing has to be passed back for this to happen -- an optional argument
  on seven tools is one an LLM forgets, and the first game through these
  tools showed exactly that. Two overrides: `full_log: true` sends the whole
  transcript (for a reply that went missing -- `log_from` past the lines
  you hold is the tell), and `log_after: <n>` names an explicit line count
  to cut at instead of the session's own. A failed call sends no transcript
  and so does not move the cursor.

  The reply carries one line of overlap on purpose. `render_log` collapses a
  burst of engine steps into one line that it rewrites *in place* as the
  burst grows, so the last line a caller holds is the one line that can
  still change under it -- resending exactly that line is what makes the
  cursor safe to splice, and is why the cursor counts lines rather than
  reusing the version number `after` already carries (the transcript is
  folded from events, and no version maps to a line count).

  A session's first read after `new_game`, `join` or `resume_game` is the
  whole transcript, which is what a fresh seat and a just-reclaimed seat
  both want. The final read of a finished game ignores the cursor too: `state_view` asks for the
  transcript with `omniscient or over` once a game is over, which lifts
  redaction across the whole history at once -- every earlier steal stops
  being "a card" and names what it was -- and those are rewrites of lines the
  caller already holds, too far back for any overlap to cover.


## 0.49.1

### Fixed

- **A trade round you opened yourself went unwatched.** `settle()` stops
  polling when the move belongs to this seat, on the assumption that the
  table cannot change until that seat acts. A trade round breaks the
  assumption: `to_move` stays on the offering seat for the whole round while
  everyone else answers. So `postRound()` broadcast the offer, called
  `settle()`, and `settle()` returned on its first line -- the accepts and
  counters then sat on the server unseen until something else forced a full
  reload. A round with anyone still to answer is now watched like any other
  seat's turn.

## 0.49.0

### Added

- `hexset.catanatron.duel --game-type=colonist-1v1`: a 15 VP /
  discard-over-9 rules variant. Rules travel on `GameState` through copies
  and the Catanatron bridge, so Heximax's win bonus and discard model match
  the table actually being played.

### Changed

- `hexset.catanatron.duel --speedups {basic,cached}`: opt-in accelerations
  for the pinned Catanatron reference bot -- board copying and a shared
  feature cache, checked against the pinned upstream implementation's hash
  before installing.
- Native state evaluation and dynamically scheduled duel games, speeding up
  mixed HexSet/Catanatron throughput runs.
- The Catanatron adapter now shares its board/seat mapping and caches an
  unchanged board's reconstruction across `CatanatronBot` instances working
  the same table.

## 0.48.3

### Fixed

- **The board's setup-turn controls were gated twice, on two different
  things.** `renderBoardButtons` read `awaiting_confirm` directly while the
  banner and the roster row read `to_move`, which is how 0.48.2 shipped with
  the buttons offered to the right seat and the banner announcing the next
  player's turn. `_public_mover` now reports the held seat as the one on
  move, so the page has one notion of whose turn it is again and the extra
  gating is gone.

- `hexset.catanatron.duel`: games were seeded by shard, so a duel's own
  worker count changed which games it played. Seeded by game index instead
  -- a run with 4 workers and one with 8 now play the same games.

- **Playing a Knight lost its board-level cancel.** A recent engine change
  (3034778) made it resolve through the same forced robber phase a seven
  does, and dropped the client-side arm/cancel path that came with the old
  one-step version, without a replacement. It's undoable again, through the
  same `#undo-build` point a build or Road Building already gets, up until
  the robber move actually lands. Undoing a played dev card also used to
  leave `dev_card_played` stuck true even after the card was refunded --
  fixed alongside, since a Knight makes the gap much more likely to bite.
  The trade-acceptance pane now says just "Accepted" instead of redrawing
  the offer's own cards a second time, and its check button is colored only
  for that accept, gray for a counter. The setup-hold banner reads
  "END TURN" rather than "END YOUR TURN".

### Added

- `tests/server/test_page_setup_turn.py`: browser coverage for the setup
  hold -- the banner, the roster highlight, the two buttons offered, and the
  release when the turn is ended. Every defect 0.48.2 shipped was in
  `index.html` and passed the whole Python suite; all three of these fail
  against the unfixed page. It also covers a reopen restoring the hold,
  which nothing else did.

- `tests/server/test_page_robber_undo.py` and
  `tests/server/test_page_trade_acceptance_colors.py`: browser coverage for
  the Knight-undo and trade-pane fixes above, for the same reason the setup
  hold's own coverage exists -- both are `index.html` rendering/interaction
  logic the Python suite can't see.

## 0.48.2

### Fixed

- **A browser seat could not take back a setup placement.** A setup road is
  the one handoff the engine makes on its own -- the snake moves
  `current_player` on as part of applying the placement, where every
  Main-phase handoff waits for that seat's own `END_TURN`. `_apply` drops the
  undo point the instant another seat moves, and since bots became peer
  clients they move in milliseconds, so the button never survived long enough
  to press.

  A browser seat now holds its setup turn open until it ends it, and the
  session offers it an ordinary `END_TURN` to do so -- the board's existing
  End Turn button, in its usual corner. Undo stays reachable for as long as
  the turn is held, which gives setup the same take-back window Main-phase
  builds already had.

  The seat holding its turn open is also the seat the view reports as
  `to_move`, rather than the one the engine's snake advanced to. "Whose move
  is it" has about a dozen askers -- the phase banner, the roster highlight,
  the board's own buttons, every other seat's client deciding whether to act
  -- and answering it once in `_public_mover` is what keeps them agreeing.
  The first cut of this answered it only for the buttons, so the banner and
  the highlighted row both announced the next player's turn while the table
  sat waiting for the seat that still had it.

- **The board's banner said "PLACE SETTLEMENT" at a seat with nothing to
  place.** A seat holding its setup turn open has already placed, and the
  engine's phase has moved on to the *next* seat's `SETUP_SETTLEMENT`, which
  is what the banner was reading. It now names ending the turn, which while
  held is the only thing the seat can do.

  Scoped to seats whose client `kind` is `"web"`, and nothing else:

  - A **bot** seat is never held. A bot decides from `onnx_record`'s
    `action_mask`, which is built from the engine's own action space, so an
    `END_TURN` only the session knows about is not in it -- a held bot seat
    would have no legal move at all, and widening the mask would be a new
    action every checkpoint has to be trained on for a button no bot presses.
  - An **LLM** seat (`kind` `"mcp"`/`"api"`) is never held either. It is a
    manual seat, but it drives itself with nobody watching, so holding it
    only stalls the table.

  This restores `awaiting_confirm`, added in `4f9dbe4` and lost in `82f4bd1`
  when the no-lobby/no-cascade rewrite removed `advance_bots()`. The original
  hold was written as "do not run the cascade driver", so it went out with
  the driver; there is no driver to hold now, so the turn itself is held.

- **A journalled game with a `PLAY_KNIGHT` from before the knight/robber
  split ("play a knight, then move the robber", commit `3034778`, 0.35.0)
  would not resume.** That change made `PLAY_KNIGHT` operand-less and moved
  the robber through a following `MOVE_ROBBER` step instead, but left
  `webplay.GameSession.restore` reading every journalled `PLAY_KNIGHT`
  line's `a`/`b` as if they were still today's shape -- always `0, 0`. A
  file written under the old rule has its knight's actual target hex and
  victim on that same line, and replaying it now checked a bare
  `Action(PLAY_KNIGHT)` against `Action(PLAY_KNIGHT, target, victim)` and
  found them unequal: `ResumeError: PLAY_KNIGHT is not legal in ROLL` (or
  `MAIN`), for a table that had done nothing wrong. Any finished game
  journalled before that split that ever played a knight could not be
  reopened at all.

  `GameSession._apply_knight` now recognises a pre-split line (its
  `MOVE_ROBBER` follow-up is missing, not journalled separately -- see
  `_is_split_knight_play`) and plays it as the two actions this engine
  wants, folding the extra step back out afterward so every
  `undo.back_to`/`note.step` recorded after it in the same file still
  lands where it was written to. A knight that won the game outright plays
  as today's rule says regardless of which shape recorded it -- no robber
  move at all -- rather than the pre-split engine's own behaviour of moving
  it anyway; the position is already decided by then, so nothing downstream
  reads the difference.

  Investigated in response to an operator report of "advance/undo missing
  from setup phase" on three specific tables; a faithful re-replay (each
  table's own `locked` seats applied at the step they actually retired,
  not pre-seeded before setup) showed the setup/lock path was not at fault
  -- all three, and no others found in a full sweep of the journal
  directory, failed on exactly this.

## 0.48.1

### Fixed

- **`webplay.GameSession.round` counted a lap as the seats the board was
  dealt for, not the seats still playing it.** A retired seat (`Game.locked`)
  is skipped by turn rotation and never takes one of the lap's turns, but the
  divisor was the dealt `num_players`, so those skipped turns were counted as
  if somebody had played them. On the shape every 1v1 has here -- a four-seat
  board with two seats locked -- a lap was reported as four turns when only
  two seats were taking them, so each player appeared twice per round and the
  count came out at half the laps actually played. A 69-turn 1v1 ended on
  round 18 rather than 35.

  Display only: `game.turns` was always right, turn rotation was always
  right, and nothing about play or the trained policy's turn feature
  (`hexset.encoding`'s `TURN_SCALE`) reads this. It reaches a viewer through
  the state view's `round` and the journal's per-event `round` field, so
  historical journals keep the numbers they were written with -- replaying
  one now renumbers its rounds, it does not rewrite the file.

  Full tables are unaffected, and are pinned by a test of their own.

## 0.48.0

### Added

- **`hexset.bench.duel --runtime <module>`**, and
  `hexset.clients.netbot.load_runtime` behind it. A `network:`/`mcts:`
  entrant needs a runtime that can open its checkpoint, which needs torch or
  onnxruntime and so cannot live here. Until now the only way to get one
  registered was for the driver to wrap this CLI in a module of its own whose
  whole body was an import (HexN carried a `hexn.duel` for exactly that, and
  nothing else). Naming the module on the command line is that import, moved
  to where the entrant is resolved.

  It reaches the pool as `hexset.arena.compete`'s `worker_initializer`, so it
  runs once in each worker -- and in the calling process at `--workers 1` --
  which is what a wrapping module could not do under spawn or forkserver,
  where a worker starts from a bare interpreter. Passed by name rather than
  as a loader, for the same reason: a bound callable does not survive the
  pickle. A duel that names no runtime imports nothing extra.

## 0.47.0

### Added

- `trade_floor` and `gate_rows` ONNX metadata keys: a checkpoint's clearing
  floor and its trade gate's per-event row bound are now read off the file
  (`hexset.clients.modelmeta.gate_config`) and carried onto the bot by
  `hexset.clients.netbot.bot_for`. Both are properties of the exported value
  head — a floor measured against one checkpoint says nothing about
  another's — so they no longer sit as one constant applied to every
  checkpoint alike. A file declaring neither is read at the previous
  behaviour, `0.0` and `32`. `trade_floor` is clamped to `[0.0, 1.0]` and
  `gate_rows` to `512`; both fall back to their defaults when unreadable,
  the same bargain `simulations` and `wave` already strike.

  Both are read off a checkpoint by name rather than required of it
  (`gate_config_of`), so a loader in another repo that has not adopted them
  still spawns a bot, at the behaviour it had before the keys existed.

### Changed

- **`GET /api/version` answers for the API, not for the build.** It returns
  `{"api": <int>}` — this API's own contract version, `hexset.server.api.
  API_VERSION`, currently `4` — in place of `{"version", "git_commit"}`. The
  route named the API and answered with the distribution's version, which is
  the wrong signal to key compatibility off: `hexset.__version__` moves for
  engine, bot and research work that never touches the wire (0.46.0 removed a
  research interface and left every route alone), so a client watching it saw
  churn it had to ignore and could have missed a break it must not. The
  integer is bumped only when a client written against the previous contract
  can break. Its history, recovered from this file, is recorded on
  `API_VERSION` itself: `1` at 0.14.0, `2` at 0.26.1 (the valuation route and
  the `confirm` flag went), `3` at 0.36.0 (the one-to-one proposal routes
  became the trade round), `4` at 0.46.0 (`search2` stopped being seatable by
  name).

- **`hexset.server.modelmeta` is now `hexset.clients.modelmeta`.** It imports
  no runtime and is no longer server-only: the runtime-neutral
  `hexset.clients.netbot` reads the gate settings from it too. Update the
  import path; nothing else about the module changed.

### Removed

- **`hexset.trading.NETWORK_GATE_ROWS`**: a network gate's cost bound living
  in the engine's trading module. Read `NetworkBot.gate_rows` instead, or
  `hexset.clients.modelmeta.DEFAULT_GATE_ROWS` for the value it held.

- **`hexset.build_info()`.** The git commit it carried only ever mattered as
  research provenance, and that already lives in
  `hexset.experiment.provenance()` alongside the rest of the fingerprint a
  result needs — the dirty flag, dependency versions and VCS revisions, the
  determinism-relevant environment, checkpoint hashes. A consumer that stamped
  `build_info()` into its own run records (HexN's `hexn.run.manifest`) reads
  `provenance()` instead and gets strictly more. `hexset._source.git_value` is
  unchanged and still serves that path.

### Fixed

- A finished game's player roster keeps the height it had while the game was
  live. Going read-only replaced every row's `<select>`/`<input>` with plain
  text carrying neither their padding nor their border, so each row lost about
  4px and a four-seat list collapsed 18px, jumping the board below it.

## 0.46.0

This release changes public APIs and benchmark result semantics. HexSet remains
in the 0.x development series. See the [migration guide](docs/research.md#removed-interfaces).

### Removed

- Search2, greedy, tiered evaluation, their presets, and retired checkpoint
  registration paths. Heximax and Catanatron are the handcrafted baselines.
- The obsolete journal-census parser, external duel backend, network evaluator
  wrappers and unused adapter action-space arguments.

### Changed

- Consolidate legal-action helpers, bot interfaces, benchmark statistics and
  provenance. Duels and profiling use the arena; census aggregation uses totals.
- Correct duel side attribution and census seat exposure. Win-rate denominators
  include unfinished games; uncertainty estimates account for paired boards.
- Include settings, checkpoint hashes and raw outcomes in benchmark JSON.
  Replay records remain the trajectory format; JSON persistence uses the standard library.
- Support custom runtime initialization in spawned workers and reject incomplete
  odd-player antithetic rotations.
- Validate complete lane action batches before advancing games, correct the
  encoder perspective during discards and seed network trade gates.
- Update research and training documentation, add a runnable bot example and
  CI coverage for core and optional integrations, and consolidate test setup.
- Install Docker dependencies from project metadata and load ONNX Runtime only
  where model inference needs it.

### Fixed

- Finished games are read-only in the browser for seated players as well as spectators.

## 0.45.1

### Fixed

- `hexset.bench.versus.PolicyPolicy.gate` seats the `NetworkBot` it builds at the game and seat it is installed for. Unseated, the gate priced every candidate at -1.0 and a network duellist never traded.

## 0.45.0

### Added

- **A lane environment hands back the games it played.** `hexset.gym.lanes.LaneEnv(..., records=True)` gives every finished `Episode` a `hexset.record.Record` of its own game — the chance stream recorded, so it replays without the seed — built on the same `hexset.record.Tape` `hexset.arena` records a tournament game with. A driver that kept a game to re-search or re-encode it now replays through `hexset.record` alone, instead of rebuilding the position from `(seed, index)` and its own action stream. Off by default: an environment nobody asked for records from deals the plain chance source it always did. `hexset.bench.versus.compete_batched(..., episodes=True, records=True)` passes it through.
- `hexset.record.replay_to(record, ply)` — the live `Game` after `ply` recorded actions, trades applied as recorded. A ply the record does not hold is refused rather than clamped.
- `hexset.bench.versus.PolicyPolicy` seats any `hexset.clients.policy.Policy` in a batched duel, gated by the checkpoint's own trade gate when the checkpoint is passed too. A policy's `gate` may be written `gate(game, seat, max_trades)` to be handed the run's trade budget, so a duel at `max_trades=0` no longer seats a trading gate into a no-trade evaluation. `Verdict.metrics()` reports `seconds`.

## 0.44.3

### Fixed

- `hexset.trading.trade_event` ends at the first revisited position instead of asserting. The assertion assumed a gate is a strict function of the position; the network gate scores each candidate in a world drawn from its belief (0.44.0), and heximax samples worlds too, so a trade that changes the ledger changes the next draw and a reverse exchange can price positive without anything being broken. Self-play against a network checkpoint tripped it on hexn's distillation test.
- **A reopened table puts every seat back as whoever held it** (`journal.players`, `api.reopened_seats`), instead of rebuilding every non-bot seat as empty. Before this, a restart handed your still-in-progress seat to whoever opened the link next, and you got it back only by the luck of `POST /api/join` picking it; now `join` finds no candidate at all and `POST /api/reclaim` is the way back into your own seat. The browser (`index.html`'s `resumeGame`) actually calls `reclaim` now — it never did — and stops attempting to join a game that is already over.
- A game *abandoned* unfinished (New Game, or a 24h eviction) stays gone. A closing line alone does not say whether a game was played out or walked away from, and inferring "finished" from one rebuilt an abandoned game as a mutable table with no journal to write to and no bots to answer, silently dropping every action taken at it.
- `POST /api/reclaim` succeeds at a game that is over: it is a handshake about who you are, not permission to act, and the token it mints is refused by the same gate that refuses no token at all. "The game is already over" is now one 409 from one place (`api.require_live`) rather than a 409 from some routes and a 400 from others.

## 0.44.2

### Fixed

- A finished game can still be opened after a restart: `GET /api/table/<code>` finds a closed game's journal (`journal.most_recent`) instead of 404-ing once the process that held it in memory is gone, and reopens it read-only.

## 0.44.1

### Fixed

- The network gate's world for a candidate is seeded by the ask -- the information set, the counterparty and the bundle, salted once per bot -- so `gains_many` and `estimate_many` about the same candidate read the same world and a round's own gain and its estimate of the other side are one judgement; the gate is a pure function of what it was asked.

## 0.44.0

### Changed

- **The network gate's continuation is the mover's, in a paired belief world.** `hexset.clients.netbot.NetworkBot` scores each candidate in one world drawn from this seat's own belief (`View.sample`, the counterparty certified to hold what the candidate says it gives), exchanged and not, and rolls out whoever is to move -- this seat as the actor, the actor when this seat responds -- so a responder prices what the actor will do with the cards. A counter that hands a seat the card completing its winning build reads as that win: zero for the responder's own gain or below, the win itself for the estimate, a pass from `default_respond`. Two worlds per candidate; the live table is never touched.

## 0.43.0

### Changed

- **A network checkpoint's trade gate prices continuations, not hands.** `hexset.clients.netbot.NetworkBot` values the live position and every post-trade position *after the seat's own best play from it*: the policy's greedy actions through the rest of the turn on an imagined copy (fresh chance, deck reshuffled), a finished game reading as its one-hot winner, at most `CONTINUATION_PLIES` (8) actions, one batched forward a ply. Two raw value estimates of nearly identical hands differ by the head's noise, and the old gate offered that noise -- at a won position it priced giving the winning card away at +0.006. Now a won position reads 1.0 before and after any trade that keeps the build and lower after one that does not; the counterparty's row reads the actor's win either way, so nobody offers or counters into it. On the g4 game of 2026-09-08 19:47Z, round 19, clio-r3's gate goes from one candidate clearing to none. No per-model floor is needed for this; the network gate's `trade_floor` stays `0.0`.

## 0.42.2

### Fixed

- `hexset.record.from_journal` folds a round's executed trades (`Journal.manual_trade`, a journal step with no action) into the preceding action's trades and maps an undo's `back_to` through journal step numbers. A served game's record used to drop every accepted offer and diverge from the table's hands at the first one; `Journal.note` steps are skipped.

## 0.42.1

### Fixed

- The win banner numbers the winner the way the roster and the log do (`playerNumber`), so a table with a closed seat reads Player 1, 2, 3 everywhere.
- A finished game's own seats see the full log the spectator link shows: the reveal that already opened hands and cards at game over now opens the transcript too.
- `POST /api/bot` refuses once the game is over, as `POST /api/leave` already did: a finished game's record is not rewritten by re-picking who played it.

## 0.42.0

### Added

- `hexset.gym.lanes.LaneEnv` takes a board *law* -- `board=` may be `board(index) -> Board | None` -- so a paired evaluation gives games `2k` and `2k+1` one board while each keeps its own dice; `LaneEnv.cohort(games)` re-arms a bounded environment for a fresh cohort without rebuilding it; `hexset.actions.mask_of(space, options)` is the legality mask over options already enumerated (`legal_mask` is now this over `legal_actions`).
- `hexset.clients.netbot.NetworkBot` gains an optional `seat`: a bot seated at one seat refuses, loudly, to answer a view of another (the guard the training side's trader carried); `seat_at(game)` seats a bot's gate without asking it to move, the public form of what `GatedSearch` and a collector used to do by poking `_seated`; `register_entrants(loader, evaluator_max_trades=...)` sets the trade switch of every "network" evaluator the runtime provides (`0` for a handcrafted search over a learned value).
- **`hexset.bench.versus.compete_batched`: `arena.compete`'s verdict over batched policies.** `compete_batched(policies, games, caster=..., players=..., seed=..., lanes=..., action_cap=..., max_trades=..., antithetic=True, learner=0)` plays a duel through `hexset.gym.lanes.LaneEnv` — one call per tick across every lane, which is what a network-backed policy needs and what `compete`'s one-position-at-a-time `Bot` cannot give it — under `compete`'s own pairing law: the two halves of an antithetic pair are the same board with the two sides' seats exchanged, so the seat term cancels per board. Returns a `Verdict` carrying the paired victory-point margin with a normal interval (`arena.mean_interval`), wins with `arena.wilson`'s interval, `arena.Standing`s, boards counted both ways only, and `truncated`/`exhausted` kept apart; `Verdict.metrics()` is the flat dict a training run logs. `BatchPolicy` is the protocol (`act(requests)` over `LaneEnv.Request`s) and `BotPolicy` puts any `hexset.bots.Bot` behind it, gate included.
- `hexset.casting.rotating(lineup)` and `hexset.casting.swapped(caster, a=0, b=1)`: `arena.compete`'s lineup rotation as a caster, and the complementary cast that exchanges two policy ids. Exchanging the ids is the complement of *any* cast, where the tournament's own `seats // 2` seat shift is only the complement of its adjacent lineup — `swapped(alternating(n))` is `alternating(n, flip=True)`.

## 0.41.0

### Added

- `hexset.encoding.to_frame`/`from_frame`: the seat-frame rotation (perspective seat to slot 0, others in turn order behind it) as two pure, inverse functions, replacing the by-hand `(seat ± perspective) % players` arithmetic `_encode_globals`/`_ledger_parts` repeated inline. One named seam for a convention a training repo previously had to write out three times over.
- `hexset.encoding.global_columns(players)`: a named `dict[str, slice]` over `global_features(players)`'s fourteen blocks (`own_hand` through `ledger`), built from the same widths `global_features` sums, so a caller reads a block by name instead of counting offsets by hand off those constants the way a checkpoint migration across encodings used to have to.
- `hexset.casting`: pure casting laws over a game index -- `alternating(players, flip=False)` (the duel caster, seat-pair swapping by parity), `paired(caster)` (games `2k`/`2k+1` share `caster(k)`), `league_rotation(learners, players, order=None)` (rotate learner ids over seats, `order` reseating who sits next to whom) -- so a training loop's seating law lives beside `arena`'s own rotation rather than being re-derived per caller.
- **`hexset.gym.lanes`: a lockstep multi-game environment.** `LaneEnv(players, seed, lanes, deal=..., action_cap=..., board=..., caster=..., bots=..., gates=...)` holds `lanes` games in flight and steps every one of them once per tick: `requests()` hands out one `Request` per live lane (lane, game index, seat, policy id, stream step, legal `options`, the live `Game` and the seat's information-set `view`), `step(actions)` applies one `Action` per request and returns the games that ended as `Episode`s -- decisions demultiplexed by seat with their index in the lane's action stream, the cleared-trade census `(step, a, b, received)`, and an `Outcome` carrying winner, per-seat terminal points, turns, actions, `truncated` and the trade count. Finished lanes refill on the spot. A `caster(index)` decides which policy id holds each seat as a pure function of the game index; ids listed in `bots` are played by a `hexset.bots` bot (one per board, `BoardBots`), and every seat -- bot or caller -- is seated as its own trade gate on `game.gates`, so the engine's trade event runs and a learner played through the gym can trade at last (`hexset.gym.aec` seats none). Numpy-free and gym-package-free: `import hexset.gym.lanes` needs neither `pettingzoo` nor `gymnasium`, which are now required only when `HexSetAEC`/`HexSetEnv` are actually reached.
- **One game law.** `hexset.arena.deal_game(seed, index, players, board=..., chance=...)`, with `deal_board`, `board_key` and `game_key`, is where a run's `index`-th game comes from; `arena._play_one` and `LaneEnv` both call it, so a tournament and a collector playing index `i` of seed `s` play the same game rather than two games derived alike. `hexset.arena.play_game(game, bots, action_cap=...)` is `play`'s loop over a game somebody else dealt.

### Changed

- **A checkpoint runtime is a `Policy`, and gets the bot, the evaluator, the searcher and the trade gate for free.** `hexset.clients.policy` states that protocol -- `act_rows`, `value_rows`, `score_rows` over live `(game, seat)` positions rather than encoded records, values in board-seat order -- and `hexset.clients.netbot` holds `NetworkBot` (with its two-sided trade gate), `NetworkEvaluator`, `LeafEvaluator`, `GatedSearch`, the `bot_for`/`evaluator_for`/`searcher_for` constructors, and `register_entrants(loader)`, which registers the arena's "network"/"mcts" entrant kinds, its "network" evaluator, its checkpoint loader and its leaf-evaluator factory in one call. `hexset.clients.onnxbot` keeps the ONNX half -- `load` and `V2Policy` -- and re-exports every name it exported before, so `spawn(path, board)` and every existing import are unchanged. The trade gate now hands the policy a copy of the seated game per candidate instead of mutating the live one, so a runtime encodes a post-trade position however it likes; the gate's own arithmetic, its `NETWORK_GATE_ROWS` cap and its verdicts are unchanged.

## 0.40.0

### Changed

- **A network checkpoint's trade gate scores both sides of an exchange.** `hexset.clients.onnxbot.NetworkBot` (and the searched `GatedSearch`) now score the post-trade position with *both* hands moved and the counterparty's ledger row updated, as a real clearing leaves it, and read the value head's per-seat vector: `gains_many` is this seat's own row after minus before, in win probability (it used to answer +1/-1 off a swap of its own hand alone); `estimate_many`, new, is the counterparty's row after minus before -- this seat's own estimate of what the exchange does to the other seat's chances -- so `default_offer`/`default_respond` offer and counter with the candidate best for the bot among those it believes the other seat gains from too, as `agents/reference/trading-final.md`'s trade round specifies. Every candidate two cards or fewer a side is scored; the rest fill `NETWORK_GATE_ROWS`. Under a strict-positivity floor the old sign gate offered an arbitrary acceptable bundle nearly every turn (clio at the table, 2026-09-08); the magnitude ranks offers and the estimate prices the risk of lifting an opponent.

## 0.39.0

### Changed

- **The clearing floor belongs to the gate, not the table.** `hexset.trading.TRADE_FLOOR` is gone; every seat's gate declares its own `trade_floor`, read by `hexset.trading.trade_floor_of` at every admission point (`clears_floor(gain, gate)`; `trade_event`, `execute_agreed`, the round's `default_offer`/`default_respond`/`default_pick`). There is no engine default: a gate that prices a candidate positive without declaring one is refused with a `TypeError`; a gate that only ever declines is never asked. heximax carries the one measured value (`hexset.bots.heximax.HEXIMAX_TRADE_FLOOR = 0.0197`, the trade lab's phase 3); `search2` carries the same number unmeasured so the frozen referent clears what it cleared; a network checkpoint's boolean gate carries `0.0`. `hexset.bots.search2.Bot` names the attribute. Registered in `agents/reference/trading-final.md`'s 2026-09-05 amendment as open; now done.

## 0.38.1

### Fixed

- One trade event a turn, and one bot broadcast a turn. A knight played in MAIN re-enters MAIN after its robber move, and both the engine's `run_trade_event` and the served table's `begin_round` fired again there, so a bot traded or offered twice in one turn. `Game.trade_event_turn` and `GameSession._broadcast_turn` key the once-a-turn rule on `(turns, current_player)`.
- A manual seat holding no cards passes on a broadcast at once instead of being asked: it could neither accept nor counter, and a bot actor's turn was held on its answer.
- The round's log line no longer spells the taken deal out again: the bundle is already in the offer or counter clause, so the close reads `Traded with Player N (...)`.

## 0.38.0

### Added

- `hexset.clients.onnxbot`: `load`, `network_bot`, `network_evaluator`, `searcher` and `spawn` take `threads`, capping onnxruntime's intra- and inter-op thread pools; `None` (the default) keeps onnxruntime's own sizing.
- MCP `new_game`/`join` take a required `model` argument (your exact model
  identifier); it becomes the seat's client secret, and
  `resume_game(code, model)` reclaims the seat with it via
  `POST /api/reclaim`.
- `POST /mcp`: MCP served over HTTP by `web.py` itself (Streamable HTTP
  transport), replacing the `python -m hexset.server.mcp` stdio program.
  `initialize` mints an `Mcp-Session-Id`; every later request on that
  session must carry it back: a missing one is a 400, an unknown one a 404
  (call `initialize` again). `DELETE /mcp` ends a session; `GET /mcp` is 405. A
  `tools/call` for `wait_for_turn` answers as `text/event-stream`; every
  other tool answers one JSON-RPC response. `Origin` is checked per the
  spec's security section: present and not 127.0.0.1/localhost/::1/this
  server's own `--host` is a 403.
- MCP `wait_for_turn(timeout?)`: blocks until `legal_actions` is non-empty,
  an offer is pending, your own trade round is fully answered, or the game
  is over -- returning at once if already true. Streamed with a keepalive
  roughly every 15 seconds while it waits.
- `GET /api/version` (no token): `{"version", "git_commit"}`.
- `POST /api/action`, `.../trade/round/answer` and `.../trade/round/choose`
  accept an optional `"version"`; a mismatch against the table's current one
  is a 409 ("the table has moved") rather than applying against a stale
  read. MCP `act`, `answer_trade` and `choose_trade` take the same optional
  `version` and refuse with the same message if it has moved since the
  `state()`/`get_table()` an index was chosen from.
- `hexset.clients.botclient`'s external join sends `client: {"id":
  sha256(secret), "kind": "api"}`; `secret` defaults to `--model`'s own
  filename stem, overridable with `--client-secret`.

### Changed

- The page's incoming-offer window is the counteroffer layout, titled "Trade Offer Received": the offer quoted on top with Accept (primary) on its row, a reply pre-loaded from the offer beneath with Send (secondary); passing is the close glyph. The separate "Trade Counteroffer" step is gone.
- Seats are the table's to change until the first move, then fixed. A closed seat stays on the roster as a picker reading "(closed)" until play starts -- it can be reopened ("(empty)", new `POST /api/open {seat}`) or given a bot -- and leaves the roster at the first move. `POST /api/close` and `/api/open` are refused (409) once play has started, so the log's Player 1, 2, 3 numbering never shifts mid-game. Picker labels read "(empty)" and "(closed)". The journal records `unlocked` alongside `locked`.
- The reply rows of "Trade Offer Received" carry a small "Counter" caption in the gutter, so the two rows read as offer and answer.

### Removed

- heximax's `omniscient` mode, the `heximax-omni` preset, `View`'s and
  `HonestEvaluator`'s `omniscient` flag, and `hexset.dataset`'s and
  `hexset.bench.fit_weights`'s `--omniscient` reading. Nothing outside
  heximax's own honesty-price readouts ever constructed an omniscient
  `View`; the engine's `game.state(seat, hidden=False)`, which the
  Catanatron adapter and the server's spectator view read, is unchanged.
  `heximax.MODES` is now `("honest", "notrade")`.
- `hexset.server.mcp`, the stdio MCP program (`python -m hexset.server.mcp`,
  `HEXSET_UI_BASE_URL`) -- MCP is `POST /mcp` on `web.py` now (see Added,
  above). Its tool layer moved to `hexset.server.mcptools`, called
  in-process rather than over its own HTTP client.
- `resume_game`'s local cache file and `HEXSET_MCP_SESSION_FILE`: nothing is
  left to cache once a seat's identity lives in an MCP session in memory,
  not a process. `resume_game` now takes `code`/`model` explicitly.
- `HEXSET_UI_BASE_URL` (MCP's own use of it -- `botclient.py`'s `--url`
  still names a server the ordinary way).

### Fixed

- A checkpoint served with `search: mcts` trades. `hexset.clients.onnxbot.searcher` returns a `GatedSearch`: `hexset.mcts.Search` with the checkpoint's own value-head gate (`accepts`/`accepts_many`), seated where `choose` last was. `Search` alone had no gate, so `valued_many` priced every candidate at -1 and a searched checkpoint never accepted or made an offer, while the same checkpoint played plainly traded.
- Closing any trade window closes it: declining a round, taking a deal, or passing on an offer dismisses the modal instead of leaving the composer up because it is still this seat's turn.
- The acceptance pane lists only the seats still in the game; a closed seat has no row.
- A closed seat leaves the roster the moment it is closed, in every phase, and the remaining seats renumber (Player 1, 2, 3), matching the log.
- Setup ending hands the first roll to the first seat still in the game. With seat 0 closed, it went to seat 0 and the table waited on it forever (`hexset.server.seating.first_unlocked`).
- `hexset.trading.default_offer` only broadcasts a candidate that clears `TRADE_FLOOR` on the actor's *own* gain as well as on the estimated counterparty gain, so a bot never offers a deal it would then refuse when accepted. Measured on heximax, every prior broadcast priced negative for itself and no accepted offer ever completed.
- A pass is an answer: a bot's or a person's pass on the open offer stays in the round's `responses` (`"kind": "pass"`, `bundle` null) and shows as "Passed" on the acceptance pane, instead of looking like a seat still to answer.
- The trade round is in the game log as one line, rewritten as it goes -- the offer, each accept or counter, then the trade taken, `declines.`, or `Everyone declines.` the moment every seat has passed -- and in the journal as discrete steps (`kind: "note"`, one per offer/answer/close; `Journal.note` / `notes_of` carry them through a restart).
- `hexset.clients.onnxbot.LeafEvaluator.terminal` scores a finished game as the one-hot winner, the win-probability scale its contract-6 value head is trained on, instead of `terminal_relative_points`; it raises if the game has not finished.
- `hexset.trading.default_respond` no longer answers `accept` to an offer the responding seat cannot cover; such an offer is countered or passed instead.

## 0.37.0

### Added

- `POST /api/games`/`POST /api/join` accept an optional `client: {"id":
  <64-hex sha256>, "kind": "web"|"api"|"mcp"}`; absent defaults to kind
  `"api"`, an unknown kind or malformed `id` is a 400.
- `POST /api/reclaim {"code", "secret"}`: mints a fresh token for the seat
  whose `client.id` equals `sha256(secret)`, replacing any token that seat
  already held. 403 with no match.
- An unnamed claimed seat's display name now follows its client's kind:
  `web` → `human`, `api` → `api`, `mcp` → `mcp`.
- The page's trade modal, in one shape for all four of its states, at one
  width in all of them. The title names the state -- Trade Offer, Trade
  Offer Received, Trade Counteroffer, Trade Acceptance -- and the one way
  out of it is a plain glyph in that title row, so Cancel, Pass, Back and
  Decline all are the same X where the heading says which it is. A trade
  being built, offer or counter alike, is two rows of five cards, give over
  get, "for" between them: a tap on a card adds one of it, and the "−" that
  appears under a card in the offer takes one back. A trade being read is a
  line of just the cards in
  it, prefixed by the colour pip of the seat *giving* -- so left of "for" is
  always what that pip hands over, and a seat's accept mirrors the offer it
  answers. Your own open offer draws as an acceptance pane: your offer, a
  rule, then one row per seat in seat order with the check that takes that
  deal, a seat still to answer left blank. Every button in the modal is one
  44px icon square, its word in the tooltip; the bank/port square keeps its
  rate. Countering keeps the offer you were sent on screen, quoted above the
  reply you are building with a quiet Accept for taking it as it stands, and
  starts that reply from it. Seats are named as the rest of the page names
  them, so a table of three `search2` bots reads as Player 2, Player 3 and
  Player 4.

### Changed

- MCP `new_game`/`join` no longer default an unnamed seat's display name to
  `"mcp"` themselves; the server does it now (see Added, above).

## 0.36.0

### Added

- The trade round is now the served table's protocol end to end
  (`docs/bot-api.md` §3). `POST /api/games/<code>/trade/round` broadcasts
  the current player's offer (1-3 cards a side) to every seat;
  `.../trade/round/answer` is a seat's accept, counter or pass;
  `.../trade/round/choose` is the actor's pick or decline. A bot actor's
  turn holds (`trade_wait`, `to_move` null) until every person at the table
  has answered its offer. MCP tools `offer_trade`, `answer_trade`,
  `choose_trade`. Bot offers, answers and picks come from
  `hexset.trading.default_offer`/`default_respond`/`default_pick` (own gain,
  and the estimated gain of the other side) unless a bot implements
  `offer`/`respond`/`pick` itself.
- The give side of the composer counts up to the bank's own rate, not to
  `MAX_TRADE_CARDS`. A 4:1 sale could not be drawn at all before, so the
  bank button fired on whatever the row happened to show and took four
  cards for it; it is now offered only when the cards on the table *are*
  the rate, and the offer-to-players button only when both sides are within
  the table's three-card rule. Whichever of the two buttons is out of
  reach says why in its own tooltip, in place of the sentence that used to
  sit above the rows.
- `hexset.trading.execute_agreed`: one execution path for every agreed
  exchange, asking a bot side's gate fresh and taking a manual side's
  submission as its consent; `execute_trade` and the round's own execution
  are both built on it.

### Changed

- The trade round's third gate method is `pick(view, responses)`, not
  `choose` -- which is every bot's action picker and collided with it.

### Removed

- `POST .../trade`, `GET .../trade/acceptable`, `POST .../trade/confirm`
  and `.../trade/decline` (the one-to-one proposal routes), the MCP tools
  over them, `webplay.bundle_from_wire`, and `docs/negotiation-interface.md`.
  `PendingGate` no longer records clearing-house candidates.

### Fixed

- `hexset.trading.default_respond` countered with exchanges it would then
  refuse: a candidate was admitted on the *actor's* estimated gain alone,
  with the responder's own gain used only to rank what was already in. When
  the actor took such a deal, `execute_agreed`'s fresh ask of that same gate
  turned it down. A counter now has to clear the floor on both sides, like
  every other admitted exchange.
- A seat's view carried two keys named `round` -- the lap number every log
  line is tagged with, and the open trade round -- so only the second
  survived and the lap number never reached a client. The board's log pane,
  which filters on it, showed nothing at all. The open round is
  `trade_round` now; `round` is the lap number again.

## 0.35.2

### Changed

- References to the sibling training package follow its rename from `hexnet` to `hexn`.
- References to the sibling training package follow its rename from `hexnet` to `hexn`.

## 0.35.1

### Changed

- References to the private training repository follow its rename to
  `dev-HexN`.
- References to the private training repository follow its rename to
  `dev-HexN`.

## 0.35.0

### Removed

- `docs/readouts/`, `docs/engine-divergence-2026-09-02.md`,
  `docs/gym-design.md`, `docs/negotiation-interface.md` and
  `docs/onnx-contract-v2.md`, and the research-only bench instruments
  `hexset.bench.shipped_hand`, `hand_valuation`, `road_sweep`,
  `production_curve`, `encode_cost`, `behaviour` and `human_agreement`
  (with `hexset.behaviour`, which only served the last two) -- this is a
  gym to import, not a lab notebook, and the measurement record and the
  dated instruments now live in the private research repo. `docs/bot-api.md`
  is the live interface reference and stays. The shareable bench is
  `duel`, `generate`, `throughput`, `trade_census`, `fit_weights`,
  `fit_duel`, `weight_sweep`, `ablate`, `baselines`, `placement_policy`
  and `profile_heximax`; `weight_sweep` now runs its own paired cell
  instead of borrowing `road_sweep`'s, and `trade_census` no longer seats
  the frozen shipped hand.

## 0.34.0

### Removed

- `hexset.bench.aivat` and `hexset.bench.trade_lab` (and `trade_lab`'s
  test) -- `hexset.arena.compete` and `catanatron/duel.py` are the only
  game loops left. `aivat` had no dependents since August and needed a
  value-function baseline the project never built; `compete`'s antithetic
  paired boards and within-game VP margins already do the variance
  reduction it was for. `trade_lab`'s three phases are closed (floor
  0.0197 recorded), its data are Record v2 files replayable without it,
  and its private seeding scheme was the coupling the collapse could not
  remove. A position judge, if wanted again, will be built on `compete`
  with a start position and a chance stream.

### Fixed

- Discarding on a seven was served as if it were a turn. Every seat over the
  limit discards at the same time, bounded only by its own hand and its own
  `discard_quota` entry, but the server gated every submission on
  `to_move(game) != seat` — always the lowest-numbered owing seat — so a
  table with seats 0 and 3 both owing cards answered seat 3 with HTTP 409
  "it is not your turn to act" until seat 0 had finished. `hexset.game`
  gains `may_act(game, seat)` (true for *every* seat still owing a discard,
  `seat == to_move(game)` in every other phase);
  `hexset.actions.legal_actions`/`legal_mask`/`apply` take an optional
  `seat`, so a discard resolves against the seat that submitted it instead
  of `players_owing_discards(game)[0]`; and `GameSession.submit`/
  `legal_wire_actions` and `Tables.record` ask `may_act`. `to_move` itself
  is unchanged — it stays the single-seat answer for the callers that want
  one (the arena, a bot runner) — but the environment and the replay layer
  no longer take it for an ordering (next entry).
- The gym and the record layer still served a seven's discards in ascending
  seat order after the server stopped, so the same round was legal at a live
  table and refused offline. `hexset.gym.aec.HexSetAEC` no longer hardcodes
  `agent_selection` to `players_owing_discards(game)[0]` during
  `Phase.DISCARD`: it keeps PettingZoo's one-active-agent-per-`step()`
  contract but chooses *among* the owing seats, by a new `discard_order`
  (`"random"`, the default, drawn from a stream seeded off `reset(seed)`
  alone; `"seat"` for the old ascending order) or by the caller through the
  new `HexSetAEC.select_agent(agent)`, which accepts any seat `may_act`
  allows. `step()` dispatches the acting seat with the action
  (`apply(game, action, seat)`) and `observe` builds the mask from
  `legal_actions(game, seat)`, so a round resolves against whoever is
  actually acting. `hexset.gym.env.HexSetEnv` passes `discard_order` through
  and hands the learner control the moment it *owes* cards rather than after
  every lower-numbered bot seat has cleared its quota. Ascending order was
  not merely arbitrary: under it seat 0 never observed another seat's
  discard before choosing its own and the last owing seat always observed
  all of them, an asymmetry no table has and every self-play corpus taught.
  `docs/gym-design.md` §2 is rewritten to match.
- `hexset.record` could not express, and so could not replay, a discard
  round that did not happen in ascending seat order — a real server-journalled
  game's, or a converted colonist.io game's. `Record` gains `actors`, sparse
  `(step, seat)` pairs naming who took a step where the position cannot say
  (in practice the `Phase.DISCARD` ones), defaulting to empty so every record
  already written reads and replays unchanged; `replay` checks each action
  with `may_act(game, actor)`/`legal_actions(game, actor)` instead of against
  `to_move`; `advance` takes the actor and passes it to `apply`; a new
  `moves(record)` yields `(actor, action, trades)` and `steps(record)` keeps
  its two-tuple shape for existing callers. `from_journal` carries the
  journal's own `actor` across for every discard, so a simultaneous round
  served by `hexset.server` now round-trips exactly. `hexset.dataset`,
  `hexset.behaviour` and `hexset.bench.human_agreement` replay through
  `moves` so a named actor is honoured rather than silently re-applied to
  the lowest owing seat.
- The player-facing transcript wrote each seat's discard line as that seat's
  submission landed, so a simultaneous round was reported as a sequence and
  half of it was shown to the table while the other half was still choosing.
  `hexset.server.webplay.render_log` now holds a round's discards back —
  one line per seat however many cards and however interleaved — and writes
  them out together, in seat order, once no seat still owes any.
  `hexset.server.journal` is unaffected and still writes every action the
  instant it is applied: it is the crash-recovery log, not the transcript.
- The web client's phase banner read "PLAYER 1'S TURN" to a seat that owed
  cards to a seven, behind that seat's own discard modal — `state.to_move`
  names only the lowest-numbered owing seat. It now shows the phase to any
  seat with a `discard_quota` entry left to clear. The modal itself needed
  no change: it opens off `state.legal_actions`, which now answers per seat.

## 0.33.0

### Changed

- `hexset.arena.compete` is now the only thing in the package that plays a
  game of HexSet: `hexset.bench.generate`, `road_sweep` (and the
  `hand_valuation` and `weight_sweep` sweeps on top of it), `throughput` and
  `trade_census` all play their games through it and read the `Tournament` it
  returns, in place of five separate copies of the board seeding, the
  antithetic pairing and the play loop. A tournament carries what those
  copies existed to collect: per-seat roads, settlements and cities, the
  seating each game used, and -- with `records=True` -- every trade the
  engine cleared, with its turn, phase, both hands and both private gains.
  A duel's seating is a lineup rather than one of two named geometries:
  `--geometry` takes any pattern of `a`/`b` slots, or a comma-separated
  lineup that can also seat entrants on neither side. `--games` must divide
  by the number of seats wherever it did not have to before, and
  `hexset.bench.throughput` reports games and turns rather than actions.

## 0.32.0

### Changed

- `hexset.fitting` fits the evaluation weights *and* the win temperature in
  one solve: a conditional logit over the four seats of a recorded position,
  labelled with the eventual winner, coefficients `w / T`, cluster-robust by
  game with an optional block bootstrap. `hexset.dataset` builds its choice
  sets by replaying records honestly through `HonestEvaluator.rows_game`
  (new: the per-seat term rows before the dot). `hexset.bench.fit_weights`
  fits several designs from one records file and scores each by held-out
  log loss beside the shipped pair; `hexset.bench.fit_duel` plays a fitted
  pair against the shipped heximax, paired.
- `Heximax.temperature` (also `heximax(temperature=...)` and
  `Entrant.temperature`) reads the `win` stance at a given temperature
  instead of `search2.WIN_TEMPERATURE`, so a candidate pair can sit at a
  table with the incumbent; `search2.win_at` is the stance at any
  temperature.
- heximax's honest evaluator values an opponent's development cards at
  their expected victory points -- held count times the VP share of the
  unseen pool (`hexset.bots.heximax.evaluate.expected_card_points`) -- where
  it used to count VP cards for the knower alone; the omniscient evaluator
  now counts every seat's real VP cards. Play-neutral against `search2` and
  the honest bot on 800 identical boards; the anchor term now means the
  same thing in every seat's row.

## 0.31.0

### Changed

- `hexset.trading.trade_round`: a second trading protocol, for a *served*
  game only (`hexset.server`) -- propose-and-respond rather than the
  engine's exhaustive clearing house (`trade_event`, unchanged and still
  what a self-play run trains against). A session that wants it seats
  `game.max_trades = 0` (the existing off switch) so the automatic event
  no-ops itself, and calls `trade_round(game, gates)` itself, as many times
  a turn as the acting seat wants -- nothing counts or caps rounds, the
  floor and the card cap already bound what one moves. `Bot` gains four new
  optional methods for it (`offer`, `respond`, `pick`, `estimate_many`),
  each with a sensible default off a plain `gains_many`
  (`hexset.trading.default_offer`/`default_respond`/`default_pick`), so
  every existing bot plays a served table unchanged; heximax and search2
  additionally implement `estimate_many` for real. `hexset.server.webplay.
  PendingGate` gains the manual-seat side of the same three methods
  (`offer`/`respond`/`pick`), recording a broadcast offer to
  `game.pending` exactly as it already does for the clearing house.

### Removed

- `hexset.tuning`, `hexset.bench.tune`, `hexset.bench.win_temperature` and
  `hexset.bench.learn_weights`, the hill-climb weight fitter and its
  supporting scripts -- superseded by the position-level likelihood fit in
  `hexset.fitting`, which weights are now fitted against.

## 0.30.1

### Fixed

- `hexset.__version__` tried installed package metadata before the source
  tree's `pyproject.toml`, so an editable install with stale dist-info kept
  reporting `0.26.0` for four releases after the tree moved on; it now reads
  `pyproject.toml` from the tree first and only falls back to installed
  metadata when there's no tree to read.

## 0.30.0

### Changed

- `hexset.mcts.Evaluator` gained a required `terminal(game) -> Sequence[float]`
  method, called for a finished-game leaf in place of the search's own
  `relative_points` formula — board-seat order, the same frame `evaluate`'s
  returned values are already in. Anything implementing the protocol needs
  it now; `hexset.clients.onnxbot.LeafEvaluator` implements it as the new
  `hexset.mcts.terminal_relative_points`, byte-identical to today's
  behaviour.

### Fixed

- A terminal leaf was scored by `Search` itself with `relative_points` — a
  zero-sum points margin — while every other leaf came back from the
  evaluator on that evaluator's own scale (a win-probability value head's
  [0, 1], for instance), so a backup could mix two scales. Terminal leaves
  now go through `Evaluator.terminal` like everything else.
- `Search(..., stance="win")` built a tree whose `Node.ranked` never
  accumulated — `STANCE_ROWS`/`_backup` only ever implemented
  `own`/`relative`/`paranoid`, so PUCT's value term was silently zero under
  `"win"`. `Search` now raises at construction for any stance outside that
  implemented set; `"win"` is the bots' own conversion of the per-seat
  vector (`hexset.bots.search2.win`), not something the tree's backup
  computes.

## 0.29.0

### Changed

- A trade moves at most `hexset.trading.MAX_TRADE_CARDS` (3) cards on either
  side -- the human corpus puts 99.1% of recorded trades at or under that.
  `_candidates` never enumerates a bigger bundle (so neither does the
  automatic event, the server's `trade/acceptable` preview, nor a pending
  offer); a manually proposed `POST .../trade` bundle exceeding it is
  refused with a `ValueError`, the same way an uncoverable one is.

## 0.28.0

### Changed

- `hexset.trading.TRADE_FLOOR` is `0.0197`, the trade gate's measured
  resolution under paired chance (trade lab phase 3): a deal clears only when
  both private gains exceed about two points of win probability. Trades
  claiming less no longer clear.

## 0.27.1

### Added

- **The trade lab's paired-chance judge** (`hexset.bench.trade_lab judge`,
  phase 3 of the registered ablation). For a sampled pre-trade position, a
  chance script (dice, steals, discards) is drawn directly from a seeded
  source, per kind, deep enough that both an untraded fork (the event
  suppressed for that turn only) and a traded fork (the historical
  clearing applied outright) can each draw from it independently all the
  way to the action cap; a fork that still outgrows the script falls back
  to a fresh live source and is counted. `trade_lab positions` samples the
  judged set (a MAIN-entry event that cleared a trade) from a bank;
  `trade_lab phase3-readouts` bins the judge's output by claimed gain,
  paired-bootstraps the realised win-probability *and* points swing for
  the actor, the counterparty and bystanders, fits a calibration slope,
  reports the pooled actor interval and its half-width, and reports the
  threshold the gate's claims hold up above (by points, the sensitive
  readout at this gate's scale). Both the bank re-emission and the judge
  subcommand are resumable and write progress as they go.
- **`hexset.bench.trade_lab`.** Lifts the one-event trade mechanic out of a
  played game for a static ablation: `bank` plays and records heximax×4
  games, and `census` replays each to every point a trade event fires,
  respawning the same bots to re-derive the published vectors, then clears
  every position under four selection rules — the shipped maximin-public
  surplus rule, and three private-gain rules (actor, egalitarian, nash) that
  skip the public-vector filter — reporting trades-per-event, bundle shape,
  surplus split, bystander win-probability damage and rule disagreement.
  Torch-free, multiprocessed like `hexset.bench.trade_census`. `census` also
  reports the gain-distribution quantiles per rule; `strategic` shades one
  rotating seat's gate (a `tau` acceptance threshold, and — for the two rules
  that read magnitudes — a selection-key exaggeration) to check whether
  honesty is a fixed point; `rollouts` judges 300 sampled executed trades per
  rule by playing the position out both traded and untraded with fresh
  `heximax` bots, comparing realised win-share swing against the gate's own
  claimed gain.

## 0.27.0

### Changed

- **A candidate must clear a floor, not just be positive, to trade.**
  `hexset.trading.TRADE_FLOOR` gates both sides of a deal — `trade_event`,
  `execute_trade` and `GET .../trade/acceptable` all read the same
  `clears_floor` predicate now, in place of a bare `> 0.0`. Ships at `0.0`
  (unchanged behaviour) until the trade lab's paired-chance judge measures
  the gate's real resolution.
- **The trade event fires once a turn, not after every MAIN action.**
  `hexset.game.run_trade_event` now runs only on the transition into
  `Phase.MAIN` (a roll or a robber move) — a build, a buy, a bank/port
  trade, or a development card no longer reopens it mid-turn. `Game.pending`
  (a manual seat's recorded offers) now survives every later MAIN action in
  the same turn instead of being cleared by the next event.

## 0.26.1

### Added

- **The human/LLM trading surface.** `GET /api/games/<code>/trade/acceptable`
  — the seat on the move's own read-only preview of every bundle a bot
  counterparty's gate already accepts right now, grouped by counterparty and
  sorted by its gain — joins the existing `POST .../trade`,
  `.../trade/confirm` and `.../trade/decline`. The page wires all three into
  the trade modal: a counterparty picker and the give/want cards for
  "Offer to players," the acceptable-deals list (its 1-for-1 entries) so a
  deal can be picked directly, and a pending-offers panel that surfaces on
  its own once a bot's trade event finds something against this seat, with
  Confirm/Decline. MCP gains matching `trade_acceptable`, `propose_trade`,
  `confirm_trade` and `decline_trade` tools.
- **`hexset.chance`: one chance source for the whole engine.** `Game.chance`
  answers `deck_order`, `roll`, `steal` and `discard` — every random draw
  the engine makes, in place of reaching into `random.Random` directly.
  `Live` is the default (byte-identical to every seed the engine has ever
  played); `Scripted` replays a recorded event stream instead of drawing,
  raising `ChanceMismatch`/`ChanceExhausted` (naming the event index) on
  divergence; `Recording` wraps either and logs every outcome; `Forced`
  pins one steal's resource for a counterfactual child
  (`hexset.bench.aivat`, `hexset.bots.heximax.search`, replacing each
  module's own `_Forced` stand-in-rng). `imagine` always hands its copy a
  fresh `Live`, never the real game's `chance`, so a search can never drain
  a replay's scripted stream or leak its own draws into one being recorded.
- **`hexset.record.from_journal`.** Converts a `hexset.server.journal` file
  into a `Record` directly — no seed, no re-running the engine to recover
  the deck, rolls or steals, since the journal already spells them out.
  The porting surface a v2 `Record` was built for.
- **`--records <path>` on `hexset.bench.duel` (arena path, `--workers > 1`)
  and `hexset.bench.trade_census`.** Appends every game played as a v2
  record. On `duel`, the recorded games are exactly the games the verdict
  counted (`arena.compete(records=True)` builds both from the same job);
  unavailable with `--workers 1`, which plays through hexnet's own batched
  collector and returns a verdict with no per-game history.
- **Catanatron's bots can sit at a HexSet table.** `hexset.catanatron.bot.
  CatanatronBot` is a `hexset.arena` bot whose brain is a Catanatron `Player`,
  registered as the `catanatron` preset (Catanatron's own AlphaBeta player at
  depth two, built exactly as `catanatron-play --players=AB:2` builds it) — so
  it can be seated from the web picker, `POST /api/bot`, an arena lineup or the
  gym, alongside `heximax` and `search2`. Each decision mirrors the live HexSet
  position into a Catanatron `Game` (`hexset.catanatron.state.to_catanatron`,
  on the map `hexset.catanatron.board.catanatron_map` builds from the HexSet
  board) and translates the answer back. The seat never trades: Catanatron's
  players have no notion of the one-event trade mechanic. The import of
  `catanatron` is lazy, so an install without the `catanatron` extra simply has
  one fewer opponent in the picker.
- **`hexset.bench.trade_census`.** Plays a lineup through `hexset.arena`
  (grouped seating, antithetic-paired boards, `road_sweep`'s convention) and
  records every `hexset.trading.Trade` as it clears — turn, phase, both
  seats' kinds, the signed 5-vector each way, each side's hand size the
  instant before the trade, and each side's public surplus — then rolls it
  up per bot: trades/turn, bundle-size distribution (1:1, 2:1, 3+:1, 2:2,
  bulk), mean cards given/received, imbalance, the share of trades made
  holding 8+ cards, and a bot-neutral value swing at the flat 4:1 bank rate.
  `--from-journals` runs the same census over `hexset.server.journal`
  files instead of playing fresh games. Torch-free; a network entrant's
  trades census the same way once `hexnet.netbot` registers it.
- **`hexset.bench.road_sweep`**: heximax-vs-heximax duels across a grid of
  challenger `road`/`card` evaluation weights, recording roads, settlements,
  cities and VP per seat alongside the win rate `hexset.bench.ablate` already
  tracked. `docs/readouts/heximax-road-sweep/` has the first sweep.
- **`hexset.bench.profile_heximax`**: plays N complete four-seat games under
  `cProfile` for one preset, reporting ms/decision (mean/p50/p95) and the top
  functions by cumulative and total time. `docs/readouts/heximax-profile/`
  has the first reading: real per-turn trade clearing, not anything inside
  the search's lookahead, is the largest cost center in a heximax game.
- **A read can wait for the next change instead of asking again.** `GET
  /api/state` and `GET /api/table/<code>` accept `?after=<version>&wait=<seconds>`
  and hold the request until the table has moved past the version the caller
  already has, up to 25 seconds; every view now carries its own `version`.
  A request without `after` answers immediately, exactly as before.
  `python -m hexset.clients.botclient --poll-interval` is now the longest one
  of those waits rather than a sleep between moves, and defaults to 10 seconds.
- Nobody moves during setup while any seat is still empty. The grace window
  that used to hold a seat open for a fixed time is gone for good, and with
  it `SEAT_GRACE_SECONDS`, `Config.seat_grace`, `POST /api/games`'s
  `seat_grace`, `--seat-grace` and `HEXSET_UI_SEAT_GRACE` — a seat resolves
  when a person opens the link and takes it, the creator picks a bot for it,
  or the creator closes it outright (`POST /api/close`), never on a clock.
  `to_move` is `null` and every view's `waiting_for` names the seats still
  open until then; once none is, play starts from seat 0 at full speed.

### Changed

- **heximax answers a whole trade event's candidates in one pass.**
  `Heximax.gains_many` batches every candidate a trade event asks about into
  one vectorised evaluation (`HonestEvaluator.score_many` over a
  `(candidates, seat, resource)` hand array) instead of re-scoring all four
  seats per candidate in pure Python; `_delta` is now `gains_many` with one
  row. Trades are unchanged — verified byte-identical over recorded games —
  and heximax×4 games run about 2.8x faster (9.5s/game -> 3.4s/game, 16-game
  `compete`).
- **Every human or LLM seat is a direct gate, unconditionally.** Seat-up
  installs `PendingGate` the instant a manual seat is claimed
  (`POST /api/games`/`/api/join`, `hexset.server.mcp`'s `new_game`/`join`) —
  there is no `confirm` flag on the wire any more to opt out of it, and no
  other gate a person or an LLM can get.
- **A seat's own `pending` offers are capped to the top 5 by the acting
  seat's gain**, descending (`GameSession.pending_for`) — a lenient bot gate
  can price a great many candidates above zero in one event, more than a
  person can usefully be shown, and this is what both `GET /api/state`'s
  `pending` block and `.../trade/confirm`/`.../decline`'s `index` now read
  from, kept in exact agreement.
- **A seat's gate returns how much a deal is worth to it, not a public
  advertisement.** `Bot.gains_many(view, received, counterparties) ->
  list[float]` replaces the published valuation vector as the trade
  mechanic's whole interface: each candidate exchange is priced in that
  seat's own value units, and a deal clears only when both sides price it
  strictly above zero. heximax and search2 price it from their own
  evaluators (win probability and the evaluator delta, respectively); a bot
  with only a boolean `accepts`/`accepts_many` gate is priced at
  `+1.0`/`-1.0` by a structural default, so nothing that traded before stops
  trading now. `RandomBot` and every other seat with no trading surface at
  all still never trades.
- **The table clears the deal fairest to the party gaining less, not the
  one with the biggest combined public surplus.** `Game.trade_rule`
  (default `"egalitarian"`) selects among every candidate both gates price
  above zero by the smaller of the two private gains; `"nash"` (their
  product) and `"actor"` (the current player's own gain) remain selectable
  for lab comparisons. Every coverable candidate now reaches the acting
  seat's gate directly — there is no public-surplus pre-filter left to rank
  candidates before a gate is asked.
- **Nothing is published any more.** A gate is a pure function of the
  current position, asked fresh at every trade event, so the engine clears
  a turn's first event eagerly (inside `enter_main`) rather than waiting for
  some later observation or publish to trigger it — every driver
  (`hexset.arena`, `hexset.record`, `hexset.bench`, the gym, the server)
  simplifies to "seat the gates and step the game."
- **`hexset.bench.trade_census` records both sides' private gains instead of
  a shared public surplus.** `TradeRecord.gain_a`/`gain_b`/`larger_gain`
  replace `surplus_a`/`surplus_b`/`larger_surplus`.
- The observation/record contract bumps to **6**: the 20-float public
  valuation block is gone from both `hexset.encoding`'s global features and
  `hexset.onnx_record`'s record. A contract-5 (or earlier) checkpoint is
  refused at load by name, the same as every previous contract retirement.
- **Playing a Knight is now two actions: play the card, then move the
  robber.** Previously one action carried a target hex and a victim
  together; now you play the Knight, and the board then asks for the robber
  move exactly the way it already does after rolling a seven — pick a hex,
  then a victim if there's a choice. A Knight that wins the game ends it the
  instant your total crosses the threshold, with no robber move at all. The
  board page no longer arms a "cancel" state for the Knight — once played,
  it resolves the same forced robber move a seven does.
- **`hexset.record.Record` is version 2: it carries its own chance.** A new
  `chance` field (the deck order, every roll, every steal, every random
  discard, as an explicit event stream) replaces depending on `seed` to
  reproduce them from the engine's random draws — the tripwire the old
  docstring warned about ("unreadable without the exact engine version"),
  and the reason nothing outside this engine's own seeded stream could ever
  become a `Record`. `seed` is now optional: present, `replay` uses it as
  an extra check that `chance` is what that seed's stream actually
  produced (`ReplayError` on divergence); absent, `replay` drives the game
  from `chance` alone. `to_json` writes `"version": 2`; `from_json` refuses
  a version-1 line by name rather than misreading it. The only version-1
  file this project shipped, the trade-lab bank, is re-emitted as version 2
  by re-running `record_game`/`write` — no format migration needed. `Record`
  also gains `first` (`Game`'s own new field, set by `start`): the setup
  snake's start seat, so `replay` reopens the same snake a game with a
  rotated deal actually played rather than assuming seat 0.
- `search2` is off the board's model picker; it stays seatable by name for API clients, tests and the training mix.
- **The Catanatron adapter's translation tables now run both ways.**
  `hexset.catanatron.names`, `.board`, `.state` and `.actions` express each
  name, enum, coordinate and action correspondence once as a bijection and use
  it in both directions, rather than carrying a second copy for the new
  direction: `board.catanatron_map` is `translate_board`'s inverse (ports
  included — HexSet spaces them evenly around the coast, so each one is
  re-seated on the coastal edge the board actually has),
  `state.to_catanatron` is `state.translate`'s, and `actions.to_catanatron`
  now answers `PLAY_KNIGHT` with whichever half of Catanatron's two-decision
  split is on the table instead of raising for its caller to resolve.
- **The Docker image installs the `catanatron` extra**, pinned to the same
  commit as `pyproject.toml`, so a deployed table can seat the `catanatron`
  opponent. This restage needs a rebuild, not just a restart.
- **`hexset.bench.trade_census` reads the true state through
  `game.state(0, hidden=False)`** rather than `game._state` — the sanctioned
  path `tests/test_view.py` pins, and (unlike `game.state(seat)`) not one of
  the pending trade event's trigger points, so the instrumentation's
  bookkeeping snapshots still fire nothing. No behaviour change: the two
  return the same object.
- A closed seat reads "closed" (the picker's option and the row), and players are numbered among the seats still in the game: close one and the table reads Player 1, 2, 3.
- **`hexset.trading._candidates` skips a zero-valuation seat's enumeration.**
  A seat that has never published (`NO_VALUATION`, all zero) can never clear
  a trade as either party — `_rank_candidates_loop`/`_rank_candidates_
  vectorized` already discard every candidate touching it (`mine <= 0.0` /
  `theirs <= 0.0`) — so `_candidates` now skips walking that seat's hand
  before generating any bundle, rather than enumerating them only to have
  ranking throw them away. Behaviour-preserving: the heximax choice census
  is byte-identical.
- **A private trade gate prices a candidate without cloning the position.**
  `hexset.bots.evaluate.hand_shifted(state, changes)` returns `state` with
  only the named seats' hands changed, sharing the board, bank, deck and
  every dev-card pile by reference rather than copying them the way
  `state.copy_state` does — a trade only ever moves two hands, so nothing
  else needs to move. `hexset.bots.search2.SearchBot.accepts` and
  `hexset.bots.heximax.search.Heximax._delta` (the shape every real caller
  uses, `target == knower`) now price a candidate this way: heximax's own
  gate additionally recomputes the shared belief's `known`/`pool` from the
  event's already-memoized pre-trade belief (`_after_trade_belief`,
  `_ShiftedBelief`) instead of rebuilding a `View` from a cloned ledger, so
  a third seat's `expected_hand` — which the pool a certified trade
  shrinks does move, under `relative`/`paranoid` stance — is priced
  correctly without a clone either. Verified exact against the prior
  clone-and-evaluate path over real self-play (both stances, both modes,
  zero mismatches across 200k+ live gate calls) and the heximax/search2
  byte-identical choice censuses. `target != knower` — a shape nothing in
  this repo calls `_delta` with — keeps the old clone-based path
  (`Heximax._delta_reference`).
- **`HonestEvaluator.progress_toward`'s inner sum is a list comprehension,
  not a generator expression**, over the identical operands in the
  identical order (`sum([min(hand[r], n) for r, n in needed]) / total`) —
  bit-identical to the generator it replaces (CPython 3.12's `sum` is
  Neumaier-compensated, so only the same values in the same order are safe
  here), just without a generator's per-item frame-switch overhead on
  `needed`'s two or three pairs. Behaviour-neutral: the byte-identical
  choice census (`test_choices_are_byte_identical_to_the_recorded_census`,
  both `heximax` and `search2`) is unchanged. Measured with
  `hexset.bench.profile_heximax` (3 games, seed 100, single process):
  `heximax` 42,695,282 -> 38,238,671 function calls over the 3 games (-10.4%);
  `heximax-notrade` 20,192,176 -> 18,685,520 (-7.5%). Wall-clock ms/decision
  moved within this box's cross-run noise (shared with a GPU training run);
  the call-count drop is the reliable signal.
- **`HonestEvaluator.belief_for`'s cache key is cheaper to build, on a hit
  or a miss.** `map(tuple, state.hands)` in place of a generator expression
  over the same hands in the same order, list comprehensions in place of
  generator expressions for each seat's ledger `known`/`unknown`, and a
  fast path that returns `()` for `certify` outright rather than draining
  an empty generator to discover it is empty (`certify` is `()` at both of
  this method's call sites today). Same key value, same cache semantics,
  same `View.__init__` fields covered (board occupancy and the robber
  included, per the method's own exactness argument) — only the
  construction is cheaper. Behaviour-neutral: the byte-identical choice
  census is unchanged. Measured with `hexset.bench.profile_heximax` (3
  games, seed 100, single process), cumulative with the `progress_toward`
  change above: `heximax` mean ms/decision 8.998 -> 8.315 (-7.6%), 6.260s
  -> 5.753s/game (-8.1%); `heximax-notrade` mean ms/decision 7.105 -> 6.679
  (-6.0%), 2.678s -> 2.532s/game (-5.5%).
- **`Heximax._marginal_gain`/`_marginal_loss`/`_delta` clone only what a
  marginal/delta check actually touches.** A new `_thin_copy` (heximax's
  own, alongside `copy_state`, not a change to it) copies `hands` and,
  where the caller mutates it, `bank`; the board, deck, dev cards, knight
  counts and (for `_delta`) the bank are shared with the real live game
  state these checks read `view.state` from, never copied, since none of
  these three methods ever mutates them. Safe only because nothing
  downstream reads `.state` back off a `belief_for` cache hit for one of
  these calls (`_thin_copy`'s own docstring records the invariant this
  depends on for the next person to touch this path). Byte-identical
  choice census and the full non-slow suite (881 passed) both unchanged.
  Measured with `hexset.bench.profile_heximax` (3 games, seed 100, single
  process, before/after run back-to-back to isolate the change from this
  box's own load swings): `heximax` mean ms/decision 8.865 -> 8.388
  (-5.4%), 5.970s -> 5.688s/game (-4.7%); `heximax-notrade` mean
  ms/decision 6.930 -> 6.776 (-2.2%), 2.624s -> 2.563s/game (-2.3%) — a
  smaller win than the other two changes above, since these three methods
  fire only during the real game-level trade event, never inside the
  search's own lookahead.
- **The player list's picker gains a third option, "none".** Choosing it
  closes that seat outright (`POST /api/close`) — the explicit gesture that
  replaces the setup snake retiring an open seat on sight. A closed seat
  reads "locked seat" exactly as one the snake used to retire did, and its
  row disappears once the match is under way. Any seated person may close
  any other seat, the same permission as picking it a bot.
- **heximax reads its evaluation as win probability.** Its default stance is
  `win` (`hexset.bots.search2.win`): the per-seat score vector read as
  `softmax(vector / WIN_TEMPERATURE)[seat]`, the seat's own chance of
  winning, rather than `relative`'s own score minus the table mean. At the
  table heximax now robs the leader two thirds of the time instead of half,
  feeds the leader less through trades, and beats the `relative` reading
  head-to-head at an equal terminal-VP margin. `WIN_TEMPERATURE` is fitted
  against real game outcomes and pinned beside the stance; `MARGINAL_SCALE`,
  the unit heximax's published trade valuation is squashed onto, is refit
  for the new stance by its recorded protocol. `search2` is unchanged and
  stays the frozen `relative` referent.

### Removed

- **The public layer.** `Game.valuations`, `Game.publish`/`publish_due`,
  `hexset.trading.publish_valuation`/`checked_valuation`/`NO_VALUATION`/
  `VALUE_SCALE`, and `Bot.valuation` (the protocol method and every
  implementation) are gone, along with the lazy first-event trigger
  machinery (`Game.event_pending`/`awaiting_publish`,
  `hexset.game.run_pending_event`) that only ever existed to let a driver
  publish before an event ran on it.
- **`PUT /api/games/<code>/valuation` and `PostedValuation`.** Forced by the
  above: there is no vector left to post.
- **The `confirm` flag.** `POST /api/games`/`/api/join` and `hexset.server.
  mcp`'s `new_game`/`join` no longer accept one: `PendingGate` is now the
  only gate a manual seat can have, so there is nothing left to opt in or
  out of. `hexset.server.mcp`'s `set_valuation` tool is gone with it — there
  is no vector left to publish.

### Fixed

- `hexset.trading.trade_event`'s safety assertion checks that an event never
  revisits a position (every hand plus the public ledger) instead of counting
  trades against the cards on the table. A legitimate event of one- and
  two-card exchanges can run longer than there are cards without repeating a
  position; the old ceiling raised on such events in self-play.
- **A manually executed trade (`POST .../trade`, a confirmed pending offer)
  now appears in the sidebar log and survives a server restart.** It moves
  cards through `hexset.game.Game.execute_trade` directly, outside the
  action that `GameSession`'s log and journal are built around, so it had
  neither: `GameSession.execute_manual_trade` gives it its own log line
  (`_trade_lines`, same as an automatically cleared trade) and its own
  journal line (`Journal.manual_trade`, replayed by `GameSession.restore`
  as its own step) — previously the cards moved live but a resumed game
  silently forgot them, since resume rebuilds hands purely from recorded
  actions and the trades attached to them.
- Road Building played before rolling now resolves its free road placements
  before dice are drawn. Only free roads are legal during that resolution;
  if no placement is possible, remaining credit expires and rolling resumes.
  Paid building still requires MAIN, and pre-roll roads do not trigger trades.
- Incremental record consumers (dataset features, behaviour and human
  agreement) now share `record.open_record`, preserving recorded chance and
  the setup start seat. Full replay rejects unused trailing chance events.
- **`hexset.arena.Entrant.stance` now defers to the bot's own default
  instead of hardcoding `"relative"`.** Every constructor of a heximax
  entrant besides its three presets (`hexset.tuning.entrant_for`/`duel`/
  `climb`/`confirm`, `hexset.bench.tune --stance`, `hexset.bench.
  road_sweep`'s challenger/baseline) still spawned heximax at `relative`
  rather than `win`, silently disagreeing with the presets that had to
  override it explicitly. `Entrant.stance` is now `None` by default,
  resolved at spawn time to each kind's own default (`"win"` for
  `heximax`, `"relative"` for `greedy`/`search`) — stated once, on the
  bot, instead of on every caller.
- **`hexset.catanatron.duel`/`.player` now register the bot presets.**
  Neither module imported `hexset.bots`, so `PRESETS["heximax*"]` was
  missing in the bridge's worker processes and `--players=DC:heximax-
  notrade,...` raised `KeyError: 'heximax-notrade'`. `hexset.catanatron.
  player` now imports `hexset.bots` at module scope.
- **Every playable development card, not only the knight, is legal before
  rolling.** Rulebook, Production Phase: "you may play one of them before
  rolling the dice" names no exception for Road Building, Monopoly or Year
  of Plenty. `legal_actions` offered only the knight in `Phase.ROLL`;
  `hexset.game.play_road_building_card`/`play_monopoly_card`/
  `play_year_of_plenty_card` each required `Phase.MAIN` outright. All four
  card plays now share one rule (`ROLL` or `MAIN`, at most one a turn, never
  a card bought this same turn), matching what `play_knight_card` already
  did. Building, buying and trading stay Action-phase only, and the turn's
  trade event still no-ops before the roll, unchanged.
- **A tile transfer during another seat's turn no longer wins the game for
  a seat that is not on the move.** Rulebook, Winning the Game: "if you have
  10 or more VPs at any point during YOUR turn." `_check_win` scanned every
  seat's total (`victory.winner`), so a settlement that broke an opponent's
  Longest Road and handed the tile to a *third*, already-loaded seat could
  end the game on that seat's behalf mid-way through somebody else's turn.
  The check now reads only `game.current_player`'s own total.
- **A seat that crosses 10 VP off-turn now wins the instant its own turn
  begins.** Follow-up to the fix above: scoping the win check to the mover
  means a seat that gained a tile transfer on someone else's turn no longer
  wins right then, but the rulebook ("on their turn") still means they win
  as soon as it *is* their turn, before taking any action. `end_turn` now
  re-runs `_check_win` for the new current player immediately after handing
  them the turn, so `is_over(game)` is already true — and no action is
  ever requested from them — the moment play reaches them. Every driver
  that ends a turn through `hexset.game.end_turn` (the arena, the gym, a
  search stepping its own `imagine`d copy) gets this for free, since it is
  the one function that does so.
- **Bots played one action a second, whatever they were actually thinking.**
  Every bot seat submitted a move and then slept a full second before looking
  at the board again, and the page waited 1.5 s between reads on top of that
  — so a search costing under a tenth of a second landed on a one-second
  boundary and a table of three bots crawled. Nothing is paced by a clock any
  more: a bot plays its whole turn back to back and then waits for the table
  to change, and the page is told the moment it does. Three `heximax` seats
  now finish the setup phase in under a second, where the same lineup took
  the better part of twenty.
- Seats retired during setup no longer show in the player list once the match is under way.
- **A new game always gave the creator the first turn, even seated away from
  Player 1.** The setup snake started wherever `POST /api/games` happened to
  land the creator's random seat instead of seat 0. It now always opens on
  seat 0, whoever holds it, and the page highlights that seat as current;
  seat 0 is held open rather than retired while it waits to be filled.
- **An empty seat's picker read "open seat" in the same white as a chosen
  bot's name.** It now reads "empty", in the same muted grey as a locked
  seat, so an unfilled seat reads as unfilled at a glance; your own row's
  "human" placeholder is styled to the normal name color instead of the
  browser's own dimmer default.

## 0.26.0

### Added

- **The browser board is the pre-tables UI again**, rebuilt from `f6856d7`
  onto the current server rather than carried forward through the lobby
  rework. Loading the page puts you in a game — the one you were last at if
  it is still going, a fresh one otherwise — and the address bar shows that
  game's code. Every seat but the creator's starts empty, the creator's row
  reads "human", and each other row is a model picker: choosing one seats
  that bot, and a person who opens the link takes a seat instead. A link to a
  game that is full or gone says so rather than dealing a different game
  under the same address.
- **Every game is public, and watching one is omniscient.** A link to a game
  with no seat left opens it to watch: the board, the log, and every seat's
  standing, updating as the game goes. A spectator is outside the game and is
  shown all of it — every hand, every development card, every true
  victory-point count, and a transcript that names the card bought, the card
  stolen and the cards discarded. Clicking a player row shows that seat's
  cards. Nothing is actionable, so the pickers, the board buttons and the
  piece supply are simply absent, which is what says you are watching.
  `GET /api/table/<code>/board` serves the layout that view is drawn on,
  alongside the token-free `GET /api/table/<code>`.
- Your own row in the player list is your name: an input standing in for the
  line, the way a bot seat's row is a picker. It reads "human" until you type
  something, and what you type reaches the other players' lists and the log.
  Blanking it puts the seat back to unnamed. (`POST /api/name`, unchanged.)
- Player-to-player trading is gone from the browser with the offers that
  backed it (`PROPOSE_TRADE`/`ACCEPT_TRADE`/`DECLINE_TRADE`, `TRADE_RESPOND`,
  the view's `offer` block). The modal a resource card opens is the bank and
  port route, which is unchanged.
- Game codes are lowercase (`abcdef`), since a code is only ever seen as a
  URL. Lookups normalise, so a code capitalised on the way into an address
  bar still opens its game, and a game journalled under a capitalised code
  still resumes.
- The browser board seats a bot from the player list: an open seat's picker
  offers every model alongside the seat's current state, and choosing one
  fills the seat for the rest of the game. `POST /api/bot` (`Tables.seat_bot`,
  formerly `swap_bot`) now takes an empty seat as well as one with a bot on
  it, refusing a person's seat and a retired one.

### Changed

- **The web page offers a person no way to trade with another seat.** The
  advertisement controls, the negotiation panel and the pending-offer cards
  are gone from the browser; the bank/port modal a resource card opens is
  unchanged. Every route behind them is untouched and still answers an LLM
  or API client: `PUT /api/games/<code>/valuation`, `POST
  /api/games/<code>/trade`, `.../trade/confirm`, `.../trade/decline`, and
  the MCP trading tools.
- **A person's seat is gated when it sits down, not when it first
  publishes.** `hexset.server.webplay.GameSession.confirm_mode(seat)`
  installs a `PendingGate` over `hexset.trading.NO_VALUATION` at seat-up,
  and `POST /api/games`/`POST /api/join` call it. The seat advertises
  nothing and accepts nothing, and a seat whose vector is all-zero is
  dropped when candidates are ranked, before any gate is asked — so no
  exchange a person is party to can clear. Bot seats are unaffected and go
  on trading with each other.
- **The browser board has no front page.** Opening `/` deals a game and
  moves to its address; that address is the whole invitation — everyone who
  opens it sits down at the same table, and the last open seat can be given
  to a bot from the player list. The deal/join/name/code-entry screen is
  gone, and with it the browser's own name field (`POST /api/name` is
  unchanged for API and MCP clients).

### Fixed

- Opening a full game's address logged a console error. The page asked for a
  seat first and read the refusal as "watch this one instead"; arriving at a
  full table is the ordinary way to reach a game you are not playing in, so
  it reads `GET /api/table/<code>` first and only asks for a seat when one
  is open.
- Trades the engine cleared on a poll were never mentioned in the game log.
  A turn's first trade event runs lazily, at whichever of the engine's
  trigger points is reached first, which on the server is a poll rather than
  an action — those exchanges reached `trades` in the state and the ledger
  but no log line. They are attributed to the last action applied, matching
  `hexset.record.record_game`.

## 0.25.2

### Fixed

- **The seat panel could not tell an occupied seat from an open or a locked
  one.** Every seat's line was drawn as a bot model picker — your own, a
  seat nobody had taken, and one the setup snake had retired — because the
  player rows stopped carrying a `human` flag when the lobby was removed and
  the page still branched on it. Each line now reads the server's own
  per-seat kind: a name for a person, a picker for a bot, and "open seat" /
  "locked seat", dimmed, for a seat nobody is in.
- **The New game button did nothing once a bot had been swapped.** Because
  every seat was drawn as a picker, swapping wrote a fourth entry into a
  lineup that has room for three, and `POST /api/games` then asked for five
  seats at a four-seat table and was refused. The lineup slot is now read off
  the bot seats themselves and cannot grow past them.
- **A bot model picker closed about a second after it opened.** The page
  rebuilds its panels on every poll (1.5 s while it is not your move), which
  replaced the open `<select>` element. The seat panel now updates its rows
  in place and never touches a picker that has focus.
- **`POST /api/bot` answered with the swapped seat's view, not the caller's.**
  Changing a bot handed the page that bot's own seat number, hand and legal
  actions until its next poll. It now answers whoever asked, like every other
  route.
- **A game opened by someone with no seat rendered nothing.** `GET
  /api/board` and `GET /api/state` are both seat-gated and an observer holds
  no token for either, so the page took a 401 where its board should have
  been and stopped at "Loading...". `GET /api/table/<CODE>/board` serves the
  (public) layout without a token, and an observer polls
  `GET /api/table/<CODE>` for state. The seat panel's bot pickers are
  disabled for a reader with no seat, which is the only thing they could
  ever have answered.

## 0.25.1

### Fixed

- **A human seat auto-cleared trades against its published vector.** `POST
  /api/games` and `POST /api/join` left a human seat's gate at
  `PostedValuation` (auto-accept) unless `confirm` was set at seat-up, so a
  bot could clear a trade against a human who never confirmed anything —
  the same gate an LLM seat gets by design, but not what the negotiation
  interface intends for a person at the web page. Both routes now default a
  request that omits `confirm` to confirm mode (`PendingGate`): a bot's
  clearing candidate lands in `pending` for the human to `confirm`/`decline`
  instead. `confirm: false` still opts a human seat back out to
  auto-accept. `hexset.server.mcp`'s `new_game`/`join` tools are unaffected
  — they now send `confirm` explicitly on every call, keeping an LLM seat's
  own opt-in default (PI ratification decision 3).

## 0.25.0

### Added

- **`hexset.trading.NETWORK_GATE_ROWS`** (`32`): the most candidates a network
  gate's `accepts_many` will score in one batched forward, beside
  `VALUE_SCALE`. `hexset.clients.onnxbot.NetworkBot.accepts_many` now scores
  only the top `NETWORK_GATE_ROWS` candidates by public rank and declines the
  rest outright; `accepts` is unchanged. The engine still asks about every
  candidate — only a network gate's own evaluation is bounded.

## 0.24.0

### Added

- **`hexset.bots.Bot.accepts_many(view, received, counterparties)`**: a seat's
  private gate answered for a whole batch of candidate bundles in one call,
  defaulting to a loop over `accepts` so an existing bot is unaffected.
  `hexset.trading.trade_event` now asks a seat's gate this way — once for the
  current player over every ranked candidate, then once per counterparty over
  the candidates it accepted — instead of once per candidate bundle.
  `hexset.clients.onnxbot.NetworkBot` overrides it with one batched graph call.

## 0.23.1

### Fixed

- **The served game never traded.** `hexset.server.webplay.GameSession.
  state_view` fired the turn's pending trade event as a side effect of
  *any* viewer's poll (a spectator's, or an acting bot's own runner
  checking whose turn it is), by reading the current player's own hidden
  view unconditionally; `Game.publish_due(seat)` was then defined as "the
  event has not run yet", so that poll made a bot seat's own publish look
  moot before it ever happened, permanently, for that turn and every turn
  after. `Game.publish_due` is now keyed off a seat's own turn-scoped
  `awaiting_publish` flag instead of the event, so an early observation no
  longer stops the seat's publish from taking effect; `state_view` now
  triggers the pending event through `hexset.game.run_pending_event`
  directly rather than reading `game.state(game.current_player)` for a
  reader who may not be that player at all.

## 0.23.0

### Added

- **A negotiation interface for human and LLM seats.** `Game.execute_trade(proposer,
  counterparty, bundle)` composes and executes any bundle both sides can
  cover directly, bypassing the automatic candidate search — legal on the
  proposer's own turn against any seat, or during another seat's turn
  against that seat only; re-validates coverage and the counterparty's
  public surplus as a hard rule, then its private gate, exactly as the
  automatic event does, but never consults the proposer's own vector or
  gate, since submitting is its own consent. `POST
  /api/games/<CODE>/trade {counterparty, give, receive}` is the HTTP entry
  point. `Game.pending` is a snapshot of the last trade event's candidates
  against a confirm-mode seat, recomputed every event and cleared by
  `end_turn`; `hexset.server.webplay.PendingGate` is such a seat's private
  gate — it never clears on its own, recording each candidate instead — and
  is installed by opting a seat into confirm mode at seat-up (`confirm` on
  `POST /api/games`/`POST /api/join`). `GET /api/state`'s `pending` block
  (filtered per viewer) and `POST /api/games/<CODE>/trade/confirm`/`.../decline`
  answer one. The web UI gained a negotiation panel below the advertisement
  toggles: a counterparty's published wants/gives as clickable chips
  composing a draft bundle, a client-side clears/affordable indicator, and
  pending-offer cards during a bot's turn. The MCP server gained
  `set_valuation`, `get_table`, `propose_trade`, `confirm_trade`,
  `decline_trade` tools and a `confirm` flag on `new_game`/`join`.

## 0.22.0

### Added

- **Trading is one event, interleaved with the turn.** `hexset.trading.trade_event`
  clears deals for the current player after the roll and the robber, and
  again after every MAIN action (build, buy, bank/port trade, a development
  card): a bundle — any signed counts on disjoint resources, each side
  bounded only by what that hand holds — executes when both seats' public
  valuation vectors say it helps them *and* both seats' private gates
  accept, best deal first, repeatedly, until nothing clears. Candidates are
  ranked by public surplus and the two private gates are asked in that rank
  order until one clears or candidates run out — no budget, and no cap on
  trades themselves: the gate must be strictly positive and is re-asked
  after every exchange, so the acting seat's own valuation strictly
  increases and the event ends on its own.

### Removed

- **The gate budget.** A registered ablation (8/16/32 candidates per
  clearing attempt vs. unbounded) found unbounded both the strongest arm
  and within cost, so the cap is gone: private gates are asked in public-
  surplus rank order until one clears or candidates run out, always.
  `hexset.trading.GATE_BUDGET`, the `gate_budget`/`order` keyword
  parameters of `trade_event`/`_best_clearing` and the ranking helpers,
  `Game.gate_budget`/`Game.bundle_order`, `Game.budget_binds` (nothing
  binds now), the `order="minimal_bundle"` ranking path,
  `hexset.arena.play`/`_play_one`/`compete`'s threading of these, and
  `hexset.bench.duel --gate-budget`/`--order` are all deleted. The maximin
  ranking and its actor's-surplus tie-break are unchanged.

## 0.21.0

### Changed

- **A seat publishes once a turn, not after every action, and the turn's
  first trade event runs lazily.** `Game.publish_due(seat)`: true exactly
  once per seat per turn, while `seat` is the current player, the phase is
  `MAIN`, and this turn's first event has not run yet.
  `hexset.arena.play`, `hexset.record.record_game`, `hexset.bench.aivat`,
  `hexset.gym`'s auto-played opponents and the server's embedded bots call
  `hexset.trading.publish_valuation` only when this is true, at the
  post-roll/robber point, instead of after every action (measured at 8.4x
  collection cost in a batched collector for an event that can only ever
  observe two publishes a turn). `enter_main` no longer runs the turn's
  first event directly — it sets `Game.event_pending`, and the event runs
  the first time the current player's own `hexset.actions.legal_actions`,
  `Game.state(seat)` at `hidden=True`, or `Game.publish` is reached,
  whichever comes first (a `hidden=False` read of the true state does not
  trigger it — that path is for reading state for a reason unrelated to
  this seat's own turn), so a driver that publishes before it ever observes
  the game trades on the vector it just published, and a seat that never
  publishes (an idle human) still gets its event on whatever is already
  standing. Every event after the first one in a turn is unaffected. A
  human still publishes whenever it likes through
  `PUT /api/games/<code>/valuation`, `publish_due` or not.
  `hexset.record.record_game` attributes a lazily-triggered first event to
  the *previous* action's step (the roll or robber resolution), matching
  what `hexset.record.advance` already replays there.
  `hexset.clients.botclient.LocalSearchBrain` now hands an embedded
  `NetworkBot` the live `Game` at construction rather than waiting for its
  first `choose()` call, so `publish_due`-gated publishing before a seat's
  very first decision of a game does not read its `valuation`/`accepts` off
  an unseated bot.

## 0.20.0

Cut with `feat(trading): gate_budget/order as trade_event parameters; the registered gate-budget ablation`; that work's entry was reworded afterwards and is filed
under the release that carried the rewording.

## 0.19.0

### Changed

- **Trade candidates are bundles, not one-for-one swaps.**
  `hexset.trading._candidates` now enumerates every signed bundle on
  disjoint resources, coverable from the true hands, rather than only
  coverable one-card-for-one-card swaps — a 2-for-1 clears as one bundle
  now, where before it could not clear at all, since no sequence of
  one-card steps that each has to satisfy both gates on its own reaches it.
  A bundle's size is bounded only by what each side's hand holds — no fixed
  cap (owner review, 2026-09-03, against an interim 1..3-cards-a-side
  limit).
- **Trade and build interleave.** The trade event runs at the start of
  `Phase.MAIN` and again after every MAIN action the current player takes —
  build, buy, a bank/port trade, a development card — on the same published
  vectors (owner review against the rulebook, 2026-09-03; replaces "one
  event before any build"). Never runs after `end_turn`, and never during
  setup, `ROLL`, `ROBBER` or discard resolution.
- **The tie-break is the acting seat's choice among fair deals, not fewer
  cards.** Rank keys: the smaller of the two public surpluses, highest
  first (unchanged, the maximin); the current player's own surplus, highest
  first — among equally fair deals the actor takes the better one for
  itself; the total surplus, highest first; a canonical bundle order, then
  the lower counterparty seat, for determinism only (owner review,
  2026-09-03, "the tie-break" — replaces fewer-cards/canonical/lower-seat as
  the whole rule).

## 0.18.0

### Added

- **An embedded ONNX seat now trades.** `hexset.clients.onnxbot.NetworkBot`
  gained `valuation`/`accepts`, both derived from the checkpoint's own value
  head with no new graph output: `valuation` is `tanh(delta_V_r /
  VALUE_SCALE)` per resource, from one batched forward over the seat's hand
  plus its five one-card imagined successors when the graph's declared batch
  dimension allows it; `accepts` is the head's strict preference for the
  concrete post-trade hand. `hexset.trading.VALUE_SCALE`, the pinned
  constant both cite.

## 0.17.0

### Added

- `Game.valuations` — every seat's public vector, five floats in `[-1, 1]`,
  all-zero until something publishes; `Game.publish(seat, vector)` is the
  one way to set one, validated and recorded, nothing else; `Game.trades`
  and `Game.trades_made` for the turn's exchanges; `Game.max_trades` (`0`
  off, `None` unbounded); `Game.gates`, the per-seat objects `trade_event`
  asks for a private judgement. `Game.num_players`.
- `hexset.bots.Bot.valuation(view)` and
  `Bot.accepts(view, received, counterparty)`, both defaulting to "this seat
  never trades", so a bot written before the mechanic keeps working.
- `PUT /api/games/<CODE>/valuation` sets the calling seat's vector; every
  seat's `valuations` and the turn's `trades` ride in the game view and in
  `GET /api/record`. The browser client gains five per-resource toggles and
  a read-out of both.
- Presets `search2-notrade` and `greedy-notrade`.

### Changed

- **The trade event reads published vectors instead of fetching them.**
  `hexset.trading.trade_event(game, gate)` drops its `valuation_of`
  parameter and reads `game.valuations` directly; a driver publishes a
  seat's vector once, right after that seat's own decision
  (`Game.publish(seat, vector)`, or `hexset.trading.publish_valuation(game,
  seat, trader)` for the common "ask the trader, then publish" case).
  `hexset.arena.play`, `hexset.record.record_game`, `hexset.bench.aivat`,
  `hexset.gym`'s auto-played opponents and the server's embedded bots all
  publish this way now. `CONTRACT_VERSION` stamps `"5"` (it stayed `"4"`
  after `RECORD_FIELDS` had already changed).
- ONNX record contract `"4"` → `"5"`: the four `offer_*` fields and
  `pair_mask` are gone, `valuations` (`players × 5` floats) is added, and a
  graph no longer needs a `pair_index` output. Contracts 2, 3 and 4 are
  refused by name.
- Flat action space 553 → 550, and `globals` 86 → 87 at four players.
- `Entrant.max_offers`, `SearchBot.max_offers`, `Heximax.max_offers` and the
  `max_offers` checkpoint metadata key are all `max_trades`; `network:<path>@0`
  replaces `@<offers>`. `hexset.server.web`'s `--max-offers` is `--no-trade`.
- `Record.offers` → `Record.trades`, with `hexset.record.steps`/`advance`
  replaying them; the server journal records a step's trades the same way.
- `hexset.server.rules` keeps only `options_for` and `is_legal`: with no
  trade action, `hexset.actions.legal_actions` is the honest list for every
  seat.

### Removed

- **The offer protocol.** `Phase.TRADE_RESPOND`, `propose_trade`/
  `accept_trade`/`decline_trade`, `Offer`, `Game.offer`/`.pending_responders`/
  `.offers_made`/`.offered`, `MAX_OFFERS_PER_TURN`, `trading.responders`/
  `well_formed`/`can_propose`/`can_accept`, `actions._offer_actions`,
  `within_offer_budget`, `Action.give`/`.want`/`.ask`, `pair_index`/
  `pair_mask`/`NUM_PAIRS`, `server.rules.fair_legal_actions`/
  `proposable_options`, `SearchBot.partner_choice` and the `greedy-partner`,
  `greedy-offers1`/`2`/`3` and `search2-offers3` presets.
- The `heximax` top-level compatibility package (`import heximax` —
  use `hexset.bots.heximax`), the `hexset.evaluate` shim (use
  `hexset.bots.evaluate`), and the `Belief` alias for `hexset.view.View`.

### Fixed

- `hexset.server.api.spawn_bot` imported `.onnxbot` from `hexset.server`,
  where the module no longer lives after the one-distribution restructure
  moved it to `hexset.clients.onnxbot`. Any `.onnx` model picked in the web
  UI returned HTTP 500 (`ModuleNotFoundError`); now it imports from
  `hexset.clients.onnxbot`.
- `hexset.catanatron.state.translate` left `Game.valuations` at its empty
  default instead of one all-zero vector per seat, so `hexset.encoding`
  raised `IndexError` reading a bridged position; `tests/catanatron`'s
  three white-box suites read the upstream `catanatron.Game`'s `state`
  field as `_state` (a leftover from a project-wide sed that meant to
  touch only `hexset.game.Game`'s newly private field).

## 0.16.0

### Added

- `hexset.gym` (`pip install -e ".[gym]"`): `HexSetAEC`, a PettingZoo
  `AECEnv` with one agent per seat and an honest `action_mask`; `HexSetEnv`,
  a single-agent Gymnasium `Env` (registered as `HexSet-v0`) with one
  learner seat and `hexset.arena` bots auto-played at the rest; `register()`.
  `import hexset` stays numpy-only — only `import hexset.gym` needs
  `pettingzoo`/`gymnasium`.
- `hexset.view.View` gained `__eq__`/`__hash__`, comparing `perspective`,
  `omniscient`, `num_players` and `signature()` rather than object identity.

## 0.15.0

Cut with `feat(gym): PettingZoo AEC environment and Gymnasium wrapper`; that work's entry was reworded afterwards and is filed
under the release that carried the rewording.

## 0.14.0

### Added

- No lobby: `POST /api/games` deals a full game immediately and seats the
  creator at a random seat; `POST /api/join` or `GET /<id>` claims an open
  seat; an empty seat locks out after a grace window. `GET /api/table/<id>`
  serves a token-free observer view once no seat is left to claim.
- **The engine moved into this repo, under `engine/`.** `engine/hexset`
  (the rules engine, bots, ledger, arena and tuning) and `engine/heximax`
  (the honest handcrafted baseline, its own top-level package) were
  imported with full commit history from `0xBrsm/dev-HexNet`.
- **A single `pip install -e .` from the repo root now provides `hexset`,
  `heximax` and `hexset_ui`.** `hexset` is no longer installed from a
  separate checkout of `dev-HexNet`; see the README's "Running it"
  section.
- **`hexset.build_info()`**: version and git commit for a consumer (e.g.
  HexNet's run manifest) to stamp into its own provenance records.
- `benchmarks.duel` verdicts now carry `turns_mean`/`turns_median`/
  `turns_max` and `exhausted` (games that ran out `MAX_TURNS` without a
  winner, distinct from `unfinished`) on the arena path.
  `hexset.arena.Tournament` gained a `turns` tuple alongside
  `winners`/`points`.
- `Game.locked`: a per-seat setup lock / seat-retirement primitive. A locked
  seat is skipped by the setup snake and turn rotation, is never `to_move`,
  and drops out of any trade offer via the new `lock_seat(game, seat)`.
  `Game.locked: frozenset[int]` defaults to empty and is preserved by
  `imagine`. `start()` gained a `first=` keyword so the setup snake can begin
  at any seat.
- `hexset.catanatron`: an optional adapter to Catanatron's arena
  (`pip install -e "src[catanatron]"`), replacing the standalone
  `catan-bridge` repo. Translates a live `catanatron.Game` into a
  `hexset.Game`/`GameState`, maps board ids both ways, resolves `hexset`
  actions onto catanatron's `playable_actions`, and exposes a
  `catanatron.models.player.Player` (registered as `DC:<entrant>`) so any
  `hexset` bot can be seated in a Catanatron duel and vice versa.
  `python -m hexset.catanatron.duel` shards a duel across worker processes
  and pins catanatron to a fixed commit; the base `hexset` package stays
  catanatron-free.
- `hexset.tuning` can now fit heximax's weight profiles: `evaluator=
  "heximax-trading"` / `"heximax-notrade"` build `kind="heximax"` entrants.
  `heximax.heximax()` gained a `weights` keyword.
- The live trade offer is now part of the observation:
  `encoding.global_features` gains 18 features at four players — the
  standing offer's give/want bundles, the proposer's seat, and who has
  answered. The information-set record grows matching fields
  (`offer_give`, `offer_want`, `offer_proposer`, `offer_answered`).
  **The ONNX contract bumps to v3** (27 inputs, outputs unchanged); a
  consumer must fill these fields before deploying a v3 checkpoint.
- `hexset.migrate`: function-preserving checkpoint migration onto a wider
  observation. New `embed_global` columns are zeroed, so a migrated
  checkpoint plays exactly as its source until trained further. A
  checkpoint from before this change cannot be loaded without migrating it
  first.
- A public-knowledge ledger, `hexset.ledger`: a new `PublicLedger`, created
  with the `Game` and carried through `imagine()`, tracks each seat's
  reconstructed hand as `known[5]` (a certified per-resource lower bound)
  plus `unknown`. A robber/knight steal credits the thief's gain to
  `unknown` only, never revealing the true resource taken.
  `encoding.global_features` gains 18 more features at four players (each
  opponent's `known`/`unknown`, seat-relative); the information-set record
  grows `ledger_known` `(players, 5)` and `ledger_unknown` `(players,)`.
  **The ONNX contract bumps to v4** (29 inputs, outputs unchanged); a
  consumer must supply the ledger fields before deploying a v4 checkpoint.
  `hexset.migrate` zero-pads the tail growth automatically.
- `hexset.heximax`: an honest handcrafted bot that reads every opponent
  through the public ledger and public counts rather than the true hand
  (`search2` reads the true hand). Plugs into the existing `Bot`/`arena`
  API via a `Belief` model, an `HonestEvaluator`, and a max^n search with
  PIMC opponent determinization. Three presets: `heximax` (honest, three
  offers a turn), `heximax-omni` (the same bot reading every true hand, for
  measuring what honesty costs), and `heximax-notrade` (no-trade weights,
  offer budget zero). The PIMC determinization count is fixed at `k=1`; the
  `heximax-k2`/`-k4`/`-k8` ablation presets are not shipped.
  `heximax` also gained protocol-free trade valuation (`deficit`/`surplus`,
  `candidate_bundles`, `score_proposal`, `accept_rule`, `counter_of`,
  `rank_partners`) and a minimal adapter, `Heximax.propose_actions`, that
  replaces the engine's one-for-one trade sample with heximax's own scored
  candidates.

### Changed

- **`Game.state` is now a method, not a field.** `game.state(seat, *,
  hidden=True)` is the access path: `hidden=True` (the default) returns
  `seat`'s information-set `View` (`hexset.view`, moved from
  `hexset.bots.heximax.belief.Belief` -- `Belief` is kept as an alias);
  `hidden=False` returns the true `GameState` (the same object every time,
  never a copy) and is the only sanctioned way to read it from outside the
  engine. The three sanctioned callers are `hexset.bots.search2`,
  heximax's own `omniscient` mode, and the Catanatron adapter when it hosts
  a Catanatron bot. `Game.set_state(state)` replaces the true state outright
  (the one write a determinizer or an undo needs) without exposing the
  now-private `Game._state` field.
- **HexSet is licensed GPL-3.0-only** (was AGPL-3.0). One licence for the
  whole distribution; third-party components are listed in `NOTICE.md`.
- **One distribution, `hexset`.** `engine/` and `src/hexset_ui/` are gone;
  everything now ships from `src/` as `hexset` (engine, bots, ledger),
  `hexset.bench`, `hexset.server`, `hexset.clients` and the sibling
  `heximax`, under one `pyproject.toml`. Update any import of `benchmarks.*`
  to `hexset.bench.*`, and of `hexset_ui.*` to `hexset.server.*` or
  `hexset.clients.*`; the PyPI/Docker distribution name changes from
  `hexset-ui` to `hexset`, and the MCP server's advertised name from
  `hexset-ui` to `hexset`. `onnxruntime` moves from a hard dependency to the
  `.[server]`/`.[clients]` extras. `hexset_ui/record.py`'s duplicate of
  `hexset.onnx_record` is deleted now that the latter no longer needs torch.
- **`hexset.bots` holds every heuristic bot.** `hexset/bots.py` and
  `hexset/evaluate.py` move into the package as `hexset.bots.search2` and
  `hexset.bots.evaluate`; `heximax` moves in alongside them as
  `hexset.bots.heximax`, split by concern (`belief`/`evaluate`/`search`/
  `trade`/`presets`). Public API unchanged: `from hexset.bots import
  SearchBot`, `from hexset.evaluate import Weights` and `import heximax`
  (now a deprecated shim) all still resolve the same names.
- The engine's own test suite now runs in place at `engine/tests` instead
  of a separate `dev-HexNet` checkout.
- **Package renamed `catan` → `hexset`**, ahead of release as its own
  public repo under GPL-3.0-only. `import catan` → `import hexset`
  throughout; `CatanNet` → `HexNet`; the `CATAN_EXPORT_COMMIT` env var →
  `HEXSET_EXPORT_COMMIT`. Every source file under `hexset/`, `benchmarks/`
  and `tests/` now carries an SPDX `GPL-3.0-only` header, and a `LICENSE`
  file is added.
- `hexset` split into `hexset` (engine, bots, ledger) and a sibling package
  `hexnet` (PPO/training research), ahead of HexSet and HexNet becoming
  separate repos. `collect`, `ddp`, `distill`, `distill_train`, `expert`,
  `export_onnx`, `league`, `migrate`, `model`, `netbot`, `policy`, `ppo`,
  `readout`, `rewards`, `schedule`, `selfplay`, `train`, `widen` and `run/`
  move to `hexnet` (e.g. `import hexset.train` → `import hexnet.train`);
  training-bound benchmarks move to `hexnet/benchmarks/`. `hexset` still
  declares only numpy and never imports `hexnet`: `hexset.arena` gained a
  registry (`register_entrant_kind`, `register_evaluator_provider`,
  `register_checkpoint_loader`, `register_leaf_evaluator_factory`) that
  `hexnet.netbot` populates on import. `relative_points` moved from
  `rewards` to `victory`; `NUM_PAIRS`/`pair_index`/`pair_mask` moved from
  `policy` to `actions` — both re-export their old names.
- `heximax` split out of `hexset` into its own top-level package
  (`hexset/heximax.py` → `heximax/__init__.py`). `hexset.arena` gained
  `register_preset` and `hexset.tuning` gained `register_heximax_evaluator`;
  `heximax` calls both on import to register its presets and evaluator
  names. `hexset` never imports `heximax`; consumers (`benchmarks.duel`,
  `hexset_ui`) import it explicitly.

### Removed

- ONNX contract 1 is no longer served. `onnxbot` refuses a contract-1 or
  contract-unspecified checkpoint by name; the server serves contracts 2, 3
  and 4 only. `encoding_v1.py` and `OnnxPolicy` are deleted.

### Fixed

- `heximax-omni` priced trades against a hand that did not exist:
  `_move_hand` folded a non-knower's hand into one all-one-resource total,
  which is exact for the honest bot but wrong once `omniscient` scores
  every hand verbatim. `_move_hand` now takes an `exact` flag, and
  `_partner_delta` passes `self.omniscient`. Only `heximax-omni`'s
  behaviour changes.

## 0.13.0

### Added

- `catan.widen`: function-preserving checkpoint widening (Net2WiderNet).
  Every trunk unit at width `d` is copied to fill width `D`; the wide net
  reproduces the narrow net's logits and values exactly. `--noise σ` adds
  Gaussian noise to the copies' incoming weights. `catan.train --resume`
  continues from a widened checkpoint with no new flag.
- `--mix` accepts a table entry, `table(a|b|c)=f`: in a share `f` of games
  the learner takes one seat and every other seat is an independent draw
  from the pool `a|b|c`, with replacement.
- `benchmarks.duel` records the seat geometry of every verdict — `blocked`
  (`[a, a, b, b]`) or `interleaved` (`[a, b, a, b]`) — as a new `geometry`
  field.
- `--geometry {blocked,interleaved}` on the arena path, default `blocked`
  (reproduces every prior verdict bit for bit). The versus path can only
  seat interleaved and refuses `--geometry blocked`.

### Changed

- An offer with no explicit `ask` is now put to the table in random order
  instead of clockwise from the proposer; `trading.responders` still uses
  clockwise order as the eligibility list.
- **The piece supply is now enforced: 15 roads, 5 settlements, 4 cities a
  player.** `state.can_place_settlement`, `can_upgrade_to_city` and
  `can_place_road` refuse a piece that is not in the box, so
  `legal_actions` stops offering builds that cannot be built. Previously
  unlimited.
- `benchmarks.duel` imports `catan.collect` and `catan.train` lazily, so
  the module loads on a box without torch.

## 0.12.0

### Added

- `benchmarks.aivat`: AIVAT variance reduction for duel verdicts —
  subtracts the chance-conditional expected value at every dice roll,
  dev-card draw, and robber steal from the observed outcome. `--check`
  replays a recorded verdict bit-identically to validate the estimator
  against it.

## 0.11.0

### Added

- `benchmarks.human_agreement`: scores the policy against recorded human
  decisions one decision at a time — **top-1 agreement** and **log-loss**,
  each against a matched null (uniform over the legal option set at that
  position). Decisions with a single legal action are excluded and counted
  separately. Results break down by `ActionType`, `Phase`, and game
  progress, and confidence intervals are clustered on the game rather than
  the position.

## 0.10.0

### Added

- `--mix` now accepts any arena entrant spec, not just two hardcoded
  names: `search2-offers3`, `mcts:<ckpt>@64`, `network:<ckpt>`, or any
  preset, resolved through `collect.named_opponent`. `collect.check_mix`
  refuses a mistyped entrant or missing checkpoint before the run starts.
- `collect.RESERVED_MIX` names `greedy` and `parent`, which keep resolving
  to the same bots as before.
- `benchmarks.mix_cost`: reports what a `--mix` costs a PPO iteration, per
  decision and per shard.

### Fixed

- The in-process collector's `--mix` fell through to the parent checkpoint
  for any name other than `greedy`. Both collectors now build opponents
  through the same function.

## 0.9.2

### Fixed

- **`benchmarks.rank` and `benchmarks.sibling` no longer freeze one chance
  outcome per sibling.** A chance child (a `MOVE_ROBBER`, `PLAY_KNIGHT` or
  `BUY_DEV_CARD` row) is now scored as the mean over `--chance-draws`
  independent draws (default 8), with the rollout budget for that child
  partitioned across its draws rather than duplicated. `--chance-draws 1`
  restores the old single-draw behaviour exactly. Every number either
  probe produced before this change was taken under the single-draw path.

### Added

- `catan.mcts.draws_hidden` and `catan.mcts.sampled_children`: the public
  predicate for which edges are chance edges, now shared by the tree and
  the probes.
- `benchmarks.rank.head_row`/`Row`, `.share`, `.lane_plan` and `.chance`
  (a payload reporting how many rows/children drew and the residual
  spread averaging leaves), plus `benchmarks.sibling.Spread.chance_children`
  and `.chance_spread`.

## 0.9.1

### Fixed

- `catan.mcts` no longer freezes the three chance edges (`MOVE_ROBBER`,
  `PLAY_KNIGHT`, `BUY_DEV_CARD`): each now uses a keyed `_Chance` slot so a
  repeated outcome reuses its child and the edge's `Q` becomes a true
  average over draws, instead of the first visit's single frozen outcome.
  Off-path (chance-free) search behaviour is unchanged.

### Added

- `catan.actions.victim_of` (was `_victim`) and `catan.mcts.HIDDEN_DRAW`
  (the three action types whose `apply` resolves a hidden card), both now
  public.

## 0.9.0

### Added

- A **quantile value head**: `ModelConfig(value_head="quantile")` widens
  the existing `"linear"` head to `players × quantiles` outputs
  (`ModelConfig.quantiles`, default 32); its forward still returns the
  `players`-vector mean, so `V`'s shape and meaning are unchanged
  everywhere else. The full tensor is exposed via `Prediction.quantiles`
  and `Evaluation.quantiles`.
- The matching quantile value loss (quantile Huber,
  `QUANTILE_HUBER_KAPPA = 1/30`) in `catan.ppo.minibatch_terms`;
  `quantile_levels` and `quantile_huber_loss` moved from
  `benchmarks.head_swap` into `catan.model`.
- `Stats.value_mse` / `Terms.value_mse`: the plain squared error of the
  mean, logged alongside `value_loss` and never differentiated.
- `catan.league --value-head` and `--quantiles`; asking for `quantile` off
  a scalar base warm-starts every level from the scalar head's own output
  (`catan.model.quantile_warm_start`).
- `catan.train --quantiles`, alongside the existing `--value-head`.

### Changed

- With any `value_head` other than `"quantile"`, a full training update
  remains bit-identical to `0.8.0`.

### Fixed

- `CatanNet._emit` now builds `logits`, `give`, `want` and the value read
  in a fixed, explicit order; the previous order-dependent gradient
  accumulation could make a default-config update diverge after one
  optimiser step.

## 0.8.0

### Added

- **Board-paired advantage baselines**: `Collector(pair_boards=True)`
  deals games `2k`/`2k+1` on the same board (independent dice), and
  `PPOConfig(pair_baseline=True)` makes each seat's policy-gradient
  terminal `r - (r + r')/2` against its paired game's same-seat reward.
  `catan.league --pair-boards` turns on both. With pairing off, dealing,
  casting, advantages and value targets are bit-identical to `0.7.2`.
- `benchmarks.noise_scale --paired`: runs the gradient-noise estimator on
  a board-paired cohort, reporting both the raw and pair-adjusted
  advantage streams from one batch.

## 0.7.2

### Fixed

- `run.manifest.freeze` read git provenance after creating the run
  directory, so a frozen run's `git_dirty` field was always `true`.
  Provenance is now read before anything is written.

## 0.7.1

### Added

- `catan.league --learner-order`: permutes the fixed table order the
  league rotation otherwise leaves invariant (`collect.league_caster`
  takes an `order`).

## 0.7.0

### Added

- `catan.league`: the table league — N learners share every game in one
  directory, each with its own `PPOConfig` overrides; a run is rated by
  its control arm, `learner0`.
- `catan.run`: a run is a directory with a frozen manifest
  (`freeze`/`load`) recording its parameters, resolved config and
  repository provenance before it starts.
- `catan.export_onnx`: `.pt` → `.onnx` conversion, behind a new `export`
  optional dependency; `torch` stays an unlisted hard dependency.
- A rolling checkpoint ring (`prune_recent`, keeps the newest N
  `recent-*.pt`) and blowout preservation (`preserve_blowout`, keeps the
  pre-update weights and offending batch when the training brake fires).
- Per-seat PPO overrides: `adam_eps` and an entropy-controller gain
  (`nudged`).
- `selfplay.owned` and a `learners` gate on `Collector`, so several
  learners can record from one game (`learners=(0,)` is the
  single-learner case).

### Fixed

- `catan.distill_train`'s manifest parameters no longer matched its
  parser; both are aligned again.

## 0.6.1

### Changed

- Duels are **antithetically paired by default** in `train.versus` and
  `arena.compete`: every board is played under both seat assignments and
  the two readings are averaged. Pass `antithetic=False` to reproduce the
  previous behaviour.
- Duels now draw a different seed per pair rather than sharing one
  `--duel-seed` across all comparisons.

### Fixed

- `benchmarks.duel` seeds a stochastic entrant from a hash of its **spec**
  rather than its argument position, so an entrant plays the same
  wherever it sits and a swapped duel measures the swap. Self-duels keep
  the positional tie-break.
- `benchmarks.duel` writes a verdict by default rather than only on
  request, and no longer writes `--json` output twice.
- `arena.wilson` no longer returns an upper bound above 1.0 at `p = 1`.
- A net is now rebuilt from the head shapes recorded in its own
  checkpoint (`catan.model.config_from_args`); `benchmarks.minibatch_iso_kl`
  and `benchmarks.noise_scale` previously could not load an
  `mlp`-policy-head checkpoint at all.

## 0.6.0

### Added

- `--critic {gae,none,aux}`: the value head's route into training as one
  flag; the head module is built in every mode so checkpoints stay
  loadable.
- `--kl-break`: a one-sided ceiling on a finished epoch's mean KL, with
  `epochs_taken` telemetry.
- `--fused`: the trunk's gather/scatter as dense row-normalised adjacency
  GEMMs; opt-in, since it only wins on GPU.
- A flat wire format for worker episodes, replacing the previous
  per-episode framing.
- `--rival`: every eval also duels a rival run's checkpoint at the
  matched iteration.
- `--detach-value` on `catan.train`.
- `benchmarks.generate --bot network:<path>`: records a trained
  checkpoint's self-play for `benchmarks.behaviour`.

### Changed

- Evaluation protocol: gates now decide on the matched-rival duel;
  `greedy` is demoted to a mix-exploitation canary; external anchors
  calibrate.

## 0.5.0

### Added

- `DistillConfig.contested_only` and `.hard_target`: train the policy
  only on rows where the search overruled it, toward its argmax rather
  than its visit distribution.
- `DistillConfig.anchor`: a cross-entropy toward the recorded prior on
  the rows `contested_only` zeroes out.
- `DistillConfig.stake_scale`: weight a contested row by what the
  correction is worth (the search's own Q-gap).
- `DistillConfig.buffer_iterations` and `.refresh_prior`: train on
  several iterations of collected rows, recomputing the filter and
  anchor against the live policy.
- `DistillConfig.pack_contested`: the policy loss term is trained on its
  own densely packed minibatches of contested rows.
- `ModelConfig.value_head` and `.policy_head`: ablatable readout shapes —
  `linear`, `mlp`, `pooled`, `mlp_pooled`, `attn` for the value head;
  `linear`, `mlp` for the policy head. Exposed as `--value-head` /
  `--policy-head` on both trainers and recorded in the checkpoint's
  `args`. Both default to `linear` and build identical modules, so every
  existing checkpoint stays loadable.
- `benchmarks.head_shape`: sweeps readout shapes by refitting a head on a
  frozen trunk.
- `catan.distill_train --collect-workers`: shards the searched collector
  across processes.
- Distillation statistics now split agreement by contested vs. settled
  rows, and report entropy, anchor loss, and contested-row counts.

### Changed

- `catan.ppo` and `catan.train` can cut the value loss off the trunk
  (`--detach-value`).
- `benchmarks.rank` gained a head learning-rate sweep.

### Fixed

- **`legal_actions` could re-enumerate a trade already declined this
  turn**, letting a seat spend its whole offer budget re-asking the same
  question. `Game.offered` now records the bundles put to the table this
  turn and the sample skips them; `imagine` copies the set. **This
  changes the action space every run measured before this change played
  under.**
- `--learning-rate` is now honoured on resume in the distillation
  trainer.
- The attention head's pooling query is now built off the head, not the
  trunk.
- The from-scratch benchmark arms deal 128 lanes rather than inheriting
  512.
- The zero-sum check now tolerates float32.
- Valued corpora are now collected from the parent checkpoint rather
  than the trained control.

## 0.4.0

### Added

- `catan.collect`: self-play collection sharded across worker processes
  with CPU inference.
- `catan.ddp`: the PPO update data-parallel across CPU worker processes,
  behind `--update-workers`.
- `catan.schedule`: the learning rate as a controller driven by
  `approx_kl` instead of a fixed schedule.
- `catan.selfplay.Collector.cohort` / `catan.train --collect-mode`: deal
  a fixed block of games and play every one to completion, instead of
  refilling a lane the moment its game ends. `--async-collect` now
  requires `--collect-mode stream`.
- TD(λ) value targets behind `--lam`.
- Opponent mixing in the collector, plus a frozen evaluation ladder
  including `search2-offers3` as a rung.
- `catan.ppo.Terms` carries `policy_term`, `value_term` and
  `entropy_term` (the loss summands, still attached to the graph) so a
  caller can differentiate one term at a time.
- `benchmarks.noise_scale`: the gradient noise scale (McCandlish et al.
  2018) from one collected batch, decomposed over the objective's terms.
- `benchmarks.duel`: two checkpoints head to head on identical boards,
  scored as paired terminal VP.
- `benchmarks.minibatch_iso_kl`: the learning rate that holds step
  length constant across minibatch sizes.
- `benchmarks.rank`: whether the value head orders siblings the way the
  truth does.
- `benchmarks.training_loop`: production-shape sync/async PPO timing.
- `catan.arena` takes `network:<path>` wherever a preset name is taken,
  and `netsearch:<path>` / `netgreedy:<path>` swap a checkpoint in as the
  leaf evaluation of the ordinary search. `network:<path>@<offers>`
  imposes an offer budget on the checkpoint. `arena.pooled` groups
  standings by base name.
- `catan.train` prints its effective device and worker counts and warns
  when running crippled (e.g. silently on CPU).

### Changed

- `catan.selfplay.Collector` encodes a worker tick as one vectorized
  NumPy batch laid out as the model's packed input.
- Board-template lookups are cached, and `catan.encoding._template`'s
  cache is raised to 4096 boards.
- Collection can overlap the GPU update behind `--async-collect`,
  training on a policy one iteration stale.
- `benchmarks.duel` defaults `--workers` to 26 unless both sides are
  bare networks.
- `lam` and `minibatch` are now columns in `log.jsonl`.
- The devcontainer now installs Python.

### Fixed

- **`--resume` was discarding `--learning-rate`**, along with:
  `--resume` with no checkpoint starting fresh in silence, RNG state
  restored as a CPU tensor on a CUDA resume, the KL gauge reporting a
  negative divergence on on-policy batches, and log scalars not being
  detached in `minibatch_terms`.
- `--workers` could not duel two checkpoints against each other (a
  naming collision shadowed the duel helper).
- A duel now sets an explicit torch thread count under a `--cpus` cap.
- The lambda sweep no longer oversubscribes the box with unneeded update
  workers.
- `summarise` is compatible with `distill_train` again.
- `tests/test_duel.py` no longer imports torch at module scope, so a
  torch-free box can collect the rest of the suite.

## 0.3.0

### Added

- `catan.placement`: a heuristic opening-placement prior over pip count,
  distinct resources reached, and whether a scarce resource is reached,
  fitted by conditional logit over four-player games.
- `benchmarks.placement_policy`: compares any arena entrant's setup picks
  against the prior's ranking of the same legal field.
- `Weights.scarce`: a new evaluation weight for reaching a scarce
  resource, converted via `evaluate.FITTED_SCARCE`.
- `board.scarce_resources`: resources with fewer hexes than the
  commonest.

### Changed

- Arena entrants can now be constructed by name.
- **Every arena number recorded before this release was measured with
  `scarce` at zero** — this release moves the baseline for all of them.

## 0.2.0

### Added

- `catan.distill`: distils the search's visit counts into the policy,
  with Dirichlet root noise and a bootstrapped value target.
- `catan.distill_train`: the expert-iteration training loop.
- `benchmarks.expert_scale`: how synchronized expert collection scales.
- `benchmarks.horizon`: what shortening the value horizon removes.
- `LeafEvaluator` supports fixed-shape leaf inference; the compiled
  search inference path is exposed.

### Changed

- Search leaves are batched across games rather than evaluated one at a
  time.
- Hidden deck shuffles are deferred until a draw actually needs them.
- Linear PUCT edge scores are cached, and the responder scan is no
  longer repeated per offer.

### Fixed

- A loaded network is now placed on the device it was asked for.
- Dirichlet root noise now defaults off (opt-in).

## 0.1.0

### Added

- `catan.board.coords`: cube hex coordinates, neighbours, distance, and
  hexagonal layout generation.
- `catan.board.topology`: vertices, edges and adjacency derived from any
  set of hex coordinates. Vertices are keyed by the three hexes touching
  them, so the key is canonical regardless of which hex reaches it.
  Disconnected and touching islands are supported.
- `catan.board.terrain`: resource and terrain types, including sea and
  gold for Seafarers.
- `catan.board.board`: terrain and number tokens, the official setup
  bags, and the variable-setup rule keeping 6 and 8 off adjacent hexes.
- `catan.board.maps`: base and mini layouts, plus multi-island layout
  construction.
- `catan.state`: occupancy, hands and bank stock; placement legality for
  settlements, cities and roads expressed as layout-agnostic graph
  queries; gross production; gold hex claim counts.
- `catan.economy`: build costs, affordability and payment, bank trades
  at the best rate the player's ports allow, and production payout
  applying the official bank shortage rule.
- `catan.board.ports`: coastlines derived from hex/edge adjacency, and
  the nine base-game ports.
- `catan.roads`: longest road as a longest trail, so loops count in
  full and an opponent's building breaks a route without invalidating
  the roads either side.
- `catan.cards`, `catan.devcards`: the 25-card deck, buying, and the
  four playable effects. Cards bought this turn are held aside until it
  ends.
- `catan.robber`: robber movement, stealing weighted by the victim's
  hand, and discarding on a seven.
- `catan.victory`: victory points, plus longest road and largest army
  with the rule that a challenger must beat the holder outright.
- `catan.game`: the turn and phase machine — snake-order setup,
  rolling, discarding, the robber, the main phase and win detection.
- `catan.actions`: a flat action space sized from the board, with
  legality masking.
- `catan.play`: a random player that plays full games end to end.
- `benchmarks.throughput`: games/sec measurement with environment
  recording.
- `catan.evaluate`: handcrafted position scoring, one score per seat
  rather than a scalar. Combines victory points, expected cards per
  turn, resource diversity, reachable production, progress towards the
  nearest purchase, roads, knights, hand size with a discard penalty,
  and port rates, as an ablatable weights dataclass.
- `catan.bots`: a `Bot` protocol the network will also satisfy, a random
  bot, and a max^n search over the evaluation (`greedy` is the one-ply
  case). Rolls are chance nodes weighted over all eleven outcomes rather
  than sampled; a beam bounds the main phase's branching.
- `catan.game.imagine`: a copy for hypothetical play, with its own
  random stream and a reshuffled deck so a search cannot read the real
  game's upcoming draw.
- `catan.game.to_move`: whose decision the legal actions are — not
  always the current player, e.g. while discarding on a seven.
- `catan.state.copy_state`, `catan.game.ROLL_ODDS`.
- `catan.arena`: head-to-head play with the lineup rotated so every
  entrant sits every seat equally, win rates reported with a Wilson
  interval, an action-per-game cap, and every seat's terminal victory
  points kept alongside the winner.
- `benchmarks.baselines`: runs a lineup and records the commit and
  environment with the result.
- `catan.tuning`: fits the evaluation weights by hill-climbing against
  the incumbent through the arena, with a `confirm` step that plays the
  fitted weights against the starting weights at a large budget.
- `benchmarks.tune`: runs the climb, reporting each duel as it resolves.
- `benchmarks.production_curve`: sweeps candidate `production` values
  against the intact weights to test whether the weight is identifiable
  from self-play at all.
- `catan.evaluate_tiered`: a second evaluation, selectable through the
  `greedy-tiered` and `search2-tiered` presets, kept as a comparison
  baseline rather than the default.
- `catan.encoding`: the heterogeneous graph the model reads. Seats are
  rotated so the player to move is always seat 0; only information the
  perspective player may legally know is encoded (own hand and cards
  exactly, opponents as counts). Board adjacency is cached per board.
- `catan.selfplay`: a vectorised rollout collector holding N games in
  flight and stepping them in lockstep, behind a `BatchPolicy` protocol.
  Trajectories are demultiplexed by seat; the decision-maker is
  `to_move` rather than `current_player`. Finished lanes are refilled in
  place, and an action cap truncates a game that will not end. No
  built-in reward — an `Outcome` reports the winner, terminal points,
  turns and truncation.
- `benchmarks.rollout`: ticks/sec and actions/sec for the collector
  under a trivial policy.
- `catan.rewards`: the scalarisation `catan.selfplay` leaves open —
  terminal victory points read against the mean of the other seats and
  scaled by the ten points a game is won on, zero-sum by construction
  with no discount factor. A truncated game is scored where it stopped.
- `catan.policy`: the torch `BatchPolicy`. One forward, one packed
  host-to-device copy and one concatenated read-back per tick.
  `PROPOSE_TRADE` is a single flat slot; the recorded `log_prob` is the
  joint over slot and offer.
- `catan.ppo`: GAE over per-seat trajectories, the clipped surrogate,
  value loss and entropy bonus. `GAMMA` is a module constant, not a
  config field. The value head is trained on terminal outcomes and never
  bootstrapped.
- `catan.train`: the runnable, resumable loop. Checkpoints are written
  to a temporary file and renamed; the game counter is saved with the
  weights so a resumed run replays its own training set. `--eval-at-start`
  duels the untrained network as a baseline, and duels fix their cohort
  of games in advance.
- `benchmarks.value_head`: what the value head explains, split by stage
  of the game.
- `catan.netbot`: a trained checkpoint as a `catan.bots.Bot`, so a
  network can enter the arena. The checkpoint is loaded once per process,
  keyed on topology and path; the offer budget defaults to the one the
  checkpoint recorded training under.
- `catan.mcts`: PUCT over a learned policy and value, with leaves
  gathered into waves and evaluated together. A node backs up a per-seat
  vector through `catan.bots.STANCES` instead of a scalar and sign flip;
  chance nodes are sampled rather than expanded eleven ways; nodes store
  their positions; terminal nodes take `catan.rewards.relative_points`
  directly. `simulations` counts descents that cross an edge.
- `catan.expert`: `SearchPolicy`, a `BatchPolicy` that runs one tree per
  decision, so expert-iteration games come out of the existing
  `Collector`. Visit counts ride to the transition on `Choice.aux` as a
  `Target`; the recorded value is the root's backed-up mean; actions are
  sampled, not argmaxed.

### Changed

- `catan.actions.Action` carries an `ask` order on `PROPOSE_TRADE`,
  naming who the proposer would rather have take the offer — an offer
  stops at the first player to accept. Enabled by
  `SearchBot(partner_choice=True)` and the `greedy-partner` preset;
  works only under the `paranoid` stance. Records carry the order, so a
  game with a choosing proposer replays.
- `catan.trading.responders` orders an offer round the table from the
  proposer rather than by ascending seat index.
- `relative` is now the default stance for `greedy` and `search2`;
  `greedy-own` and `search2-own` reproduce the old behaviour.
- `benchmarks.throughput.environment` reports whether the working tree
  was dirty; runners default `--workers` to every core.
- `catan.tuning` fits either evaluation, taking an evaluator name and
  resolving the matching `Weights` class and stance; `TUNABLE` becomes
  `tunable(weights)`.
- `benchmarks.ablate` takes an `--evaluator`, so the tiered evaluation's
  terms can be ablated the same way the default's are.
- `catan.bots.SearchBot` takes a `stance` saying how a seat turns the
  per-seat vector into the one number it maximises: `own` (plain max^n),
  `relative` (subtracts the mean of the other seats) or `paranoid`
  (subtracts the best of them). Selectable through the `greedy-relative`,
  `greedy-paranoid` and `search2-relative` presets.
- `catan.arena` entrants carry which evaluation to score with, so two
  evaluations can be played against each other directly.
- `catan.game.roll_dice` takes an optional explicit roll, so a search
  can enumerate the outcomes instead of sampling one.
- `catan.arena` entrants are now a frozen `Entrant` description rather
  than a bot-building closure (`FACTORIES` → `PRESETS` and `spawn`), so
  `compete` can fan out over a process pool and a lineup can go into a
  run manifest verbatim.

## Before 0.13 — hexset-ui 0.1.0

`hexset_ui`'s first release, before it consumed the `hexset` package (it
carried a private copy of the engine; see `docs/engine-divergence-2026-09-02.md`).

### Added

- A browser game of humans against bots: HTTP API with a lobby, MCP server,
  static web client, and ONNX Runtime inference for exported networks.
