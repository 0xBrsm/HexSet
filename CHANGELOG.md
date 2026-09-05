# Changelog

All notable changes to HexSet are recorded here — the `hexset` engine, its
bots (`hexset.bots`), `hexset.bench`, `hexset.server`, `hexset.clients` and
`hexset.gym`, shipped from `src/` as one distribution.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

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

### Removed

- **The public layer.** `Game.valuations`, `Game.publish`/`publish_due`,
  `hexset.trading.publish_valuation`/`checked_valuation`/`NO_VALUATION`/
  `VALUE_SCALE`, and `Bot.valuation` (the protocol method and every
  implementation) are gone, along with the lazy first-event trigger
  machinery (`Game.event_pending`/`awaiting_publish`,
  `hexset.game.run_pending_event`) that only ever existed to let a driver
  publish before an event ran on it.
- **`PUT /api/games/<code>/valuation` and `PostedValuation`.** Forced by the
  above: there is no vector left to post. A human/LLM seat's own trading
  surface is next-task work (`agents/reference/trading-final.md`, item 5);
  `PendingGate` (confirm mode) is unaffected and still records candidates to
  `game.pending`, now from a batched `gains_many` call instead of one
  `accepts` call per candidate.

### Added

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

### Fixed

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

### Changed

- **Playing a Knight is now two actions: play the card, then move the
  robber.** Previously one action carried a target hex and a victim
  together; now you play the Knight, and the board then asks for the robber
  move exactly the way it already does after rolling a seven — pick a hex,
  then a victim if there's a choice. A Knight that wins the game ends it the
  instant your total crosses the threshold, with no robber move at all. The
  board page no longer arms a "cancel" state for the Knight — once played,
  it resolves the same forced robber move a seven does.

  Under the hood: `PLAY_KNIGHT` dropped its operands, so the ONNX action-space
  contract bumps to `"6"` — shared with the trading redesign in flight, so a
  checkpoint traced against contract `"5"` or older is refused by name rather
  than fed a mismatched action space. The Catanatron adapter now maps the two
  hexset decisions onto Catanatron's own `PLAY_KNIGHT_CARD`/`MOVE_ROBBER`
  pair one-to-one, rather than folding them into one hexset action. Old
  recorded games whose Knight actions carried a target/victim no longer
  replay; the only such records this project shipped are the trade-lab bank,
  re-emitted separately.
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

### Added

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

  **This route is not authenticated and cannot be** — holding the link is the
  whole qualification, and everyone playing holds the link. A seat that opens
  its own game's public view is reading every opponent's hand. Every route
  that *acts* still answers a token and still gets its own seat's honest view
  (`state_view` refuses `omniscient` alongside a seat outright), so nothing a
  bot or a training run reads is affected; the exposure is to people, at a
  table, who choose to look.
- Your own row in the player list is your name: an input standing in for the
  line, the way a bot seat's row is a picker. It reads "human" until you type
  something, and what you type reaches the other players' lists and the log.
  Blanking it puts the seat back to unnamed. (`POST /api/name`, unchanged.)
- Player-to-player trading is gone from the browser with the offers that
  backed it (`PROPOSE_TRADE`/`ACCEPT_TRADE`/`DECLINE_TRADE`, `TRADE_RESPOND`,
  the view's `offer` block). The modal a resource card opens is the bank and
  port route, which is unchanged.
- Nobody moves during setup while any seat is still empty. The grace window
  that used to hold a seat open for a fixed time is gone for good, and with
  it `SEAT_GRACE_SECONDS`, `Config.seat_grace`, `POST /api/games`'s
  `seat_grace`, `--seat-grace` and `HEXSET_UI_SEAT_GRACE` — a seat resolves
  when a person opens the link and takes it, the creator picks a bot for it,
  or the creator closes it outright (`POST /api/close`), never on a clock.
  `to_move` is `null` and every view's `waiting_for` names the seats still
  open until then; once none is, play starts from seat 0 at full speed.
- Game codes are lowercase (`abcdef`), since a code is only ever seen as a
  URL. Lookups normalise, so a code capitalised on the way into an address
  bar still opens its game, and a game journalled under a capitalised code
  still resumes.
- The browser board seats a bot from the player list: an open seat's picker
  offers every model alongside the seat's current state, and choosing one
  fills the seat for the rest of the game. `POST /api/bot` (`Tables.seat_bot`,
  formerly `swap_bot`) now takes an empty seat as well as one with a bot on
  it, refusing a person's seat and a retired one.
- **`hexset.trading.NETWORK_GATE_ROWS`** (`32`): the most candidates a network
  gate's `accepts_many` will score in one batched forward, beside
  `VALUE_SCALE`. `hexset.clients.onnxbot.NetworkBot.accepts_many` now scores
  only the top `NETWORK_GATE_ROWS` candidates by public rank and declines the
  rest outright; `accepts` is unchanged. The engine still asks about every
  candidate — only a network gate's own evaluation is bounded.
- **`hexset.bots.Bot.accepts_many(view, received, counterparties)`**: a seat's
  private gate answered for a whole batch of candidate bundles in one call,
  defaulting to a loop over `accepts` so an existing bot is unaffected.
  `hexset.trading.trade_event` now asks a seat's gate this way — once for the
  current player over every ranked candidate, then once per counterparty over
  the candidates it accepted — instead of once per candidate bundle.
  `hexset.clients.onnxbot.NetworkBot` overrides it with one batched graph call.
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
- **An embedded ONNX seat now trades.** `hexset.clients.onnxbot.NetworkBot`
  gained `valuation`/`accepts`, both derived from the checkpoint's own value
  head with no new graph output: `valuation` is `tanh(delta_V_r /
  VALUE_SCALE)` per resource, from one batched forward over the seat's hand
  plus its five one-card imagined successors when the graph's declared batch
  dimension allows it; `accepts` is the head's strict preference for the
  concrete post-trade hand. `hexset.trading.VALUE_SCALE`, the pinned
  constant both cite.
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
- `hexset.gym` (`pip install -e ".[gym]"`): `HexSetAEC`, a PettingZoo
  `AECEnv` with one agent per seat and an honest `action_mask`; `HexSetEnv`,
  a single-agent Gymnasium `Env` (registered as `HexSet-v0`) with one
  learner seat and `hexset.arena` bots auto-played at the rest; `register()`.
  `import hexset` stays numpy-only — only `import hexset.gym` needs
  `pettingzoo`/`gymnasium`.
- `hexset.view.View` gained `__eq__`/`__hash__`, comparing `perspective`,
  `omniscient`, `num_players` and `signature()` rather than object identity.
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
### Fixed

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
- `hexset.server.api.spawn_bot` imported `.onnxbot` from `hexset.server`,
  where the module no longer lives after the one-distribution restructure
  moved it to `hexset.clients.onnxbot`. Any `.onnx` model picked in the web
  UI returned HTTP 500 (`ModuleNotFoundError`); now it imports from
  `hexset.clients.onnxbot`.
- `heximax-omni` priced trades against a hand that did not exist:
  `_move_hand` folded a non-knower's hand into one all-one-resource total,
  which is exact for the honest bot but wrong once `omniscient` scores
  every hand verbatim. `_move_hand` now takes an `exact` flag, and
  `_partner_delta` passes `self.omniscient`. Only `heximax-omni`'s
  behaviour changes.
- `hexset.catanatron.state.translate` left `Game.valuations` at its empty
  default instead of one all-zero vector per seat, so `hexset.encoding`
  raised `IndexError` reading a bridged position; `tests/catanatron`'s
  three white-box suites read the upstream `catanatron.Game`'s `state`
  field as `_state` (a leftover from a project-wide sed that meant to
  touch only `hexset.game.Game`'s newly private field).

### Changed

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
- ONNX contract 1 is no longer served. `onnxbot` refuses a contract-1 or
  contract-unspecified checkpoint by name; the server serves contracts 2, 3
  and 4 only. `encoding_v1.py` and `OnnxPolicy` are deleted.

### Changed

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
