# Changelog

Changes to the HexSet distribution. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 1.0.0

The first stable release: the engine, the gym and the served table, with no
playing bots of its own. From here on HexSet follows Semantic Versioning for
its public interface: every module with no leading underscore in its path, as
far as its `__all__` lists, and the HTTP, MCP and ONNX contracts
(`docs/guide.md`, "Public interface"). Everything else is internal.

Breaks clients that import the bots from `hexset.bots`, construct an
`Entrant` without a `kind`, name `heximax`/`rehex` without loading a runtime
that registers them, set a removed `TradeParams` field, import an
underscore name from `hexset.trading`, play a move the engine now refuses
(any build but the owed Road Building road, a robber move that skips a
victim or goes to sea), register two different bots under one name, start
the server with `--checkpoint`, run `python -m hexset.clients.botclient --model`,
import `BotRunner`, `LocalSearchBrain`, `LocalTransport` or `RecordBrain`
from `hexset.clients.botclient`, or import a module that is now internal by
its old name (below). A checkpoint that declares a removed `TradeParams` field
in its metadata loads as before, the key ignored like any unknown one.

### Removed

- **HexSet no longer ships playing bots.** Heximax and Rehex
  (`hexset.bots.heximax`, `hexset.bots.rehex`, `hexset.bots.winning`,
  `hexset.bots.evaluate`), their bench tools (`hexset.bench.ablate`,
  `weight_sweep`, `profile_heximax`) and `docs/heximax.md` leave the
  distribution. The engine, the gym and the served table are what HexSet is;
  a bot is something a consumer brings, and keeping two particular bots
  inside the engine made every change to them an engine release and every
  engine test lean on them. `hexset.bots` keeps what any bot composes: `Bot`,
  `TradeGate`, `TradesBy`, `RandomBot`, `determinized` and `stances`.
- **The opening-placement heuristic goes with them**: `hexset.placement`
  (pips plus fitted weights for resource variety and scarcity, and
  `PlacementBot`, which seats any bot's setup settlements by it), the
  `random-placement` preset, `Entrant.placement` and
  `bench.placement_policy`. It is a strategy, and only those bots read it.
- **So do the readings only a bot's strategy used**: `View.expected_hand`,
  `table_holding`, `steal_odds` and `deck_odds`, `victory.takes_army` and
  `road_to_take`, `roads.longest_with_new_roads`, `board.scarce_resources`
  and `chance.Forced`.
- **`Entrant` loses `expansion_value` and `fragility`**, which only one
  bot's factory read, and **its `kind` no longer defaults to `"heximax"`**:
  it is required.
- **`hexset.arena.no_trade()` is gone**; it was one bot's config with its
  offers at zero, and belongs to that bot.
- **`hexset.bots.stances` keeps the three stances a tree search backs up**
  (`own`, `relative`, `paranoid`). The `win` stance, `win_at`,
  `WIN_TEMPERATURE` and `win_this_round` are calibrated to one bot's
  evaluation units and leave with it; `mcts.Search` never accepted `win`.
- **`TradeParams.weighted_offers`, `offer_cards` and `max_take_cards`**,
  with the code paths that read them: expected-gain ranking in
  `fragments.choose_initial`/`choose_remainder` and
  `TradeProtocol.accept_share_many`; the proposal-width filter on `menu` and
  `counter_menu`; the take-side check in a gate's own caps. Nothing set any
  of them, and each is a branch every trade decision walks past.
  `max_give_cards` and the fragment plan are unchanged.
- **The `discard` chance kind and the `discard` method of every `Chance`
  source** (`random_discard` with it). Discards are always a seat's own
  action.
- **Dead engine helpers:** `victory.winner`, `devcards.play_road_building`,
  `robber.DISCARD_THRESHOLD`, `economy.BANK_TRADE_RATIO` (use
  `board.ports.BASE_TRADE_RATIO`), `record.append`,
  `record.steps` (use `moves`), `play.summarise`, `View.exact`,
  `Search.rank` and `mcts.terminal_relative_points` (use
  `victory.relative_points`).
- **Dead trading helpers:** `execute_trade` and `Game.execute_trade` (use
  `execute_agreed(..., ask_actor=False, ask_counterparty=True)`),
  `offer_candidates`, `PARAM_NAMES`, the `Gate` alias and the `"response"`
  role alias in `consent_gain`.
- **`Verdict.exhausted`, and the duel verdict's `exhausted` and `truncated`
  keys.** They were always zero once a stuck game raises.
- **`hexset.catanatron.state.translate`**, a test oracle only; it lives
  under the tests.
- **`hexset.clients.netbot.load_runtime`** (use `hexset.arena.load_runtime`),
  the re-exports of netbot names and `options_for` from
  `hexset.clients.onnxbot`, and `modelmeta.DEFAULT_TRADE_FLOOR` and
  `DEFAULT_GATE_PLIES` (use `hexset.trading.UNLIMITED`).
- **`hexset.server.seating`'s copies of the engine's seat rules**
  (`start_at`, `locked_of`, `settle`, `snapshot`, `next_unlocked`,
  `lock_seat`); use `hexset.game.start(first=, locked=)` and `lock_seat`.
  Also `webplay.ROUND_NOTE_KINDS`.

### Fixed

- **The robber must rob when it can.** `move_robber_to` refuses a victim who
  is not on the hex or holds no cards, and accepts no victim only when
  nobody there can be robbed.
- **A planning gate can counter an open offer.** Open ("any card") offers
  reach the protocol's new `respond_any` under `fragment_trades`, so the
  fragment plan names the cards. Before, the engine default answered them.
- **Lineups with retired seats open equally for each side.** The first turn
  rotates among the playing entrants (`deal_seats`, `deal_game(first=)`).
  Before, side A of `a,b,retired,retired` took three first turns in four.
  Results for full tables are unchanged.
- **The antithetic pair's second game swaps the sides for every duel
  geometry.** `compete(complement=)` names the second seating, and
  `hexset.bench.duel` builds the side swap from `--geometry`. The old
  half-turn shift mapped `abab` onto itself and moved side A of
  `a,b,random,random` onto a third party's seat. A complement that would
  move a retired entrant onto a playing seat is refused.
- **`compete_batched` stops on a game that reaches the turn cap**, raising
  `Exhausted` the way `compete` does, instead of scoring it as a loss for
  the side it stalled.

### Changed

- **The public interface is the code's own.** Every public module lists it
  in `__all__`, and a module wholly internal is named with an underscore:
  the trading submodules (`hexset.trading._engine`, `_params`, `_policy`,
  `_fragments`), the server's session, journal, seating and bot runner
  (`hexset.server._webplay`, `_journal`, `_seating`, `_runner`),
  `hexset.clients._modelmeta`, and the catanatron adapter's mapping modules
  (`hexset.catanatron._actions`, `_board`, `_names`, `_state`). The board
  layout and the action and trade wire formats move from the session into
  the public `hexset.server.wire`. The rules modules (`state`, `economy`,
  `devcards`, `roads`, `robber`, `victory`, `ledger`, `play`) are public:
  every bot reads them. `tests/test_public_surface.py` checks each public
  module's `__all__` and that every function and class it lists has a
  docstring.
- **Registration is one name, one bot.** Registering a different factory,
  preset or spec under a taken name raises; the same one again does nothing.
  Shipped names (`random`, `retired`, `catanatron`) and prefixes (`network:`,
  `mcts:`, `catanatron:`) cannot be registered. Spec prefixes resolve
  longest first, after presets. The `network` and `mcts` kinds take the
  checkpoint loader installed last.
- **`compete` on a lineup with retired seats needs a `complement`** (or
  `antithetic=False`) for antithetic pairing; the duel builds one.
- **While a Road Building road can still be placed, only that road may be
  built.** The engine enforces what `legal_actions` already offered.
- **On a board with no desert the robber starts beside it** (`state.OFF_BOARD`,
  `-1` in the state, the record and the API), and the first seven or knight
  places it on any terrain hex, as Seafarers has it. It started on hex 0,
  blocking a producing hex from the first roll. **The robber never goes to
  sea**: a sea hex is not a legal robber target (`robber.may_stand`).
- **`hexset.trading`'s public surface is its `__all__`.** The underscore
  helpers, `show_offer`, `show_response`, `side_counts` and
  `InitialSelection` are no longer exported; they stay importable from the
  submodules as internals. `TradeParams.enumerate_give` is now
  `enumeration_cards`. `trade_event`'s `gate` argument is optional.
- **`Entrant.options` is stored sorted by key**, so equal settings make
  equal, equally hashed entrants that journal as one run. A mapping is
  accepted, and a repeated key is refused.
- **A new served game seats no bots it was not asked for.** A request that
  names none gets an empty table; `server.web --checkpoint` and
  `Config.default_bots` are gone.
- **`hexset.clients.botclient` is an example client of the API, with no
  model.** It meets every duty `your_move` names: a build when one is legal,
  else any action; an accept of any offer it covers, a counter naming cards
  for an open offer, a pass otherwise; one offer of its own a turn, and a
  pick among the answers. It no longer loads an ONNX checkpoint (`--model`,
  `--device` and `RecordBrain` are gone): no checkpoint ships to load, and a
  client playing its own network reads `GET /api/record` itself. The
  server's own bot seats run from `hexset.server._runner` (`BotRunner`,
  `LocalSearchBrain`, `LocalTransport`), moved out of the example.
- **MCP's `your_move` is the server's.** It no longer says `act` while
  setup waits on an empty seat or the previous seat holds its setup turn
  open, when `POST /api/action` refuses the move.
- **The served picker lists what was registered**: every runtime preset,
  then `catanatron` where its extra is installed, then the ONNX files.
- **`HexSetEnv` defaults to three `random` opponents** and refuses `retired`
  as one; `bench.baselines --lineup` and `bench.generate --bot` have no
  default. `hexset.gym.__all__` names only what imports on a base install.
- **The engine's tests seat a bot they register themselves**
  (`tests/traders.py`), the way a consumer's would.
- **Package metadata for an index:** SPDX license (`GPL-3.0-only`) with its
  license file, the README as the long description, authors, keywords,
  classifiers and project URLs. The build requires `setuptools>=77`.

### Added

- **`your_move` and `waiting_on` on every seat's view** (`GET /api/state`
  and every seat POST): `discard`, `answer_trade`, `choose_trade`, `act`,
  `wait` or `game_over`, from the one function MCP, the example client and
  the server's own bot seats all read (`hexset.server.api.your_move`).
  Before, each client derived it on its own; the old example client ignored
  offers waiting on its seat, so a bot's offer held the table for good.
- **Every API client is tested playing a whole game beside trading bots**
  (`tests/server/test_api_clients.py`): two example clients, and an MCP seat
  through the tools alone.
- **A bot registers itself, and every command can load it.** A runtime is a
  module whose import calls `register_entrant_kind`, `register_preset` and
  `register_spec(prefix, parse)` for a parsed spelling such as
  `mybot:depth=3`. `hexset.arena.load_runtime(*modules)` imports runtimes,
  in code or as `compete`'s worker initializer, and every command that names
  bots -- `bench.duel`, `baselines`, `generate`, `trade_census` and
  `server.web` -- takes a repeatable `--runtime`; so does `HexSetEnv`
  (`runtime=`). An unknown name says to load the runtime that registers it.
  `unregister_entrant_kind`, `unregister_preset` and `unregister_spec` take
  one back.
- **`Entrant.options`**, sorted `(key, value)` pairs read with
  `Entrant.option(key, default)`, carries settings only one kind's factory
  reads, so a bot's own knobs need no field on the engine's entrant.
- **`hexset.arena.registered_presets()`** lists the presets runtimes added,
  and the arena has an `__all__`, with `seat_bots`, `run_seated`,
  `deal_seats` and `half_turn` public.
- **`hexset.bots.seat_at(bot, game)`** seats every part of a bot that keeps
  a game: a network gate, a search's, both halves of a `TradesBy`.
- **A determinized world can deal another seat's development cards by the
  caller's own reading of how long each has been kept**: `View.sample(rng,
  hold)`, `bots.determinized.distinct_worlds(..., hold)` and
  `mcts.Search(hold=...)` take any `view.HoldReading`, whose `weight(card,
  age)` scales the unseen cards of each kind. The engine reads no hidden
  card by itself: without one, every unseen card is dealt alike.
- **`TradeProtocol.respond_any`**, and `"respond_any"` in `HOOKS`.
- **Check-only helpers** that raise `ValueError` without writing anything:
  `state.check_road`, `check_settlement`, `check_city`,
  `economy.check_afford`, `devcards.check_play` and
  `board.terrain.check_resource`.
- **`--game-type`, `--turn-cap` and `--trade-mode`** on `baselines`,
  `generate` and `trade_census`.
- **A `catanatron:` spec** (`depth`, `worlds`, `temperature`, `select`).
- **`turn_cap` on `HexSetAEC` and `HexSetEnv`**, and **`Outcome.exhausted`**,
  which separates a game the engine ended at the turn cap from one the
  action cap cut short (`truncated`).
- **`onnxbot.spawn(players=)`** refuses a checkpoint trained for another
  player count.


## 0.105.0

No engine changes.

## 0.104.0

No engine changes.

## 0.103.0

### Added

- **`winning.expected_points` and `winning.leaders`**: a seat's expected
  points as the reader sees them, and the other seats with the most.

## 0.102.0

### Changed

- **`View.turn` carries the game's turn count** (`Game.turns`), public like
  `mover`.

## 0.101.1

No engine changes.

## 0.101.0

### Added

- **`victory.takes_army` and `victory.road_to_take`**: whether one more knight
  gives a seat Largest Army, and the road length that takes Longest Road for
  it. Two bots each had their own copy; both use these now.
## 0.100.0

No engine changes.

## 0.99.0

No engine changes.

## 0.98.0

No engine changes.

## 0.97.0

### Added

- **Counters to counters in the engine.** `Game.counter_steps` (default 1,
  a counter the actor takes or leaves) and `trade_round(...,
  counter_steps=)`. Where the actor picks nothing, `trading.counter_thread`
  carries each seat's counter on, in seat order: the side it was put to
  answers it with its own `respond` -- accept, counter or pass -- the mover
  and that seat in turn, up to `counter_steps` counters, a repeated bundle a
  pass. Every counter is between the mover and one seat. The served table
  still runs one. `trading.CounterThread` holds a thread's rules;
  `hexset.seat.Seat.counter(other, received, any=)` plays one a step at a
  time at a hosted table (`Seat(..., counter_steps=)`, default 1).

## 0.96.0

No engine changes.

## 0.95.0

No engine changes.

## 0.94.0

### Added

- **A served checkpoint can name its trader.** A `trader` key in an `.onnx`
  file's metadata (`clients.modelmeta.trader_config`) names a bot, as a
  lineup names it, that answers every trade for the checkpoint while it
  plays every move. `onnxbot.spawn` -- the web board's model picker -- and
  a lineup's `network:`/`mcts:` entrants seat it through
  `hexset.arena.traded`, the one wrap `Entrant.trader` also uses. An
  entrant's own `<entrant>~<trader>` replaces the file's. A checkpoint that
  names none trades through its own gate.

## 0.93.0

### Removed

- **`TradeParams.weighted_offers` stays.** A gate without
  `accept_share_many` reads a counterparty's chance of taking an offer as
  the sign of its estimate.
- **The pre-#789 control.** `config.PRE_789_TRADE`, `Entrant.fragment_trades`
  and `Entrant.trade_override` (`Entrant.trade` is read directly), and
  `--fragment-trades-a`/`--fragment-trades-b` on `hexset.bench.duel`, whose
  verdict no longer carries `fragment_trades_a`/`fragment_trades_b`.

## 0.92.0

### Removed

- `--pin-weights-a`/`--pin-weights-b` on `hexset.bench.duel`. The engine's
  `observe_trade` hook is unchanged.

## 0.91.0

### Added

- **One bot's moves over another's trading.** `hexset.bots.TradesBy(mover,
  trader)` answers `choose` with the mover and every trade hook with the
  trader, read and written through, so a run's `retune` reaches the seat
  that actually trades. Any bot can move and any gate can trade; a search
  bot used as a trader is only asked about exchanges and never searches a
  move. An entrant names its trader with `Entrant.trader`, spelled
  `<entrant>~<trader>` in a lineup. An entrant with a trader sets no
  bargaining limits of its own; the trader's are the ones that apply.

### Changed

- **One place for a gate's bargaining limits.** `trading.DeclaredTrade`
  carries `trade_params`, `max_offers`, `trade_floor`, `gate_plies`,
  `responder_card_risk`, `fragment_trades` and the offer budget over a
  gate's own `trade`, each setter writing through `retune`. The search
  bot, `NetworkBot` and `GatedSearch` read them from it instead of from three
  copies. Every value they answer is unchanged.

## 0.90.0

No engine changes.

## 0.89.1

No engine changes.

## 0.89.0

### Fixed

- **A replayed record is the position that was played, offers included.**
  Since 0.88.0 every offer and answer goes on the public ledger -- wants,
  wastes, and the cards an offer certifies -- but a `Record` kept only the
  actions and the exchanges, so a replay (and every dataset or journal built
  from one) reached a different ledger than the game had. A record now
  carries `shown`: `(step, order, seat, received)` for each offer or answer,
  `order` the step's exchanges it came after, and `advance` puts each back
  between the same exchanges (`record.shown_by_step`). `Tape.step` takes the
  turn's `game.shown` slice alongside the exchanges, as the arena loop and
  the training lanes now pass it. A record written before this field has
  none, and replays as it always did.

## 0.88.0

### Added

- **The public ledger keeps what each seat has shown it wants and will give
  up** (`SeatLedger.want`, `SeatLedger.waste`, `PublicLedger.show`). Every
  offer, acceptance and counter is public, and until now none of it reached
  the ledger. What a seat asks for becomes a want and what it gives a waste,
  each at the most it has shown; the cards clear them -- a want when the
  seat receives that resource, a waste when it parts with it -- so they carry
  across turns until met. A steal moves neither. The round drivers and the
  served table (restored journals included) show what they see; a hosted
  seat's client reports its table's offers and answers, and a chat ask for
  a card, through `Seat.shown`.
- **An offer certifies the cards it gives.** Where the referee has checked
  the seat covers what it gives, `known` rises to it out of the seat's
  untyped cards. A counter asks only for what the ledger certifies, so a
  seat can now counter for the cards another has just offered.
- **`TradeParams.fit_offers`.** Offers and counters that fit what their
  counterparty has shown (`trading.fit`: one for giving it a resource it
  wants, one for asking only for cards it has offered away, within the
  counts offered) go first; where nobody has shown anything, or nothing
  fitting clears the cutoff, the choice is the one without it. It reads
  what a seat has shown the table rather than a model of what it would
  accept.

The ledger block of `hexset.encoding` is unchanged in shape, but its `known`
counts now include the cards offers certify.


## 0.87.0

No engine changes.


## 0.86.0

### Added

- **Reading a table, the piece-supply corner shows the open seat's
  pieces.** A spectator, anyone at a finished game, and anyone stepping
  through a finished game's rounds already click a player row to see that
  seat's cards; the settlements, cities and roads that seat had left to
  place now show in the board's bottom-left corner too, in its colour and
  counted off the round on screen. With no row open the corner stays
  empty, as the hand area does. A seat playing a live game still sees its
  own supply with nothing to click. Before this the corner was blank for
  every reader, so a replay could not say how close a seat was to running
  out of a piece.


## 0.85.0

### Removed

- **The `reject` round note, and answers naming an earlier offer.** A
  journalled record now writes its trade rounds as a table here plays them
  -- one offer and each seat's one answer, a refusal a silent `pass`,
  `nobody` once every seat has passed -- so a recorded game's log reads
  like one dealt here. Nothing wrote a `reject` but a record, and no table
  here answers an offer after another has opened, so the kind and its
  standalone lines ("Player 3 declines Player 1's offer of ...") go.


## 0.84.0

### Added

- **A journalled record carries its trade rounds** (`journal_of(...,
  notes=...)`). Each round step -- an offer, a counter, a seat's answer --
  is placed after the action it followed and after the exchanges already
  cleared on it, and each exchange is now a line of its own after its
  action, as a manual trade is. The log then reads a recorded game's
  negotiation the way it reads one played here: the offer, the answers,
  and the trade that closed it, on one line.
- **`reject`: one seat turning an offer down** (`ROUND_NOTE_KINDS`). A table
  here keeps its passes silent until every seat has passed; a game recorded
  elsewhere may have seen each refusal, and the log reads it ("Player 3
  declines."). Like an `accept`, it carries the offer it answers.

### Changed

- **An answer to an earlier offer names it.** An `accept` or `reject` of an
  offer other than the one its line opened with stands on its own line,
  naming the offer ("Player 3 declines Player 1's offer of 1 Wood for 1
  Ore"), rather than joining the latest offer's line.


## 0.83.0

### Added

- **Open trade offers: "any" cards, an invitation to counter.** An offer
  may carry any cards on one side (`Offer.any`, signed towards the actor
  like its bundle): cards the actor gives, of the answerer's naming ("any
  card for your ore"), or cards it takes, of the answerer's choosing ("my
  sheep for any card"). An open offer is never accepted as it stands, only
  countered, and a counter to it is an ordinary one that the actor picks or
  refuses like any other. Every exchange that executes is concrete.
  - The engine answers an open offer for any gate (`default_respond_any`)
    with the counter it would send any offer, else a pass: an open offer
    only skips acceptance. A gate may answer open offers itself through a
    `respond_any(view, offer)` hook; an acceptance of an open offer is read
    as a pass. `Seat.answer` answers them the same way.
  - Served tables: `POST .../trade/round` takes `give_any` or `want_any`, on
    one side, each side naming at least one card or holding the any cards;
    the actor must hold only the cards it names. `pending` and
    `trade_round` carry the offer's `any` when it has any; an `accept` of an
    open offer is refused (409), and any counter is taken. The journal keeps
    an offer's any cards, and the log reads them ("offers any 1 card for 1
    Ore"). The MCP `offer_trade` tool takes `give_any`/`want_any`, and an
    open offer in `pending` is shown with its any cards and `can_accept`
    false.
  - The web page draws an any card as "?", the sixth card of an offer's
    rows, on one side at a time. An incoming open offer's accept is
    disabled; it is answered by a counter. Bots do not propose open offers
    yet.

## 0.82.1

### Fixed

- **A trade filed after a bank trade or a build is in the log.** The log
  wrote the exchanges cleared inside an action for every action but those
  two, whose lines are rewritten in place as a run of them grows. A game
  dealt here never files a trade there, but a journalled record
  (`journal.journal_of`) files each exchange after whichever action it
  followed, and those exchanges were applied to the hands and missing from
  the log. They are now written after the run's line, and end the run.


## 0.82.0

### Added

- **Any recorded game opens in the page's replay** (`journal.journal_of`). A
  `Record` that replays is written as a server journal filed under a join
  code, by playing it through a `GameSession` with the journal attached, so
  every line is the one a live table would have written. `/<code>` then
  shows the game finished, and the replay bar steps it round by round. A game
  from any source can be read the way a game dealt here is, with no second
  viewer to keep in step with the page.
- **A journal with no seed opens from itself.** Resuming and replaying used
  to rebuild the board and every roll from the header's seed, which only a
  game dealt here has. A header with `seed: null` opens from its own board
  fields, and the deck and the effects its lines recorded are its chance
  stream (`record.from_events(events, partial=True)`).
- **Journals name their ruleset.** The header carries `rules`, and
  `record.from_journal` reads it back, so a game under other rules (a duel's,
  say) survives the round trip. A header without it is a standard game.


## 0.81.0

No engine changes.


## 0.80.1

No engine changes.


## 0.80.0

### Added

- **Weighted offers** (`TradeParams.weighted_offers`, off by default). A
  fragmented gate ranks its own offers by expected gain: its gain times the
  chance the counterparty takes the offer. Any offer with some chance of
  being taken may open; the fragment cutoff still gates the gain alone.


## 0.79.1

A patch for the prose, as 0.58.1 was. Comments, docstrings, the README, the
guide and earlier entries of this changelog had again taken on where rules
came from rather than what they are -- measurements, the runs and dates
behind fitted constants, review and incident notes. Each rule and constant
now says what it is, and nothing else changed: no engine behaviour, no
constant, no test's assertions. A stray verdict file under `runs/` is gone,
and `runs/` is ignored.


## 0.79.0

No engine changes.


## 0.78.0

### Changed

- **An acceptance is priced by who took the offer again.** An offer goes
  out with some seat in mind, but since 0.74 `default_pick` executed any
  acceptance of it, from whichever seat took it -- a trade the actor would
  never have proposed to that seat, the leader say, went through because the
  leader said yes. `default_pick` now prices each acceptance by its taker,
  as it does a counter, and picks the best above the actor's floor or none.
  What 0.74 settled stands: nobody re-prices an agreed trade when the cards
  move, and a responder's accept binds it.
- **An offer asks only for what the other seats hold between them.** Since
  0.69 the offer menu (`open_candidates`) asked for any cards, so an offer
  could reach a seat not known to hold the card -- and could also ask for
  cards no seat held at all. Every card is in the bank or a hand, so the
  other seats hold what the bank and the actor do not. The menu now asks
  for no more of a resource than that. Which seat holds it still narrows
  nothing.

### Added

- `GameState.dev_ages`: each held development card's age in its owner's
  turns, public -- bought at 0, a turn older at each of its owner's turn ends,
  a play taking off the youngest card that could have been played. One entry
  per seat; `None` for a seat whose history the state was built without (the
  Catanatron bridge's).
- `roads.longest_with_new_roads`: the longest route a seat could have with
  at most `j` new roads, for every `j` up to a budget, by the rules of
  `longest_road` (which is its `j = 0`), optionally with each route's new
  edges.
- `View.mover` and `View.card_played`: the seat on turn and whether it has
  played its development card this turn, set by `View.from_game` (`None`
  and `False` on a view built from a bare state). Not part of the
  information set's identity.


## 0.77.0

### Added

- **Every run keeps its games as they finish (`compete(journal=...)`).**
  `arena.compete` held every game until it returned, so a crash, a kill, a
  dead worker, a stuck game or a container restart lost the whole run, and
  nothing could be read part-way. With `journal` a run writes a header line
  with its settings and then one line per game -- its outcome and, under
  `records=True`, its record -- appended and fsynced the moment the game
  finishes, in whatever order the workers finish them; on an error it keeps
  the games still running before it raises, and a game that exhausts the turn
  cap is written before `Exhausted`. `resume=True` checks the journal's
  header against the run and plays only the games it lacks, so the result is
  the run's own; a journal that already holds a run is otherwise refused.
  `read_journal` and `journal_tournament` read one part-way. `Outcome` now
  carries its `index` and `board_index`. The format lives in
  `hexset.gamelog`, which any other batch can use the same way.
- **The bench CLIs journal by default, with `--resume`.** `bench.duel`
  (beside its verdict, named for the pairing, seed and games), `generate`,
  `baselines` and `trade_census`.
- `hexset.record.append`: one record, fsynced.

### Changed

- `hexset.record.read` skips a torn last line instead of raising, and
  `record.write` flushes after every record.

## 0.76.0

No engine changes.

## 0.75.1

### Fixed

- **`hexset.catanatron.speedups` is gone: it could only break the reference.**
  It sped up the Catanatron reference by patching its board and state copies
  and its evaluator in-process, guarded by digests of the source it was
  written against. The pinned reference (`0xBrsm/catanatron@bd1dd7ec`,
  submitted upstream as catanatron#389) now does that work itself -- a
  board-feature cache inside its alpha-beta player and cheaper copies, on by
  default -- and the patch no longer fits it: its evaluator takes no `cache=`,
  so a search run under `catanatron_speedups()` failed on its first leaf.
  Nothing entered it after the Catanatron bridge and its `--speedups` flag
  left in 0.56.0, so no seat was running it; its tests called the evaluator
  directly and never searched. The reference is as fast as it was, and the pin
  no longer needs its digests recomputed.

## 0.75.0

### Added

- **`roads.longest_road(state, player, extra_edge=)`** measures the route as
  if one more edge were the player's, without writing it into the state. A
  bot asking "does this road lengthen my longest road" had to place the road
  on the state it was handed -- the live game's, behind a seat's view -- and
  take it off again.

## 0.74.0

### Added

- **An observed game can be played forward.** `state.Hidden` (0.51) gave a
  seat's game a way to say "n cards, types unknown", but only as a snapshot:
  every hand-moving transition read or wrote a composition, so the moment
  anybody rolled, built or traded, a game holding hidden piles raised. A bot
  seated at a table hosted elsewhere had to rebuild its whole position from
  the host's snapshot at every decision -- every field the engine keeps (the
  setup snake, the card-per-turn flag, the plays tallied against the deck,
  the turn counter) re-derived by hand, and any gate state keyed on the game
  lost each time. Now `game.observe(game, seat)` gives that seat's game, and
  `actions.apply` plays it forward for every seat's actions: a
  `HiddenHand` takes each public change by name (`move`, with the identity
  kept in `flow` for `PublicLedger.apply_hand_diff`), hidden development
  cards and the deck move by count, and what the seat cannot draw -- the
  dice, a steal, a purchase, a Monopoly's takings -- comes from a
  `chance.Hosted` source the caller feeds (`expect`), with `chance.UNSEEN`
  for a draw the seat was not shown. The host is the referee: applying
  another seat's action takes its word, `legal_actions` answers for the
  observing seat only, and a win on hidden victory point cards is the host's
  to announce. Played in lockstep with the true game, an observed game stays
  exactly `observed_by` of it, ledger included, across whole two-, three- and
  four-seat games. See `docs/hosted.md`.
- **A bot can take a seat at a table somebody else hosts.** The engine's own
  tables drive every decision around a gate -- when the turn's trade event
  opens, the offer budget, the repeat rule, `trade_now`, who hears how a round
  went, what the table's trading is published as -- and a client mirroring a
  foreign table had to copy all of it, and a copy drifts from the engine it
  copies. `hexset.seat.Seat` is the observed game plus the seat's bot,
  with that bookkeeping done by the engine: the client feeds it the host's
  events (`play`, `traded`) and carries its decisions (`offer`, `pick`,
  `close_round`, `answer`, `choose`, `discard`) onto the host's wire, and an
  event the game cannot take raises `OutOfStep` so the client can `resync`
  from the host's own statement of the position. Every verb returns what the
  engine's own stage returns on the true game -- the same offer from the
  served bot included. See `docs/hosted.md`.
- `Chance.draw` and `Chance.surrender` are the two questions only an observed
  state asks; `devcards.buy_drawn` buys from a deck known by its length.

### Changed

- `robber.steal` returns `chance.UNSEEN` for a steal whose card this state
  cannot name, and `game.buy_development_card` for another seat's purchase
  on an observed state. On a referee's state nothing changes.
- `GameSession.state_view` renders an observed game. A finished game
  discloses every seat's cards and true points, and on an observed state
  another seat's are counts: it now discloses what the state holds rather
  than raising `HiddenRead` at every game's end. A referee's game discloses
  exactly as before.

## 0.73.0

### Changed

- **An agreement binds: nobody re-prices a trade at execution.** A round's
  exchange used to be put to both gates again at the moment cards moved
  (`execute_agreed(ask_actor=True, ask_counterparty=True)` in the engine's
  round, the bot sides in the served table's), and `default_pick` declined an
  acceptance of the actor's own offer whenever the actor now priced it under
  its floor. At a synchronous table the second asking returned the first
  answer and changed nothing; wherever time passes between answer and
  execution -- a served table a person takes their time at, a table hosted
  elsewhere -- it let a seat walk away from a deal it had made. Now each side
  judges once: a responder by its answer, the actor by its offer and by its
  pick of a counter. `default_pick` ranks acceptances by the actor's gain but never
  declines one; a counter still has to clear the actor's floor. Execution
  checks the rules only -- both hands still cover the exchange. Each bot
  side's gain is still priced onto the executed `Trade` for the record
  (`execute_agreed`'s new `price_actor`/`price_counterparty`), so a trade
  census reads what it did. `execute_trade`, a bundle composed by hand and put
  to a bot, still asks the bot: it has not agreed to anything yet. At the
  engine's own synchronous table a deterministic gate that offers only what
  clears its own floor, and answers as it prices, plays exactly as before.

## 0.72.0

### Added

- **A seat can limit what it proposes without limiting what it accepts.**
  `TradeParams.offer_cards` is the width of everything a gate *asks of
  others* -- the offers it broadcasts and the counters it answers with -- and
  nothing it signs. The only card limits before were `max_give_cards` and
  `max_take_cards`, which are refusals at consent: a seat that should post
  one-for-one offers had to declare both at one and then turned down every
  two-for-one it was offered as well, or have its menu cut by a filter outside
  the gate after the gate had chosen (a plan whose fragment was two-for-one
  then offered nothing, and said nothing about it). Now the engine's menus
  (`menu`, `counter_menu`) are built at the narrower width, a gate's own
  `candidates` are held to it, and the fragmented protocol plans each fragment
  at it, so a two-card target comes out as two one-for-ones. Undeclared, as on
  every shipped config, nothing changes. Read from and written to checkpoint
  metadata as `offer_cards`.

## 0.71.0

### Added

- **A table can require two road pieces for Road Building.** The printed rule
  plays the card with a single road piece left, placing a single road, and the
  engine offered it whenever a seat had any piece at all. A served table can
  refuse exactly that road: the card is spent, the road is refused as having
  no pieces, and the rest of the turn is lost. `Rules.road_building_min_roads`
  (1 or 2, default 1) is how many pieces must be left: `legal_actions` offers
  the card, and `play_road_building_card` accepts it, only at or above that.
  Every shipped ruleset keeps 1, so no existing table plays differently; a client mirroring
  such a table asks for 2.

## 0.70.0

### Added

- **A gate can choose when in its turn it trades.** The turn's one trade event
  used to open on entering the main phase, always, so a seat could only ever
  go to the table before its first action. A bot that trades after building
  -- spend first, then offer what it still needs, the way a rules bot does
  when its hand is big -- had no way to say so. A gate that implements
  `trade_now(game) -> bool` is now asked at every main-phase decision point
  until the event is used: on entering the main phase and after each of its
  own main-phase actions (`actions.apply`), never between a Road Building
  card's free roads. The event runs at the first yes; a turn with no yes does
  not trade. Still one event a turn, and a gate without `trade_now` opens on
  entering the main phase exactly as before, so the search bot, the network bots
  and every existing table are unchanged. Records file the exchanges under the
  step they cleared inside, which is what `Tape.step` already did, so a
  mid-turn trade replays in place. See `docs/trading.md`, "When the turn's
  trade opens".

## 0.69.0

An offer is broadcast: a player asks the table for the card it needs, and
whoever holds it may take it. HexSet's bots could not. The actor's menu held
only exchanges asking for what the ledger certified some other seat held
(`known_candidates`), or what a sampled belief world dealt it (a search
bot's `offer_worlds`, and the fragmented policy's parent pool), so an offer
for a card nobody was *known* to hold never reached the table. The menu now
reads nothing but the actor's own hand.

**Breaks a client:** `hexset.trading.sampled_candidates` is removed. `menu`
is now the offer menu only; a counter's menu is `counter_menu`.

### Changed

- **Offers ask for any cards.** The engine's default offer menu is
  `open_candidates`: every give the actor's exact hand covers, against any
  cards asked for, at most the gate's enumeration width a side, on disjoint
  resources -- listed once per unlocked seat so a gate can still price who
  takes it. Nothing a counterparty holds, is certified to hold, or could be
  sampled to hold narrows it. `offer_candidates` and `menu` return it for a
  gate with no `candidates` hook of its own. What stays is the physical
  check: a responder accepts only what its own hand covers, and
  `execute_agreed` re-checks both true hands when the cards move, so an
  offer nobody can cover goes unanswered and moves nothing.
- **Counters are unchanged, and now have their own menu.** A counter is put
  to the actor alone, so it still asks only for what the ledger certifies
  the actor holds (`known_candidates`). `default_respond` reads it through
  the new `counter_menu(gate, view, actor)`, which calls a gate's own
  `candidates` hook with `turn=None` -- the convention the fragmented
  policy already used to tell a counter from an offer.
- **A gate with no offer limit asks for a lot more.** `max_offers=None`
  still means "offer while the menu holds something not yet offered this
  turn", and the menu is now every ask the hand can make, not the handful
  the ledger certified. A gate with no opponent model (`estimate_many`) that
  values many asks positively can now run hundreds of rounds a turn; declare
  a budget.
- **`choose_initial` indexes the pool by bundle.** Each partition's opening
  fragment is one lookup instead of a pass over the whole scored pool; the
  choice is identical. The open pool is several times the size of the one
  it replaces, which would have made that pass the fragmented policy's
  cost.

### Added

- **`hexset.trading.has_room(view, seat, received)`**: whether a seat's
  public hand has room for its side of an exchange -- every card it is not
  certified to hold fits among its untyped cards. For a gate's opponent
  model, never for the menu.

### Fixed

- **A network gate never values an exchange its counterparty has no room
  for.** A candidate asking a seat for more untyped cards than its public
  hand holds is priced `-1.0` without building a position, and at
  `gate_plies > 0` only candidates the certified sampled world actually
  covers are rolled forward. Before, `_after` could write a negative ledger
  row, and the continuation could move cards a sampled hand did not have;
  the certified menus never asked for such an exchange, the open one does.

## 0.68.0

A recorded game is supposed to be the unrecorded game with a tape running.
Under the duel variant it was not: `record.recording` wrapped independent dice
whatever the rules said, so every recorded `balanced_dice` game rolled a
different game from the one `play_game` deals, under a record whose `rules`
named the dice deck.

**Breaks a client:** `record.recording(rng)` now requires the ruleset,
`recording(rng, rules)`. A caller passing `chance=recording` to `deal_game`
or `start` raises `TypeError` instead of silently recording the wrong dice;
pass `chance=lambda rng: recording(rng, game_type.rules)`.

### Fixed

- **Recorded games roll the dice their rules ask for.** `recording` wraps
  `chance.for_rules(rules, rng)` -- `Balanced` under `balanced_dice`, `Live`
  otherwise -- the same choice `game.start` makes, now made in one place.
  Reached from `bench.duel --records` (`arena._play_and_record`), from
  `record_game` (`bench.generate`) and from recording gym lanes. On a
  duel-variant table a recorded game now plays move for move as the
  unrecorded one; before, a game could even stall to the turn cap recorded
  and finish in 101 turns unrecorded (seed 97000, game 1658).

  What this touches: any duel-variant result produced with recording on --
  every `bench.duel --records` verdict and every `bench.generate` corpus
  under `--game-type duel-variant` -- was played with independent dice.
  Standard games are unaffected: their rules ask for `Live`, which is what
  they got.

### Added

- **`Record.balanced_dice`**: whether the record's chance stream came from
  the dice deck. The replay seed cross-check regenerates the draws from it.
  It is a fact about the stream rather than a restatement of `rules`,
  because a record written before this release can name `balanced_dice` and
  still hold independent rolls. Absent reads as `False`, which is what every
  such record holds, so they keep replaying.

## 0.67.0

### Not a behaviour change

Every bot the arena builds plays exactly as it did: the shipped presets take
the same pair and the same `-0.25`, and every in-tree custom-weight caller was
updated to name the values it was already getting.


## 0.66.0

The rules were a contract every *caller* honoured and three readers did not.
0.65.0 made a game type checkable at the door; this closes the three places
downstream that still read a 10-point standard game off a module constant, and
the one place a played game forgot which rules it was played under.

### Changed

- **A record carries its ruleset.** `Record.rules` is sealed from the game and
  dealt back by `open_record`, so a `duel-variant` game replays as one. It did
  not: `open_record` re-dealt every record under `STANDARD`, which meant a
  15-point game was replayed at 10 and every tool built on replay -- the
  dataset, the fit, any re-encode -- read it wrong. The field defaults to
  `STANDARD` and is optional on read, because every record written before it
  existed is a standard game.

  `record_game` takes a `game_type`, and `hexset.bench.generate` takes
  `--game-type`, which is what makes a variant dataset reachable at all.

- **`Rules.seven_odds`, read instead of assumed.** `bots.evaluate`'s robber
  risk priced a seven at 6/36 whatever the dice were. The dice deck rolls
  0.16532 -- the repeat discount falls hardest on the commonest sum -- so
  `seven_before_next_turn` and `hand_terms` take the rate as an argument and
  the evaluators pass the position's own. Sub-1% on the standard rules, where
  nothing moves: this is a gauge correction, not a lever. The lever is counting
  the deck, which needs state no bot carries yet.

- **`victory.relative_points` is scaled by the game's own target.** It divided
  by 10 always, so the same finish in a 15-point game returned a reward half
  again as large. Reached from `mcts.terminal_relative_points` and the
  `relative_points` reward mode in `hexset.gym`, both of which now pass the
  position's `winning_points`.

- **`fitting`'s `remaining_production` reads each game's own target.**
  `ChoiceSet` carries `winning_points`, filled from the record. The feature is
  production times the points left to win, and computing "points left" against
  a constant 10 floors it at zero from the tenth point of a 15-point game --
  erasing exactly the runway the feature exists to measure.

None of this moves a standard game: `STANDARD.seven_odds` is 6/36,
`WINNING_POINTS` is 10, and a record without a `rules` key reads as standard.


## 0.65.0

A game type is now a contract -- a ruleset together with the table sizes it is
played at -- and the turn cap is a property of the run rather than of the
module. Both come from the same finding: the engine dealt a four-seat game
under the 1v1 ladder settings without complaint, and then let it run for a
thousand turns before scoring it as a loss.

### Added

- **`rules.GameType`, and the two types that ship.** `STANDARD_GAME` is 10 / 7
  with no shield and independent dice at 2, 3 or 4 seats; `DUEL_VARIANT_GAME` is
  15 / 9 with the shield on and the dice deck, at 2 seats and only 2. A type is
  checked against the seats that actually *play* -- dealt minus retired -- so a
  four-seat deal with two seats closed is a two-seat game and the duel accepts
  it, while the same deal with nobody retired is a four-seat game and it does
  not. Anything outside the two is a `GameType` a consumer declares; the engine
  holds it to what it declared rather than refusing it.

  `GAME_TYPES` was defined in `rules.py` and read nowhere in the package. It is
  now what `start` reads, and `hexset.bench.duel` takes `--game-type`.

  There is no four-player 15-point game: the 15-point game is the duel.

- **`arena.RETIRED`, a first-class entrant that never plays.** `_play_one`
  reads the closed seats off the lineup before the deal, which is what makes
  the contract checkable; `compete` checks it once before any worker starts.
  A ranked duel is now `--geometry a,b,retired,retired --game-type
  duel-variant` through the stock bench, with no runtime shim patching
  `arena.start` -- which is the pattern that let the invalid format exist.

- **`arena.Exhausted`.** A game that reaches the turn cap with nobody at the
  win threshold stops the run, naming the seed, index and seating to replay it.
  It was previously scored as a loss for whoever sat side A: silent,
  directional, and about ten ordinary games of compute apiece.

- **`game.UNSTRUCTURED_TURN_CAP` and `game.NO_TURN_CAP`.** Horizons for the two
  callers that are not measuring agents trying to win.

### Changed

- **`MAX_TURNS` is 300, from 1000, and is a default rather than a law.**
  `Game.turn_cap` carries it, `start`/`deal_game`/`play`/`compete`/
  `record_game`/`LaneEnv` take it, and `imagine` copies it so a search keeps
  the horizon its game has. The number is read off what games take: the
  longest barely moves with table size, so a flat total is the right shape.
  Random play is a separate case -- four random bots run to a median of 285
  and a maximum of 702 -- and `play_random_game` therefore defaults to
  `UNSTRUCTURED_TURN_CAP`. A cap loose enough for the worst imaginable
  player catches no bugs at all.

- **`start` takes `game_type` and `locked` in place of `rules`.** Retiring at
  the deal rather than afterwards is what lets the contract be checked, and it
  points the setup snake past closed seats from the outset. `lock_seat` still
  retires a seat that leaves *during* a game, which does not reopen the
  contract. `new_game` still takes a bare `Rules` and checks nothing: it builds
  a state, not a playable game.

- **`onnxbot.load` caps onnxruntime at one thread by default.** It defaulted
  to `None` -- onnxruntime's own choice, one intra-op thread per visible core
  -- with a docstring telling a caller that had already sharded across
  processes to pass 1. The sharded caller in this package never did:
  `clients/netbot.register_entrants` builds its loaders as
  `loader(path, topology)`, so every arena worker claimed the whole box
  against every other worker. Measured on 200 games: 129 s capped, 509 s
  uncapped, identical results. A caller that genuinely wants the box passes
  `threads=None`.

### Fixed

- **`actions.Stuck` raised `NameError` instead of reporting.** Its message
  called `to_move`, which `hexset/actions.py` never imported, so the one path
  that says "this seat has no legal move" died on the way to saying it. Never
  exercised until a game could end with no winner under it.

- **Replaying a record re-applied the turn cap to it.** A record made under one
  cap could not be replayed under a tighter one: the replay ended the game
  early and then rejected the record's own remaining actions as illegal in
  `GAME_OVER`, which is a false accusation against a valid record. A replay is
  driven by a recorded action list that already knows where its game ended, so
  it now runs uncapped.


## 0.64.1

Two fixes for a table with retired seats -- a four-seat game with two seats
closed, which is how the server runs a 1v1 against a four-player model.

### Fixed

- **Setup handed the first real turn to a retired seat.** `_advance_setup`
  read the head of the snake queue when setup ended, on the reasoning that
  `lock_seat` moves the snake off `current_player` first. That holds only for
  a seat retired *while it was on the clock*. Retire the seats at the front of
  the order before setup -- what the arena does when it seats fewer players
  than the table holds -- and the snake runs on the seats behind them, then
  hands the first turn back to a seat that never placed. The handoff now skips
  retired seats.

- **The Catanatron mirror contained seats that cannot act.** Nothing in
  `hexset/catanatron/` knew about `game.locked`, so a four-seat table with two
  closed seats was mirrored as a four-player game in which two colours never
  move, and the reference engine searched against opponents that do not exist.
  It now mirrors the seats that can still act, plus any retired seat that left
  pieces behind, since those still occupy the board.

  `seating()` takes the hexset seats being mirrored, `mirrored_seats()` reads
  them back, and the places that assumed seat number equals catanatron's own
  colour index -- `current_player_index`, `current_turn_index`,
  `discard_counts`, and the `P<n>_` feature keys -- now go through the index
  of the mirrored order. They coincide only when every seat is mirrored.

## 0.64.0

Behaviour change for `duel-variant`: the 1v1 game type now carries a friendly
robber and balanced dice, neither of which it modelled. It spelled the ladder
variant as two numbers; it is four settings.

### Added

- **`Rules.friendly_robber`, and `DUEL_VARIANT` sets it.** The robber may not
  take a hex occupied by a seat at or below `robber.FRIENDLY_ROBBER_POINTS`
  (two) *public* points: a seat that is visibly behind can be neither blocked
  nor robbed. Victory-point cards stay hidden and so do not lift the shield.

  `duel-variant` previously carried two of the variant's settings, 15 winning
  points and a 9-card discard limit, so every `duel-variant` result taken
  before this ran a different game from the ladder it is named for.

  The rule reuses the machinery a mirrored table already used:
  `robber.allowed_targets` returns a host's own list where it gave one (it has
  already applied that table's rules) and the game type's otherwise, and
  `legal_actions` and `move_robber_to` both ask it. `Rules` travels on
  `GameState`, so a search copy and the Catanatron bridge see the same table --
  `state_to_catanatron` now sets the reference engine's `friendly_robber` from
  the rules rather than hardcoding it off.

- **`Rules.balanced_dice` and `chance.Balanced`, and `DUEL_VARIANT` sets it.**
  The fourth setting is a dice deck: all 36 two-dice combinations, drawn and
  discarded, reshuffled whole at 12 cards remaining, with a 30% discount on a
  card repeating the previous number. `start()` builds the source the rules
  ask for; an explicit `chance` (a replay) still wins.

  The reshuffle threshold is the part worth knowing: a deck is never played
  out, so the 36 combinations are not a quota over any window and no 36-roll
  window holds one of each. The discount, not the deck, is what suppresses
  repeats -- independent dice repeat 11.3% of the time, this source 6.4%.

  Still not modelled, and tested so it stays visible: the variant also damps
  seven streaks and steers each player's share of the sevens, specified as
  behaviour with no numbers. This source rolls the textbook seven rate, so a
  `duel-variant` game is closer to a ladder game than it was and still not one.

### Changed

- **The reference-engine pin moves to a build carrying upstream's performance
  work**, six commits ahead of and zero behind the previous pin, which
  remains its merge base. The pin returns to an upstream commit once that work
  lands there.

  Those commits rewrite every module `hexset/catanatron/speedups.py`
  hash-guards (`state`, `features`, `value`, `board`, `minimax`), so the five
  digests are recomputed. The guard did its job: it refuses an unsupported
  source rather than silently patching one.

  One of them was written for this bump. Hoisting `Game.seed`/`Game.random`
  out of the reference engine's `initialize` branch -- needed, because a
  mirror is built with `initialize=False` and `copy()` reads both -- put an
  unseeded `random.randrange` on the `random` module's *shared* stream, once
  per mirror: once per decision, plus the search's own copies.
  `test_arena_reference_bots_use_independent_seeded_search_streams` caught it.
  That default now draws from a throwaway generator.

## 0.63.1

### Added

- **`compete(progress=)`** and a running tally from `hexset.bench.duel`. The
  arena calls `progress(done, games, outcome)` in the calling process as each
  game's outcome arrives, and the duel prints one line to stderr at most every
  `--progress-seconds` (default 60; 0 after every game; `--no-progress` off)
  with the games in, side A's running win share, the mean paired VP and a
  straight-line time left (withheld until the worker pool has drained its
  first wave, which it would otherwise overstate). A 464-game neural duel used
  to write nothing for an hour; a long run is no longer silent until it
  finishes.

## 0.63.0

Behaviour change for every `mcts:` entrant: the search no longer roots its
tree on the true state.

### Fixed

- **`hexset.mcts.Search` searched the real position.** Every decision built
  its root with `imagine(game, ...)`, a copy of the true `GameState`, and
  nothing in the module ever sampled the mover's `View`. The tree therefore
  enumerated opponents' moves from their actual hands and development cards,
  resolved steals and dev-card draws against the real piles, and backed exact
  hidden outcomes up to the root -- a hidden-information leak in the arena's
  `kind="mcts"` entrant (`GatedSearch`, `netbot.searcher_for`,
  `onnxbot.searcher`) and in anything else that seated the tree. A
  composition-preserving permutation of the other seats' hidden cards, which
  leaves the acting seat's `View` bit-identical, moved 2 of 6 mid-game
  decisions.

  The root is now determinized: `k` samples are drawn from the mover's own
  `View`, folded to the distinct worlds among them by `holdings_signature`,
  and each world gets its own tree. Root visit counts, priors and values are
  combined weighted by each world's share of the draws, and the chosen
  action comes from that combined root. The tree itself -- PUCT, chance
  edges, backup -- is unchanged.

### Added

- **`Search(k=)`**, determinized worlds per decision, default `1`, threaded
  through `netbot.searcher_for`, `onnxbot.searcher`, `Entrant.k` and the
  entrant spec: `mcts:<path>@<simulations>w<wave>:k=<n>`. `simulations` is
  the budget of one world's tree, so combined visit counts still sum to it
  and `k=1` costs exactly what today's search costs.
- **`Search(hidden=False)`**, the omniscient root, off by default. It is an
  analysis tool: no bot the arena seats reaches it.
- **`tests/test_information_set.py`**, a permanent gate. The permutation
  probe and the `observed_by` probe run against every bot the arena can seat
  -- a search bot, the `mcts` entrant, the tree itself over a hand-reading
  evaluator, the one-forward network bot and a determinized `CatanatronBot`
  -- at a main-phase, a discard and a trade-response decision, with a control
  that fails on the omniscient root.

### Changed

- **`CatanatronBot`'s `worlds=0` default is documented as omniscient.** It
  mirrors every seat's true hand to the reference player; `worlds>0` is the
  information-set read. The default is unchanged: the `catanatron` entrant is
  the published reference baseline.

## 0.62.1

No engine changes.

## 0.62.0

No engine changes.

## 0.61.0

No engine changes.

## 0.60.0

Breaks clients: `hexset.server.web` loses `--max-trade-cards` and `--no-trade`,
the served state loses `max_trade_cards`, and `Game.max_trades`,
`Game.max_trade_cards` and the `max_trades`/`max_offers` arguments of
`play_game`, `compete`, `record_game` and `LaneEnv` are gone. The wire no
longer caps a trade's size. A checkpoint's `max_trades` metadata key is still
read, as `max_offers`.

One trade round, one menu hook. The fragmented protocol used to reach the
table through its own `propose` hook and an engine entry point,
`propose_offer`, beside the round every other gate played; both are gone.
The protocol now supplies `candidates(view, counterparties, turn=,
already_offered=)` -- the one fragment its plan calls for next, against the
seat it was planned with -- and `offer` takes it as it stands, so
`hexset.trading.offer` is the one opening for every gate: known menu,
sampled menu (`offer_worlds`), or planned fragment. `allow_repeated_offer`
is honoured there, `trade_round_finished` in `resolve_offer`, and the served
table runs the same stages. `hexset.trading.install` no longer shadows an
unwanted hook with `None`; it removes the instance attribute, so a bot's own
class-level `candidates` shows through when the protocol is not planning.

### Added

- **Every trade gate limit is one model parameter set:
  `hexset.trading.TradeParams`.** What a seat is willing to move and in
  which direction, the floor its gains are held to, what a response costs it
  per outgoing card, how many offers it makes in a turn, whether it plans
  one target across fragments and on what cutoff, and its continuation
  budget -- one frozen object a handcrafted search bot and a neural
  checkpoint both carry. These were constants inside one search bot,
  readable by that one bot and by nothing else, which is why a checkpoint
  could not be asked to bargain the same way. **A limit a model does not
  declare is a limit it does not have**: every field defaults to "none of my
  own", and nothing outside the model overrides it. `UNLIMITED` is that
  empty declaration and the only set `hexset.trading` ships; what a
  particular bot bargains like is that bot's own config (for a checkpoint,
  its ONNX metadata -- `max_offers`, `max_give_cards`,
  `responder_card_risk`, `fragment_trades` and the rest, see
  [docs/onnx.md](docs/onnx.md)). An arena entrant names one through
  `Entrant.trade`. See [docs/trading.md](docs/trading.md).
- **A shared negotiation protocol: `hexset.trading.TradeProtocol`.** The
  offer/answer/sign half of a gate, over any object that can value a batch
  of candidates, so `NetworkBot` and `GatedSearch` play the same protocol as
  the search bot rather than a cut-down one. `hooks_for` says which of the
  engine's duck-typed hooks a parameter set needs and `install` binds them:
  a set that constrains nothing installs none and is answered by
  `default_offer`/`default_respond`/`default_pick` exactly as before, so
  carrying parameters is not a behaviour change in disguise. Existing
  checkpoints are read at `UNLIMITED` and bargain unchanged.
- **A proposal protocol beside the index one.** A gate exposing `propose`
  builds its own candidate pool and returns a signed bundle, so a policy that
  enumerates in a world sampled from its own belief is never handed the true
  hands. `trading.propose_offer` validates the result, `trading.resolve_offer`
  runs the round, and `trading.consent` lets a gate price its own consent and
  be told which side of the exchange it is on.
- **The served path runs the gate's offer policy.**
  `GameSession.begin_round`/`_resolve` in `hexset.server.webplay` now call
  `trading.propose_offer` and loop up to the actor gate's own
  `trade_offer_budget`, where they used to build every broadcast from the
  true state directly and cap every seat at one a turn.

### Changed

- **A table decides nothing about trading but the mechanism.**
  `Game.max_trades` and `Game.max_trade_cards` are gone. How many offers a
  turn holds is the acting seat's own `TradeParams.max_offers`, read as
  `trade_offer_budget`: `0` is the no-trade arm and is never asked, under
  either mechanism, and an undeclared budget is genuinely unlimited -- the
  actor offers while it has an offer it has not already made this turn,
  which `already_offered` terminates. **A gate that declares nothing
  therefore offers more than it did**, where the table used to cap it at one
  round a turn; strength numbers measured before this do not carry over for
  such a gate. How many cards an exchange moves is each seat's own
  `max_give_cards`/`max_take_cards`, refused at its own consent; the
  referee's card check, the served `max_trade_cards`, `View`'s cap and the
  wire's size check are gone, and a table of manual seats has no card limit
  at all. There is no deployment override either: `--max-trade-cards`,
  `--no-trade` and `Config.max_offers` are removed, and so are
  `Entrant.max_offers` and the `max_offers`/`max_trades` arguments to
  `play_game`, `compete`, `record_game` and `LaneEnv`. A run seats bots
  built to bargain as it wants (`Entrant.trade`, or the bot's own
  `max_offers=N`), or retunes a built gate with `hexset.trading.retune`. A
  no-trade arm is a bot built not to trade (the bot's own no-trade spec,
  `network:<path>@0`, both naming a `TradeParams`) rather than a table
  forbidding it, and `network:<path>@N` for other N raises.
  `Checkpoint.max_trades` is gone -- a checkpoint spells it `max_offers` in
  its metadata, and an exported file still carrying the old key is read
  rather than ignored. `imagine` no longer carries a cap;
  `hexset.mcts.Search.max_trades` was stored and never read. What the engine
  keeps is `ENUMERATION_CARDS`: how wide it enumerates candidates for a gate
  that declares no cap, a search bound that refuses nothing.
- **`hexset.trading` is a package.** `engine` is the referee, `params` is
  what a gate declares about itself, `policy` is the protocol those
  parameters drive, and `fragments` is the planning arithmetic (moved out of
  a bot's own package). Everything is re-exported, so `from hexset.trading
  import ...` names what it always did.
- **A bot's trade settings read through its parameters.** `NetworkBot` keeps
  `max_offers`, `trade_floor` and `gate_plies` as properties over one
  `trade` field rather than as separate copies of it.
  `hexset.clients.modelmeta.GateConfig` is now `TradeParams` (`.plies` is
  `.gate_plies`), and `Entrant.fragment_trades` defaults to `None`, meaning
  "the bot's own", rather than `True`.
- **`trading.trade_round` gains `offers_made`.** `already_offered` is a set, so
  a permitted repeat does not grow it; without a separate record a cleared
  second fragment was dropped from the round's result.
- **One word for one thing: an *offer* is what a seat puts up, and to
  *broadcast* it is to put that one offer to every other seated gate.** The
  same quantity was being counted as offers in the parameters
  (`max_offers`, `trade_offer_budget`, `already_offered`) and as broadcasts in
  the server (`_broadcasts_made`, a "broadcast budget"), which reads as two
  budgets to anyone configuring a model. The noun is an offer throughout;
  `_broadcast`/`_try_broadcast` keep the name because they are the act.
- **An auto-clearing table refuses a gate that prices its own consent.** The
  clearing house never asks for it, so such a gate's limits would silently not
  apply.
- **`hexset.bench.duel`'s `--fragment-trades-a/-b` take `on` or `off`.** They
  used to be switches that could only force the policy on, which stopped
  meaning anything once it became the default; `off` is how a duel seats the
  pre-#789 control on one side.

### Fixed

- **A served bot with no declared offer limit went silent instead of
  unlimited.** `trade_offer_budget` is `-1` for a gate that declares no limit
  of its own -- every checkpoint that says nothing about `max_offers`, which
  is every checkpoint exported so far -- and the session read it as a budget
  already spent, so the seat never opened a round at all. It is the one bug a
  bot that simply does not want to trade looks exactly like, and the browser
  is where it would have been found. The engine's own loop
  (`_run_trade_rounds`) read the same `-1` correctly throughout.
- **A planning gate with no declared offer limit never offered either.**
  `TradeProtocol.propose` made the same comparison against `-1`, so a
  checkpoint asking for `fragment_trades` without `max_offers` planned a
  target every turn and put none of it to the table. An undeclared budget now
  offers the whole plan and stops when the plan is spent. An exchange that
  executed on a bundle other than the planned fragment used to end the target
  by writing the budget into the attempt counter, which only worked because
  the same comparison misread it; a finished target is now a flag.

## 0.59.1

One trade round, wherever it is driven. The engine's turn loop and the
served table each had their own copy of the round's opening -- the actor's
menu and its gate's pick -- and the served copy had drifted: no proposal
validation, no observer notification, and both copies built the actor's
menu from the other seats' *true* hands (the clearing house's enumeration,
reused where a seat is choosing). The round is now three stages in
`hexset.trading`, named for the gate verbs they run one level up --
`offer`, `respond`, `pick` -- and `trade_round` is those stages run at
once; the served table runs the same three around its pause for seats that
answer later.

### Changed

- **The actor's menu comes from its own view.** Two menus ship, both built
  from what the seat can know. `known_candidates` enumerates the exact hand
  against what the ledger certifies each other seat holds -- the bound the
  responders' counters already used. `sampled_candidates` enumerates it
  against hands drawn from the seat's belief, so the bot also asks for what
  the untyped cards *could* be, and the table varies from turn to turn. A
  gate picks its menu by supplying `candidates(view, counterparties)`; the
  round reads it through `menu` for offers and counters alike, and a gate
  without one gets the known menu. Only the clearing house (`trade_event`)
  still enumerates from the true hands, where no seat is choosing. The
  known menu measured strength-neutral against the old true-hand one over
  2000 paired arena games (50.8% [48.8, 52.8], paired VP -0.02).
- **The served table publishes trade events.** A closed round reaches every
  bot's `observe_trade` exactly as the engine's own loop publishes its event
  (`hexset.game.publish_trade_event`), so a bot's public-activity model sees
  the same stream at either table.
- **`webplay.GameSession`** opens, answers and picks through the shared
  stages; its own code is now only the pause -- the `awaiting` set, the
  cardless-manual-seat pass, and a manual side's consent at execution.

### Added

- **`hexset.trading.known_candidates` / `sampled_candidates` /
  `offer_candidates` / `menu` / `offer` / `respond` / `pick`**, the gate hook
  **`candidates(view, counterparties)`**, and **`hexset.game.event_hand_sizes`
  / `publish_trade_event`**.


## 0.59.0

A minor bump for one new field on `Game`: a host's rule for where the robber
may go on the move at hand. Some tables forbid placements the rulebook
allows -- a "friendly robber" rule keeps the robber off hexes next to
a seat showing two or fewer points -- and tell a client only which hexes
remain. A policy offered the rulebook's set at such a table picks placements
the table will not take, and the table's own clock then places the robber
instead. `legal_actions` had no way to be told; now it has one.

### Added

- **`Game.robber_allowed`: the hexes the robber may go to this move, or
  `None` for the rulebook's "anywhere but where it stands".** `legal_actions`
  offers `MOVE_ROBBER` only onto those hexes; `move_robber_to` refuses any
  other target and clears the field once the move is made, so the rule
  cannot outlive the prompt it answered; `imagine` copies it, so a search's
  worlds see the same table. Whoever mirrors such a table sets it from what
  the table says before asking for a move. This is the list a table hands
  over, not the rule that produced it -- a later robber phase inside a
  search still uses the rulebook -- which is the right fidelity for a rule
  the engine is not the authority on.
## 0.58.2

### Fixed

- **`hexset.clients.onnxbot.spawn` forwards its `rng` through the plain
  checkpoint branch.** `spawn()` accepted an `rng` and dropped it when the
  checkpoint carried no searches, so a plain network checkpoint played by the
  web path built unseeded trade belief worlds while the same checkpoint in the
  arena path was seeded -- the two disagreed on a seed that was supposed to fix
  them both. `network_bot()` now takes `rng` and passes it to the bot it
  builds. The parameter is optional and defaults to the previous behaviour, so
  no caller has to change.

## 0.58.1

A patch for the documentation, and for the prose inside the code. The
published tree had accumulated comments written for whoever wrote the line
rather than whoever reads it -- incident narrative, campaign measurements,
fitted-coefficient provenance, commit hashes and PR numbers -- and a `docs/`
tree that had grown by accretion rather than by index. No engine behaviour
changes here: every module's code is byte-identical to 0.58.0 with docstrings
stripped.

### Changed

- **Comments and docstrings across 140 modules and tests are pruned to what a
  reader of the code needs**: contracts and invariants, units and shapes,
  index and frame conventions, parameter semantics, sharp edges, and short
  docstrings on public API. 11,226 lines removed against 3,504 rewritten.
  Where a rule was justified by a measurement, the rule stays and the
  measurement goes. Verified by comparing each file's AST with docstrings
  stripped against 0.58.0: no signature, assertion, marker, skip condition,
  argparse default or weight value moved. MCP tool descriptions and argparse
  help are user-facing and were left alone, as is `catanatron/speedups.py`'s
  licence attribution.
- **`docs/` is indexed rather than accreted.** `workflows.md` becomes
  `guide.md` and is the entry point; `server.md` splits into operating
  instructions (`server.md`), the JSON routes (`api.md`) and the MCP tools
  (`mcp.md`); evaluation splits out of the guide into `evaluation.md`. The
  README carries the index and stops repeating what the documents say.
- **The Dockerfile moves to the published root.** It was under `docker/`,
  one directory deep in a distribution that has no other build directory; a
  build referencing `docker/Dockerfile` needs its path updated.
  `compose.example.yaml` is updated with it.
- **Third-party attribution moves into the README's licence section.**
  Dependencies are declared in `pyproject.toml` and none are vendored;
  `catanatron/speedups.py` names the upstream revisions it reproduces. See
  `Removed`.
- **The two refused-contract fixtures are stubs, not real exports.**
  `tests/clients/fixtures/dev-contract2.onnx` and `tiny.onnx` were genuine
  downstream exports carrying real weights. Nothing loads a refused contract
  to play, so between them that was 1.1 MB of parameters no test ran, and
  their metadata named a training checkpoint no test read.
  `tests/clients/fixtures/build_refused_stubs.py` now writes both in
  `build_stub.py`'s idiom -- same declared names, shapes and dtypes, uniform
  over the legal mask, zero value, no learned parameters -- taking the pair
  from 860 KB to 3 KB and making them rebuildable with the `export` extra,
  which the previous fixtures were not.

### Removed

- **`NOTICE.md`.** It recorded a one-time third-party audit taken at the
  GPL-3.0-only relicensing (0.14.0) and had since drifted: it named a
  Catanatron revision the pin had long since moved off (at 0.49.1), and
  claimed the Dockerfile specified that revision, which it never did. It also
  never shipped -- `license = { file = "LICENSE" }` declares only the licence
  and there is no `MANIFEST.in`, so no wheel or sdist carried it. The two
  facts in it that `pyproject.toml` does not state -- nothing is vendored, and
  the browser interface loads no third-party scripts, stylesheets or web fonts
  -- are now in the README.
- **`examples/`.** The directory promised a collection and held one file,
  and that file was the install smoke check wearing an example's clothes: CI
  ran it after pytest because it was the only executable proving a fresh
  install imports, plays a game and replays a record. It is now
  `tests/test_bot_integration.py`, which asserts what the script printed, and
  its walkthrough is an inline snippet in the guide -- the page that already
  explained the same code in prose. CI drops the extra step; pytest covers
  it.

### Fixed

- **`tests/clients/test_contract_dispatch.py` no longer fails where
  `onnxruntime` is installed and `onnx` is not.**
  `test_an_unknown_contract_is_refused_by_name` imported `onnx` in its body
  without a guard, so it failed rather than skipped under the `clients` extra
  without `export`, which is a supported combination -- a failure in an
  otherwise green module, since the rest of the file needs only
  `onnxruntime`. It now uses `pytest.importorskip`, as
  `tests/clients/test_botclient.py` already did for the same dependency.

## 0.58.0

A minor bump for a collapse: the engine had drifted to three copies of its
game loop -- `arena.play_game`, `arena`'s own recording loop for
`compete(records=True)`, and `record.record_game` -- kept in step by hand
rather than by construction, and, inside each, three more hand-copied
lines seating `game.gates`/`trade_mode`/`max_trades` before the loop ran.
That second triplet was the actual bug: `record_game` carried one of the
three lines and not the other two, so every recording ran at whatever
`Game` defaulted to regardless of what its bots could do. Both collapse
now: one loop (`arena._run`), stepped with a `Tape` and a `ClearedTrade`
census as optional hooks a caller attaches rather than separate copies of
the `while not is_over(...)` that drives it; and one seating function
(`arena._seat`), called by all three sites instead of hand-copied at each.
`record.play_random_game` is deliberately not part of either collapse: it
steps uniformly random legal actions with no bots and no gates, which is a
different thing from a bot-driven game, not a fourth copy of one.

### Added

- **`record.record_game` takes `trade_mode`/`max_trades`.** It previously
  had no way to be told how to trade and every recording ran at `Game`'s
  own defaults regardless of what its bots could do; those defaults --
  `trade_mode="round"`, `max_trades=1` -- are this function's new keyword
  defaults too, so an existing call is unaffected. A `RandomBot` seat
  implements none of `gains_many`/`accepts_many`/`accepts` and so can never
  clear a trade; passing `max_trades=0` for such a lineup now skips the
  engine enumerating and discarding every coverable bundle each turn, which
  `tests/test_record_engine.py` measured at roughly 1.8x faster wall clock
  for a suite entirely seated with `RandomBot`.

### Changed

- **`record_game` and `compete(records=True)`'s recording loop now run
  through the same step function `play_game` does** (`arena._run`), and seat
  their bots through the same function too (`arena._seat`), rather than each
  carrying its own copy of `while not is_over(...)` and its own three lines
  of `game.gates`/`trade_mode`/`max_trades`. `arena.py`'s long-standing
  comment that `--records` "must record exactly the game `play` would have
  played" used to be a promise kept by hand across those copies; it is now
  true by construction, and says so. Verified byte-identical against `main`:
  recording the same six seeds/seatings (`RandomBot` and a search bot, three
  seeds each) produces the same actions, the same cleared trades, and the
  same chance stream.

## 0.57.1

A patch: `View.unseen_dev_cards`/`deck_odds` were biased high on three of
the five card types; they are exact now.

### Fixed

- **`View.unseen_dev_cards` no longer overstates Road Building, Year of
  Plenty and Monopoly.** Knights leaving play were always subtracted
  (`state.knights_played`); the other three playable types had no
  counterpart, so a card stayed in the unseen estimate forever once played
  -- `deck_odds` kept assigning it a share of the next draw, and anything
  sampling an opponent's hand or the deck from a `View` (a search bot's
  Monopoly targeting and `p_holds`/`sample` among them) could deal a card
  that was, in fact, gone. Fixed by `state.dev_cards_played`, a table-wide
  tally of the three types maintained the same way and at the same call
  sites `knights_played` already was
  (`devcards.play_year_of_plenty`/`play_monopoly`,
  `game.play_road_building_card`), and subtracted in `unseen_dev_cards`
  alongside it. `hexset.catanatron.state.translate` now reads the same count
  exactly off Catanatron's own per-type play counters rather than defaulting
  it to zero. No observation the network sees changes -- neither
  `encoding.py` nor `onnx_record.py` reads this quantity; it is a `View`-
  and `GameState`-internal correction only.

## 0.57.0

This release breaks any caller holding the higher-threshold 1v1 preset's
previous constant or its previous `GAME_TYPES` key (see **Changed**): both are
gone with no compatibility alias, so such a caller fails at the point of use
with an `ImportError`/`KeyError` rather than quietly playing a different game.
Under 0.x semantic versioning a minor bump is the signal for that.

### Changed

- **The 15-VP, 9-card-discard-limit 1v1 preset is `hexset.rules.DUEL_VARIANT`,
  registered as `GAME_TYPES["duel-variant"]`.** The variant itself is
  unchanged. The constant and registry key it previously carried are gone
  with no compatibility alias, so a caller holding the old ones gets an
  `ImportError`/`KeyError` rather than a silently different game.

## 0.56.1

A patch: the engine's own games are unchanged; a state assembled by hand is
now refused where it was silently mis-scored.

### Fixed

- **`update_longest_road`'s incremental paths refuse a cache that was never
  filled** (`hexset.victory.StaleRoadLengths`). 0.51.0's `road_lengths`
  cache (#157) is refreshed for the builder alone and the card awarded
  from it, so a `GameState` whose `edge_owner` was written directly -- a
  mirror of another server's board, which `new_game` hands a cache of
  zeros -- had every other seat reading as roadless: the next road built
  in a search took Longest Road from a seat holding seven, and an
  opponent's fifth took it from the builder's ten, without an engine test
  failing, since the arena builds every state through `place_road`. The
  check is exact: one road is a route of length one, so
  a roaded seat cached at zero (the seat about to be counted excepted) or
  a cache of the wrong size cannot be a maintained cache. One pass over
  the edges, before the route search. The remedy for a caller that
  assembles its own state is one from-scratch `update_longest_road(state)`
  before the first incremental update; `state.py`'s field comment says so.

## 0.56.0

A minor bump under 0.x for a removal: the Catanatron-hosted bridge is gone
and only the adapter, which seats a Catanatron player at a HexSet table,
remains. Nothing a client of the engine, the arena or the served bots sees
changes.

### Removed

- **The Catanatron bridge:** `hexset.catanatron.duel`, `.player`
  (`DevCatanPlayer`, the `DC:` player-registry entrant) and `.register`, along
  with the bridge-only `.speed_benchmark` and `.throughput_benchmark` duel
  profiling tools and their tests. This ran games from the Catanatron side,
  with a HexSet entrant wrapped as a Catanatron `Player`; the reverse adapter
  (`hexset.catanatron.bot`, the `catanatron` arena entrant) is unaffected and
  keeps running games from the HexSet side, as does the general-purpose
  `hexset.catanatron.speedups` acceleration module both directions could use.
  Two paths to the same opponent were producing non-comparable numbers, and
  the two engines' rules have since converged (dev-card offers before the
  roll, free-road exclusivity, the standard piece supply) to the point the
  bridge no longer earned its keep; `arena.compete` is the project's only
  game loop. The bridge is not retrievable from this
  repository's history.

### Fixed

- `hexset.catanatron.bot`'s module docstring, which still described the
  dev-card, free-road and piece-supply offer gaps the bridge's removal made
  moot -- HexSet's rules converged with Catanatron's on all three. Only the
  one-event trade mechanic remains unrepresented on a Catanatron seat.

## 0.55.0

### Changed

- **The network trade gate is one value forward over every coverable hand
  again.** At `gate_plies == 0`, the default every checkpoint gets unless
  its file asks otherwise, `hexset.clients.netbot.NetworkBot` prices each
  candidate exchange as the head's own reading of the hand after the cards
  move, alongside the live position, in one batched forward. Nothing stands
  in front of the head. The rollout gate stays behind `gate_plies`
  for a file that wants to pay for it,
  and now rolls every coverable candidate rather than the filter's
  survivors.

### Removed

- **The affordability filter.** 0.52.0 zeroed out, unscored, any candidate
  that left the set of buyable kinds (road, settlement, city, dev card)
  unchanged. The filter cost strength for a saving the forward did not need.
  With it go the position that read as changing nothing at a won table: the
  head prices that candidate now, at whatever its own resolution says.
  Nothing a client sees changes; a checkpoint's metadata keys are the same.


## 0.54.0

This release changes the MCP tool contract in ways an existing MCP client
will notice (see **Changed**): every acting tool now blocks until the
caller's next move rather than answering the instant after the action, the
`undo` and `get_table` tools are gone, `models` is `bots` and its `model`
argument `identity`, and replies are compacted server-side. Under 0.x
semantic versioning a minor bump is the signal for that.

### Added


- **Game replay: step any game a round at a time.** A strip along the foot of
  the board — first / previous / ROUND / next / live — at every game and for
  everyone at it, players as much as spectators: looking back at what just
  happened is not only a watcher's business. A seat that steps back keeps its
  own view of the round and loses only its controls, which come down for as
  long as it is looking; `>|` hands the turn straight back. While a finished game
  is being stepped through the title keeps naming the winner rather than
  reverting to the phase of the round on screen: who won is a fact about the
  game, not about the position being read. Round N is the position at
  the *end* of round N — what happened in that round has happened, the way
  the sidebar log reads a round — so `|<` is the finished opening and the
  last round is the game's final position. `>|` leaves replay and goes back
  to following the game live rather than pinning you to a snapshot that
  quietly goes stale. Stepping is by round; a turn at a time is what the log
  is already for.

  Served by `GET /api/table/<code>/replay?round=N`, which replays the journal
  into a throwaway session and reads it through a stand-in `Table`, so a
  past position carries every field the live public view does. The live
  session is never touched: reading a game in progress cannot move it.

  A past round of a game still being played is read as whoever is asking. A
  seat gets its own seat's view of it, hiding every other hand exactly as its
  live view does; only a reader with no seat at that game gets the omniscient
  one the public view beside this already hands out. Answering omniscient to a
  token holder would have made stepping back one round a way of reading the
  table's hidden cards and then playing on with them.

  Once a game has been *played out*, every round of it is read in full by
  everyone, seat or not — the same release the finished live view already
  makes, reaching back over the whole game. Disclosure follows the game rather
  than the position, and the position cannot tell: a session rebuilt at round
  twelve reads as unfinished however the game later ended, so the journal's
  own `result` line is what decides it. Without that, stepping back into a
  finished game would have been the one thing that put a reader back behind a
  redaction they had already been released from.

  Either way a replayed round is inert: it carries no `legal_actions` and no
  `can_undo`, which is the whole of what the page wires a click to. The
  stand-in is a real session at a real round, so a seat asking for a round it
  held would otherwise be handed the moves it had *then* — and every one of
  them would post to the live table, at some later position entirely.

  Nothing is cached, on either side. An `undo` rewrites a game's history
  rather than appending to it (`replayable_rounds` drops the steps taken
  back), so a round number is not a stable name for a position, and a cache
  keyed on one would serve a board that never existed.

### Changed


- **`pending_free_roads` reads the piece limit once, not once per edge.** It
  asked `can_place_road` per candidate, and that opens with a `road_count` --
  itself a scan of every edge -- so enumerating owed roads was quadratic in
  the board. The supply check is hoisted out and the per-edge test is
  `road_placeable`, which is the same rule without it. Both readers are on
  the hot path: `legal_actions` calls this in ROLL and, since Road Building
  began resolving in MAIN as well, there too, and a search bot's rollouts
  call `legal_actions` millions of times a game. 2.6x on a mid-game
  four-player board, 243µs to 92µs a call.

- **A finished game reads as an observer's, seat or no seat.** Holding a seat
  at a game that was over left the page still dressed as that seat's: its
  colour on the accent, its hand pinned open in the card pane, its piece
  supply still counted out. Every hand is revealed once a game is over, so
  the seat's holder now reads the table the way a spectator does -- nobody's
  cards until a row is picked. The seat is still theirs, and it is still what
  keeps their own name on their own row; it just stops being the view's
  default.

- **The trade modal remembers where it was dragged.** The offset reset to
  centre every time a mode opened, which made moving the box a per-decision
  convenience rather than a preference. It now survives both the next open
  and a reload (`hexset.modaldrag`), and is clamped against the viewport on
  the way in, so a position saved on a wide window cannot strand the box off
  the edge of a narrow one.

- **The seat picker lists its models by name.** The options came out in
  whatever order `/api/models` returned them, which is `MODEL_OPTIONS` order,
  and finding one by eye is the whole job the control has. `(empty)` and
  `(closed)` stay pinned above the list: they are what the seat *is*, not
  something that could be seated in it.

- **The trade modal's buttons are smaller and its close glyph larger.** The
  action squares go 44px to 40px and their glyphs 22px to 20px (16px on the
  bank/port square, which gives up room to the rate printed under it); the
  close X goes the other way, 20px to 24px over a 19px glyph. The body
  carries more vertical padding to go with it.

- **`summary.afford` says why an affordable build is not offered.** `why`:
  `phase` (not this seat's main phase), `pieces` (none of that piece left),
  `deck` (no development card left) or `spot` (nowhere to put it). Asked
  for by the first Sonnet seat, which met `ok: true, legal: false` with no
  way to tell a distance-rule dead end from a wrong phase.

- **MCP `trade_ratios` on every reply; `race.leader` is `top_opponent`.** The
  first Sonnet seat, sent its ratios only when they changed, assumed 4:1 and
  missed its own port for several turns -- 60 bytes a reply was the wrong
  saving. `leader` named the leading *opponent* and read as the overall
  leader; the new name says what it is. The `robber` table's `hits` and
  `options` encodings are restated in the `state` description.

- **MCP replies kept `winner` when seat 0 won.** The trim that drops a null
  `winner` dropped seat 0 as well; caught by the first Terra game, which
  seat 0 won.

- **MCP `game_over` reply reports `usage`.** `calls` and `bytes`, what this
  session was sent over the game as the client received it, counted
  server-side so a seat's payload cost can be compared across clients
  independently of what the model spends reasoning. `board`'s description
  now says to read it once after taking a seat, before planning beyond what
  `summary` shows.

- **`summary.roads` flags a network-joining road.** A legal road with both
  ends already the seat's own carries `link`; its `to` is a vertex the seat
  holds, not ground it reaches.

- **The MCP `game_over` reply no longer pushes the whole transcript.** It
  used to, so the client could replace its copy with the redaction-lifted
  history -- 5-6k tokens a seat rarely reads. The final reply now carries
  the usual slice; `state(full_log=true)` fetches the un-redacted whole on
  request.

- **`summary.spots` keeps `type` when settlements and cities mix.** The
  table's `KIND:` prefix assumed one kind per list; mid-game, once both are
  affordable, a settlement spot was labelled `BUILD_CITY:`. A mixed list
  now carries `type` as a column.

- **`summary.roads` plans two roads.** Each legal road also names `then`,
  the best settleable vertex one road further on, with `then_pips` -- the
  two-road plan every setup road and most early ones are, so a seat no
  longer reads the board to make it. Booleans in reply tables print as
  `1`/`0`.

- **Fewer MCP round-trips that decide nothing.** A seat's own trade round
  is settled inside the wait once everyone has answered and there is
  nothing to choose: every answer a pass closes it, exactly one accept as
  offered with no counter executes it. `offer_trade` takes `to`, the seats
  the offerer would trade with best first, and then the best-ranked listed
  accepter is executed and an unlisted one (the leader, say) refused; a
  counter from anyone always comes back. The `undo` tool and `can_undo` are
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
  every seat's is zero; and `trade_ratios` was sent only when it differed
  from the last reply -- reverted below, after a seat missed its port.

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
  The compaction lives in the server, so every MCP client gets it. Also
  fixes the MCP layer annotating -- and stripping `x`/`y` from -- the
  table's own `layout` dict in place, which the browser draws from; it works on a copy now.

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

### Fixed

- **Road Building resolves when played: both roads before anything else in
  MAIN.** While free roads were owed, MAIN offered every other build, card
  and trade alongside them and withheld only `END_TURN`, so a seat could
  interleave a whole turn between the two placements. The ROLL branch
  already resolved the card on the spot; MAIN now does the same and offers
  the road placements alone. Sending a settlement or a development-card buy
  past owed roads is a rules violation. With nowhere legal to place a road
  the credit stays unusable and the turn continues as normal -- the
  stranded-credit escape hatch is unchanged.

- **Clicking a player at a finished game to read their cards moved the rest
  of the page.** `renderHand` had exactly two shapes: a one-line hint when
  nobody was picked, or the full two-column card grid once somebody was.
  `#hand` sits in `#play-area`, sized to its own content, with `#log` taking
  whatever's left below it, so swapping one shape for the other shoved `#log`
  up or down by however much a header row and a card grid cost over a single
  line of text. The pane now always lays out the same two columns the
  viewer's own hand uses during play; with nobody picked they are invisible
  and CLICK A PLAYER TO SEE THEIR CARDS sits centred over the space they
  hold, and picking a row reveals them -- no title over the cards, the
  highlighted roster row says whose they are.

- **The Undo corner button painted over an open modal instead of under it.**
  `#undo-build` carried a flat `z-index: 60`, above `#modal`'s `55`, so it
  would reach through *any* modal -- a purpose it only ever needed for the
  "Steal from" robber-victim modal, whose own hex a self-played Knight
  stays undoable behind. Every other modal (trade, discard, Monopoly, Year
  of Plenty) now covers it like it covers the rest of the board: the corner
  sits at `z-index: 40` by default and only gets `.reach-modal`'s `60` back
  while `modalMode` is `"steal"`.


## 0.53.0

### Added

- **The network trade gate's continuation rollout is back, as a
  per-checkpoint setting.** `gate_plies` (metadata key `gate_plies`,
  default `0`) asks `NetworkBot` to roll the mover's own greedy policy
  forward that many plies from each surviving candidate before valuing
  it, in one belief world drawn per ask from the asking seat's own
  information set (certified to cover what each candidate says the
  counterparty gives) -- never on a hand the seat cannot read -- same
  footing as `search`/`simulations` is for `hexset.mcts.Search`:
  search on the trade decision, read off the file the same way. `0`, the
  default every checkpoint gets unless it asks otherwise, is today's
  single forward over the affordability filter's own after-position --
  training collection runs at the default, and a served file that wants
  the rollout asks for it in its own metadata.

- **A duel can be played under the automatic clearing house.**
  `hexset.arena.compete` takes `trade_mode`/`max_trades` and hands them to
  every game it plays, and `hexset.bench.duel` exposes them as
  `--trade-mode {round,auto}` and `--max-trades` (`-1` for the unbounded
  house studies before 0.50 were recorded under), recording both in the
  verdict -- so a policy trained against the clearing house can be read in
  its native environment.


## 0.52.0

### Changed

- **Longest Road no longer recounts every seat's roads on every placement.**
  `victory.update_longest_road` called `roads.road_lengths` -- an exhaustive
  route search over every seat's edges, from scratch -- after each of the
  four places a road or settlement can go down
  (`game.place_initial_settlement`/`build_settlement`/
  `place_initial_road`/`build_road`), measured at ~25% of a 4-seat
  self-play game (9.2M recursive calls a game). Roads are edge-disjoint, so
  a road just placed by seat `p` can only extend `p`'s own route; a
  settlement can only *cut* a route passing through its vertex, and only a
  foreign one, since a builder's own building never blocks the builder's
  own route. `GameState.road_lengths` now caches each seat's length, and
  `update_longest_road` recomputes only the seat(s) a placement could have
  changed before awarding from the cache -- unchanged for any caller that
  mutates `edge_owner`/`vertex_owner` directly rather than through
  `place_road`/`place_settlement`, which still gets the historical
  from-scratch recompute. `hexset.catanatron.state.translate` (rebuilt
  fresh from a mirrored catanatron game every decision, so it inherits no
  running cache) computes the field once off its own topology instead.

- **The network trade gate values a candidate exchange in one forward
  again, filtered by what it lets the seat buy.** `NetworkBot` no longer
  samples a belief world, imagines two games or rolls a mover's greedy
  policy forward a few plies to price a trade -- the continuation rollout,
  and the paired belief worlds it drew, are gone. Every candidate the
  asking seat can cover is read from its own frame -- its own hand exactly,
  the counterparty's known lower bound moved the same way -- and an
  affordability filter decides first whether the trade changes anything the
  seat can buy (a road, a settlement, a city, a development card): a
  candidate that leaves every one of those exactly as it was is priced at
  `0.0` outright and never reaches the head at all, so it can never be
  offered, accepted or countered with. `gate_rows` -- the cap on how many
  candidates one batched forward scored -- is gone with the rollout it used
  to bound; every candidate a seat can cover is now considered, since the
  filter is the bound.

### Removed

- `NetworkBot.gate_rows` and the checkpoint metadata key of the same name
  (`hexset.clients.modelmeta.GateConfig.rows`, `DEFAULT_GATE_ROWS`,
  `MAX_GATE_ROWS`). A checkpoint exported with `gate_rows` in its metadata
  still loads without complaint; the field is simply never read.


## 0.51.0

### Changed

- **The trade gate answers a repeated ask from its last evaluation, and
  enumerating a board is no longer quadratic in its piece limits.**
  `gains_many` and `estimate_many` are the seat's own row and the
  counterparty's row of one evaluation, and `hexset.trading.default_offer`
  and `default_respond` each ask for both, back to back, over the identical
  candidates at the identical position -- `hexset.clients.netbot.NetworkBot`
  now serves the second ask from a one-entry memo keyed on the position and
  the ask, instead of rebuilding every imagined world and continuation.
  And `can_place_road` rescanned every edge to count the player's roads once
  per candidate edge (`can_place_settlement`/`can_upgrade_to_city` the same
  over vertices); the piece limit is a property of the player, so
  `actions._building_actions` reads it once and calls the new
  `state.road_placeable`/`settlement_placeable`/`city_upgradeable` -- the
  public predicates without that check -- per placement. Neither changes
  what the gate sees.

- **Road Building is not offered to a seat with no road pieces left.** The
  card places roads; at fifteen on the board there is nothing to place.

- Size-only readers go through `hexset.economy.hand_size` and
  `hexset.devcards.dev_count` rather than summing a hand or a
  development holding, so the robber, the discard rule, the encoding, the
  record, the arena census and the Catanatron bridge's ledger read on an
  observed state too.

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

### Added

- **Hidden hands and cards in `GameState`.** A state written by a seat
  rather than by the referee can now say "n cards, types unknown":
  `hands[seat]`, `dev_cards[seat]` and `new_dev_cards[seat]` accept a
  `HiddenHand`/`HiddenCards` count and `deck` a `HiddenDeck` length
  (`hexset.state`). Sizes, `len` and the `[:]` copy idiom work; indexing,
  iterating or summing one raises `HiddenRead` instead of returning a
  fabricated number. `observed_by(state, seat)` downgrades a true state to
  what one seat can see. A `View`, `View.sample`, a search bot and its
  evaluator, `hexset.encoding` and `hexset.onnx_record` all read the same
  off an observed state as off the truth it was observed from; an identity
  read -- an opponent's `holdings`, `card_points`, `victory_points`,
  `is_over`, or `legal_actions` for a seat whose hand is hidden -- raises. A
  live adapter therefore needs no placeholder composition for the hands it
  cannot see.

- Opt-in sampled-world voting for model/search callbacks, with a decision-local
  cache and frequency-weighted votes (`hexset.bots.determinized`). The default
  key includes the sampled development deck; models that cannot observe it can
  explicitly use `holdings_signature` for greater reuse. `CatanatronBot` exposes
  the same optional vote while preserving its default reference behavior.
  See [the cache contract and examples](docs/worlds.md).


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

- `hexset.catanatron.duel --game-type=duel-variant`: a 15 VP /
  discard-over-9 rules variant. Rules travel on `GameState` through copies
  and the Catanatron bridge, so a bot's win bonus and discard model match
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
  whole body was an import -- a training project would carry a `duel` module
  for exactly that, and nothing else. Naming the module on the command line is that import, moved
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
  became the trade round), `4` at 0.46.0 (a search bot stopped being seatable by
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
  `build_info()` into its own run records reads `provenance()` instead and gets strictly more. `hexset._source.git_value` is
  unchanged and still serves that path.

### Fixed

- A finished game's player roster keeps the height it had while the game was
  live. Going read-only replaced every row's `<select>`/`<input>` with plain
  text carrying neither their padding nor their border, so each row lost about
  4px and a four-seat list collapsed 18px, jumping the board below it.

## 0.46.0

This release changes public APIs and benchmark result semantics. HexSet remains
in the 0.x development series.

### Removed

- Retired checkpoint registration paths.
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

- `hexset.trading.trade_event` ends at the first revisited position instead of asserting. The assertion assumed a gate is a strict function of the position; the network gate scores each candidate in a world drawn from its belief (0.44.0), and a search bot samples worlds too, so a trade that changes the ledger changes the next draw and a reverse exchange can price positive without anything being broken. Self-play against a network checkpoint tripped it in a distillation test.
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

- **A network checkpoint's trade gate prices continuations, not hands.** `hexset.clients.netbot.NetworkBot` values the live position and every post-trade position *after the seat's own best play from it*: the policy's greedy actions through the rest of the turn on an imagined copy (fresh chance, deck reshuffled), a finished game reading as its one-hot winner, at most `CONTINUATION_PLIES` (8) actions, one batched forward a ply. Two raw value estimates of nearly identical hands differ by the head's noise, and the old gate offered that noise -- at a won position it priced giving the winning card away at +0.006. Now a won position reads 1.0 before and after any trade that keeps the build and lower after one that does not; the counterparty's row reads the actor's win either way, so nobody offers or counters into it. No per-model floor is needed for this; the network gate's `trade_floor` stays `0.0`.

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

- `hexset.encoding.to_frame`/`from_frame`: the seat-frame rotation (perspective seat to slot 0, others in turn order behind it) as two pure, inverse functions, replacing the by-hand `(seat ± perspective) % players` arithmetic `_encode_globals`/`_ledger_parts` repeated inline. One named seam for a convention a training application previously had to write out three times over.
- `hexset.encoding.global_columns(players)`: a named `dict[str, slice]` over `global_features(players)`'s fourteen blocks (`own_hand` through `ledger`), built from the same widths `global_features` sums, so a caller reads a block by name instead of counting offsets by hand off those constants the way a checkpoint migration across encodings used to have to.
- `hexset.casting`: pure casting laws over a game index -- `alternating(players, flip=False)` (the duel caster, seat-pair swapping by parity), `paired(caster)` (games `2k`/`2k+1` share `caster(k)`), `league_rotation(learners, players, order=None)` (rotate learner ids over seats, `order` reseating who sits next to whom) -- so a training loop's seating law lives beside `arena`'s own rotation rather than being re-derived per caller.
- **`hexset.gym.lanes`: a lockstep multi-game environment.** `LaneEnv(players, seed, lanes, deal=..., action_cap=..., board=..., caster=..., bots=..., gates=...)` holds `lanes` games in flight and steps every one of them once per tick: `requests()` hands out one `Request` per live lane (lane, game index, seat, policy id, stream step, legal `options`, the live `Game` and the seat's information-set `view`), `step(actions)` applies one `Action` per request and returns the games that ended as `Episode`s -- decisions demultiplexed by seat with their index in the lane's action stream, the cleared-trade census `(step, a, b, received)`, and an `Outcome` carrying winner, per-seat terminal points, turns, actions, `truncated` and the trade count. Finished lanes refill on the spot. A `caster(index)` decides which policy id holds each seat as a pure function of the game index; ids listed in `bots` are played by a `hexset.bots` bot (one per board, `BoardBots`), and every seat -- bot or caller -- is seated as its own trade gate on `game.gates`, so the engine's trade event runs and a learner played through the gym can trade at last (`hexset.gym.aec` seats none). Numpy-free and gym-package-free: `import hexset.gym.lanes` needs neither `pettingzoo` nor `gymnasium`, which are now required only when `HexSetAEC`/`HexSetEnv` are actually reached.
- **One game law.** `hexset.arena.deal_game(seed, index, players, board=..., chance=...)`, with `deal_board`, `board_key` and `game_key`, is where a run's `index`-th game comes from; `arena._play_one` and `LaneEnv` both call it, so a tournament and a collector playing index `i` of seed `s` play the same game rather than two games derived alike. `hexset.arena.play_game(game, bots, action_cap=...)` is `play`'s loop over a game somebody else dealt.

### Changed

- **A checkpoint runtime is a `Policy`, and gets the bot, the evaluator, the searcher and the trade gate for free.** `hexset.clients.policy` states that protocol -- `act_rows`, `value_rows`, `score_rows` over live `(game, seat)` positions rather than encoded records, values in board-seat order -- and `hexset.clients.netbot` holds `NetworkBot` (with its two-sided trade gate), `NetworkEvaluator`, `LeafEvaluator`, `GatedSearch`, the `bot_for`/`evaluator_for`/`searcher_for` constructors, and `register_entrants(loader)`, which registers the arena's "network"/"mcts" entrant kinds, its "network" evaluator, its checkpoint loader and its leaf-evaluator factory in one call. `hexset.clients.onnxbot` keeps the ONNX half -- `load` and `V2Policy` -- and re-exports every name it exported before, so `spawn(path, board)` and every existing import are unchanged. The trade gate now hands the policy a copy of the seated game per candidate instead of mutating the live one, so a runtime encodes a post-trade position however it likes; the gate's own arithmetic, its `NETWORK_GATE_ROWS` cap and its verdicts are unchanged.

## 0.40.0

### Changed

- **A network checkpoint's trade gate scores both sides of an exchange.** `hexset.clients.onnxbot.NetworkBot` (and the searched `GatedSearch`) now score the post-trade position with *both* hands moved and the counterparty's ledger row updated, as a real clearing leaves it, and read the value head's per-seat vector: `gains_many` is this seat's own row after minus before, in win probability (it used to answer +1/-1 off a swap of its own hand alone); `estimate_many`, new, is the counterparty's row after minus before -- this seat's own estimate of what the exchange does to the other seat's chances -- so `default_offer`/`default_respond` offer and counter with the candidate best for the bot among those it believes the other seat gains from too, as the trade round specifies. Every candidate two cards or fewer a side is scored; the rest fill `NETWORK_GATE_ROWS`. Under a strict-positivity floor the old sign gate offered an arbitrary acceptable bundle nearly every turn; the magnitude ranks offers and the estimate prices the risk of lifting an opponent.

## 0.39.0

### Changed

- **The clearing floor belongs to the gate, not the table.** `hexset.trading.TRADE_FLOOR` is gone; every seat's gate declares its own `trade_floor`, read by `hexset.trading.trade_floor_of` at every admission point (`clears_floor(gain, gate)`; `trade_event`, `execute_agreed`, the round's `default_offer`/`default_respond`/`default_pick`). There is no engine default: a gate that prices a candidate positive without declaring one is refused with a `TypeError`; a gate that only ever declines is never asked. A network checkpoint's boolean gate carries `0.0`.

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

- `View`'s `omniscient` flag, and `hexset.dataset`'s and
  `hexset.bench.fit_weights`'s `--omniscient` reading. Nothing outside a
  search bot's own honesty-price readouts ever constructed an omniscient
  `View`; the engine's `game.state(seat, hidden=False)`, which the
  Catanatron adapter and the server's spectator view read, is unchanged.
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
- `hexset.trading.default_offer` only broadcasts a candidate that clears `TRADE_FLOOR` on the actor's *own* gain as well as on the estimated counterparty gain, so a bot never offers a deal it would then refuse when accepted.
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
  them, so a table of three bots reads as Player 2, Player 3 and
  Player 4.

### Changed

- MCP `new_game`/`join` no longer default an unnamed seat's display name to
  `"mcp"` themselves; the server does it now (see Added, above).

## 0.36.0

### Added

- The trade round is now the served table's protocol end to end
  (`docs/onnx.md` §3). `POST /api/games/<code>/trade/round` broadcasts
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

- Documentation wording only; no change to the distribution.

## 0.35.1

### Changed

- Documentation wording only; no change to the distribution.

## 0.35.0

### Removed

- `docs/readouts/`, the engine-divergence note, `docs/gym-design.md`,
  `docs/negotiation-interface.md` and `docs/onnx-contract-v2.md`, and the
  research-only bench instruments `hexset.bench.shipped_hand`,
  `hand_valuation`, `road_sweep`, `production_curve`, `encode_cost`,
  `behaviour` and `human_agreement` (with `hexset.behaviour`, which only
  served the last two) -- this is a gym to import, not a lab notebook, and
  the measurement record and the dated instruments are kept outside this
  repository. `docs/onnx.md` is the live interface reference and stays. The
  shareable bench is `duel`, `generate`, `throughput`, `trade_census`,
  `fit_weights`, `fit_duel` and `baselines`; `trade_census` no longer seats
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
  game's, or an imported game's. `Record` gains `actors`, sparse
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
  `hand_valuation` and weight sweeps on top of it), `throughput` and
  `trade_census` all play their games through it and read the `Tournament`
  it returns, in place of five separate copies of the board seeding, the
  antithetic pairing and the play loop. A tournament carries what those
  copies existed to collect: per-seat roads, settlements and cities, the
  seating each game used, and -- with `records=True` -- every trade the
  engine cleared, with its turn, phase, both hands and both private gains. A
  duel's seating is a lineup rather than one of two named geometries:
  `--geometry` takes any pattern of `a`/`b` slots, or a comma-separated
  lineup that can also seat entrants on neither side. `--games` must divide
  by the number of seats wherever it did not have to before, and
  `hexset.bench.throughput` reports games and turns rather than actions.

## 0.32.0

No engine changes.

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
  every existing bot plays a served table unchanged; the search bots
  additionally implement `estimate_many` for real.
  `hexset.server.webplay.PendingGate` gains the manual-seat side of the same
  three methods (`offer`/`respond`/`pick`), recording a broadcast offer to
  `game.pending` exactly as it already does for the clearing house.

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
- `Search(..., stance=...)` built a tree whose `Node.ranked` never
  accumulated for a stance it did not implement — `STANCE_ROWS`/`_backup`
  only ever implemented `own`/`relative`/`paranoid`, so PUCT's value term
  was silently zero under any other. `Search` now raises at construction for
  any stance outside that implemented set.

## 0.29.0

### Changed

- A trade moves at most `hexset.trading.MAX_TRADE_CARDS` (3) cards on either
  side.
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
  played game for a static ablation: `bank` plays and records games of four
  search bots, and `census` replays each to every point a trade event fires,
  respawning the same bots to re-derive the published vectors, then clears
  every position under four selection rules — the shipped maximin-public
  surplus rule, and three private-gain rules (actor, egalitarian, nash) that
  skip the public-vector filter — reporting trades-per-event, bundle shape,
  surplus split, bystander win-probability damage and rule disagreement.
  Torch-free, multiprocessed like `hexset.bench.trade_census`. `census` also
  reports the gain-distribution quantiles per rule; `strategic` shades one
  rotating seat's gate (a `tau` acceptance threshold, and — for the two
  rules that read magnitudes — a selection-key exaggeration) to check
  whether honesty is a fixed point; `rollouts` judges 300 sampled executed
  trades per rule by playing the position out both traded and untraded with
  fresh search bots, comparing realised win-share swing against the gate's
  own claimed gain.

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
  divergence; `Recording` wraps either and logs every outcome; `Forced` pins
  one steal's resource for a counterfactual child (`hexset.bench.aivat` and
  a search bot's search, replacing each module's own `_Forced`
  stand-in-rng). `imagine` always hands its copy a fresh `Live`, never the
  real game's `chance`, so a search can never drain a replay's scripted
  stream or leak its own draws into one being recorded.
- **`hexset.record.from_journal`.** Converts a `hexset.server.journal` file
  into a `Record` directly — no seed, no re-running the engine to recover
  the deck, rolls or steals, since the journal already spells them out.
  The porting surface a v2 `Record` was built for.
- **`--records <path>` on `hexset.bench.duel` (arena path, `--workers > 1`)
  and `hexset.bench.trade_census`.** Appends every game played as a v2
  record. On `duel`, the recorded games are exactly the games the verdict
  counted (`arena.compete(records=True)` builds both from the same job);
  unavailable with `--workers 1`, which plays through a batched collector and
  returns a verdict with no per-game history.
- **Catanatron's bots can sit at a HexSet table.**
  `hexset.catanatron.bot.CatanatronBot` is a `hexset.arena` bot whose brain
  is a Catanatron `Player`, registered as the `catanatron` preset
  (Catanatron's own AlphaBeta player at depth two, built exactly as
  `catanatron-play --players=AB:2` builds it) — so it can be seated from the
  web picker, `POST /api/bot`, an arena lineup or the gym, alongside the
  other bots. Each decision mirrors the live HexSet position into a
  Catanatron `Game` (`hexset.catanatron.state.to_catanatron`, on the map
  `hexset.catanatron.board.catanatron_map` builds from the HexSet board) and
  translates the answer back. The seat never trades: Catanatron's players
  have no notion of the one-event trade mechanic. The import of `catanatron`
  is lazy, so an install without the `catanatron` extra simply has one fewer
  opponent in the picker.
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
  trades census the same way once a runtime registers it.
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
  strictly above zero. The search bots price it from their own evaluators; a
  bot with only a boolean `accepts`/`accepts_many` gate is priced at
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
- **`hexset.trading._candidates` skips a zero-valuation seat's
  enumeration.** A seat that has never published (`NO_VALUATION`, all zero)
  can never clear a trade as either party —
  `_rank_candidates_loop`/`_rank_candidates_vectorized` already discard
  every candidate touching it (`mine <= 0.0` / `theirs <= 0.0`) — so
  `_candidates` now skips walking that seat's hand before generating any
  bundle, rather than enumerating them only to have ranking throw them away.
  Behaviour-preserving: the search bot's choice census is byte-identical.
- **The player list's picker gains a third option, "none".** Choosing it
  closes that seat outright (`POST /api/close`) — the explicit gesture that
  replaces the setup snake retiring an open seat on sight. A closed seat
  reads "locked seat" exactly as one the snake used to retire did, and its
  row disappears once the match is under way. Any seated person may close
  any other seat, the same permission as picking it a bot.

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
  instead of hardcoding `"relative"`.** Every constructor of an entrant
  besides the bot's own presets
  (`hexset.tuning.entrant_for`/`duel`/`climb`/`confirm`, `hexset.bench.tune
  --stance`, `hexset.bench.road_sweep`'s challenger/baseline) still spawned
  it at `relative` rather than its own default, silently disagreeing with
  the presets that had to override it explicitly. `Entrant.stance` is now
  `None` by default, resolved at spawn time to each kind's own default --
  stated once, on the bot, instead of on every caller.
- **`hexset.catanatron.duel`/`.player` now register the bot presets.**
  Neither module imported `hexset.bots`, so the bot presets were missing
  from `PRESETS` in the bridge's worker processes and naming one in
  `--players=DC:...` raised `KeyError`. `hexset.catanatron.player` now
  imports `hexset.bots` at module scope.
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
  Every bot seat submitted a move and then slept a full second before
  looking at the board again, and the page waited 1.5 s between reads on top
  of that — so a search costing under a tenth of a second landed on a
  one-second boundary and a table of three bots crawled. Nothing is paced by
  a clock any more: a bot plays its whole turn back to back and then waits
  for the table to change, and the page is told the moment it does. Three
  search-bot seats now finish the setup phase in under a second, where the
  same lineup took the better part of twenty.
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
  cap.
- **Trade and build interleave.** The trade event runs at the start of
  `Phase.MAIN` and again after every MAIN action the current player takes —
  build, buy, a bank/port trade, a development card — on the same published
  vectors (replaces "one event before any build"). Never runs after
  `end_turn`, and never during setup, `ROLL`, `ROBBER` or discard resolution.
- **The tie-break is the acting seat's choice among fair deals, not fewer
  cards.** Rank keys: the smaller of the two public surpluses, highest
  first (unchanged, the maximin); the current player's own surplus, highest
  first — among equally fair deals the actor takes the better one for
  itself; the total surplus, highest first; a canonical bundle order, then
  the lower counterparty seat, for determinism only (replaces
  fewer-cards/canonical/lower-seat as the whole rule).

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
- `Entrant.max_offers` and the `max_offers` checkpoint metadata key are both
  `max_trades`; `network:<path>@0` replaces `@<offers>`.
  `hexset.server.web`'s `--max-offers` is `--no-trade`.
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
  `pair_mask`/`NUM_PAIRS` and `server.rules.fair_legal_actions`/
  `proposable_options`.
- The `Belief` alias for `hexset.view.View`.

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
- **The engine moved into this repo, under `engine/`.** `engine/hexset` (the
  rules engine, bots, ledger, arena and tuning) was imported with its full
  commit history.
- **A single `pip install -e .` from the repo root now provides `hexset`,
  its bot package and `hexset_ui`.** `hexset` is no longer installed from a
  separate checkout; see the README's "Running it" section.
- **`hexset.build_info()`**: version and git commit for a consumer, a
  training project's run manifest say, to stamp into its own provenance records.
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

### Changed

- **`Game.state` is now a method, not a field.** `game.state(seat, *,
  hidden=True)` is the access path: `hidden=True` (the default) returns
  `seat`'s information-set `View` (`hexset.view`, moved from a bot's own
  `Belief` -- `Belief` is kept as an alias); `hidden=False` returns the true
  `GameState` (the same object every time, never a copy) and is the only
  sanctioned way to read it from outside the engine. The three sanctioned
  callers are one search bot, another's own `omniscient` mode, and the
  Catanatron adapter when it hosts a Catanatron bot. `Game.set_state(state)`
  replaces the true state outright (the one write a determinizer or an undo
  needs) without exposing the now-private `Game._state` field.
- **HexSet is licensed GPL-3.0-only** (was AGPL-3.0). One licence for the
  whole distribution; third-party components are declared in
  `pyproject.toml` and summarised in the README.
- **One distribution, `hexset`.** `engine/` and `hexset_ui/` are gone;
  everything now ships as `hexset` (engine, bots, ledger), `hexset.bench`,
  `hexset.server`, `hexset.clients` and the sibling bot package, under one
  `pyproject.toml`. Update any import of `benchmarks.*` to `hexset.bench.*`,
  and of `hexset_ui.*` to `hexset.server.*` or `hexset.clients.*`; the
  PyPI/Docker distribution name changes from `hexset-ui` to `hexset`, and
  the MCP server's advertised name from `hexset-ui` to `hexset`.
  `onnxruntime` moves from a hard dependency to the `.[server]`/`.[clients]`
  extras. `hexset_ui/record.py`'s duplicate of `hexset.onnx_record` is
  deleted now that the latter no longer needs torch.
- The engine's own test suite now runs in place at `engine/tests` instead
  of a separate checkout.
- **Package renamed `catan` → `hexset`**, ahead of release as its own
  public repo under GPL-3.0-only. `import catan` → `import hexset`
  throughout; the `CATAN_EXPORT_COMMIT` env var →
  `HEXSET_EXPORT_COMMIT`. Every source file under `hexset/`, `benchmarks/`
  and `tests/` now carries an SPDX `GPL-3.0-only` header, and a `LICENSE`
  file is added.
- `hexset` split into `hexset` (engine, bots, ledger) and a sibling training
  package (PPO/training research), ahead of the two becoming separate
  repositories. `collect`, `ddp`, `distill`, `distill_train`, `expert`,
  `export_onnx`, `league`, `migrate`, `model`, `netbot`, `policy`, `ppo`,
  `readout`, `rewards`, `schedule`, `selfplay`, `train`, `widen` and `run/`
  move to the training package (`import hexset.train` no longer resolves);
  training-bound benchmarks move with them. `hexset` still declares only
  numpy and never imports that package: `hexset.arena` gained a
  registry (`register_entrant_kind`, `register_evaluator_provider`,
  `register_checkpoint_loader`, `register_leaf_evaluator_factory`) that
  the training package populates on import. `relative_points` moved from
  `rewards` to `victory`; `NUM_PAIRS`/`pair_index`/`pair_mask` moved from
  `policy` to `actions` — both re-export their old names.
- `hexset.arena` gained `register_preset`; a bot package calls it on import
  to register its presets. `hexset` never imports a bot package; consumers
  (`benchmarks.duel`, `hexset_ui`) import it explicitly.

### Removed

- ONNX contract 1 is no longer served. `onnxbot` refuses a contract-1 or
  contract-unspecified checkpoint by name; the server serves contracts 2, 3
  and 4 only. `encoding_v1.py` and `OnnxPolicy` are deleted.

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

- `benchmarks.human_agreement`: scores the policy against the decisions in
  human game records, one decision at a time — **top-1 agreement** and
  **log-loss**, each against a matched null (uniform over the legal option set at that
  position). Decisions with a single legal action are excluded and counted
  separately. Results break down by `ActionType`, `Phase`, and game
  progress, and confidence intervals are clustered on the game rather than
  the position.

## 0.10.0

### Added

- `--mix` now accepts any arena entrant spec, not just two hardcoded
  names: `mcts:<ckpt>@64`, `network:<ckpt>`, or any
  preset, resolved through `collect.named_opponent`. `collect.check_mix`
  refuses a mistyped entrant or missing checkpoint before the run starts.
- `collect.RESERVED_MIX` names a preset name and `parent`, which keep resolving
  to the same bots as before.
- `benchmarks.mix_cost`: reports what a `--mix` costs a PPO iteration, per
  decision and per shard.

### Fixed

- The in-process collector's `--mix` fell through to the parent checkpoint
  for any name other than one preset name. Both collectors now build opponents
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
  external anchors calibrate.

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
- Opponent mixing in the collector, plus a frozen evaluation ladder.
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
- `catan.bots`: a `Bot` protocol the network will also satisfy and a
  random bot.
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
- `benchmarks.tune`: runs the climb, reporting each duel as it resolves.
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
  stops at the first player to accept. Records carry the order, so a
  game with a choosing proposer replays.
- `catan.trading.responders` orders an offer round the table from the
  proposer rather than by ascending seat index.
- `benchmarks.throughput.environment` reports whether the working tree
  was dirty; runners default `--workers` to every core.
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
carried a private copy of the engine).

### Added

- A browser game of humans against bots: HTTP API with a lobby, MCP server,
  static web client, and ONNX Runtime inference for exported networks.
