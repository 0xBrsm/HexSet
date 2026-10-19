# HTTP API

JSON over HTTP against a running server ([server.md](server.md)). Request
bodies are JSON objects. Creating, joining or reclaiming a seat returns a
`token` beside the seat's state; every seat route takes that token in the
`X-HexSet-Token` header.

| Method and route | Request or result |
| --- | --- |
| `GET /api` | `{"api": 4, "docs": "<this page's URL>"}` |
| `GET /api/version` | `{"api": 4}`, this API's contract version |
| `GET /api/models` | `{"models": [...], "game_types": {...}}`, the names `bots` and `model` accept, with each bot's profile ([below](#opponent-names)) |
| `GET /api/rankings` | The rankings ([below](#rankings)); 404 where the server keeps none |
| `POST /api/games` | Optional `{"name": "Alice", "bots": ["mybot"], "board_mode": "spiral", "game_type": "standard", "client": {...}}`; deals an unranked game |
| `POST /api/join` | `{"code": "abcdef", "name": "Alice", "client": {...}}`; claims a random open seat |
| `POST /api/reclaim` | `{"code": "abcdef", "secret": "..."}`; a fresh token for the seat whose client id matches |
| `POST /api/ranked` | `{"game_type": "standard", "client": {...}}`; a ticket for a ranked game ([below](#ranked-play)) |
| `GET /api/ranked/<ticket>?wait=N` | `waiting`, or once matched the seat's token and state |
| `POST /api/ranked/<ticket>/leave` | Leaves the queue |
| `GET /api/state` | The seat's view, including `legal_actions` |
| `GET /api/board` | Static board layout |
| `GET /api/record` | The seat's ONNX input record, `options` and `space`; 409 unless the seat may act |
| `POST /api/action` | `{"action": <legal_actions entry>, "version": n}` |
| `POST /api/undo` | Take back the seat's last action; see `can_undo` |
| `POST /api/name` | `{"name": "Alice"}`; an empty name restores the seat's default |
| `POST /api/bot` | `{"seat": 1, "model": "mybot"}`; seats a bot on an empty seat or replaces a bot |
| `POST /api/close` | `{"seat": 1}`; closes an empty seat, or one a bot holds, before the first move |
| `POST /api/open` | `{"seat": 1}`; reopens a closed seat, or takes the bot off one, before the first move |
| `POST /api/leave` | Retires the caller's seat for the rest of the game; 400 while a trade round awaits it or is its own |
| `GET /api/table/<code>` | Spectator view: every hand, omniscient |
| `GET /api/table/<code>/board` | Board layout |
| `GET /api/table/<code>/replay?round=N` | The position at the end of round `N` |

`/api`, `version`, `models`, `rankings`, `games`, `join`, `reclaim`, the `ranked` routes and the three
`/api/table/<code>` reads take no token. Codes are six characters from
`23456789abcdefghjkmnpqrstuvwxyz`, matched case-insensitively. Names are
trimmed and cut to 40 characters; an unnamed seat is named `human`, `api` or
`mcp` after its client's kind ([below](#client-identity-and-seat-recovery)).

## Errors

Every refusal is `{"error": "<message>"}` with one of these statuses:

| Status | When |
| --- | --- |
| 400 | Malformed body or field (a seat that is not an integer, a name that is not a string, a bad `client`), an illegal or out-of-turn action, an unknown model, a bot that will not load |
| 401 | A seat route without a token |
| 403 | An unknown token; a reclaim secret that matches no seat |
| 404 | An unknown game code or route |
| 409 | A stale `version`; an action while seats are unresolved (`waiting_for`) or a bot's offer awaits answers (`trade_wait`); any seat POST once the game is over; no open seat to join; seat changes after the first move or at a table with fixed seats; the last seat leaving; a trade-round conflict (below) |
| 429 | `POST /api/games` from an address over its new-game limits (below) |
| 500 | An unexpected server error |

`POST /api/games` validates every bot before dealing: a name not offered at
the game's type in `/api/models`, more bots than the game type leaves seats for, or a bot that
will not load is a 400, and no game is created. `board_mode` names a
`hexset.board.board.BOARD_MODES` entry, `spiral` (the server's `--board`)
when omitted, and `game_type` a `hexset.rules.GAME_TYPES` entry, `standard`
when omitted; an unknown name is a 400. Every game has four seats. A
`standard` game starts with all four open, and closing seats plays it at two
or three. A `duel-variant` game (15 points, discard above 9, friendly robber,
balanced dice) is played at two: seats 2 and 3 are closed at the deal, the
creator sits at seat 0 or 1, at most one bot is named, and its seats are
fixed, so `/api/close`, `/api/open` and `/api/bot` on a closed seat refuse. `POST /api/bot` refuses a seat held by a person
(400) and a closed seat after the first move or at a table with fixed seats
(400), and changes nothing when the bot will not load.

## State

`GET /api/state`, and the response of every seat POST, is the seat's view:

| Field | Meaning |
| --- | --- |
| `code`, `seat`, `version` | Game code, this seat, the table's change counter |
| `phase`, `current_player`, `to_move` | Engine phase, the seat whose turn it is, the seat expected to act (`null` while `waiting_for` or `trade_wait` is non-empty) |
| `seats` | `{seat, kind, name}` per seat; `kind` is `empty`, `bot` or `player` |
| `waiting_for` | Empty, unclosed seats holding up setup |
| `claimed_seats`, `locked`, `started` | Occupied seats, closed or retired seats, whether the first move is made |
| `game_type`, `board_mode`, `seats_fixed`, `fixed_seats` | The game type and board mode the table was dealt with (`board_mode` is `null` for a board no mode dealt); whether the type is played at one seat count only (duel), and the seats its deal closed for good. A table's other seats can be closed before the first move, down to one person practising alone |
| `legal_actions` | The seat's moves now, as `{"type", "a", "b"}` objects; empty off turn |
| `can_undo`, `awaiting_confirm` | Whether `/api/undo` succeeds now; the seat holding its setup turn open |
| `players` | Per seat: `name`, `bot`, `victory_points`, `hand_size`, `dev_card_count`, `knights_played`, `road_length`, award flags, `last_roll`, ledger `known`/`unknown`; `hand` and `dev_cards` for this seat only |
| `bank`, `robber`, `vertex_owner`, `vertex_building`, `edge_owner` | Board and bank state; `robber` is `-1` while it stands beside a board with no desert, before its first move |
| `discard_quota`, `dev_cards_remaining`, `last_roll`, `round`, `winning_points` | Per-seat cards owed to a seven, deck size, dice, round, points to win |
| `trade_ratios` | This seat's bank rate per resource |
| `trades`, `pending`, `trade_round`, `trade_wait` | This turn's exchanges and the trade round (below) |
| `your_move`, `waiting_on` | What the table wants from this seat now (below); while that is `wait`, the seats it waits on |
| `winner`, `game_over`, `log` | Result and the transcript, redacted for this seat |

`your_move` is the first of these that holds, in this order:

| Value | The seat owes |
| --- | --- |
| `game_over` | Nothing: the game has ended |
| `discard` | A seven's discard, from `legal_actions`, whoever is `to_move` |
| `answer_trade` | An answer to the first offer in `pending` |
| `choose_trade` | A pick among the answers to its own round, or a decline |
| `act` | An action from `legal_actions`; only while this seat is `to_move` |
| `wait` | Nothing yet; `waiting_on` names the seats the table waits on |

A seat whose own round still awaits answers may act; the round closes with
its turn. A client that meets every value but `wait` can play any table:
nothing else stops play on its seat.

`victory_points` counts hidden victory-point cards only for this seat; other
seats show public points. Once the game is over every seat's hand, cards and
points are shown to every reader, and the log is unredacted.

Reads take `?after=<version>&wait=<seconds>` (`/api/state` and
`/api/table/<code>`): the response waits until `version` exceeds `after` or
`wait` runs out, capped at 25 seconds.

Submit an entry from the latest `legal_actions` unchanged. With `version`
the action is refused (409) if the table changed since that read; without
it the action is checked against the current position. A setup road leaves
the seat holding its setup turn (`awaiting_confirm`), with `END_TURN` as its
only action, so that the placement stays undoable.

`POST /api/undo` takes back the seat's most recent setup placement, build,
bank trade, or Road Building or Knight play while it is still the last
action in the game and no trade round of the seat's own is open (400
otherwise).

`GET /api/table/<code>/replay?round=N` clamps `N` to the game's rounds; a
missing or non-integer `round` is a 400. A token for a seat at that game reads
the round as that seat; any other reader, or no token, reads it omniscient.
The replay view has empty `legal_actions`, `can_undo` false, `your_move`
`wait` (`game_over` at a finished game's end), and a `replay` object with
`round`, `last_round` and `finished`.

## Trade routes

Every trade route takes `X-HexSet-Token`. Bundles are five integers in
resource order (`Wood`, `Brick`, `Sheep`, `Wheat`, `Ore`), signed from the
offer actor's side: positive counts go to the actor, negative counts leave
it. `[-1, 0, 0, 1, 0]` means the actor gives one Wood and receives one Wheat.

| Route | JSON body |
| --- | --- |
| `POST /api/games/<code>/trade/round` | `{"give": [1,0,0,0,0], "want": [0,0,0,1,0]}`, unsigned counts; optional `give_any` or `want_any` |
| `POST /api/games/<code>/trade/round/answer` | `{"actor": 0, "received": [-1,0,0,1,0], "kind": "accept"}`; `kind` is `accept`, `counter` or `pass`; a counter adds a signed `bundle`; optional `version` |
| `POST /api/games/<code>/trade/round/choose` | `{"seat": 1, "bundle": [-1,0,0,1,0]}` executes that answer; `{"decline": true}` closes the round; optional `version` |

Opening a round:

- Only the seat whose turn it is, in `MAIN`, with no Road Building roads left
  to place (409 otherwise).
- `give` and `want` are on disjoint resources, at least one card each (400).
- The seat must hold `give` (400).
- A second offer replaces the seat's open round.

`"give_any": n` adds `n` cards the actor gives, named by the answerer ("any
card for your Ore"); `"want_any": n` adds `n` cards the actor takes, chosen
by the answerer ("my Sheep for any card"). At most one of the two, and the
side holding them may name no cards. Such an offer is open: it is never
accepted as it stands, only countered with named cards, and the actor picks
or declines the counter like any other. The offer then carries `any`, signed
like the bundle (negative: the actor gives), in `pending` and `trade_round`.

`pending` in a seat's state lists the offers awaiting its answer, as `actor`
and `bundle`. Answer by echoing them as `actor` and `received`. 409 for:

- an offer no longer open, or a seat the round is not waiting on (the actor
  included, or a seat that already answered);
- an accept of an open offer;
- an accept or counter the answering seat cannot cover.

A counter's `bundle` is signed towards the actor, with cards both ways on
disjoint resources (409 otherwise).

The actor's `trade_round` holds `offer`, `responses` (each `seat`, `kind`,
`bundle`; a pass has a null `bundle`) and `awaiting`. The actor chooses a
recorded accept or counter by its exact `seat` and `bundle`. Choosing with
no open round, with Road Building roads left to place, or naming no recorded
answer is a 409.

Bot seats answer an offer at once. A manual seat with no cards passes
automatically.
A bot's own offer waits for every seat that answers through this API: those
seats are in `trade_wait`, `to_move` is `null`, and `/api/action` is a 409
until they answer. A bot offers on entering `MAIN` (or, for a gate with
`trade_now`, at the first main-phase point it says yes) and keeps offering
until its own `max_offers` is spent (no limit when it declares none) or it
has no offer it has not made this turn; it resolves each round after the
last answer. Executed trades appear in `trades`
and in the journal.

The MCP tools `offer_trade`, `answer_trade` and `choose_trade` drive these
routes with named resource counts and response indices ([mcp.md](mcp.md)).

## Client identity and seat recovery

`/api/games` and `/api/join` accept an optional client object:

```json
{"client": {"id": "<64 lowercase hex characters>", "name": "deepvine", "version": "2.3",
            "owner": "octocat", "url": "https://github.com/octocat/deepvine",
            "description": "MCTS at 400 simulations a move over a 1.2M-parameter self-play net."}}
```

`id` is the SHA-256 hex digest of a secret the client keeps, hashed as UTF-8.
It is the client's identity: without it the seat has none to recover, and
the rankings have nothing to count it under. Keep one secret for a bot
across its games, versions included.

The rest is the client's profile, each field a string trimmed and cut to its
length. A client of the HTTP API must send its `id`, `name`, `version`,
`owner` and `description` to deal or join (a 400 names what is missing): the
price of the service is that every game an API client plays counts, under an
id that says what it is. Only `url` is optional. The page and the MCP
endpoint are not asked (below, and [mcp.md](mcp.md)).

| Field | Length | What to put there |
| --- | --- | --- |
| `name` | 40 | What the client is called; also the seat's name when the body gives none |
| `version` | 40 | Its version, as its author numbers them |
| `owner` | 40 | Who runs it: a Discord or GitHub username is best |
| `url` | 200 | An http(s) page about it: its repository, a write-up, a model card. Refused (400), not cut, when longer or not http(s) |
| `description` | 100 | What it is, in one sentence: its architecture and its main parameters ("A rules-based reflex agent for 4p and duels."). Refused (400), not cut, when longer |

A profile is whatever the latest request carrying its `id` sent: each
`/api/games` or `/api/join` replaces it whole, so a field left out is cleared.
`POST /api/name` on a seat it holds changes its `name` alone, the rest kept.
The page remembers the name its player last gave and sends it with every
deal and join.
It is shown beside the id's rows in the rankings. Nothing verifies it.
Anything else in the object is ignored.

The seat's kind is the server's, decided from the request, never from the
body: `mcp` for a seat the MCP endpoint claims; `web` for a request the
browser marks as sent by this server's own page (`Sec-Fetch-Site:
same-origin`, or, from a browser that sends none, an `Origin` naming the host
the request was addressed to); `api` for anything else. A `web` seat is held
on its setup turn until it ends it, which gives the page's undo button its
time; an `api` seat's setup turn passes on its own. The marks are what a
browser sends, so a client that sets those headers itself is taken for the
page.

The journal also keeps, with each seat a request claims, where the request
came from: the client's address, `via` (`http` or `mcp`, the route it arrived
on) and its `User-Agent`. None of it is shown to any seat or spectator.

## New-game limits

A served table spends server time on every bot seated at it, so
`POST /api/games` is limited per client address, whichever interface it
comes through: at most `--games-per-period` new games in any rolling
`--period-days` (default 100 in 30 days, counted across restarts from the
journals), and at most `--live-tables-per-ip`
unfinished games in play at once (default 5). A game stops counting toward the
second once it ends or nobody has read it for 15 minutes. Over either limit
the deal is a 429 and nothing is created; a deal refused for another reason
does not count. Loopback addresses (`--limit-exempt`) are under no limit.

A 429 for the period limit reads `game limit: <n> per <days> days for this
address; next slot <UTC time>`, with `; contact: <contact>` when the server sets
`--contact`. Its body adds `next_slot` (and `contact`), and `Retry-After`
gives the seconds to the next slot.

A new game's response carries `allowance`: `games_left`, `games_per_period`,
`period_days`, `live_tables`, `live_tables_limit` (`null` where there is no
limit), plus `contact` once ten or fewer games are left. When the caller has
another game in play it also carries `notice`: tables share one CPU core, so
games played in parallel each run slower. `GET /api/limits` returns the same
`allowance` without dealing a game (`{}` for a caller under no limit).

A client the server's owner has issued an API key sends it on
`X-HexSet-Key` with `POST /api/games`. Its deals count against the key's own
allowance per period instead of its address's (the unfinished-games limit is
unchanged), and the journal names the key's holder, never the key. An
unknown key is a 401.

`POST /api/reclaim` takes the code and the secret, finds the seat whose `id`
matches, and returns a fresh token; the old token stops working. It works on
live and finished games and refuses bot seats and closed or retired seats.
The journal records the id, never the secret or the token.

## An example client

`hexset.clients.botclient` is a client of these routes alone, for reading
alongside this page. It needs no model and imports nothing from the engine
but the token header's name:

```sh
python -m hexset.clients.botclient --url http://127.0.0.1:8770 --game abcdef --owner octocat
python -m hexset.clients.botclient --url http://127.0.0.1:8770 --ranked duel-variant --owner octocat
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--url` | required | Server base URL |
| `--game` | `--game` or `--ranked` | Join this game by its code: unranked |
| `--name` | `botclient` | Its profile's `name`, and the seat's |
| `--seed` | none | Seed for its random choices |
| `--poll-interval` | `10.0` | Longest one long-poll waits, in seconds |
| `--ranked` | none | Instead of `--game`: queue for a ranked game of this type (`standard`, `duel-variant`) and play the seat dealt (`ranked_seat`) |
| `--client-secret` | `--name` | Secret behind the seat's client id |
| `--owner` | required | Its profile's `owner` |
| `--version` | `1` | Its profile's `version` |
| `--description` | this example's own | Its profile's `description` |
| `--info-url` | none | Its profile's `url` |

It joins, then meets whatever `your_move` names. On `act` and `discard` it
plays a build when one is legal, else any legal action, and once a turn in
`MAIN` it first offers one card of the kind it holds most of for one it holds
none of, acting again only once every seat has answered. On `answer_trade` it
accepts an offer it covers, counters an open offer with named cards it holds,
and passes otherwise; on `choose_trade` it executes the first answer it
covers, else declines. A refused request is not repeated: it waits for the
table to move and reads it again.

A client playing its own network reads `GET /api/record`, the
[record contract](onnx.md#record-inputs)'s input for this seat, beside `/api/state`.

## Ranked play

Only ranked games reach the [rankings](#rankings), and only the server
deals them: a client queues for a game type and is seated where the queue
puts it, with no say in the table, its opponents, the board or the seat
order. A game dealt by `POST /api/games`, or joined by its code, is unranked
and never counts. Ranked play is for clients of the HTTP API, which must say what they
are ([above](#client-identity-and-seat-recovery)), and for models over MCP
(`ranked_game`, [mcp.md](mcp.md)); the page does not queue.

`POST /api/ranked {"game_type": ..., "client": {...}}` returns
`{"ticket", "game_type", "status": "waiting"}`. It needs a `client.id`, the
identity every ranked game counts under, and is admitted against the
caller's limits as a deal is (New-game limits, above). `GET
/api/ranked/<ticket>?wait=N` long-polls up to 25 seconds: `waiting`, or once
matched `"status": "matched"` with the seat's `token` and its state, as
`POST /api/join` answers; from there the seat plays as any other. A ticket
not read for 60 seconds is dropped, and its deal given back; `POST
/api/ranked/<ticket>/leave` drops it at once. A client may hold any number of
tickets and play any number of ranked games at once.

Every 10 seconds the server matches each game type's queue. The oldest
ticket is seated with the waiting clients it has met least in ranked games,
never two seats for one client id, until the table is full. One seat left
goes to the server's own bot (`--ranked-bot`, [server.md](server.md)), never
more: like any player it sits once at a table, so a table short by two or
more waits for players. One
version of a client, its `id` at one profile `version`, sits at no more than
100 ranked tables with that bot; past that it waits for other clients, so a
new version starts afresh.

The server may also enter its own bots (`--ranked-entrant`,
[server.md](server.md)). Each holds a ticket at every game type it plays, one
ranked game of each at a time, is matched like any client, under the id
`bot:<name>` and its profile's version, and is ranked as one. Its seats are
not fill: a client seated with entrants alone has not met the fill bot.

A ranked table's seats are fixed: `/api/bot`, `/api/open` and `/api/close`
are a 409 there. It runs on a clock: when a move takes longer than 60
seconds, the game is aborted and its journal names the seats it was waiting
on, which the ranker is handed with every seat's points. A ranked game a
server restart cut short is closed at start and counts for nothing: its
seats get a 404 and queue again.

## Rankings

`GET /api/rankings` is the rankings the server's ranker made of its
ranked games ([server.md](server.md#rankings)); without a ranker it is a
404.

```json
{"updated": "YYYY-MM-DDTHH:MM:SSZ", "games": 12, "min_games": 10,
 "boards": [{"key": "duel", "title": "Duel", "note": null,
             "columns": [{"key": "rating", "label": "Rating"}],
             "rows": [{"name": "claude-opus-5-5", "id": "<64 hex>", "games": 40,
                       "rating": 1612}]}]}
```

`updated` is when it was last ranked (null until the first ranking finishes),
`games` how many ranked games it was ranked over, and `min_games` the
fewest games a row shown has played. Each board's `rows` are in the order
shown, each a `name`, an `id` and its `games`, either of the last two
null, and a value (text, a number, true/false or null)
for each of `columns`; `pinned: true` marks a rating the ranker holds fixed.
An `id` is a client id, the one a seat claims with
(above), and a row with one also carries that id's latest profile:
`version`, `owner`, `url` and `description`, null where it said none, and
its profile's `name` in place of the row's when it gave one. An `id` of
`bot:<entry>` is a server bot's row, carrying the profile the server lists
it with ([below](#opponent-names)).

## Opponent names

`GET /api/models` lists every preset a runtime registered, in registration
order (with the `catanatron` extra installed, `catanatron` among them), then
each `*.onnx` in the models directory as `<name>-<version>`, from the `name`
and `version` its metadata declares ([onnx.md](onnx.md#serving-a-model)); the
filename plays no part. `models` is every entry once; `game_types` maps each
`hexset.rules.GAME_TYPES` name to the entries a table of that type seats, each
with the `name`, `version` and `commit` a model file declares, and the
`owner`, `url` and `description` it declares too, or the profile a preset was
registered with ([guide.md](guide.md)):

```json
{"models": ["rehex", "mynet-1.0", "mynet-1.0d"],
 "game_types": {"standard": {"rehex": {"version": "2.11.1", "owner": "octocat",
                                       "description": "A rules-based reflex agent for 4p and duels."},
                             "mynet-1.0": {"name": "mynet", "version": "1.0", "commit": "0123abcd"}},
                "duel-variant": {"rehex": {"version": "2.11.1", "owner": "octocat",
                                           "description": "A rules-based reflex agent for 4p and duels."},
                                 "mynet-1.0d": {"name": "mynet", "version": "1.0d", "commit": "4567cdef"}}}}
```

A preset is offered at every game type. A model file plays only the game types
its `modes` declare, and the standard game alone when it declares none; files
may share a name as long as their versions differ. `bots` and `/api/bot` take
an entry, and one offered only at another game type is a 400.

`mybot` in the routes above stands for a preset: it exists only on a server started with the runtime that
registers it (`python -m hexset.server.web --runtime mybots`). Without a
runtime the list is `catanatron` and the ONNX files; see the
[guide](guide.md#bundled-opponents). A registered spec's other spellings
(such as `mybot:depth=3`) are available through the Python and evaluation
interfaces, not the server.
