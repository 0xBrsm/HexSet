# HTTP API

JSON over HTTP against a running server ([server.md](server.md)). Request
bodies are JSON objects. Creating, joining or reclaiming a seat returns a
`token` beside the seat's state; every seat route takes that token in the
`X-HexSet-Token` header.

| Method and route | Request or result |
| --- | --- |
| `GET /api/version` | `{"api": 4}`, this API's contract version |
| `GET /api/models` | `{"models": [...]}`, the names `bots` and `model` accept |
| `POST /api/games` | Optional `{"name": "Alice", "bots": ["mybot"], "board_mode": "spiral", "game_type": "standard", "client": {...}}`; deals a game |
| `POST /api/join` | `{"code": "abcdef", "name": "Alice", "client": {...}}`; claims a random open seat |
| `POST /api/reclaim` | `{"code": "abcdef", "secret": "..."}`; a fresh token for the seat whose client id matches |
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

`version`, `models`, `games`, `join`, `reclaim` and the three
`/api/table/<code>` reads take no token. Codes are six characters from
`23456789abcdefghjkmnpqrstuvwxyz`, matched case-insensitively. Names are
trimmed and cut to 40 characters; an unnamed seat is named `human`, `api` or
`mcp` after its client kind.

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

`POST /api/games` validates every bot before dealing: a name not in
`/api/models`, more bots than the game type leaves seats for, or a bot that
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
{"client": {"id": "<64 lowercase hex characters>", "kind": "api"}}
```

`id` is the SHA-256 hex digest of a secret the client keeps, hashed as UTF-8.
`kind` is `web`, `api` or `mcp`. Without the object the seat is `api` with no
recoverable identity. `mcp` is only what the MCP endpoint's own seats are: a
request to `/api/*` that claims it is journalled as `api`.

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
python -m hexset.clients.botclient --url http://127.0.0.1:8770 --game abcdef
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--url`, `--game` | required | Server base URL, game code |
| `--name` | none | Display name |
| `--seed` | none | Seed for its random choices |
| `--poll-interval` | `10.0` | Longest one long-poll waits, in seconds |
| `--client-secret` | `--name`, else `botclient` | Secret behind the seat's client id |

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

## Opponent names

`GET /api/models` lists every preset a runtime registered, in registration
order (with the `catanatron` extra installed, `catanatron` among them), then
each `*.onnx` in the models directory by filename stem. `mybot` above stands
for such a preset: it exists only on a server started with the runtime that
registers it (`python -m hexset.server.web --runtime mybots`). Without a
runtime the list is `catanatron` and the ONNX files; see the
[guide](guide.md#bundled-opponents). A registered spec's other spellings
(such as `mybot:depth=3`) are available through the Python and evaluation
interfaces, not the server.
