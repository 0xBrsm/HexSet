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
| `GET /api/menu` | `{"tabs": [{"label": "...", "url": "..."}]}`, the tabs the server adds to the page's menu for this client ([below](#menu-tabs)) |
| `POST /api/games` | Optional `{"name": "Alice", "bots": ["mybot"], "board_mode": "spiral", "game_type": "standard", "client": {...}}`; deals an unranked game |
| `POST /api/join` | `{"code": "abcdef", "name": "Alice", "client": {...}}`; claims a random open seat |
| `POST /api/page` | `{"turnstile": "<answer>"}`: the page's check, for the registered guide; `{"verified": bool}` ([below](#who-may-use-the-guide)) |
| `POST /api/reclaim` | `{"code": "abcdef", "secret": "...", "probe": false}`; a fresh token for the seat whose client id matches |
| `POST /api/ranked` | `{"game_type": "standard", "client": {...}}`; a ticket for a ranked game ([below](#ranked-play)) |
| `GET /api/ranked/<ticket>?wait=N` | `waiting`, or once matched the seat's token and state |
| `POST /api/ranked/<ticket>/leave` | Leaves the queue |
| `GET /api/state` | The seat's view, including `legal_actions` |
| `GET /api/board` | Static board layout; `spots` names each vertex by its numbers, no two alike (`hexset.board.spots.spot_names`) |
| `GET /api/record` | The seat's ONNX input record, `options` and `space`; 409 unless the seat may act |
| `GET /api/chance` | The seat's chance of winning as the table's model rates it ([below](#win-chance)); 409 at a ranked table |
| `POST /api/action` | `{"action": <legal_actions entry>, "version": n}` |
| `POST /api/undo` | The web page's: take back the seat's last action; see `can_undo` |
| `POST /api/guide` | The web page's: turn on the seat's guide ([below](#guide)); 409 at a ranked table, at a table more than two seats play, or where no guide is offered |
| `POST /api/hint` | Show the seat its guide's move for the current decision ([below](#guide)); 409 while the guide is off |
| `POST /api/takeback` | The web page's: take the game back to just before the seat's last move ([below](#guide)) |
| `POST /api/flag` | The web page's: flag one of the guide's cards for review, or unflag it ([below](#guide)); a finished game's too |
| `POST /api/playfrom` | The web page's: `{"step": N}`; a new table playing the game on from the position after `N` steps ([below](#guide)) |
| `POST /api/name` | `{"name": "Alice"}`; an empty name restores the seat's default |
| `POST /api/bot` | `{"seat": 1, "model": "mybot"}`; seats a bot on an empty seat or replaces a bot |
| `POST /api/close` | `{"seat": 1}`; closes an empty seat, or one a bot holds, before the first move |
| `POST /api/open` | `{"seat": 1}`; reopens a closed seat, or takes the bot off one, before the first move |
| `POST /api/leave` | Retires the caller's seat for the rest of the game; 400 while a trade round awaits it or is its own |
| `GET /api/table/<code>` | Spectator view: every hand, omniscient |
| `GET /api/table/<code>/board` | Board layout |
| `GET /api/table/<code>/replay?round=N` | The position at the end of round `N`; `?step=N` the position after `N` steps |
| `GET /api/table/<code>/timeline` | The game's steps, and a seat's guide chance and graded moves at each, for a replay strip |

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
| 403 | An unknown token; a reclaim secret that matches no seat (without `probe`) |
| 404 | An unknown game code or route |
| 409 | A stale `version`; `GET /api/chance` at a ranked table; an action while seats are unresolved (`waiting_for`) or a bot's offer awaits answers (`trade_wait`); any seat POST once the game is over; no open seat to join; seat changes after the first move or at a table with fixed seats; the last seat leaving; a trade-round conflict (below) |
| 429 | `POST /api/games` from an address over its new-game limits, or `POST /api/ranked` from one at its unfinished-games limit (below) |
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
| `seats` | `{seat, kind, name}` per seat; `kind` is `empty`, `bot` or `player`. A bot seat whose profile gives a `label` that differs from its name also carries it as `label`, the name the page shows |
| `waiting_for` | Empty, unclosed seats holding up setup |
| `claimed_seats`, `locked`, `started` | Occupied seats, closed or retired seats, whether the first move is made |
| `game_type`, `board_mode`, `seats_fixed`, `fixed_seats` | The game type and board mode the table was dealt with (`board_mode` is `null` for a board no mode dealt); whether the type is played at one seat count only (duel), and the seats its deal closed for good. A table's other seats can be closed before the first move, down to one person practising alone |
| `legal_actions` | The seat's moves now, as `{"type", "a", "b"}` objects; empty off turn |
| `can_undo`, `awaiting_confirm` | Whether `/api/undo` succeeds now (always false for an API or MCP seat); the seat holding its setup turn open |
| `players` | Per seat: `name`, `bot`, `victory_points`, `hand_size`, `dev_card_count`, `knights_played`, `road_length`, award flags, `last_roll`, ledger `known`/`unknown`; `hand` and `dev_cards` for this seat only |
| `bank`, `robber`, `vertex_owner`, `vertex_building`, `edge_owner` | Board and bank state; `robber` is `-1` while it stands beside a board with no desert, before its first move |
| `discard_quota`, `dev_cards_remaining`, `last_roll`, `round`, `winning_points` | Per-seat cards owed to a seven, deck size, dice, round, points to win |
| `trade_ratios` | This seat's bank rate per resource |
| `trades`, `pending`, `trade_round`, `trade_wait` | This turn's exchanges and the trade round (below) |
| `your_move`, `waiting_on` | What the table wants from this seat now (below); while that is `wait`, the seats it waits on |
| `winner`, `game_over`, `log` | Result and the transcript, redacted for this seat |
| `feed` | `{"epoch", "events"}`: the newest actions, one entry each, hidden as the log is (below) |

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

`feed.events` holds the newest 96 actions, each numbered `n` by its place in
the game, with `round`, `actor` and `kind`: `roll` (`value`, and `gains`, seat
to cards gained by resource), `build` (`piece`, `at` a vertex or edge,
`setup`, and a setup settlement's `gains`), `buy` (`card`), `play` (`card`;
Monopoly's `resource` and `taken`, seat to cards; Year Of Plenty's
`resources`), `robber` (`hex`, `victim`, `stole`, `card`), `bank` (`gave`,
`got`), `discard` (`card`), `end`, `round` (a trade round's step: `step`,
`seat`, `bundle`, and an offer's `any` cards where it asks for any) and
`trade`. An entry may carry `trades`, the exchanges
cleared with it. A bought, stolen or discarded `card` is `null` except to the
seat that holds it (and the thief, for a steal); a discard round still open is
left out until every seat has discarded. An undo rewrites the game's actions
and starts a new `epoch`.

An entry that changed a seat's row in `players` carries `after`: seat to the
keys of that row the action changed (`victory_points`, `knights_played`,
`road_length`, `longest_road`, `largest_army`, `hand_size`, `dev_card_count`,
`hand`, `dev_cards`), each with its value after the action, hidden as the row
itself is. Each of an entry's `trades` carries its own `after`, what that
exchange changed. An entry that changed the bank carries `bank`, its
resource counts after the action, keyed as the view's `bank`. A client that
applies each entry's `after` to the rows of the view it held before, and its
`bank` to the view's, shows the table as it stood after that entry.

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
otherwise). It is the web page's, for a click taken back: a seat an API or
MCP client claimed gets a 403 and plays each move as meant.

`GET /api/table/<code>/replay?round=N` clamps `N` to the game's rounds; a
missing or non-integer `round` is a 400. `?step=N` in its place reads the
position after `N` steps (0 is the deal), clamped to the steps played, and the
round is the last step's. A token for a seat at that game reads
the round as that seat; any other reader, or no token, reads it omniscient.
The replay view has empty `legal_actions`, `can_undo` false, `your_move`
`wait` (`game_over` at a finished game's end), and a `replay` object with
`round`, `last_round`, `step`, `last_step`, `finished` and `guide_chance`. Read with a seat's token
at an unranked game, it also carries `chance`, `{"rater", "chance"}` as `GET
/api/chance` gives them, read at that round, and `guide_chance` is the seat's
guide `chance` as it stood at that round's end, from only what the guide read
up to there: from the live table's guide where the seat's guide is on, else
from the graded moves in the journal; `null` for anyone else or with nothing
read.

`replay.hint` is the hint the seat's guide gave at the decision after that
step, in the shape of the guide's live `hint` ([below](#hints-and-notes)),
rebuilt by the guide from what it kept of that reading, with no new reading.
It is the hint of the note the game's line kept at that step (matched by its
`id`), shown only where what the seat could see there
(`hexset.game.observed_digest`) is what it saw when the hint was given, so a
hint for a move later taken back never shows on the line that replaced it.
Its `id` is the note's. `null` for anyone else, at a ranked game, at another
seat's step, at a forced move, and where the guide kept nothing of it.

`replay.played` lists every move the seat played at that decision, with the
note its guide gave it, kept or taken back: the notes of the game's line
there, and those of moves taken back at the same position, where no undo
after the move went back before that step or the seat saw the same there.
The newest note of each move, oldest first, each with `taken_back`. Empty
for anyone else, at a ranked game and where none was played or graded.

A seat's replay view at an unranked game also carries `replay.stats`, its
guide's dice and robber numbers (`guide.stats`) as they stood after that
step. The page's review covers the whole game at every position, its
graph's playhead at the position shown.

`GET /api/table/<code>/timeline` gives `steps`, the steps played,
`last_round`, and `items`, one a step, `{"round", "actor", "type"}` (`type`
an action type's name, `TRADE` for a trade). Read with a seat's token at an
unranked game it also gives `chance`, the seat's guide chance after each step
from the deal to the newest (`steps` + 1 values, `null` before anything is
read; from the live guide where the seat's guide is on, else from the graded
moves in the journal), and `marks`, `{"step", "class"}` for each graded move
of the seat's; `null` and empty for anyone else. The page's replay bar
draws it as a strip: the chance as a filled shape, a tick under each
brilliant move and each error in its grade's colour, and the playhead at the
position shown. The strip spans about a typical game (250 steps for a
duel), so a game in play grows from its left, and a longer game fits it. A
tap or drag on the strip, a swipe across the board (left for the next move,
right for the one before; from the game, right steps back into it), the
arrow keys, the mouse wheel over the strip itself (a move a notch, a
trackpad's small deltas added up to a notch; anywhere else the wheel
scrolls as ever), a tap or drag on the review's graph, or a row of the
guide's review (the position just before that move) moves the board. The first time a game is stepped back into, the
strip says how: "← → TO STEP" on a mouse or trackpad, "SWIPE THE BOARD TO
STEP" on a touch screen, until the keys, the wheel or a swipe is used. The strip has a pill on each side and no other
controls: on its right the round ("R12"), which turns red ("R8»") while a
past position is shown and returns to the game when tapped; on its left the
guide's win chance (below). Both keep one width and place at every step.

## Win chance

`GET /api/chance` returns `{"version": 12, "rater": "mynet", "chance":
0.4312}`: the seat's chance of winning as one model at the table rates it,
read at table `version`. The rater is the server's `--chance-model`
([server.md](server.md)) at every game type it is offered at, whether or not
it sits at the table; elsewhere, and without one, it is the lowest-numbered
bot seat still in the game that plays a model file from `/api/models` rather
than a registered bot. The read is one evaluation of that model's value head,
encoded from the asking seat's own side, so it uses only what that seat may
see; it runs no search. `rater` is the model's shown name (its profile's
`label`, else its entry), `null` where no model rates the table. Once the game
is over, the read is of its last position still in play, rebuilt from the
journal: the position before the step that ended it, or the nearest earlier
one a replay can stop at. `chance` is `null` where no model rates the table,
for a seat that has left, and where the model cannot be read. A table is read
once per `version`: asking again before the table changes returns the same
answer. Ranked tables refuse the route with a 409. The page does not show it:
the page's win chance is the guide's ([below](#guide)).

## Guide

A guide is a coach for a person playing on the server's page: it reads each
of the seat's decisions, offers a hint, and grades each move the seat plays.
HexSet does not include one. A server offers a guide where a runtime it loads
(`--runtime`) registers one (`hexset.guides.register_guide`) and
`--guide-model` ([server.md](server.md)) names the model the guide reads
with, at every game type that model is offered at. It is offered to a seat
the server's own page claimed at an unranked duel (a table two seats play,
the rest locked). The seat's view then carries `guide`: `{"offered": true,
"takeback": false, "on": false}` until the seat turns it on with `POST
/api/guide`, and `{"offered": false, "takeback": false, "on": false}` where
there is none to turn on. `takeback` says whether the seat has a move to take
back. An API or MCP seat gets no `guide` and a 403 from every guide route.

Once it is on, the view's `guide` is:

| field | meaning |
|---|---|
| `on` | `true` |
| `step` | The step the view is at, which `reading` and `hint` are of |
| `reading` | The current decision: `none` (no decision with a choice), `pending` while the guide reads it, or `ready` once a hint shows at once when asked for |
| `hint` | `null` until `POST /api/hint` asks for it, then `{"id", "move", "options", "why", "objective", "groups"}` ([below](#hints-and-notes)) |
| `notes` | Every graded move of the seat's and every forced one, oldest first |
| `taken_back` | The seat's graded moves since taken back, each a note with its class |
| `grading` | How many of the seat's moves are still being graded |
| `pending` | The newest of those, `{"step", "played"}`, while its note has not landed, else `null` |
| `flags` | The guide's cards the seat has flagged for review now, by the page's key for each card (`POST /api/flag`) |
| `stats` | The game's dice and robber numbers so far in cards, read from its journal (takebacks included), `{"you", "them", "rounds"}`: `you` the seat's and `them` its opponent's (the opponents' mean at more seats), each `{"dice", "robber", "net", "on", "moves", "discarded"}`: `dice` the cards the rolls other than 7 paid, the robber's blocks put back, less what they pay on average with no robber on the board, `robber` cards taken by the seat's steals less cards the robber blocked and stole from it, `net` the two together, `on` robber moves by another seat onto the seat's buildings of `moves`, every robber move by another seat, `discarded` cards given up to sevens; `null` without a journal |
| `chance` | The seat's win chance as the guide last read it, `{"value", "step", "after"}`: `after` true where it is the score of a move the seat played, false where it is the position's; `null` before anything is read |

### Hints and notes

A hint is `{"id", "move", "options", "why", "objective", "groups"}`: `id` a
number no other note or hint of the table's guide shares, `move` the first
option's move, and `options`, the moves to play, three at most, best first,
each `{"action", "score", "loss", "lead", "bound", "class", "why"}`: `score`
the seat's win chance after that move as the guide reads it, `loss` the
guide's own move's score less it and `lead` the reverse, `bound` the
smallest difference the guide tells apart (`null` for the guide's own move),
`class` the grade the move would get if played, and `why` its reasons. The
hint's `why` says why the moves shown beat the rest; `groups` lists, for two
or more options the page may show on one card, `{"of", "why"}`: `of` their
places in `options` and `why` the reasons they share. During a discard read
whole, a hint also carries `given`, the count of each resource given so far.

A note is `{"id", "seat", "step", "round", "played", "best", "loss", "score",
"class", "objective", "bound", "better", "better_classes", "why",
"better_whys", "better_groups"}`: the move played and the guide's own (wire
actions), `loss` and `score` as a hint option's, and for a flagged move
`better`, the moves that were better, with their classes, reasons and the
reasons they share; a brilliant move also carries `lead`, its gain over the
guide's own move. A `why` is `{"text", "subject", "predicate"}`: the
sentence the page shows, and the subject and predicate it is written from.

`objective` is what a grade is on: `"win"` (win chance) or `"mix"` (win
chance with the points gap), where an option or note on `"mix"` also carries
`unit`, `"win"` or `"vp"`, the part that decided it, and `weight`, the win
chance a point is worth there.

`class` is one of `forced` (the only legal move), `best` (the guide's own
move, with nothing as good), `good` (as good as the guide's own move, as far
as the guide can tell), `brilliant` (clearly better than it), `mistake`
(clearly worse) and `blunder` (much worse).

### Who may use the guide

The registered guide decides who may use it (`hexset.guides`): it may refuse
`POST /api/guide`, `POST /api/hint`, `POST /api/takeback` and `POST
/api/playfrom` with an error of its own, a 428 where the page must reload or
pass a check (with `"check": "turnstile"` for the latter) and a 429 with
`Retry-After` for a limit. It may stamp the page as it is served: a page
token in the `hexset-page` meta tag, which the page then sends on every API
request in `X-HexSet-Page`, and a Cloudflare Turnstile site key in the
`hexset-turnstile` meta tag, which the page runs, sending the answer to
`POST /api/page`. Where it requires a page token, a request counts as the
page's (a `web` seat) only with one. With no guide registered, the page
carries neither and the server asks for nothing.

### Take back, play from here and flags

`POST /api/takeback` takes the game back to just before the seat's most recent
move, however many moves ago and whoever has moved since; asked again, it goes
back one move of the seat's further each time. A Knight is taken back with the
robber move it opened, and a discard to a seven as a whole: all the seat's
cards of it, and any other seat's cards given between them. It needs no guide
turned on, only a table that offers one. The journal records an undo back to
that step. Where the moves taken back include a roll, a development card drawn
or a steal, the game's chance is drawn afresh from there, written to the
journal as a `reseed` line: new dice and steal streams and the deck's remaining
cards reshuffled, so nothing seen in the moves taken back comes round again.
The table plays on from the position the journal replays to, and a trade round
open at the time goes with the moves.

`POST /api/playfrom` with `{"step": N}` opens a new table under a code of its
own that plays the game on from the position after `N` steps: the same
board, the same seats, each bot the same entry and spec, the seat asking
holding its seat there with the guide on. It works on a finished game too,
and leaves the game asked from as it is. The new game's journal is a copy of
the game's up to that step (`hexset.server._journal.branched`): every line
written before the line's step `N`, moves taken back and their undo lines
included, without the closing line or the trade-round notes of step `N` on,
then the `guide` lines of the steps before `N`, a `branch` line naming the
game (`from`, its id, and `code`), the step and its round (`round`, the
last step's before it), and a `reseed` line at `N`: new dice and steal
streams, and the development cards still in the deck reshuffled. The answer
is `POST /api/games`'s: the new seat's `token` with its view. Every view of
the new table, a reopened one's included, carries `branch`, `{"code",
"step", "round"}`: the game it came from and the position. Like `POST
/api/takeback` it is the web page's, at an unranked table that offers a guide
(403 or 409 otherwise); a step outside `0` to the steps played, or not a
whole number, is a 400; a position the game does not replay to or one the
game is over at is a 409. A step between a Knight and its robber move is a
position like any other: the Knight played, its robber move owed.

`POST /api/flag` with `{"card", "of", "id", "step", "round", "text",
"grade", "played", "best", "flagged"}` flags one of the guide's cards for
review, or takes the flag off: `card` the page's key for the card, `of`
`"note"` or `"hint"` and `id` that note's or hint's, `text` the card as
shown, `played` and `best` wire moves or `null`, and `flagged` true to flag
and false to take the flag off. The guide keeps each flag beside the game's
journal; the view's `guide.flags` lists the cards flagged now.

The game's journal records each use, as a `trainer` line naming the seat and
`guide`, `hint` or `takeback`, and each note, as a `guide` line. A table
reopened from its journal, after a restart, keeps those notes: the guide,
turned on again, starts from every `guide` line not taken back. A finished
game's table with no guide of the seat's on (one reopened from its journal)
shows the guide the journal saved, read-only: `guide` is `on` with `saved`
true, `reading` `"none"`, no `hint`, `takeback` false, every note not taken
back, the `chance` the last graded move left and `stats`, where the seat
turned its guide on during the game.

### On the page

The page shows the guide to a seat its table offers one to as round buttons
in the board's top-left corner: a compass that turns the guide on and off
(remembered by the browser) and Hint. Off, nothing of the guide shows: no
bubbles, no Hint and no win chance. While it is on, the board's undo button
is its take back, the replay bar's strip shows the seat's win chance over the
game, and the pill on its left the percent at the playhead. A hint rings the
moves it shows on the board, and after each move a bubble at the move gives
its grade; a bubble's card says why. Once a move is graded, the pill opens
the review: a graph of the win chance over the game and its key moments,
each a step of the replay. Every hint and note card has a review flag in its
corner (`POST /api/flag`).

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
A secret that matches no seat is a 403, or, with `"probe": true`, a 200 with
`{"token": null}`, for a client that asks on every arrival whether a seat at
the game is its own.
The journal records the id, never the secret or the token. A seat's token
also stops working when the server restarts or drops an idle game from
memory ([server.md](server.md#saved-games)): reclaim gets a fresh one.

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
identity every ranked game counts under. Ranked play counts against no
game allowance, an address's or an API key's: a ticket is refused (429) only
from an address at its unfinished-games limit (New-game limits, above), and
a ranked game it plays counts toward that limit as any game does. `GET
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
more waits for players.

A player here is an `id` at one profile `version`: a client, a server
entrant (below) or the fill bot at the version it plays. Two players play no
more than 100 completed ranked games together: a game counts while it is in
play and once it is played out to a result, and one aborted on its clock or
cut short by a server restart does not. Past that the queue seats neither
with the other, and a ticket with no one left to meet waits; a new version
of either starts afresh.

The server may also enter its own bots (`--ranked-entrant`,
[server.md](server.md)). Each holds a ticket at every game type it plays, one
ranked game of each at a time, is matched like any client, under the id
`bot:<name>` and its profile's version, and is ranked as one, under the
same cap. Its seats are not fill: a client seated with entrants alone has not
met the fill bot.

A ranked table's seats are fixed: `/api/bot`, `/api/open` and `/api/close`
are a 409 there. It runs on a clock: when a move takes longer than 60
seconds, the game is aborted and its journal names the seats it was waiting
on, which the ranker is handed with every seat's points. Only a move restarts
the clock. The trading between two moves (opening, answering, picking or
declining a trade round) counts as one action, and so does a rename: they
share the 60 seconds of the move that follows them. A ranked game a
server restart cut short is closed at start and counts for nothing: its
seats get a 404 and queue again.

## Menu tabs

`GET /api/menu` lists the tabs the server adds to the game page's menu
for the asking client, in order: each a `label` and the `url` the page
opens in a frame of the menu. A server given no `--menu-tab`, or whose
`--menu-tab-for` networks do not hold the client's address, returns
`{"tabs": []}` ([server.md](server.md#menu-tabs)).

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

A board's `pairs` are its rows head to head, as the ranker counted them:
`{"a": <id>, "b": <id>, "games", "a_wins", "b_wins", "expected",
"expected_low", "expected_high"}`, one per pair of rows shown that the
ranker gave, `expected` its chance of `a` beating `b` and the other two the
range it gives around it (each null where it gave none); an empty list for
none.

## Opponent names

`GET /api/models` lists every preset a runtime registered, in registration
order (with the `catanatron` extra installed, `catanatron` among them), then
each `*.onnx` in the models directory by the `name` its metadata declares
([onnx.md](onnx.md#serving-a-model)), one file per name at each game type; the
filename plays no part. `models` is every entry once; `game_types` maps each
`hexset.rules.GAME_TYPES` name to the entries a table of that type seats, each
with the `name`, `version` and `label` a model file declares (nothing else it
declares is served), or the profile a preset was registered with
([guide.md](guide.md)):

```json
{"models": ["mybot", "mynet"],
 "game_types": {"standard": {"mybot": {"version": "1.0", "owner": "octocat",
                                       "description": "A rules bot for every game type."},
                             "mynet": {"name": "mynet", "version": "1.0", "label": "My Net"}},
                "duel-variant": {"mybot": {"version": "1.0", "owner": "octocat",
                                           "description": "A rules bot for every game type."},
                                 "mynet": {"name": "mynet", "version": "1.0d", "label": "My Net"}}}}
```

A preset is offered at every game type, or at those the server's
`--preset-modes` names for it. A model file plays only the game types its
`modes` declare, and the standard game alone when it declares none. An entry
is a model's `name` alone: one name is one bot, which may be one file at some
game types and another at the rest, as `mynet` is above. Two files offering
one name at one game type are neither offered there, and a file whose name
is a registered preset's is not offered at all. `bots` and `/api/bot` take
an entry, and one offered only at another game type is a 400.

`mybot` in the routes above stands for a preset: it exists only on a server started with the runtime that
registers it (`python -m hexset.server.web --runtime mybots`). Without a
runtime the list is `catanatron` and the ONNX files; see the
[guide](guide.md#bundled-opponents). A registered spec's other spellings
(such as `mybot:depth=3`) are available through the Python and evaluation
interfaces, not the server.
