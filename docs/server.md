# Server operation and client interfaces

The browser, HTTP clients, MCP clients, and embedded bots use the same
server action validation. Each participant has a seat token. Bot seats run
in background clients; submitting an action returns the state after that
action, and subsequent bot moves arrive through state updates.

## Configuration

Start the server with `python -m hexset.server.web`. Defaults are
`127.0.0.1:8770`, CPU inference, and an automatically opened browser.

| Option | Purpose |
| --- | --- |
| `--host`, `--port` | Listening address and port |
| `--no-browser` | Suppress opening a browser |
| `--seed` | Board and game random seed |
| `--checkpoint` | Opponent name used to fill the default bot lineup |
| `--device` | Inference provider selection; defaults to `cpu` |
| `--no-trade` | Disable trading for bot seats |
| `--games-dir` | Journal directory; an empty string disables journaling |

`HEXSET_UI_MODELS_DIR` selects the model directory; it defaults to `models/`
under the repository root. `HEXSET_UI_GAMES_DIR` selects the journal directory
when `--games-dir` is omitted; it defaults to `games` relative to the working
directory. An empty environment value disables journaling.

## Docker

The example image installs NumPy, ONNX Runtime, and the pinned Catanatron
dependency from `pyproject.toml`. The image includes HexSet; the
example Compose file overrides source and models with bind mounts and runs
as UID 10001.
Prepare a writable journal directory before starting it:

```sh
cp compose.example.yaml compose.yaml
mkdir -p games
sudo chown 10001 games
docker compose up -d --build
```

Alternatively, set `user:` in `compose.yaml` to a user that owns `games/`.
Open `http://localhost:8770`. The Compose file publishes port 8770 on the
host; edit the port mapping for your deployment.

`compose.yaml` is gitignored. Python source changes require
`docker compose restart`; the HTML file is read on each request. Dependency
or Dockerfile changes require a rebuild. The container uses a read-only root
filesystem, read-only source and model mounts, and a writable journal mount.

## Saved games and visibility

The server writes one JSON Lines journal per game. Journals contain complete
state, including hidden cards and random outcomes. After a restart, journal
replay restores unfinished games and completed games requested for viewing.
Games abandoned through New Game or inactivity eviction remain closed.
Disabling journaling disables recovery. If the journal directory is not
writable, the server logs the error and continues without recording.

Reopened games retain their original seat assignments, names, and client
identities. Seat tokens are held in memory and are replaced through
`/api/reclaim` after a restart. Manual seats do not reopen for unrelated
players. The browser retains a client secret and uses it to reclaim its seat.

Before the first move, unused seats can be closed or reopened. Once play
starts, participation is fixed. Leaving permanently retires the caller's seat;
it does not make the seat available to someone else. Pieces and cards remain
on the board. An unresolved trade round involving that seat must be handled
before leaving.

Completed games are read-only for both participants and spectators. The
browser reveals the full log and hands, and all seated mutation routes
return 409. A participant may still reclaim a finished seat to view the game.

Public spectator routes require only the game code and expose every hand,
development card, and true victory-point count. Players can access these
routes too. Seat tokens protect actions and seat-specific responses; they
do not prevent a player from inspecting the public view.

## HTTP API

Send JSON request bodies. Creating or joining a game returns a `token` and
seat state. Supply that token in the `X-HexSet-Token` header on subsequent
seat requests.

| Method and route | Request or result |
| --- | --- |
| `GET /api/version` | `{"api": <int>}`, this API's contract version; no token required |
| `GET /api/models` | Available opponent names |
| `POST /api/games` | Optional `{"name": "Alice", "bots": ["heximax"]}`; creates a game |
| `POST /api/join` | `{"code": "ABCDEF", "name": "Alice"}`; claims an open seat |
| `POST /api/reclaim` | `{"code": "ABCDEF", "secret": "..."}`; replace a matching seat's token |
| `GET /api/state` | Seat-specific state and current `legal_actions` |
| `GET /api/board` | Static board layout |
| `GET /api/record` | ONNX input record and action-space dimensions |
| `POST /api/action` | `{"action": ...}` with an entry from `legal_actions` |
| `POST /api/undo` | Undo the seat's last eligible build or bank trade; check `can_undo` |
| `POST /api/name` | `{"name": "Alice"}`; rename the seat |
| `POST /api/bot` | `{"seat": 1, "model": "heximax"}`; fill an open seat or replace a bot |
| `POST /api/close` | `{"seat": 1}`; close an unused seat before the first move |
| `POST /api/open` | `{"seat": 1}`; reopen a closed unused seat before the first move |
| `POST /api/leave` | Permanently retire the caller's seat |
| `GET /api/table/<code>` | Public, omniscient game state; no token required |
| `GET /api/table/<code>/board` | Public board layout; no token required |

The models, create, join, and reclaim routes also require no seat token.
State reads support `?after=<version>&wait=<seconds>` to wait for a change. Use fresh
`legal_actions` when submitting a move. Include the state's `version` in
`/api/action`, trade answers, and trade choices to reject stale submissions
with 409. Without a version, a request is validated against the current
position rather than the position the client last read.

Player trading uses three additional routes documented in the
[trade-round reference](bot-api.md#trading).

## Client identity and seat recovery

`/api/games` and `/api/join` accept an optional client object:

```json
{"client": {"id": "<64 hexadecimal SHA-256 characters>", "kind": "api"}}
```

Hash a retained client secret as UTF-8 to obtain `id`. Valid kinds are `web`,
`api`, and `mcp`; omitting the client object defaults to `api` without a
recoverable identity. Unnamed seats default to `human`, `api`, or `mcp`
according to kind.

`POST /api/reclaim` receives the game code and original secret. It finds the
seat with a matching hash and returns a fresh token, invalidating the old
one. Reclaim works on live and completed games but refuses retired seats and
embedded bots. A journal records the identity hash, not the secret or token.

## MCP

The HTTP server serves MCP at `http://127.0.0.1:8770/mcp`. Configure an MCP
client with that Streamable HTTP URL; no separate stdio command is provided.

An `initialize` request returns an `Mcp-Session-Id` header. Send it with all
subsequent requests. A missing session ID returns 400; an unknown ID returns
404 and requires reinitialization. `DELETE /mcp` ends the session.
`GET /mcp` returns 405. Every `tools/call` is answered as an SSE stream
(`text/event-stream`): `: keepalive` comment lines while the call waits,
then one `message` event carrying the JSON-RPC response. `initialize`,
`tools/list` and `ping` are plain JSON responses.

Available tools are `bots`, `new_game`, `join`, `resume_game`, `board`,
`state`, `wait_for_turn`, `act`, `discard`, `undo`, `leave_game`,
`get_table`, `offer_trade`, `answer_trade`, and `choose_trade`. `bots` lists
the opponent names `new_game`'s `opponents` accepts (`GET /api/models`).

`new_game` and `join` require an `identity` string naming the caller (an
LLM would typically pass its model id). The server trims and lowercases
that string, then hashes it as the seat's client identity.
`resume_game(code, identity)` uses the same string to reclaim the seat
after an MCP session or server restart. There is no local session cache.
The identity is whatever string the caller supplies; nothing verifies it.

`act(index)` submits one entry from the latest `state().legal_actions`.
Ending a turn requires `END_TURN`. Every acting tool (`new_game`, `join`,
`resume_game`, `act`, `discard`, `undo`, `offer_trade`, `answer_trade`,
`choose_trade`) replies at the caller's next decision, not the instant
after the action: it blocks until `your_move` is something other than
`wait`, for at most `timeout` seconds (default and cap 600; `0` replies at
once), so a seat that ends its turn gets back the table as it stands when
play returns to it. Two forced moves are played inside that wait rather
than handed back to decide: a `ROLL` that is the only legal action (a
seat holding a Knight still chooses), and a `pass` on any broadcast offer
the seat's hand cannot cover.

`discard(cards)` plays a whole seven's worth of discards in one call:
`cards` is a resource -> count dictionary that must total exactly this
seat's `discard_quota`, no more and no less. The engine still takes one
card at a time (`act(index)` on a single `DISCARD` entry still works, one
card per call); `discard` is the same submissions made for the caller,
each matched against the freshest `legal_actions` after the previous one
landed.

`board` replies as text, not JSON: an incidence encoding with one line per
hex (id, resource, pips, vertex ids), one per vertex (id, pips, resources,
port, neighboring vertex ids) and the edge ids as `id:v0-v1`, with render
geometry and the constant name tables left out. In every state reply
`legal_actions`' per-type groups and `summary.spots`/`summary.robber` are
tables, `(keys):row|row` with comma-separated cells (`-` null, `;`
between list items; `robber` hits as `seat:Ns+Nc`, options as
`index:victim`), and JSON is written without spaces. The MCP layer
annotates a copy of the table's layout; the HTTP `GET /api/board` the
browser draws from is untouched. `wait_for_turn` does only the waiting, and is needed
only after a reply whose `timeout` ran out. `state`, `get_table`, `board`,
`bots` and `leave_game` reply immediately. In MCP replies `legal_actions` is grouped
by action type; each entry carries the flat `index` to act on and a named
operand (`edge`, `vertex`, `hex` and `victim`, or the resource names for
bank trades, discards, Monopoly and Year of Plenty), and `legal_count` is
the flat total. Board occupancy comes as `buildings` (vertex, seat, kind)
and `roads` (edge ids, one list per seat) rather than the HTTP API's dense
`vertex_owner`, `vertex_building` and `edge_owner` arrays. MCP replies also
drop the HTTP view's `version`, `claimed_seats`, `waiting_for`, `trade_wait`
and per-player `last_roll`; fold `seats` into `players` as each entry's
`kind`; and send `hand`, `known`, `dev_cards` and `bank` sparse, a missing
name meaning zero.

`get_table` returns the same reply as `state`. Trading tools use
resource-name dictionaries and indices into that reply's `pending` and
`trade_round.responses`, translating them to signed HTTP bundles. Pass the chosen
`legal_actions` entry, with its group key as `type`, as `act`'s `expect` to
refuse if that index now names a different action. The trade tools take no
staleness guard: the server already rejects an answer or a choice that does
not match the exact open offer. The HTTP API's whole-table `version` is not
exposed as an MCP argument, since it changes whenever any seat does
anything, including other seats answering the same trade round.

Every state-returning tool also carries `summary`, derived server-side from
the view and the fixed board: `afford` (per build, whether the hand covers
it, what it is short, and whether `legal_actions` offers it now), `race`
(points and distance to the win, the leading opponent by public points, and
for each award the caller's count, the holder's, and the count that would
take it), and, only when such a move is legal, `spots` (every settlement or
city placement with the vertex's pips, resources and port, best first) and
`robber` (each hex the robber may move to, its pips, whose buildings it
hits, and an `act` index per victim). Every entry names the `legal_actions`
index it corresponds to. Whichever of `legal_actions`'
SETUP_SETTLEMENT/BUILD_SETTLEMENT/BUILD_CITY or MOVE_ROBBER groups `spots`
or `robber` covers is then dropped from `legal_actions` -- the same
entries, indexed the same way, so keeping both said nothing twice;
`legal_count` still counts them. The state also carries `winning_points`,
the rule the game is played to.

The transcript `log` is sent incrementally. Each MCP session remembers how
many lines it has been sent, and every state-returning reply carries only
the lines added since that session's previous reply plus the one trailing
line that may have been rewritten in place, with `log_from` naming the
index the slice starts at and `log_total` the whole length. A new seat, a
reclaimed seat and the final read of a finished game get the whole
transcript. `full_log: true` forces that on any call, for a client that lost
a reply; `log_after: <n>` overrides the cursor with an explicit line count.

Every state-returning tool answers with `your_move`: `act`, `discard`,
`answer_trade` or `choose_trade` names the tool the table wants from the
caller now, `wait` means none does (only after a `timeout` ran out, or
while seats are still open) and `waiting_on` lists the seats it is waiting
for, and `game_over` is the end. It is derived from `legal_actions`,
`pending`, `trade_round`, `discard_quota` and `to_move`, which remain
available, and from `waiting_for` and `trade_wait`, which do not.

`wait_for_turn(timeout=...)` waits until `your_move` is anything but
`wait`: legal actions, a pending offer, a fully answered round, or game
completion gives the caller something to handle. The response uses Server-Sent Events, with keepalives during the
wait. Use this tool to wait for other seats without repeatedly reading state.

## External ONNX clients

A compatible policy model can claim an open seat from a separate process:

```sh
python -m hexset.clients.botclient \
  --url http://127.0.0.1:8770 \
  --game ABCDEF \
  --model models/policy.onnx
```

Replace the game code and model path with your own. The client defaults to
an identity derived from the model filename stem; `--client-secret` overrides
that value. This client reads
`/api/record` and submits actions through the seat API. It does not participate
in player trading and rejects models with `search=mcts`; search models must
run as embedded opponents. See the [ONNX model contract](bot-api.md).

The `heximax` opponent uses the adaptive policy. Endpoint pinning is available
through the [Python and evaluation interfaces](heximax.md), without adding
separate bot choices to the server picker.
