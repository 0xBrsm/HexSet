# MCP interface

The server speaks MCP over Streamable HTTP at `http://127.0.0.1:8770/mcp`
(the protocol versions in `hexset.server.web.MCP_PROTOCOL_VERSIONS`; a client
asking for another gets `DEFAULT_MCP_PROTOCOL_VERSION`). There is no stdio
transport. The tools act through [the JSON API](api.md); the process is
[server.md](server.md).

## Session

| Request | Answer |
| --- | --- |
| `POST /mcp` `initialize` | JSON result with an `Mcp-Session-Id` header; send it on every later request |
| `POST /mcp` without `Mcp-Session-Id` | 400 |
| `POST /mcp` with an unknown `Mcp-Session-Id` | 404; call `initialize` again |
| `POST /mcp` notification (no `id`) | 202, no body |
| `POST /mcp` `tools/list`, `ping` | JSON result |
| `POST /mcp` `tools/call` | `text/event-stream` |
| `POST /mcp` any other method | JSON-RPC error `-32601` |
| `DELETE /mcp` | 200, ends the session; 404 for an unknown one |
| `GET /mcp` | 405 |

A request whose `Origin` names a host other than `127.0.0.1`, `localhost`,
`::1` or the server's `--host` is a 403; a request without `Origin` is
allowed. A session holds one seat, in memory only.

A `tools/call` stream sends a `: keepalive` comment every 15 seconds while
the call waits, then one `message` event with the JSON-RPC response and
closes. The result is one text content item: compact JSON, or plain text for
`board`. A call the tools refuse is a result with `isError: true`.

## Tools

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `bots` | none | `{"models": [...]}`, the names `opponents` accepts |
| `new_game` | `identity`, `opponents?`, `name?` | Deal a game and take a random seat |
| `join` | `code`, `identity`, `name?` | Take a random open seat at a game |
| `resume_game` | `code`, `identity` | Reclaim the seat `identity` holds, from a new session |
| `state` | none | The seat's state reply |
| `board` | none | The board, as text |
| `wait_for_turn` | none | Wait until the table wants something of this seat |
| `act` | `index`, `expect?` | Play one `legal_actions` entry |
| `discard` | `cards` | Play a whole discard in one call |
| `offer_trade` | `give`, `want`, `give_any?`, `want_any?`, `to?` | Broadcast an offer |
| `answer_trade` | `index`, `kind`, `give?`, `receive?` | Accept, counter or pass `pending[index]` |
| `choose_trade` | `index?`, `decline?` | Execute `trade_round.responses[index]`, or decline |
| `leave_game` | none | Retire the seat for the rest of the game |

Shared arguments:

| Argument | Taken by | Meaning |
| --- | --- | --- |
| `timeout` | the acting tools and `wait_for_turn` | Seconds to wait for this seat's next decision; default and cap 600, `0` replies at once |
| `log_after` | the acting tools, `wait_for_turn`, `state` | Override the transcript cursor: the line count held |
| `full_log` | the acting tools, `wait_for_turn`, `state` | Send the whole transcript |

The acting tools are `new_game`, `join`, `resume_game`, `act`, `discard`,
`offer_trade`, `answer_trade` and `choose_trade`. Resource counts are objects
from resource name (`Wood`, `Brick`, `Sheep`, `Wheat`, `Ore`) to a count. An
argument a tool does not take refuses the call.

`opponents` names one bot per seat, from `bots`. Omitted or empty, the game
has no bots; empty seats stay open for others to `join` by `code`.

`identity` names the caller (a model id, for example). The tool layer trims
and lowercases it, uses the result as the client secret, and sends its
SHA-256 as the seat's client id with kind `mcp`; `resume_game(code,
identity)` reclaims the seat with the same string. Nothing verifies it.

There is no `undo` tool.

## Turn handling

Every state reply carries `your_move`, the server's own field from
[`GET /api/state`](api.md#state), which every API client reads alike:

| Value | Meaning |
| --- | --- |
| `discard` | This seat owes a discard: call `discard` |
| `answer_trade` | An offer awaits this seat's answer |
| `choose_trade` | This seat's own round is fully answered |
| `act` | This seat is to move and has `legal_actions` |
| `wait` | Nothing is wanted; `waiting_on` lists the seats the table waits for |
| `game_over` | The game has ended |

`game_over` is checked first, then the others in the order above. The server
derives them from `legal_actions`, `pending`, `trade_round` and
`discard_quota`, which stay in the reply, and from `to_move`, `waiting_for`
and `trade_wait`, which do not.

An acting tool replies at this seat's next decision, not the instant after
its action: it blocks until `your_move` is not `wait`, for at most `timeout`
seconds. A seat that ends its turn gets the table back when play returns to
it. An acting reply therefore shows `wait` only when `timeout` ran out first,
for example while open seats hold up setup. `state`, `board`, `bots` and
`leave_game` reply at once; `wait_for_turn` only waits.

Inside that wait the tool layer plays three forced moves itself:

- a `ROLL` that is the only legal action (a seat holding a Knight still
  chooses);
- a pass on an offer while the seat's hand is empty;
- the seat's own fully answered round when nothing is left to choose (see
  `to` below).

An offer the hand cannot cover still reaches the seat, with
`can_accept: false`; it can still be countered.

## Acting

`act(index)` plays entry `index` of the flat `legal_actions` list behind the
latest reply; the same indices appear in `summary.spots`, `summary.robber`
and `summary.roads`. `END_TURN` is an action like any other. `expect`, the
chosen entry plus its group key as `type` (`{"type": "BUILD_ROAD", "edge":
17}`), refuses the call if `index` now names a different action.

`discard(cards)` is only for the `DISCARD` phase. `cards` must total the
seat's `discard_quota` exactly and be held. The tool then submits one
`DISCARD` per card, each against the freshest `legal_actions`; `act` on a
single `DISCARD` entry works too.

## Trading

Every reply carries `can_offer`: true on the seat's own turn in `MAIN`, with
at least one legal action and no round of its own open. `offer_trade` still
refuses an offer the hand cannot cover.
`give_any`/`want_any` add that many cards of the answerer's choosing to one
side, which makes the offer one that is only countered
([api.md](api.md#trade-routes)).

The trade tools take resource-name objects and indices into the latest
`pending` and `trade_round.responses`, and translate them to the signed
bundles of the HTTP routes. Replies show trades from this seat's side:

| Field | Shape |
| --- | --- |
| `pending[]` | `actor`, `you_give`, `you_receive`, `you_give_any`/`you_receive_any` for an open offer, `can_accept` |
| `trade_round` | `you_give`, `you_receive`, `you_give_any`/`you_receive_any` for an open offer, `responses[]` (`seat`, `kind`, `you_would_give`, `you_would_receive`), `awaiting` |
| `trades[]` | `a`, `b`, `a_gave`, `a_got` |

The trade tools send no `version`: the server already refuses an answer or a
choice that does not match the exact open offer or response.

`offer_trade`'s `to` lists the seats the offerer would trade with, best
first. Once every seat has answered and no counter came back:

- with `to`, the best-ranked listed accept executes; with no listed accept
  the round closes;
- without `to`, a single accept executes, no accept closes the round, and two
  or more come back as `your_move: choose_trade`.

Any counter returns the round as `choose_trade`.

## State replies

`board` is text: a `hexes: id resource pips vertices` block, one line per hex;
a `vertices: id pips resources port neighbors` block, one line per vertex, with
pips summed and resources de-duplicated over the touching hexes; and a
`piece_supply` line. There is no edge list; `summary.roads` names each legal
road's far vertex.

In a state reply, `legal_actions` is grouped by action type, and each group
is a table. Each entry carries the flat `index` and a named operand: `edge`,
`vertex`, `hex` and `victim` (`null` for nobody), `give`/`want` for
`BANK_TRADE`, `resource` for `PLAY_MONOPOLY` and `DISCARD`, `resources` for
`PLAY_YEAR_OF_PLENTY`. When `summary.spots` or `summary.robber` is present,
the groups it covers (`SETUP_SETTLEMENT`, `BUILD_SETTLEMENT`, `BUILD_CITY`;
`MOVE_ROBBER`) are dropped from `legal_actions`.

A table is `(keys):row|row`: cells comma-separated, `-` for null, `1`/`0` for
booleans, `;` between list items, `k:v+k:v` for an object. A `spots` table of
one action type is prefixed `TYPE:`. `robber` hits read `seat:Ns+Nc`
(settlements and cities) and options `index:victim`. The reply's JSON has no
spaces.

Relative to the HTTP view, an MCP reply:

- drops `version`, `claimed_seats`, `waiting_for`, `trade_wait`, `to_move`,
  `awaiting_confirm` and `can_undo`;
- drops `locked` and `trades` while empty, `winner` while null, `started`
  while true, and `discard_quota` while every entry is zero;
- drops each player's `last_roll`, `longest_road` and `largest_army`
  (`summary.race` names the award holders);
- folds `seats` into `players` as each entry's `kind`;
- sends `hand`, `known`, `dev_cards` and `bank` sparse, a missing name
  meaning zero;
- replaces `vertex_owner`, `vertex_building` and `edge_owner` with
  `buildings` (`vertex`, `seat`, `kind`) and `roads` (edge ids, one list per
  seat).

## Summary

Every state reply for a seated reader carries `summary`, computed from the
view and the board:

| Key | Contents |
| --- | --- |
| `afford` | Per build (`road`, `settlement`, `city`, `dev_card`): `ok` (the hand covers it), `missing`, `legal` (offered now; omitted when `legal_actions` is empty), and, when affordable but not offered, `why`: `phase`, `pieces`, `deck` or `spot` |
| `race` | `points`, `to_win`, `top_opponent` (`seat`, `points`, by public points), and per award `yours`, `held`, `holder`, `holder_has`, `need` |
| `spots` | Each legal settlement or city: `index`, `vertex`, `pips`, `resources`, `port`, `port_matches`; most pips first |
| `robber` | Each legal robber hex: `hex`, `resource`, `pips`, `hits`, `options`; most pips first |
| `roads` | Each legal road: `index`, `edge`, `to` (the end outside this seat's network), its `pips`, `resources`, `port`, `settle` (a settlement fits there), `link` (both ends already this seat's), `then`/`then_pips` (the best settleable vertex one road on); settleable ends first, then most pips |

`spots`, `robber` and `roads` appear only when such a move is legal.

## Transcript and usage

The transcript `log` is sent incrementally. Each session counts the lines it
has been sent; a reply carries the lines added since, plus the last line
already sent, which can be rewritten in place. `log_from` is the index the
slice starts at and `log_total` the full length. A new or reclaimed seat gets
the whole transcript. `log_after: n` replaces the session's count for one
call, and `full_log: true` sends everything.

At the end of the game every earlier line is re-rendered unredacted, so each
steal names its card. The final reply still carries only the usual slice;
`full_log: true` fetches the unredacted whole.

The `game_over` reply also carries `usage`: `calls`, the replies this
session was sent, and `bytes`, their total size as compact JSON (text for
`board`), the final reply included.
