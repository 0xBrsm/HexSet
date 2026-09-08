# The bot API

This is the complete interface a `.onnx` file must satisfy to plug in as an
opponent. It is the only thing `src/hexset/clients/onnxbot.py` reads — the file
does not need access to this repo's source, only to what is written here plus
the public [ONNX](https://onnx.ai/) format itself. `hexset.heximax` (the
default handcrafted opponent) and `hexset.bots`' `search2` implement the same
interface a different way and are the reference for what "correct" means when
in doubt.

Two independent parts make up the contract:

1. **Self-description** — `metadata_props` on the ONNX model, read once at
   load time.
2. **The graph itself** — named inputs in, named outputs out. Which shape
   this takes depends on the `contract` key below.

## 1. Self-description (`metadata_props`)

| key | meaning | default |
| --- | --- | --- |
| `players` | table size the graph was traced for | required |
| `num_hexes` / `num_vertices` / `num_edges` | board-shape fingerprint; a mismatched board fails the load rather than running on meaningless input | required |
| `contract` | which graph shape below applies: `6`, the record shape | refused if absent — see below |
| `max_trades` | `0` to switch trading off for this checkpoint | trading on |
| `search` | `mcts` to search over the model's own priors; anything else plays one forward pass | none |
| `simulations` | descents per decision, when `search=mcts` (clamped to 4096) | 128 |
| `wave` | leaves batched per expansion, when `search=mcts` (clamped to 256) | 16 |
| `iteration` | informational only; not read for behaviour | 0 |

Unreadable or missing optional keys fall back to their default rather than
failing the load — a typo'd hint costs the hint, not the whole opponent. See
`src/hexset/server/modelmeta.py` for the exact clamping.

Inference device (`cpu`/GPU) is deliberately **not** a metadata key — it is a
fact about the machine serving the game, not the checkpoint.

**The `contract` number is assigned by the exporter, not by this repo.**
`hexset.export_onnx._CONTRACT_VERSION` is the one definition; `hexset.server`
reads it and never writes it. A graph declares the fields it wants and is fed
exactly those, so a graph that predates a field this record has gained still
loads and plays. An unknown number is refused at load with the number named,
rather than failing later on its first move with a missing-input error.

**Only contract 6 is served.** 2, 3 and 4 are the offer protocol's
contracts — 3 added four live-offer fields, 4 the two public-knowledge ledger
fields, and all three declare a `pair_mask` input and a `pair_index` output
for the one-for-one give/want heads. Trading is now one engine event with no
actions at all (see §4), so those graphs describe a game this engine does not
play: there is no honest way to feed them, and they are refused by name.
Contract 5 is refused too now, for two independent reasons that happened to
land together: it declared a `valuations` field for the one-event mechanic's
public valuation vector, and that public layer is gone outright
(`agents/reference/trading-final.md`, item 1) rather than replaced; and the
knight two-step fix (a knight is played, then the robber moves through the
same phase a seven enters, rather than one action carrying both) dropped
`PLAY_KNIGHT`'s operands, shrinking the flat `ActionSpace` a contract-5
graph's `action_mask`/`prior` were traced against. Either change alone would
have forced the bump. Contract 1 — the original shape, where the engine
encoded the position into feature tensors and the graph was a bare
policy/value head masked in Python — went the same way on 2026-09-02.

## 2. The graph — the record contract (`6`)

The engine builds a **record**: the position stated in the rules' own terms,
already filtered to what the perspective seat may legally know. The graph
owns everything downstream of that — encoding, masking, normalising,
argmax, un-rotating back to board-seat order. Built by
`hexset.onnx_record.record_from_game`, whose module docstring carries the
field-by-field derivation.

Leading batch axis `B` on every tensor.

**Inputs:**

| Name | Shape | dtype |
| --- | --- | --- |
| `terrain` | `(B, num_hexes)` | int64 |
| `token` | `(B, num_hexes)` | int64 |
| `port_code` | `(B, num_vertices)` | int64 |
| `robber` | `(B,)` | int64 |
| `vertex_owner` | `(B, num_vertices)` | int64 |
| `vertex_building` | `(B, num_vertices)` | int64 |
| `edge_owner` | `(B, num_edges)` | int64 |
| `bank` | `(B, NUM_RESOURCES)` | int64 |
| `knights_played` | `(B, players)` | int64 |
| `award_points` | `(B, players)` | int64 |
| `longest_road_holder` | `(B,)` | int64 |
| `largest_army_holder` | `(B,)` | int64 |
| `phase` | `(B,)` | int64 |
| `free_roads` | `(B,)` | int64 |
| `deck_size` | `(B,)` | int64 |
| `turns` | `(B,)` | int64 |
| `perspective` | `(B,)` | int64 |
| `own_hand` | `(B, NUM_RESOURCES)` | int64 |
| `hand_totals` | `(B, players)` | int64 |
| `own_dev` | `(B, NUM_DEV_CARDS)` | int64 |
| `dev_totals` | `(B, players)` | int64 |
| `ledger_known` | `(B, players, NUM_RESOURCES)` | int64 |
| `ledger_unknown` | `(B, players)` | int64 |
| `action_mask` | `(B, space.size)` | bool |

**Outputs:**

| Name | Shape | Note |
| --- | --- | --- |
| `action_index` | `(B,)` int64 | argmax over the masked distribution |
| `prior` | `(B, space.size)` | normalised over legal actions, zero elsewhere |
| `value` | `(B, players)` | board-seat order, already un-rotated |

`NetworkBot` reads `action_index`; searches read `prior`/`value`. One graph
serves both.

**The engine drift this section used to list is gone.** This server no longer
carries its own copy of the engine: it depends on the `hexset` package (now
one distribution together with the gym, see the CHANGELOG's "one
distribution" entry), so what it plays is exactly what dev-HexN plays.

**The one mask difference that used to remain is gone too.** The
`action_mask` served here was built over an *honest* trade sample, because
the engine's own `legal_actions` filtered the offer sample by opponents'
true hands and telling a human that would give away a specific opponent's
hand. There is no offer sample: trading is not an action, no remaining
action's legality depends on another seat's hand, and there is now one list,
`hexset.actions.legal_actions`, for every seat.

## 3. Trading

**Status, 2026-09-06: the trade round is the served table's protocol.**
The engine's automatic clearing house (`hexset.trading.trade_event`) stays
the training protocol -- the arena, the bench, the gym and every self-play
run play under it -- but a served table (`hexset.server`) switches it off
(`Game.max_trades = 0`) and runs the **trade round** instead
(`hexset.trading`, "The trade round"; `agents/reference/trading-final.md`).

One round: the current player broadcasts one offer to every other seat;
each seat answers once -- accept, counter with a bundle it would take
instead, or pass; the actor executes one answer or declines them all. A
trade moves 1-3 cards a side on disjoint resources (`MAX_TRADE_CARDS`),
and a bot side's own gate must clear `TRADE_FLOOR` at execution, re-asked
fresh. Bundles on the wire are five signed counts in `RESOURCE_NAMES`
order, always signed towards the offer's actor: positive is what the actor
receives.

**Bots.** A bot's offer is the candidate maximising its own gain among
those it estimates the counterparty accepts (`default_offer`: own gain via
`gains_many`, the counterparty's via `estimate_many` -- heximax and search2
evaluate the exchange from the other seat's frame; a gate without it uses
its own gain as the estimate). A bot answers an offer by accepting when its
own gain clears the floor, else countering with its best coverable bundle
it estimates the actor accepts, else passing (`default_respond`); as actor
it picks the answer with the highest own gain above the floor
(`default_pick`). A bot may implement `offer`, `respond`, `pick` and
`estimate_many` itself; `hexset.bots.search2.Bot` lists the signatures.

**Manual seats (a person at the page, an LLM over `POST /mcp`)**
are `PendingGate`s: nothing is ever agreed on their behalf. A bot's
broadcast is recorded against each manual seat in `GET /api/state`'s
`pending` (`{"actor": <seat>, "bundle": [...]}`), and **the bot's turn
holds until every manual seat has answered** -- `trade_wait` lists the
seats being waited on and `to_move` reads `null` meanwhile -- so a person
gets to accept or counter before the bot picks. Three routes, all
seat-token gated:

- **`POST /api/games/<code>/trade/round`** -- `{"give": [5 ints], "want":
  [5 ints]}`, unsigned counts. The current player's broadcast; 409 off its
  turn or outside MAIN, 400 for a bundle it cannot cover. Bots answer
  synchronously. The view's `trade_round` block carries the offer, the accepts
  and counters so far (`responses`, each `{"seat", "kind", "bundle"}`), and
  the manual seats still to answer (`awaiting`).
- **`POST .../trade/round/answer`** -- `{"actor", "received", "kind":
  "accept"|"counter"|"pass", "bundle"?}`: the exact offer from `pending`
  echoed back; a counter's `bundle` is signed towards the actor like
  `received`. 409 once that offer is no longer open. With a bot actor, the
  round resolves the moment the last manual seat answers.
- **`POST .../trade/round/choose`** -- `{"seat", "bundle"}` executes that
  recorded answer exactly; `{"decline": true}` closes the round. Only the
  actor may call it; the round also closes when the turn ends.

`POST /api/action`, `.../trade/round/answer` and `.../trade/round/choose` all
take an optional `"version"`, compared against the table's current one — a
mismatch is a 409 rather than the request applying against a state that has
since moved.

An executed trade is logged (`log`, `trades`) and journalled like any other
move, so it survives a restart. A checkpoint served externally
(`hexset.clients.botclient.RecordBrain`) is never seated as a gate and does
not trade.

## 4. Identity (manual seats only)

`POST /api/games` and `POST /api/join` take an optional `client: {"id":
<64-hex sha256>, "kind": "web"|"api"|"mcp"}` — a hash of a secret only the
caller holds. No `client` at all defaults to kind `"api"`; an unknown kind
or a malformed `id` is a 400. `POST /api/reclaim {"code", "secret"}` mints a
fresh token for the seat whose `client.id` equals `sha256(secret)`, once the
original token is gone (a server restart, most often) — MCP's `resume_game`
(`hexset.server.mcptools`) uses this to get its seat back by the same `model`
string `new_game`/`join` were called with. None of this applies to a `.onnx`
bot: a checkpoint is never seated as a manual gate and has no client identity.

## What is never part of this contract

`onnxbot.py`'s job stops at reading these names and shapes. It never imports
or inspects anything else about how a checkpoint was produced, and a
checkpoint's author never needs this repo's source to write one — only this
document, the ONNX spec, and the topology fingerprint of the board they are
targeting.
