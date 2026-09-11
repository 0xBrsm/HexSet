# ONNX model contract

HexSet loads ONNX opponents through `hexset.clients.onnxbot`. A model must
provide compatible metadata, record inputs, and named outputs. Only contract
`6` is supported. An absent or different contract number is rejected;
changing the number alone does not make an older graph compatible.

The contract is defined by `CONTRACT_VERSION`, `RECORD_FIELDS`, and
`record_shapes` in [onnx_record.py](../src/hexset/onnx_record.py), together with
the action layout in [actions.py](../src/hexset/actions.py). Heuristic bots
implement Python interfaces; they do not consume ONNX tensors.

## Model metadata

ONNX `metadata_props` values are strings.

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `contract` | Record and action-space version | Required: `"6"` |
| `players` | Number of seats used by the graph | Required integer; embedded policies check the game's player count when choosing |
| `num_hexes` | Board hex count | Required integer |
| `num_vertices` | Board vertex count | Required integer |
| `num_edges` | Board edge count | Required integer |
| `max_trades` | `0` disables this bot's trading | Absent or empty: no override; otherwise must parse as an integer |
| `search` | `mcts` enables search using the graph's priors and values | Any other value: direct policy inference |
| `simulations` | Search descents per decision | 128; maximum 4096 |
| `wave` | Leaves evaluated per search wave | 16; maximum 256 |
| `trade_floor` | This checkpoint's measured clearing floor, in win probability | 0.0; clamped to `[0.0, 1.0]` |
| `gate_rows` | Candidates this checkpoint's trade gate scores per event | 32; maximum 512 |
| `iteration` | Training iteration, retained as model information | 0; must parse as an integer when present |

The embedded loader checks the three board counts against the target
topology. These counts do not verify connectivity or index ordering: the
graph must also use the same board indexing as the engine.

Search settings are read only for `search=mcts`. Missing, malformed, zero,
or negative `simulations` and `wave` values use their defaults; values above
the maximum are capped. `trade_floor` and `gate_rows` fall back the same
way. This fallback does not apply to other integer metadata, where
malformed values fail loading.

`trade_floor` and `gate_rows` describe the exported value head, not the
Python adapter: a floor measured against one checkpoint says nothing about
another's, so each file carries its own. A checkpoint whose resolution has
not been measured declares no floor, and any strictly positive gain clears.

Inference device selection belongs to the host's `--device` option. It is
not read from model metadata.

## Record inputs

The record describes the position from one seat's perspective. Its own hand
and development cards are exact. Opponents contribute card totals and the
public resource ledger, rather than hidden card identities. Seat arrays
remain in board-seat order; `perspective` identifies the perspective seat.
The graph handles rotation, feature encoding, masking, and normalization.

Every tensor has a leading batch axis `B`. `NUM_RESOURCES` and
`NUM_DEV_CARDS` are both 5. A graph may declare a subset of record fields;
the loader feeds only the names it requests. Unknown input names fail when
an inference request is built.

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

Resource indices are `WOOD=0`, `BRICK=1`, `SHEEP=2`, `WHEAT=3`, `ORE=4`.
Development-card indices are `KNIGHT=0`, `VICTORY_POINT=1`,
`ROAD_BUILDING=2`, `YEAR_OF_PLENTY=3`, `MONOPOLY=4`.

Terrain indices are `FOREST=0`, `HILLS=1`, `PASTURE=2`, `FIELDS=3`,
`MOUNTAINS=4`, `DESERT=5`, `SEA=6`, `GOLD=7`. Port codes are `-1` for no
port, `0` for a generic port, or `1 + resource_index` for a resource port.
Owners and award holders use `-1` when unassigned. Building values are
`0` for empty, `1` for a settlement, and `2` for a city.

Phase values are `SETUP_SETTLEMENT=0`, `SETUP_ROAD=1`, `ROLL=2`,
`DISCARD=3`, `ROBBER=4`, `MAIN=5`, `GAME_OVER=6`. `own_dev` and
`dev_totals` include newly purchased cards; the action mask determines
which cards may currently be played. `award_points` contains points from
Longest Road and Largest Army, not total victory points.

## Graph outputs

| Name | Shape and dtype | Meaning |
| --- | --- | --- |
| `action_index` | `(B,)`, int64 | Selected legal action's flat index |
| `prior` | `(B, space.size)`, floating point | Probability distribution over legal actions, zero on illegal actions |
| `value` | `(B, players)`, floating point | Per-seat win probabilities in board-seat order |

Direct action selection reads `action_index`. Embedded network trading
also reads `value`. MCTS reads `prior` and `value`, so a model intended for
all embedded modes should export all three outputs. The graph selects a
legal action; the server validates the resulting action.

MCTS terminal evaluation uses a one-hot winner vector in board-seat order,
or all zeros when the game ends without a winner. The value head must use
win probabilities on the same scale.

Use a dynamic batch axis for search and trade continuation evaluation, which
call the policy on batches of positions. Direct policy inference uses a batch
of one. Value-only reads can fall back to individual calls for fixed-batch-one
graphs, but batched action and prior reads must match the declared shape.

### Action indices

`hexset.actions.build_space(V, E, H, P)` builds the flat space from vertex,
edge, hex, and player counts. Blocks appear in this order; each block's
offset is the sum of all preceding sizes.

| Action | Block size | Index within block |
| --- | --- | --- |
| `ROLL` | 1 | 0 |
| `END_TURN` | 1 | 0 |
| `BUY_DEV_CARD` | 1 | 0 |
| `PLAY_ROAD_BUILDING` | 1 | 0 |
| `SETUP_SETTLEMENT` | V | Vertex index |
| `SETUP_ROAD` | E | Edge index |
| `BUILD_ROAD` | E | Edge index |
| `BUILD_SETTLEMENT` | V | Vertex index |
| `BUILD_CITY` | V | Vertex index |
| `MOVE_ROBBER` | H × (P + 1) | Hex index × (P + 1) + victim; victim P means no victim |
| `PLAY_KNIGHT` | 1 | 0; moving the robber is a separate decision |
| `PLAY_MONOPOLY` | 5 | Resource index |
| `PLAY_YEAR_OF_PLENTY` | 15 | Index into sorted resource pairs with repetition: (0,0), (0,1), …, (4,4) |
| `BANK_TRADE` | 25 | Given resource × 5 + received resource |
| `DISCARD` | 5 | Resource index; discards one card |

Thus `space.size = 3V + 2E + H(P + 1) + 55`. Use `action_mask` to select
legal entries. Player trading has no entry in this space.

## Trading

Server games set `Game.trade_mode = "external"` and drive offer–response
rounds across requests. Arena simulations use the same round protocol,
driven synchronously by the engine with one broadcast per turn by default.
`Game.max_trades` caps broadcasts in `"round"` mode and completed exchanges
in `"auto"` mode; `0` disables the engine's driver and `-1` removes its cap.
External callers manage their own limits. To reproduce the old exhaustive
clearing behavior, set `trade_mode="auto"` and `max_trades=-1`.
The Gym wrappers have additional limitations described in the
[README](../README.md#training-environments).

A server round has three steps:

1. The current player broadcasts an offer to the other active seats.
2. Each seat accepts, counters with another bundle, or passes.
3. The current player executes one response or declines the round.

An exchange moves 1–3 cards per side, on disjoint resources. Both hands must
cover the exchange. A bot's gate is checked again at execution and must
return a gain strictly above that gate's `trade_floor`. There is no table-wide
default: a gate that returns positive gains must declare a nonnegative floor.
A manual seat's explicit submission supplies its consent.

The Python bot protocol supports `gains_many(view, received, counterparties)`
for valuing exchanges. Bots may also implement `offer`, `respond`, `pick`,
and `estimate_many`; defaults in `hexset.trading` evaluate offers and
responses using the bot's own gain and an estimate of the other seat's gain.
Boolean `accepts` or `accepts_many` gates are mapped to gains of +1 or −1;
a bot with no supported gate declines trades.

`hexset.clients.netbot.NetworkBot` supplies the shared network trade gate,
including for ONNX policies and embedded MCTS. Its `trade_floor` and
`gate_rows` come from the checkpoint it was loaded from — see
[model metadata](#model-metadata) — defaulting to `0.0` and `32` for a file
that declares neither.
For each candidate it samples a belief world consistent with the seat's
information, then compares continuations with and without the exchange.
Both hands and the ledger change in the exchanged world. The current mover's
policy plays up to eight actions in each continuation, stopping at an
end-turn choice or a finished game. The value head scores the resulting
positions from the evaluating seat's perspective. Its own row supplies the
gain; the counterparty's row supplies the estimate of the other side's gain.

Every candidate with at most two cards per side is evaluated. Larger bundles
fill any remaining places up to the gate's `gate_rows`; unscored candidates
are declined. So `gate_rows` is not a hard cap when there are more small
bundles.
These are Python adapter behaviors, not additional graph inputs or outputs.
The external `RecordBrain` client has no trade gate and rejects search models.

A different inference runtime can implement the `Policy` protocol and use
the same bot, trade gate, and search. See [runtime integration](training.md#model-runtimes).
The ONNX loader also accepts a Python `threads` argument to cap both ONNX
Runtime thread pools; it is not model metadata.

### HTTP trade routes

All routes require `X-HexSet-Token`. Bundles are five signed integers in
resource order, from the offer actor's perspective: positive counts are
received by the actor, negative counts are given away. For example,
`[-1, 0, 0, 1, 0]` means the actor gives one wood and receives one wheat.

| Route | JSON body |
| --- | --- |
| `POST /api/games/<code>/trade/round` | `{"give": [1,0,0,0,0], "want": [0,0,0,1,0]}`; unsigned counts |
| `POST /api/games/<code>/trade/round/answer` | `{"actor": 0, "received": [-1,0,0,1,0], "kind": "accept"}`; kind is `accept`, `counter`, or `pass`; counters also supply a signed `bundle` |
| `POST /api/games/<code>/trade/round/choose` | `{"seat": 1, "bundle": [-1,0,0,1,0]}` to execute that response, or `{"decline": true}` |

Only the current player in MAIN may open a round. `pending` in a manual
seat's state contains offers requiring its answer, as `actor` and `bundle`.
Echo that exact offer as `actor` and `received` when answering; stale offers
are rejected. Counter bundles remain signed toward the original actor.

The actor's `trade_round` state contains the offer, `responses` (each with
`seat`, `kind`, and `bundle`), and `awaiting` seats. The actor chooses an
exact recorded accept or counter; a pass has `kind="pass"` and a null bundle.
A bot actor waits for every manual seat to answer:
`trade_wait` identifies those seats and `to_move` is null while it waits.
A cardless manual seat passes automatically. Bots broadcast at most once per
turn and resolve their round after the final required answer. Manual players
may open a replacement round on their own turn. Executed trades are
logged and journaled for replay.

`/api/action`, `/trade/round/answer`, and `/trade/round/choose` accept an
optional `version` field (trade routes use the full game-specific prefix
above). A mismatch with the table's current version returns 409. All acting
routes refuse once the game is over.

MCP over HTTP provides `offer_trade`, `answer_trade`, and `choose_trade` over these
routes, using named resource counts and response indices. See
[client interfaces](server.md#mcp).
