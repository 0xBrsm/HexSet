# ONNX model contract

HexSet loads ONNX checkpoints through `hexset.clients.onnxbot` (the
`clients` extra). A checkpoint provides metadata, record inputs and named
outputs. The only supported contract is `6`; a file with no `contract` key or
any other value is refused at load. Changing the number alone does not make
an older graph compatible.

The contract is `CONTRACT_VERSION`, `RECORD_FIELDS` and `record_shapes` in
[onnx_record.py](../hexset/onnx_record.py), with the action layout in
[actions.py](../hexset/actions.py).

## Model metadata

ONNX `metadata_props` values are strings.

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `contract` | Record and action-space version | Required: `"6"` |
| `players` | Seat count the graph was trained for | Required integer; seating it at a game of another count raises, and so does `spawn` given the table's count |
| `num_hexes` | Board hex count | Required integer |
| `num_vertices` | Board vertex count | Required integer |
| `num_edges` | Board edge count | Required integer |
| `iteration` | Training iteration, kept as model information | 0; must be an integer when present |
| `search` | `mcts` plays the graph through search over its priors and values | Any other value: one forward per decision |
| `simulations` | Search descents per decision | 128; at most 4096 |
| `wave` | Leaves evaluated per search wave | 16; at most 256 |
| `max_offers` | Offers this gate makes on its own turn; `0` disables its trading | Absent: no limit of its own; clamped to 0..8. `max_trades` is read as `max_offers` when `max_offers` is absent |
| `trade_floor` | Clearing floor on this checkpoint's own gains, in win probability | 0.0; clamped to `[0.0, 1.0]` |
| `gate_plies` | Plies rolled forward over the exchanged hand before it is valued | 0; at most 16 |
| `max_give_cards` | Cards this gate parts with in one exchange | Absent: no limit of its own; at least 1 |
| `responder_card_risk` | Charge per outgoing card on a response | 0.0; clamped to `[0.0, 1.0]` |
| `fragment_trades` | Plan one gated target a turn and broadcast it as fragments | `0`; `1`/`true`/`yes`/`on` enable it |
| `fragment_threshold` | The proposer cutoff a full target is gated on | 0.0; clamped to `[0.0, 1.0]` |
| `max_fragments` | Fragments one target may be broadcast as | 1; 1 or 2 |
| `fragment_cards` | Cards either side of one fragment may move | Absent: 3; at least 1 |
| `fit_offers` | Order counters, and under `fragment_trades` offers, by what the counterparty has shown it wants and gives up | `0`; `1`/`true`/`yes`/`on` enable it |
| `trader` | A bot, named as a lineup names it, that answers this checkpoint's trades while the checkpoint plays its moves | Absent or empty: the checkpoint's own gate |

The loader checks the three board counts against the target topology. The
counts do not verify connectivity or index ordering: the graph must use the
engine's board indexing.

Loading raises for a missing or unsupported `contract`, a missing or
non-integer `players` or board count, a board count that does not match the
topology, a present `iteration` that is not an integer (each naming the file),
and a `fragment_cards` wider than a declared `max_give_cards` under
`fragment_trades`. Every other key falls back: search keys are read only
under `search=mcts`, and a missing, malformed, zero or negative
`simulations` or `wave` takes its default; a trade key takes its default
when absent or unreadable and is clamped to its bounds; unknown keys are
ignored.

`trader` is resolved when the checkpoint is spawned (`onnxbot.spawn`, an
arena entrant, a served seat), not by `load`. A name this process cannot
build, usually a bot whose runtime was not loaded, raises there, naming the
file; load the registering module with `--runtime <module>` (in code,
`hexset.arena.load_runtime`).

The trade keys together are one `hexset.trading.TradeParams`, the object a
Python bot declaring its bargaining carries ([trading.md](trading.md)).
`trade_floor` and `gate_plies` describe this checkpoint's value head. A file
declaring no trade key is read at `hexset.trading.UNLIMITED`: floor 0, so any
strictly positive gain clears, no response charge, no offer budget, no
planning and no card caps. `TradeParams.as_meta()` writes a parameter set as
metadata for an exporter; `fragment_trades` alone is not a fragmented
policy, which is the whole set.

The device is the host's `--device` option and the `device` argument of
`load`/`spawn`; metadata never sets it. `load(..., threads=1)` caps both ONNX
Runtime thread pools (`threads=None` is ONNX Runtime's own choice); it is
not metadata either.

## Record inputs

The record states the position from one seat: its own hand and development
cards exactly, every other seat by card totals and the public resource
ledger. Seat arrays are in board-seat order; `perspective` names the seat.
The graph does rotation, feature encoding, masking and normalization.

Every tensor has a leading batch axis `B`. `NUM_RESOURCES` and
`NUM_DEV_CARDS` are both 5. A graph may declare a subset of the record
fields; the loader feeds only the names it declares. A declared input the
record does not carry raises when the first inference request is built.

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

| Encoding | Values |
| --- | --- |
| Resources | `WOOD=0`, `BRICK=1`, `SHEEP=2`, `WHEAT=3`, `ORE=4` |
| Development cards | `KNIGHT=0`, `VICTORY_POINT=1`, `ROAD_BUILDING=2`, `YEAR_OF_PLENTY=3`, `MONOPOLY=4` |
| Terrain | `FOREST=0`, `HILLS=1`, `PASTURE=2`, `FIELDS=3`, `MOUNTAINS=4`, `DESERT=5`, `SEA=6`, `GOLD=7` |
| Port codes | `-1` none, `0` generic, `1 + resource` for a resource port |
| Owners, award holders | Seat, or `-1` for none |
| Robber | Its hex, or `-1` while it stands beside a board with no desert, before its first move |
| Buildings | `0` empty, `1` settlement, `2` city |
| Phase | `SETUP_SETTLEMENT=0`, `SETUP_ROAD=1`, `ROLL=2`, `DISCARD=3`, `ROBBER=4`, `MAIN=5`, `GAME_OVER=6` |

`own_dev` and `dev_totals` include cards bought this turn; the action mask
says which may be played. `award_points` is the points from Longest Road and
Largest Army, not total victory points.

`action_mask` is the perspective seat's own legal actions. A record for a
seat with no move of its own (a value-only row the trade gate prices for a
seat that is not acting, or a finished game) carries the fixed mask
`hexset.onnx_record.NO_MOVE`, `END_TURN` alone, never another seat's moves,
which would encode that seat's hand. The graph needs at least one legal
entry to normalise its prior over; `value` must not depend on the mask.

## Graph outputs

| Name | Shape and dtype | Meaning |
| --- | --- | --- |
| `action_index` | `(B,)`, int64 | Flat index of the selected legal action |
| `prior` | `(B, space.size)`, floating point | Distribution over legal actions, zero on illegal ones |
| `value` | `(B, players)`, floating point | Per-seat win probabilities in board-seat order |

| Reader | Outputs |
| --- | --- |
| One forward per decision | `action_index` |
| The network trade gate | `value` |
| Search (`search=mcts`) | `prior`, `value` |

A checkpoint meant for every mode exports all three. The server validates
the selected action like any submitted move.

Search scores a finished game as a one-hot winner vector in board-seat
order; the value head must use win probabilities on that scale.

Search and trade-gate continuation call the graph on batches of positions,
so export a dynamic batch axis. One forward per decision uses a batch of one.
Value reads fall back to one call per row for a graph with a fixed batch
size; action and prior reads do not.

### Action indices

`hexset.actions.build_space(V, E, H, P)` builds the flat space from vertex,
edge, hex and player counts. Blocks appear in this order; each block's offset
is the sum of all preceding sizes.

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

So `space.size = 3V + 2E + H(P + 1) + 55`. Player trading has no entry in
this space.

## Trade gate

`hexset.clients.netbot.NetworkBot` (one forward per decision) and
`GatedSearch` (search) put the shared network trade gate in front of the
graph: each candidate exchange is valued as the change in this seat's
`value`, read after `gate_plies` plies (one forward at `0`). The checkpoint's
trade keys configure it, and it plays the protocol in
[trading.md](trading.md). The trade round it answers is
[trade rounds](guide.md#trade-rounds); its runtime is
[model runtimes](training.md#model-runtimes). Neither adds graph inputs or
outputs.

## Check an exported model

`load` validates metadata and board counts against the topology it is given,
so loading a file is the check:

```python
from hexset.arena import deal_board
from hexset.clients.onnxbot import load

board = deal_board(seed=0, index=0)
loaded = load("models/policy.onnx", board.topology)
print(loaded.players, loaded.iteration, loaded.trade_floor)
```

A missing or unsupported contract, a missing or malformed required key, or a
board count that does not match the topology raises here rather than
mid-game.
