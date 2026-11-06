# ONNX model contract

HexSet loads ONNX checkpoints through `hexset.clients.onnxbot` (the
`clients` extra). A checkpoint provides metadata, the inputs it declares and
named outputs. Its `contract` metadata names the graph's shape, and the
loader serves nine (`hexset.server.constants.RECORD_CONTRACTS`):

| Contract | Graph | Run by |
| --- | --- | --- |
| `6`, `7`, `8`, `8.1` | A **record graph**: the record of one position in, one action out ([record inputs](#record-inputs)) | `onnxbot.V2Policy` |
| `9`, `9.1`, `9d`, `9d.1`, `11d` | A **decision graph**: the contract-9 observation and the decision's arrays in, a whole decision out ([decision graphs](#decision-graphs-contract-9)) | `onnxbot.DecisionPolicy` |

A file with no `contract` key, or any other value, is refused at load, naming
the contracts this server serves. Changing the value alone does not make a
graph compatible.

The record contract is `CONTRACT_VERSION` (`"8.1"`), `RECORD_FIELDS` and
`record_shapes` in [onnx_record.py](../hexset/onnx_record.py). The
observation is [contract9.py](../hexset/contract9.py) and the decision
arrays [decisions.py](../hexset/decisions.py). The action layout is
[actions.py](../hexset/actions.py).

## Model metadata

ONNX `metadata_props` values are strings. Unknown keys are ignored.

### The graph and its board

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `contract` | The graph's shape (table above) | Required: one of the nine |
| `players` | Seat count the graph was trained for | Required integer. A decision graph of contract `9d`, `9d.1` or `11d` must say `2`. See [seat counts](#seat-counts) |
| `num_hexes` | Board hex count | Required integer |
| `num_vertices` | Board vertex count | Required integer |
| `num_edges` | Board edge count | Required integer |
| `iteration` | Training iteration, kept as model information | 0; must be an integer when present |

The loader checks the three board counts against the target topology. The
counts do not verify connectivity or index ordering: the graph must use the
engine's board indexing.

### Serving

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `name` | The model's name; the server's picker offers the file by it ([serving a model](#serving-a-model)) | Absent: the server does not list the file |
| `modes` | Game types it plays, comma separated `hexset.rules.GAME_TYPES` names | `standard`; seating it at a table of another type raises, and so does `spawn` given the table's `game_type` |
| `version` | The model's version, shown with it and journalled with each game it plays | Free text |
| `commit` | The code commit it was built and served from | Free text; kept as model information |
| `owner`, `url`, `description` | Shown with it in the model list and on its rankings row | Optional; a value a client's profile would refuse leaves the field out |

### How it plays

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `search` | `mcts`: PUCT search over its priors and values ([the determinized root](training.md#the-determinized-root)). `rollout`: paired rollouts of its own policy ([paired rollouts](training.md#searching-by-paired-rollouts)) | Any other value: one forward per decision |
| `simulations` | Search descents per decision, under `search=mcts` | 128; at most 4096 |
| `wave` | Leaves evaluated per search wave, under `search=mcts` | 16; at most 256 |
| `worlds` | Shared-chance worlds each candidate plays out in, under `search=rollout` | 32; at most 256 |
| `top` | Candidates played out, the policy's own move first, under `search=rollout` | 3; at least 2, at most 16 |
| `decisive` | The prior on one move at which a root is played without searching, under either search | Absent: every root with a choice is searched. A value that is not a number in (0, 1] fails the load |
| `horizon` | Turns after the root at which a playout stops and is scored by the value head, under `search=rollout` | Absent: playouts run to the game's end. A value that is not a whole number of 1 or more fails the load |
| `setup` | `full`: every opening root is played out whatever `decisive` says, under `search=rollout` | Absent: `decisive` applies everywhere. Any other value fails the load |
| `mix` | A temperature in win chance: a searched root plays a candidate drawn from the mixed strategy over the candidates' scores and prior (`hexset.rollout.mixed_strategy`), under `search=rollout` | Absent or `0`: the best-scoring candidate. A value that is not a finite number of 0 or more fails the load |
| `anchor` | The exponent on the prior that `mix` is anchored to: `1` the prior itself, `0` every candidate alike, under `search=rollout` | Absent: `1`. A value that is not a finite number of 0 or more fails the load |
| `det` | `1`: each searched root draws its worlds, their seeds and its `mix` draw from a stream seeded by what the deciding seat can see (`hexset.game.observed_digest`) and `salt`, under `search=rollout` | Absent or `0`: the searcher's own running stream. Any other value fails the load |
| `salt` | A whole number mixed into `det`'s seed | Absent: `0`. A value that is not a whole number, or one given without `det = 1`, fails the load |
| `opponents`, `opponent_encoder` | Two files beside this one that play every other seat of a playout, under `search=rollout` ([opponent models](#opponent-models)) | Absent: the file's own policy plays every seat. One without the other, or a name with a path in it, fails the load |
| `own_calibration` | `"a,b"`: the seat's own win chance read as `sigmoid(a + b * logit(p))` (`hexset.mcts.calibrate_own`), the other seats scaled so the vector keeps its sum. Under `search=mcts`, at every evaluated position and in the trade gate; otherwise in the trade gate and robber deals alone | Absent: no recalibration. A value that is not two finite numbers with `b > 0` fails the load |
| `finish_decisions` | Read the graph's values where a setup turn or Road Building card is finished, not inside it (`hexset.clients.finishing.Finishing`): for a value head taught only where decisions start | Off; `1`/`true`/`yes`/`on` enable it |
| `committee` | Member files beside this one, joined by `+`, that vote with it ([committees](#committees)) | Absent: the file plays alone |
| `closer` | A bot, named as a lineup names it, whose move is played in place of the checkpoint's wherever the closer has one (`hexset.arena.closed`, `hexset.bots.Closes`) | Absent or empty: none |
| `robber_tip` | Under one forward per decision, how far a robber move leans towards the leader (the other seat with the highest value): the leader's best robber move by prior is played in place of the chosen one where its prior is at least this fraction of the chosen one's (`hexset.clients.netbot.tip_robber`) | `0.0`, off; held to `[0.0, 1.0]`; an unreadable value is off |

Search keys are read only under the `search` that uses them, so a stale
`simulations` does not turn a policy into a search.

### How it trades

The trade keys together are one `hexset.trading.TradeParams`, the object a
Python bot declaring its bargaining carries ([trading.md](trading.md#tradeparams)).
A key that is absent or does not read takes its `TradeParams` default; a
number is clamped to its bounds.

| Key | Meaning | Default or validation |
| --- | --- | --- |
| `max_offers` | Offers this gate makes on its own turn; `0` disables its trading | Absent: no limit of its own; clamped to 0..8. `max_trades` is read as `max_offers` when `max_offers` is absent |
| `trade_floor` | Clearing floor on this checkpoint's own gains, in win probability | 0.0; clamped to `[0.0, 1.0]` |
| `trade_floor_relative` | Scale the floor by the seat's share of the table's victory points | Off; `1`/`true`/`yes`/`on` enable it |
| `gate_plies` | Plies rolled forward over the exchanged hand before it is valued | 0; at most 16 |
| `gate_own_frame` | Value every position from this seat's own frame, whoever is to move | On; `0`/`false`/`no`/`off` disable it |
| `gate_frames` | On another seat's turn, the frames a position is valued from and averaged: `one`, `pair` or `all` | `one`; any other word reads `one` |
| `max_give_cards` | Cards this gate parts with in one exchange | Absent: no limit of its own; at least 1 |
| `responder_card_risk` | Charge per outgoing card on a response | 0.0; clamped to `[0.0, 1.0]` |
| `open_asks` | Put repeated one-for-one asks for a card up as one open ask ([open offers](trading.md#open-offers)) | On; `0`/`false`/`no`/`off` disable it |
| `fragment_trades` | Plan one gated target a turn and broadcast it as fragments | Off; `1`/`true`/`yes`/`on` enable it |
| `fragment_threshold` | The proposer cutoff a full target is gated on | 0.0; clamped to `[0.0, 1.0]` |
| `max_fragments` | Fragments one target may be broadcast as | 1; 1 or 2 |
| `fragment_cards` | Cards either side of one fragment may move | Absent: 3; at least 1 |
| `fit_offers` | Order counters, and under `fragment_trades` offers, by what the counterparty has shown it wants and gives up | Off; `1`/`true`/`yes`/`on` enable it |
| `trader` | A bot, named as a lineup names it, that answers this checkpoint's trades while the checkpoint plays its moves | Absent or empty: the checkpoint's own gate |
| `trader_veto` | Every exchange the `trader` would make must also clear this checkpoint's own gain above its `trade_floor` (`hexset.bots.Vetoed`) | Off; `1`/`true`/`yes`/`on` enable it |
| `trader_veto_pick` | `gain`: the veto also picks which taker of the seat's own offer to deal with, the one its own value gains most from | Absent: the trader picks |

`trade_floor` and `gate_plies` describe this checkpoint's value head. A file
declaring no trade key is read at `hexset.trading.UNLIMITED`: floor 0, so any
strictly positive gain clears, no response charge, no offer budget, no
planning, no card caps, and open asks on. `TradeParams.as_meta()` writes a
parameter set as metadata for an exporter; `fragment_trades` alone is not a
fragmented policy, which is the whole set.

`trader` and `closer` are resolved when the checkpoint is spawned
(`onnxbot.spawn`, an arena entrant, a served seat), not by `load`. A name
this process cannot build, usually a bot whose runtime was not loaded,
raises there, naming the file; load the registering module with
`--runtime <module>` (in code, `hexset.arena.load_runtime`). A
`trader_veto` on a checkpoint whose own gate does not trade
(`max_offers=0`) raises there too.

### Load errors and fallbacks

`load` raises for:

- a missing `contract` or one the server does not serve;
- a missing or non-integer `players` or board count, or a board count that
  does not match the topology;
- a present `iteration` that is not an integer;
- a `decisive`, `horizon`, `setup`, `mix`, `anchor`, `det`, `salt` or `own_calibration` that does not read,
  and a lone `opponents` or `opponent_encoder`: these change what the seat
  does, so the load fails rather than playing on a guess;
- a `committee` that names a member not beside the file, or a member that
  declares a committee of its own;
- a decision graph that declares an input HexSet does not supply, an input
  at a width HexSet does not supply it at, no per-row input, no `decision`
  or `value` output, or some but not all of the whole-option outputs;
- a contract `9d`, `9d.1` or `11d` graph whose `players` is not 2;
- a `search` of `mcts` or `rollout` on a graph that declares no `prior`;
- a `fragment_cards` wider than a declared `max_give_cards` under
  `fragment_trades`.

Everything else falls back. A missing, malformed, zero or negative search
budget (`simulations`, `wave`, `worlds`, `top`) takes its default; a trade
key takes its default when absent or unreadable and is clamped to its
bounds; an unreadable `robber_tip` is off.

The device is the host's `--device` option and the `device` argument of
`load`/`spawn`; metadata never sets it. `load(..., threads=1)` caps both ONNX
Runtime thread pools (`threads=None` is ONNX Runtime's own choice); it is
not metadata either. `spawn(..., workers=N)` plays a rollout file's worlds
out in `N` worker processes beside the searching one
(`hexset.rollout.RolloutPool`, each loading the file at one thread), and
`spawn(..., pools=SharedPools())` shares one such pool among every bot
spawned from one file; other files ignore both.

### Seat counts

`spawn(..., players=n, game_type=t)` checks the table. A checkpoint fits a
table dealt its own `players`, or one dealt more where the game type plays
exactly its `players` live seats and retires the rest at the deal
(`onnxbot.seats_fit`): a two-seat net sits at a duel dealt four with two
seats retired. A decision graph reads such a table over the live seats
alone (`hexset.encoding.SeatFrame`, `hexset.decisions.frame_for`). Any
other count raises, and so does a game type its `modes` do not name.

## Serving a model

The server names a model file by what it declares, never by its filename:
its picker entry is its `name`, `modes` are the game types a table of which
may seat it, `version` is shown with it and journalled with each game it
plays, and `commit` the code it was built from. A model plays the standard
game alone unless its `modes` say otherwise, so no file is offered at every
game type by default. One name is one bot, which may be one file at some game
types and another at the rest. A file declaring no `name`, or a registered
bot's name, is not listed, and two offering one name at one game type are
neither offered there. A file may also declare an `owner`, a `url` and a
one-sentence `description` (at most 100 characters), shown with it in the
model list and on its rankings row; one a client's profile would refuse is
left out.

`hexset.clients.onnxmeta` reads and writes these keys without a model runtime,
copying the graph unchanged:

```sh
python -m hexset.clients.onnxmeta stamp policy.onnx models/mynet-duel.onnx \
    --name mynet --modes duel-variant --version 1.0 --commit 0123abcd \
    --owner octocat --description "A small policy net with 128-visit search."
python -m hexset.clients.onnxmeta show models/mynet-duel.onnx
```

## Record inputs

The record states the position from one seat: its own hand and development
cards exactly, every other seat by card totals and the public resource
ledger. Seat arrays are in board-seat order; `perspective` names the seat.
The graph does rotation, feature encoding, masking and normalization.

Every tensor has a leading batch axis `B`. `NUM_RESOURCES` and
`NUM_DEV_CARDS` are both 5. A graph declares the record fields it reads, and
the loader feeds only those. A declared input the record does not carry
raises when the first inference request is built.

Each contract keeps every field of the one before. Contract 6's record is
the rows marked 6 and `action_mask`; contract 7 adds `roads_to_take` and
`to_move`; contract 8 adds seven fields; contract 8.1 adds the table's rules
and the next roll's odds. A contract-8 graph typically declares 27 fields:
all but `knights_played`, `longest_road_holder`, `largest_army_holder`,
`hand_totals`, `ledger_known` and `ledger_unknown` up to contract 8.

| Name | Shape | dtype | From contract |
| --- | --- | --- | --- |
| `terrain` | `(B, num_hexes)` | int64 | 6 |
| `token` | `(B, num_hexes)` | int64 | 6 |
| `port_code` | `(B, num_vertices)` | int64 | 6 |
| `robber` | `(B,)` | int64 | 6 |
| `vertex_owner` | `(B, num_vertices)` | int64 | 6 |
| `vertex_building` | `(B, num_vertices)` | int64 | 6 |
| `edge_owner` | `(B, num_edges)` | int64 | 6 |
| `bank` | `(B, NUM_RESOURCES)` | int64 | 6 |
| `knights_played` | `(B, players)` | int64 | 6 |
| `award_points` | `(B, players)` | int64 | 6 |
| `longest_road_holder` | `(B,)` | int64 | 6 |
| `largest_army_holder` | `(B,)` | int64 | 6 |
| `phase` | `(B,)` | int64 | 6 |
| `free_roads` | `(B,)` | int64 | 6 |
| `deck_size` | `(B,)` | int64 | 6 |
| `turns` | `(B,)` | int64 | 6 |
| `roads_to_take` | `(B, players)` | int64 | 7 |
| `to_move` | `(B,)` | int64 | 7 |
| `perspective` | `(B,)` | int64 | 6 |
| `own_hand` | `(B, NUM_RESOURCES)` | int64 | 6 |
| `hand_totals` | `(B, players)` | int64 | 6 |
| `own_dev` | `(B, NUM_DEV_CARDS)` | int64 | 6 |
| `dev_totals` | `(B, players)` | int64 | 6 |
| `ledger_known` | `(B, players, NUM_RESOURCES)` | int64 | 6 |
| `ledger_unknown` | `(B, players)` | int64 | 6 |
| `reach` | `(B, players, num_vertices)` | int64 | 8 |
| `own_matured` | `(B, NUM_DEV_CARDS)` | int64 | 8 |
| `dev_card_played` | `(B,)` | int64 | 8 |
| `expected_hand` | `(B, players, NUM_RESOURCES)` | float32 | 8 |
| `dev_ages` | `(B, players, 3)` | float32 | 8 |
| `knights_needed` | `(B, players)` | int64 | 8 |
| `unseen_dev` | `(B, NUM_DEV_CARDS)` | int64 | 8 |
| `winning_points` | `(B,)` | int64 | 8.1 |
| `discard_limit` | `(B,)` | int64 | 8.1 |
| `friendly_robber` | `(B,)` | int64 | 8.1 |
| `balanced_dice` | `(B,)` | int64 | 8.1 |
| `roll_odds` | `(B, 11)` | float32 | 8.1 |
| `action_mask` | `(B, space.size)` | bool | 6 |

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

`roads_to_take` is `hexset.roads.roads_to_take_all`: per seat, the new roads it
needs to take Longest Road, `0` for the holder and `ROADS_TO_TAKE_CAP` (4) for
four or more or out of reach. `to_move` is `hexset.game.to_move`, the seat
whose decision it is, which differs from `perspective` when the record is a
value read for another seat.

Contract 8's fields are in board-seat order where they are per seat, and read
what the perspective may know. `reach` is `hexset.roads.reach_all`: per seat,
per vertex, the fewest new roads its network needs to touch the vertex, `-1`
where it cannot. `own_matured` is the perspective's cards held since before
this turn, and `dev_card_played` whether it has played one this turn.
`expected_hand` is each seat's expected count of each resource on the
perspective's view (`hexset.view.View.expected_counts`), its own row exact.
`dev_ages` is each seat's longest-held cards' ages in its own turns, oldest
first, `-1` past the cards it holds (`hexset.encoding.dev_age_rows`).
`knights_needed` is the knights each seat must still play to take Largest
Army, `0` for the holder and `-1` where the perspective can prove it out of
reach (`hexset.encoding.knights_needed_all`). `unseen_dev` is the cards the
perspective has not seen, by kind (`View.unseen_dev_cards`).

Contract 8.1's fields are the table's rules and the dice:
`winning_points` and `discard_limit` as numbers, `friendly_robber` and
`balanced_dice` as 0 or 1, and `roll_odds` the chance of each sum 2 to 12 on
the next roll of the seat on turn (`hexset.game.roll_odds`): the dice's
odds, or under balanced dice the dice deck's odds for that roller, 0 for a
sum the deck has run out of.

`action_mask` is the perspective seat's own legal actions. A record for a
seat with no move of its own (a value-only row the trade gate prices for a
seat that is not acting, or a finished game) carries the fixed mask
`hexset.onnx_record.NO_MOVE`, `END_TURN` alone, never another seat's moves,
which would encode that seat's hand. The graph needs at least one legal
entry to normalise its prior over; `value` must not depend on the mask.

### Record graph outputs

| Name | Shape and dtype | Meaning |
| --- | --- | --- |
| `action_index` | `(B,)`, int64 | Flat index of the selected legal action |
| `prior` | `(B, space.size)`, floating point | Distribution over legal actions, zero on illegal ones |
| `value` | `(B, players)`, floating point | Per-seat win probabilities in board-seat order |

| Reader | Outputs |
| --- | --- |
| One forward per decision | `action_index` |
| The network trade gate | `value` |
| Search (`search=mcts` or `rollout`), a robber tip, a committee | `prior`, `value` |

A checkpoint meant for every mode exports all three. The server validates
the selected action like any submitted move.

Search scores a finished game as a one-hot winner vector in board-seat
order; the value head must use win probabilities on that scale.

Search and trade-gate continuation call the graph on batches of positions,
so export a dynamic batch axis. One forward per decision uses a batch of one.
Value reads fall back to one call per row for a graph with a fixed batch
size; action and prior reads do not.

## Decision graphs (contract 9)

A decision graph answers a seat's whole decision (`hexset.decisions`): a
setup settlement with its road, the setup snake's double turn, a Road
Building card with its roads, or a whole discard is one choice, and every
other action is a decision of its own. HexSet plays the chosen decision a
move at a time (`hexset.decisions.Plans`); a position inside a decision with
no plan of its own is asked as a decision of its legal moves alone
(`decisions.atomic`).

### Inputs

The graph declares the inputs it reads, by name, from
`onnxbot.DECISION_INPUTS`; the loader refuses any other name, and needs at
least one per-row input (`observation`, `flat` or `setup`). `B` is the rows
in the batch.

| Name | Shape | dtype | What it is |
| --- | --- | --- | --- |
| `observation` | `(B, width)` | float32 | The position from the row's own seat: `contract9.flatten(contract9.encode9(game, seat, contract))` |
| `flat` | `(B, space.size)` | bool | The legal flat actions, robber victims seat-relative (the row's seat first, the others in turn order) |
| `setup` | `(B, incidences)` | bool | The legal (settlement, road) pairs of a setup turn, over `contract9.incidences(topology)` (144 on the standard board) |
| `road_building` | `(Kr, 2)` | int64 | Every row's Road Building candidates end to end: the two edges, `-1` for a road not placed |
| `road_building_row` | `(Kr,)` | int64 | The row each Road Building candidate belongs to |
| `discard` | `(Kd, 10)` | int64 | Every row's whole discards end to end: the hand kept, then the cards given, five counts each |
| `discard_row` | `(Kd,)` | int64 | The row each discard belongs to |
| `double_rows` | `(D,)` | int64 | The rows whose decision is the snake seat's double turn |

A graph that declares `double_rows` is asked the double turn as one
decision, an ordered pair of setup turns; one that does not is asked it as
two setup turns, the second on the board after the first. A value-only row
has empty masks and no candidates.

The observation is rows of fields laid out by `contract9.layout(players,
hexes, vertices, edges, contract)`: seat tokens (the row's seat first),
the table, hexes, vertices, vertex and road owners as pointers to seat rows,
and per-seat reach. On the standard board:

| Contract | Seats | Seat token | Table | Hex | Width |
| --- | --- | --- | --- | --- | --- |
| `9` | 4 | 39 | 20 | 7 | 1029 |
| `9.1` | 4 | 40 | 35 | 8 | 1067 |
| `9d` | 2 | 22 | 20 | 8 | 828 |
| `9d.1` | 2 | 32 | 20 | 8 | 848 |
| `11d` | 2 | 35 and 27 | 3 | 17 | 1000 |

`9.1` appends a live-seat flag to each seat token, and the table's rules and
next-roll odds to the table token. `9d` is the two-seat observation, defined
only for two seats: each hand is exact, and only the development deck is
hidden. `9d.1` appends each seat's production and bank rates to its token.
`11d` is also two-seat: my token (35 wide) and the opponent's (27 wide)
have widths of their own, the table is a board block of three columns (the
turn count, the next roll's chance of a seven, whether the dice have been
thrown), and a hex reads its number one-hot.
[training.md](training.md#decision-observations) and the module docstring
of `hexset.contract9` give every column.

### Outputs

| Name | Shape and dtype | Meaning |
| --- | --- | --- |
| `decision` | `(B, 2)`, int64 | Required. Each row's chosen `(kind, index)` over its decision: `kind` a `hexset.decisions.DecisionKind` (`FLAT=0`, `SETUP=1`, `DOUBLE=2`, `ROAD_BUILDING=3`, `DISCARD=4`), `index` into that block (a flat action, an incidence, `first * incidences + second`, or a candidate's position among the row's own) |
| `value` | `(B, players)`, floating point | Required. Each seat's win chance in the row's own frame: the row's seat first, the others in turn order |
| `prior` | `(B, space.size)`, floating point | Optional. A distribution over the flat actions the row's `flat` mask allows. Without it the graph cannot be searched |
| `option_flat`, `option_setup`, `option_double`, `option_road_building`, `option_discard` | `(B, space.size)`, `(B, incidences)`, `(D, incidences, incidences)`, `(Kr,)`, `(Kd,)` | Optional, all or none (`option_double` only where the graph declares `double_rows`). Each whole option's probability in its row's decision, every block together, zero where it is not legal (`decisions.OPTION_OUTPUTS`) |

HexSet reads nothing else a decision graph answers. `decisions.option_of`
turns a `(kind, index)` back into the option, a robber victim back into a
board seat. A graph that answers the whole-option outputs lists a decision's
whole options (`DecisionPolicy.option_rows`, `decisions.listed`), so paired
rollouts can search over whole decisions.

Where the native core is built, a playout of a decision graph that does not
declare `double_rows` holds its copies natively
(`DecisionPolicy.native_contract`, [native.md](native.md)).

## Committees

A file whose `committee` key names other files beside it, joined by `+`
(`b.onnx+c.onnx`), plays as a product-of-experts vote
(`hexset.clients.committee.Committee`). The file itself is the first member
and carries the seat: its player count, game types, trade gate, search and
closer are the seat's. Each member is a file this package loads on its own,
of any contract, and declares no committee of its own; a member is offered
at no table unless it is stamped with a name itself.

Every legal move is scored by each member's prior, and the move with the
highest mean log prior is played, so a move needs every member's support.
The committee's prior, which a search or a robber tip reads, is that product
renormalised over the options (the members' arithmetic mean where every
option has a zero product); its values are the members' values averaged seat
by seat. The vote reads each member's `prior` and `value`.

## Opponent models

Paired rollouts play every seat of a playout with the searching file's own
policy. A `search=rollout` file may name two other files beside it for the
other seats (`hexset.clients.conditioned`):

- `opponents`: a record graph that also declares an input `condition`, one
  float32 vector per row saying how the seat to move plays. Its move is
  sampled from its `prior` at temperature 1.
- `opponent_encoder`: a sequence graph, read on each other seat's moves as
  the table makes them. It declares record fields (the record from the
  moving seat), `action` (the move's action index) and `state`, and answers
  `condition` and `next_state`. A seat not yet seen has a zero state and no
  condition.

The encoder reads each move in a world drawn from the searching seat's own
view, never the table's true state. The searching file still plays its own
seat in every playout, reads every value and scores every root. The two
`condition` widths must agree.

## Trade gate

`hexset.clients.netbot.NetworkBot` (one forward per decision), `GatedSearch`
(MCTS) and `GatedRollouts` (paired rollouts) put the shared network trade
gate in front of the graph: each candidate exchange is valued as the change
in this seat's `value`, read after `gate_plies` plies (one forward at `0`).
The checkpoint's trade keys configure it, and it plays the protocol in
[trading.md](trading.md). The trade round it answers is
[trade rounds](guide.md#trade-rounds); its runtime is
[model runtimes](training.md#model-runtimes). Neither adds graph inputs or
outputs.

## Action indices

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
this space. A record graph's robber victims are board seats; a decision
graph's are seat-relative.

## Check an exported model

`load` validates metadata, board counts and a decision graph's declared
inputs and outputs against the topology it is given, so loading a file is
the check:

```python
from hexset.arena import deal_board
from hexset.clients.onnxbot import load

board = deal_board(seed=0, index=0)
loaded = load("models/policy.onnx", board.topology)
print(loaded.players, loaded.iteration, loaded.trade_floor, loaded.search)
```

Everything in [load errors](#load-errors-and-fallbacks) raises here rather
than mid-game, except a `trader` or `closer` this process cannot build,
which raises at `spawn`.
