# The bot API

This is the complete interface a `.onnx` file must satisfy to plug in as an
opponent. It is the only thing `src/hexset_ui/onnxbot.py` reads — the file
does not need access to this repo's source, only to what is written here plus
the public [ONNX](https://onnx.ai/) format itself. `search2.py` (the
handcrafted opponent) implements the same interface a different way and is
the reference for what "correct" means when in doubt.

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
| `contract` | `1` or `2` — which graph shape below applies | `1` |
| `max_offers` | trade-offer budget the run trained under | engine's cap |
| `search` | `mcts` to search over the model's own priors; anything else plays one forward pass | none |
| `simulations` | descents per decision, when `search=mcts` (clamped to 4096) | 128 |
| `wave` | leaves batched per expansion, when `search=mcts` (clamped to 256) | 16 |
| `iteration` | informational only; not read for behaviour | 0 |

Unreadable or missing optional keys fall back to their default rather than
failing the load — a typo'd hint costs the hint, not the whole opponent. See
`src/hexset_ui/modelmeta.py` for the exact clamping.

Inference device (`cpu`/GPU) is deliberately **not** a metadata key — it is a
fact about the machine serving the game, not the checkpoint.

## 2. The graph — `contract=2` (current)

The engine builds a **record**: the position stated in the rules' own terms,
already filtered to what the perspective seat may legally know. The graph
owns everything downstream of that — encoding, masking, normalising,
argmax, un-rotating back to board-seat order. Built by
`src/hexset_ui/record.py:build_record`; the full field-by-field derivation
lives in [`onnx-contract-v2.md`](onnx-contract-v2.md).

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
| `offer_give` | `(B, NUM_RESOURCES)` | int64 |
| `offer_want` | `(B, NUM_RESOURCES)` | int64 |
| `offer_proposer` | `(B,)` | int64 |
| `offer_answered` | `(B, players)` | int64 |
| `ledger_known` | `(B, players, NUM_RESOURCES)` | int64 |
| `ledger_unknown` | `(B, players)` | int64 |
| `action_mask` | `(B, space.size)` | bool |
| `pair_mask` | `(B, NUM_PAIRS)` | bool |

**Outputs:**

| Name | Shape | Note |
| --- | --- | --- |
| `action_index` | `(B,)` int64 | argmax over the masked distribution |
| `pair_index` | `(B,)` int64 | argmax over the masked off-diagonal offers |
| `prior` | `(B, space.size)` | normalised over legal actions, zero elsewhere |
| `pair_prior` | `(B, NUM_PAIRS)` | normalised over legal offers |
| `value` | `(B, players)` | board-seat order, already un-rotated |

`NetworkBot` reads `action_index`/`pair_index`; searches read
`prior`/`pair_prior`/`value`. One graph serves both.

**Known remaining engine drift**, not fixed by the fields above: dev-hexset's
`Game` also carries an `offered: set[(give, want)]` that prunes a turn's
repeat trade proposals from the sample offered, and its `propose_trade`
draws the neutral trade-responder order from the game's own RNG rather than
clockwise from the proposer (`hexset_ui/trading.py`'s `responders` still
uses clockwise). Both are behavioural, not tensor-shape, differences — a
checkpoint fed hexset-ui's record sees the same *fields* dev-hexset produces,
but the *play* they were trained against can still diverge until these two
land here as well.

## 3. The graph — `contract=1` (legacy)

The original shape: the engine encodes the position into feature tensors
itself (`encoding.py`), and the graph is a bare policy/value head. Masking,
softmax, the give/want outer sum, and un-rotating `value` back to board-seat
order all happen in `onnxbot.py`, not in the graph.

**Inputs**, all float32, leading batch axis `B`:

| Name | Shape |
| --- | --- |
| `hexes` | `(B, num_hexes, hex_features)` |
| `vertices` | `(B, num_vertices, vertex_features)` |
| `edges` | `(B, num_edges, edge_features)` |
| `globals` | `(B, num_globals)` |

Feature widths depend on `players`; see `encoding.py` for the exact layout.
This encoder is a from-scratch reimplementation of the training repo's own
encoder and the two must stay bit-identical — the reason `contract=2` exists
is to retire that obligation, so treat this shape as legacy and prefer
`contract=2` for anything new.

**Outputs:**

| Name | Shape | Note |
| --- | --- | --- |
| `logits` | `(B, space.size)` | pre-mask, pre-softmax action-slot scores |
| `give` | `(B, NUM_RESOURCES)` | give-side offer logits |
| `want` | `(B, NUM_RESOURCES)` | want-side offer logits |
| `value` | `(B, players)` | **perspective-rotated** (seat 0 = perspective); `onnxbot.py` un-rotates it before handing it back |

## What is never part of this contract

`onnxbot.py`'s job stops at reading these names and shapes. It never imports
or inspects anything else about how a checkpoint was produced, and a
checkpoint's author never needs this repo's source to write one — only this
document, the ONNX spec, and the topology fingerprint of the board they are
targeting.
