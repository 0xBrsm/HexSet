# The bot API

This is the complete interface a `.onnx` file must satisfy to plug in as an
opponent. It is the only thing `src/hexset_ui/onnxbot.py` reads — the file
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
| `contract` | which graph shape below applies: `2`, `3` or `4` for the record shape, `1` (or absent) for the legacy feature-tensor shape | `1` |
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

**The `contract` number is assigned by the exporter, not by this repo.**
`hexset.export_onnx._CONTRACT_VERSION` is the one definition; `hexset_ui`
reads it and never writes it. The record shape has three numbers because it
grew twice: `2` is the original 23 fields, `3` adds the four live-offer
fields, `4` adds the two public-knowledge ledger fields. A graph declares the
fields it wants and is fed exactly those, so all three load and play off the
one record the engine builds — a checkpoint does not have to be re-exported
to keep working. An unknown number is refused at load with the number named,
rather than being routed to the legacy path and failing on its first move.

## 2. The graph — the record contracts (`2`, `3`, `4`)

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

**The engine drift this section used to list is gone.** `hexset_ui` no longer
carries its own copy of the engine: it depends on the `hexset` distribution,
so the `offered` re-proposal filter and the RNG-drawn trade-responder order
are simply what this server plays now, exactly as dev-hexset does. See
[`engine-divergence-2026-09-02.md`](engine-divergence-2026-09-02.md) for the
full account of what the copy held and how each difference was resolved.

**One difference remains, deliberately, and it is not tensor-shaped.** The
`action_mask`/`pair_mask` a checkpoint is served here are built over the
*honest* trade sample (`hexset_ui.rules.fair_legal_actions`): every
one-for-one offer the mover's own hand affords, with no filter for whether
some opponent could cover it. The engine's own `legal_actions` filters by
opponents' true hands, and dev-hexset's training record uses that. So a
checkpoint served here sees `want` slots it never saw enabled in training.
This is on purpose — the alternative tells a human, on every turn, exactly
what is in a specific opponent's hand — and it now applies to *every* seat,
embedded bots included, rather than only to the ones on the wire. The cost
has not been measured; the audit document asks the PI for a before/after
duel.

## 3. The graph — `contract=1` (legacy, frozen)

The original shape: the engine encodes the position into feature tensors
itself (`encoding_v1.py`), and the graph is a bare policy/value head.

`encoding_v1.py` is **frozen** at this layout and is not kept in step with
`hexset.encoding`, which has since widened its global feature block (86 floats
against this one's 50). That is the whole point of the `contract` key: a
contract-1 file keeps its contract-1 features for as long as it is served.

Masking, softmax, the give/want outer sum, and un-rotating `value` back to
board-seat order all happen in `onnxbot.py`, not in the graph.

**Inputs**, all float32, leading batch axis `B`:

| Name | Shape |
| --- | --- |
| `hexes` | `(B, num_hexes, hex_features)` |
| `vertices` | `(B, num_vertices, vertex_features)` |
| `edges` | `(B, num_edges, edge_features)` |
| `globals` | `(B, num_globals)` |

Feature widths depend on `players`; see `encoding_v1.py` for the exact layout.
This encoder was a from-scratch reimplementation of the training repo's own
encoder, and keeping the two bit-identical is the obligation the record
contracts exist to retire — which they have: the two have already diverged,
and only the `contract` key keeps old files playable. Treat this shape as
legacy and export anything new against the current record contract.

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
