# ONNX contract v2 — moving the model inside the model file

## The goal, in one line

hexset-ui is an **interface**: it emits state, it accepts an action. `search2.py`
and any `.onnx` file are two implementations of that one interface. Nothing in
this repo knows how a network reads a position.

Today that is not true. `encoding.py` is a 340-line Python reimplementation of
the training repo's observation encoder, and `onnxbot.py` carries the masking,
log-softmax, give/want factorisation and seat un-rotation in numpy. Both have to
stay bit-identical with `0xBrsm/dev-catan` forever, in a language that repo does
not run. This plan moves them into the graph.

**Read this before believing the docstrings.** `onnxbot.py:3-19` and
`export_onnx.py:20-28` in dev-catan both argue *for* the current split — "masking
stays in Python because a position's legal moves are a fact about the rules". The
premise is right and the conclusion does not follow: legality is a rules fact, so
the engine computes the mask and *passes it in as a graph input*. The graph masks,
normalises and picks. Those docstrings are the thing being changed; do not treat
them as the specification.

## What is and is not model internals

The line is **the rules**, not the network.

| Stays in hexset-ui | Moves into the `.onnx` file |
|---|---|
| The rules, legal-move enumeration, the flat `ActionSpace` | Feature encoding (all of `encoding.py`) |
| The information set — what a seat may legally know | Masking, log-softmax, give/want outer sum |
| `mcts.py` and `search2.py` | argmax over the masked distribution |
| `modelmeta.py` — a checkpoint declaring how it wants to be played | Un-rotating `value` back to board-seat order |
| Loading the session, reading metadata, `session.run` | The static topology, as baked initialisers |

**`mcts.py` stays, and this is deliberate.** MCTS expands nodes by applying the
transition function, so it can only live where the rules live. That makes it a
*sibling of `search2.py`* — a search over the rules that consults an evaluator —
not a smuggled piece of the network. One evaluator is handcrafted
(`search2.py:127`), the other is the checkpoint. A file declaring
`search=mcts, simulations=256` is stating how it wants to be played; that is
configuration, not inference. `modelmeta.py` is unchanged by this work.

## The information-set record — the actual interface

The engine feeds the graph the position **stated in the rules' own terms, filtered
to what the perspective seat may know**. This is not encoding: the engine says
what is true and visible, the graph decides how to read it.

Filtering in the engine is load-bearing, not incidental. `encoding.py`'s docstring
names *information-set correctness* as a property enforced by construction — own
hand and dev cards exact, opponents by count alone. If the record carried full
state and trusted the graph to ignore the hidden parts, that property would become
unauditable. Keep it in the engine, where a test can pin it.

Per row (leading batch axis `B` everywhere, dynamic):

**Board** — varies per game, so these are inputs, not initialisers:

| Name | Shape | dtype | Source |
|---|---|---|---|
| `terrain` | `(B, num_hexes)` | int64 | `board.terrain` |
| `token` | `(B, num_hexes)` | int64 | `board.tokens`, 0 for none |
| `port_code` | `(B, num_vertices)` | int64 | `-1` none, `0` generic, `1+r` resource `r` |

`pips(token)` becomes a constant lookup table inside the graph; do not
precompute it engine-side.

**Position:**

| Name | Shape | dtype |
|---|---|---|
| `robber` | `(B,)` | int64 |
| `vertex_owner` | `(B, num_vertices)` | int64 (`NO_OWNER = -1`) |
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

**Information set** — already filtered, in **board-seat order**; the graph does the
rotation, so `perspective` is what tells it how far to rotate:

| Name | Shape | dtype | Note |
|---|---|---|---|
| `own_hand` | `(B, NUM_RESOURCES)` | int64 | exact, perspective seat |
| `hand_totals` | `(B, players)` | int64 | totals for every seat |
| `own_dev` | `(B, NUM_DEV_CARDS)` | int64 | `dev_cards + new_dev_cards`, perspective seat |
| `dev_totals` | `(B, players)` | int64 | totals for every seat |

**Legality:**

| Name | Shape | dtype |
|---|---|---|
| `action_mask` | `(B, space.size)` | bool |
| `pair_mask` | `(B, NUM_PAIRS)` | bool |

**Baked as initialisers, not inputs:** the `StaticGraph` adjacency. It is fixed per
topology, which is exactly what the existing `num_hexes`/`num_vertices`/`num_edges`
metadata fingerprint already guards.

**Outputs:**

| Name | Shape | Note |
|---|---|---|
| `action_index` | `(B,)` int64 | argmax over the masked distribution |
| `pair_index` | `(B,)` int64 | argmax over the masked off-diagonal offers |
| `prior` | `(B, space.size)` | normalised over legal actions, zero elsewhere |
| `pair_prior` | `(B, NUM_PAIRS)` | normalised over legal offers |
| `value` | `(B, players)` | **board-seat order**, already un-rotated |

One forward serves both callers: `NetworkBot` reads `action_index`/`pair_index`,
the searches read `prior`/`pair_prior`/`value`. `_board_order` disappears.

**Scope cut, deliberate:** the stochastic path goes. `OnnxPolicy`'s own docstring
says every checkpoint served here plays argmax and the Gumbel branch is an unused
port of what the training interface supports. Do not reimplement sampling in the
graph. If it is ever wanted, add a temperature input then.

**Not eliminated:** `award_points` (longest road, largest army) is a rules
computation the engine must supply, so dev-catan and hexset-ui still have to agree
on it. That is the residual shared surface — far smaller than a duplicated
encoder, but not zero. Freeze and version it with the rest of the record.

## Phases

Phases 1–2 are in **dev-catan** (`src/catan/`), 3–6 in **hexset-ui**. Keep both
repos green at every step; no phase may leave `pytest` failing.

### Phase 1 — a torch encoder in dev-catan

`catan.encoding.encode` is numpy and cannot be traced. Reimplement it as a
`nn.Module` taking the record above and returning the four feature tensors
`CatanNet` already consumes. It is all gather, one-hot, scale and concat — there
is no data-dependent control flow in the original, and there must be none here.

Do **not** hand-build the graph with `onnx.helper` and `onnx.compose`. The torch
route gives a directly testable object and reuses the tracing path the exporter
already trusts.

Check against `encoding.encode_batch` as well as `encode` — the searches encode a
whole wave at once, and the batched path is the one the graph actually replaces.

*Done when:* a new test drives real positions from a random playout through both
`encoding.encode` and the torch module and asserts the four arrays match exactly
(these are constructions from integers, not accumulations — no tolerance).

### Phase 2 — exporter v2

Extend `_ExportWrapper` to `record -> encoder -> CatanNet -> heads`, where "heads"
is the masking, log-softmax, give/want outer sum, argmax and un-rotation lifted
out of `onnxbot.py`. Update `_INPUT_NAMES`, `_OUTPUT_NAMES` and `_shapes`.

`_sample_inputs` must change: random gaussians were valid when every input was a
float feature tensor, but the record is integers with meaning — `vertex_owner`
must be in `[-1, players)`, masks must have at least one true entry per row.
Generate sample records from real playouts instead.

Write `contract=2` into the metadata props. Keep `players`, the topology
fingerprint, `max_offers`, `iteration`, `search`, `simulations`, `wave` as they
are — `modelmeta.py` does not change.

Extend `_verify_parity` to compare the full v2 graph against the numpy path
(`encoding.encode` + the current `onnxbot.py` math) on real positions. Keep the
existing `rtol=1e-3, atol=1e-4` for `prior`/`value`; `action_index` and
`pair_index` must match **exactly**.

*Done when:* `python -m catan.export_onnx --checkpoint ../runs/lam095/latest.pt
--out lam095-v2.onnx` succeeds with parity green, and again with
`--search mcts --simulations 256`.

### Phase 3 — the record builder in hexset-ui

New `src/hexset_ui/record.py`: `Game + seat + options -> dict[str, np.ndarray]`,
exactly the table above. Pure engine code — it imports nothing model-shaped and
knows no feature layout. It may import `victory.award_points`.

Reuse the existing `action_mask` and `_pair_mask` from `onnxbot.py` by moving them
here; they were always engine code.

*Done when:* a test asserts the record contains no hidden information — for a
position where an opponent holds a known hand, no array in the record distinguishes
it from a different opponent hand with the same total.

> Note: a `record.py` was deleted in `f986911`/`7e4de9e` for unrelated reasons
> (training residue). Reusing the name is fine; check the history so you do not
> resurrect anything from it.

### Phase 4 — hexset-ui speaks both contracts

`onnxbot.load` reads `contract` from metadata (absent = 1). Keep the v1 path
working untouched; add a v2 path that builds a record, runs the graph and reads
`action_index`/`prior`/`value` straight out. Both `models/*.onnx` generations must
play.

*Done when:* the existing test suite passes unchanged against a v1 file, and the
same suite parameterised over a v2 file passes too.

### Phase 5 — the parity gate

A test that plays N seeded games with a v1 file and the same checkpoint exported
as v2, and asserts **the identical action sequence**. This is the gate: do not
proceed to phase 6 until it is green.

### Phase 6 — delete

Drop `encoding.py`, the v1 branch of `onnxbot.py`, and from it
`_masked_log_softmax`, `_pair_logits`, `_sample`, `_pair_index`, `_one_hot`,
`_board_order`, `NUM_PAIRS`, `_OFF_DIAGONAL`, `Request`, and the `greedy`/`rng`
members of `OnnxPolicy`. Rewrite the module docstring: it currently defends the
design being removed.

*Done when:* `onnxbot.py` is roughly 120 lines — load the session, check the
fingerprint, read metadata, run, hand back what came out — and `grep -ri "logit\|
softmax\|feature\|encode" src/hexset_ui/` returns nothing outside `record.py`'s
mask helpers.

## Standing rules for whoever picks this up

1. **The engine may say what is true and visible. It may not say how the model
   reads it.** Every judgement call resolves against that sentence.
2. If a phase needs a new number baked into hexset-ui that also exists in
   dev-catan, you have taken a wrong turn. The `NUM_PAIRS` comment at
   `onnxbot.py:52` — "redefined rather than shared because that module imports
   torch" — is the exact smell this work exists to remove.
3. `search2.py` never changes. It is the reference implementation of the
   interface; if a change to it seems necessary, the interface is wrong.
4. Phases 1–2 land in dev-catan and must be pushed before phase 4 can be tested.
   The two repos are versioned only by the `contract` metadata prop — there is no
   shared package and there must not be one.
