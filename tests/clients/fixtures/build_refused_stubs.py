"""Build the fixtures that exist only to be refused.

    python build_refused_stubs.py    # -> dev-contract2.onnx, tiny.onnx

Contract 6 is the only one served. Contract 2 is one of the offer-protocol
contracts and contract 1 predates the record entirely; both are refused by
name (`hexset.server.constants.RECORD_CONTRACTS`). Nothing loads either to
play, so these graphs exist only to be turned away, and the suite asks three
things of them: the contract number reported, the declared input count, and
that the loader names the number it actually found.

Both replace genuine downstream exports that used to sit here with real
weights. Those carried the exporter's provenance in their metadata -- a
training checkpoint path -- which no test read and which the publish gate
could not see, because `git grep -I` skips binaries. Between them they were
1.1 MB of weights nobody ran. A fixture built by a script in the tree is
auditable, diffable and small.

Semantics match `build_stub.py`'s: uniform over the legal mask, first legal
slot, zero value. Deliberately legal, so nothing built against them chases a
phantom engine bug.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

NUM_HEXES, NUM_VERTICES, NUM_EDGES = 19, 54, 72
PLAYERS, RESOURCES, DEV = 4, 5, 5
# Contract 2's flat action space, and its one-for-one give/want pair space.
SPACE, PAIRS, MAX_OFFERS = 553, 25, 3
B = "batch"

INPUTS = [
    ("terrain", TP.INT64, [B, NUM_HEXES]),
    ("token", TP.INT64, [B, NUM_HEXES]),
    ("port_code", TP.INT64, [B, NUM_VERTICES]),
    ("robber", TP.INT64, [B]),
    ("vertex_owner", TP.INT64, [B, NUM_VERTICES]),
    ("vertex_building", TP.INT64, [B, NUM_VERTICES]),
    ("edge_owner", TP.INT64, [B, NUM_EDGES]),
    ("bank", TP.INT64, [B, RESOURCES]),
    ("knights_played", TP.INT64, [B, PLAYERS]),
    ("award_points", TP.INT64, [B, PLAYERS]),
    ("longest_road_holder", TP.INT64, [B]),
    ("largest_army_holder", TP.INT64, [B]),
    ("phase", TP.INT64, [B]),
    ("free_roads", TP.INT64, [B]),
    ("deck_size", TP.INT64, [B]),
    ("turns", TP.INT64, [B]),
    ("perspective", TP.INT64, [B]),
    ("own_hand", TP.INT64, [B, RESOURCES]),
    ("hand_totals", TP.INT64, [B, PLAYERS]),
    ("own_dev", TP.INT64, [B, DEV]),
    ("dev_totals", TP.INT64, [B, PLAYERS]),
    ("action_mask", TP.BOOL, [B, SPACE]),
    ("pair_mask", TP.BOOL, [B, PAIRS]),
]

OUTPUTS = [
    ("action_index", TP.INT64, [B]),
    ("pair_index", TP.INT64, [B]),
    ("prior", TP.FLOAT, [B, SPACE]),
    ("pair_prior", TP.FLOAT, [B, PAIRS]),
    ("value", TP.FLOAT, [B, PLAYERS]),
]

nodes = []
initializers = [
    numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
    numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
]


def normalise(mask_name, out_prior, out_index, tag):
    """Uniform over the legal mask; first legal slot as the choice."""
    f, total, safe = f"{tag}_f", f"{tag}_total", f"{tag}_safe"
    nodes.append(helper.make_node("Cast", [mask_name], [f], to=TP.FLOAT))
    nodes.append(helper.make_node("ReduceSum", [f, "axis1"], [total], keepdims=1))
    # A row with no legal entry would divide by zero, and the NaN would
    # surface as a wrong action rather than a crash.
    nodes.append(helper.make_node("Max", [total, "one"], [safe]))
    nodes.append(helper.make_node("Div", [f, safe], [out_prior]))
    nodes.append(helper.make_node("ArgMax", [f], [out_index], axis=1, keepdims=0))


normalise("action_mask", "prior", "action_index", "act")
normalise("pair_mask", "pair_prior", "pair_index", "pair")

# Value rides the batch axis off an input so its shape follows B, then is
# zeroed: a stub states no opinion about who is winning.
nodes.append(helper.make_node("Cast", ["hand_totals"], ["ht_f"], to=TP.FLOAT))
nodes.append(helper.make_node("Mul", ["ht_f", "zero"], ["value"]))

graph = helper.make_graph(
    nodes,
    "hexset-contract-2-stub",
    [helper.make_tensor_value_info(n, t, s) for n, t, s in INPUTS],
    [helper.make_tensor_value_info(n, t, s) for n, t, s in OUTPUTS],
    initializer=initializers,
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
)
model.doc_string = (
    "Contract-2 STUB. No learned parameters, and no provenance: contract 2 is "
    "refused by name, so this graph exists only to be turned away. Built by "
    "build_contract2_stub.py."
)

# `exporter_commit` and `checkpoint_sha256` are present because the suite
# asserts an export-shaped fixture carries them, and are obviously not real:
# the point of this rebuild is that no fixture names a real checkpoint. There
# is deliberately no `source_checkpoint` key.
meta = {
    "contract": "2",
    "players": str(PLAYERS),
    "num_hexes": str(NUM_HEXES),
    "num_vertices": str(NUM_VERTICES),
    "num_edges": str(NUM_EDGES),
    "max_offers": str(MAX_OFFERS),
    "exporter_commit": "0" * 40,
    "checkpoint_sha256": "0" * 64,
    "stub": "uniform-over-legal",
}
for k, v in meta.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = k, v

onnx.checker.check_model(model)
out = pathlib.Path(__file__).with_name("dev-contract2.onnx")
onnx.save(model, str(out))
print(f"written {out.name}: {len(INPUTS)} inputs")


# ---------------------------------------------------------------------------
# Contract 1: the pre-record graph. Observation in, raw heads out, masked and
# softmaxed in Python against the frozen `encoding_v1` feature layout. It
# declares no `contract` key at all -- that is the thing under test, since a
# file with no key is refused by name exactly like a wrong one.
# ---------------------------------------------------------------------------
C1_INPUTS = [
    ("hexes", TP.FLOAT, [B, NUM_HEXES, 11]),
    ("vertices", TP.FLOAT, [B, NUM_VERTICES, 14]),
    ("edges", TP.FLOAT, [B, NUM_EDGES, 5]),
    ("globals", TP.FLOAT, [B, 50]),
]
C1_OUTPUTS = [
    ("logits", TP.FLOAT, [B, SPACE]),
    ("give", TP.FLOAT, [B, RESOURCES]),
    ("want", TP.FLOAT, [B, RESOURCES]),
    ("value", TP.FLOAT, [B, PLAYERS]),
]

# Every head is zeros shaped off `globals`, so each output follows the batch
# axis without carrying a weight. `Shape`/`ConstantOfShape` beats a constant
# here: a fixed-batch constant would not load for a batch of anything else.
c1_nodes = [
    helper.make_node("Shape", ["globals"], ["g_shape"], start=0, end=1),
]
c1_init = []
for name, _, shape in C1_OUTPUTS:
    width = shape[1]
    c1_init.append(
        numpy_helper.from_array(np.array([width], dtype=np.int64), f"{name}_w")
    )
    c1_nodes.append(
        helper.make_node("Concat", ["g_shape", f"{name}_w"], [f"{name}_shape"], axis=0)
    )
    c1_nodes.append(
        helper.make_node(
            "ConstantOfShape",
            [f"{name}_shape"],
            [name],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "v"),
        )
    )

c1_graph = helper.make_graph(
    c1_nodes,
    "hexset-contract-1-stub",
    [helper.make_tensor_value_info(n, t, s) for n, t, s in C1_INPUTS],
    [helper.make_tensor_value_info(n, t, s) for n, t, s in C1_OUTPUTS],
    initializer=c1_init,
)
c1_model = helper.make_model(
    c1_graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
)
c1_model.doc_string = (
    "Contract-1 STUB. No learned parameters, no provenance, and deliberately "
    "no `contract` metadata key: a file with no key is refused by name just "
    "like a wrong one, which is what the suite checks. Built by "
    "build_refused_stubs.py."
)
# No `contract` key, and no `source_checkpoint`. The rest is the shape
# metadata a contract-1 export carried.
for k, v in {
    "players": str(PLAYERS),
    "num_hexes": str(NUM_HEXES),
    "num_vertices": str(NUM_VERTICES),
    "num_edges": str(NUM_EDGES),
    "max_offers": str(MAX_OFFERS),
    "stub": "zero-heads",
}.items():
    entry = c1_model.metadata_props.add()
    entry.key, entry.value = k, v

onnx.checker.check_model(c1_model)
c1_out = pathlib.Path(__file__).with_name("tiny.onnx")
onnx.save(c1_model, str(c1_out))
print(f"written {c1_out.name}: {len(C1_INPUTS)} inputs")
