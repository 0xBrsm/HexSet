"""Build a contract-v2 stub graph: right names, right shapes, no learned weights.

Semantics are deliberately trivial and deliberately *legal*: uniform over the
legal mask, first legal slot as the chosen action, zero value. A stub that
could emit an illegal action would send whoever builds against it chasing a
phantom engine bug, so the mask is the only thing this graph reads.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

NUM_HEXES, NUM_VERTICES, NUM_EDGES = 19, 54, 72
PLAYERS, SPACE, PAIRS = 4, 553, 25
RESOURCES, DEV = 5, 5
B = "B"

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
    ("offer_give", TP.INT64, [B, RESOURCES]),
    ("offer_want", TP.INT64, [B, RESOURCES]),
    ("offer_proposer", TP.INT64, [B]),
    ("offer_answered", TP.INT64, [B, PLAYERS]),
    ("ledger_known", TP.INT64, [B, PLAYERS, RESOURCES]),
    ("ledger_unknown", TP.INT64, [B, PLAYERS]),
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

one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
zero = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero")

nodes = []


def normalise(mask_name, out_prior, out_index, tag):
    """mask -> uniform-over-legal prior, and the first legal slot."""
    f = f"{tag}_f"
    total = f"{tag}_total"
    safe = f"{tag}_safe"
    nodes.append(helper.make_node("Cast", [mask_name], [f], to=TP.FLOAT))
    nodes.append(
        helper.make_node("ReduceSum", [f, f"{tag}_axis"], [total], keepdims=1)
    )
    # A row with no legal entry would divide by zero. The engine should never
    # ask, but a NaN here would surface as a wrong action rather than a crash.
    nodes.append(helper.make_node("Max", [total, "one"], [safe]))
    nodes.append(helper.make_node("Div", [f, safe], [out_prior]))
    nodes.append(
        helper.make_node("ArgMax", [f], [out_index], axis=1, keepdims=0)
    )


axis1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "act_axis")
axis1b = numpy_helper.from_array(np.array([1], dtype=np.int64), "pair_axis")

normalise("action_mask", "prior", "action_index", "act")
normalise("pair_mask", "pair_prior", "pair_index", "pair")

# Value rides the batch axis off an input so the shape follows B, and is
# multiplied to zero: a stub must not look like it has an opinion.
nodes.append(helper.make_node("Cast", ["hand_totals"], ["ht_f"], to=TP.FLOAT))
nodes.append(helper.make_node("Mul", ["ht_f", "zero"], ["value"]))

graph = helper.make_graph(
    nodes,
    "hexset-contract-v2-stub",
    [helper.make_tensor_value_info(n, t, s) for n, t, s in INPUTS],
    [helper.make_tensor_value_info(n, t, s) for n, t, s in OUTPUTS],
    initializer=[one, zero, axis1, axis1b],
)

model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
)
model.doc_string = (
    "Contract-v2 STUB. No learned parameters. Uniform over the legal mask, "
    "first legal slot as the action, zero value. For building and testing the "
    "v2 loader path only -- never for play or measurement."
)

meta = {
    "contract": "2",
    "players": str(PLAYERS),
    "num_hexes": str(NUM_HEXES),
    "num_vertices": str(NUM_VERTICES),
    "num_edges": str(NUM_EDGES),
    "max_offers": "",
    "iteration": "0",
    "search": "none",
    "stub": "uniform-over-legal",
}
for k, v in meta.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = k, v

onnx.checker.check_model(model)
onnx.save(model, str(pathlib.Path(__file__).with_name("stub-v2.onnx")))
print("written")
