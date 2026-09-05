"""Build a record-contract stub graph: right names, right shapes, no weights.

    python build_stub.py 6              # 25 inputs -> stub-contract6.onnx
    python build_stub.py 6 --partial    # a graph that declares only a subset
    python build_stub.py 6 --valued     # value = a fixed linear read of own_hand
    python build_stub.py 6 --valued --fixed-batch   # --valued with batch dim pinned to 1

Semantics are deliberately trivial and deliberately *legal*: uniform over the
legal mask, first legal slot as the chosen action, zero value. A stub that
could emit an illegal action would send whoever builds against it chasing a
phantom engine bug, so the mask is the only thing this graph reads.

`--valued` breaks the "zero value" half of that on purpose: it reads
`own_hand` through a fixed per-resource weight and broadcasts the result to
every seat, so a caller that prices one more card of each resource against
the current hand (`NetworkBot.accepts`/`accepts_many`) sees five *different*,
non-zero deltas instead of a graph that "has no opinion" by construction.
Still deterministic and still legal-only over the policy heads -- only
`value` moves. `--fixed-batch` pins every declared shape's batch axis to the
literal `1` instead of the dynamic `B` symbol, which is what
`V2Policy._batchable` has to detect and fall back from when a caller wants
more than one row scored at once.

Three files besides the plain stub, because each difference is the whole
point of its fixture: a loader that feeds a graph every field it happens to
have rather than the ones the graph *declares* breaks on the shorter one
(`onnxbot.V2Policy._run`) -- `--partial` therefore drops the two ledger
fields, which is the shape a graph trained before them would have; `--valued`
gives the value head something to say; `--fixed-batch` gives it a batch axis
that refuses to be anything but one row.

These stubs are NOT genuine exports. `hexset.export_onnx` needs torch, which
this repo does not install, so no real contract-5 file exists here to test
against; the field names, shapes and dtypes below are pinned against
`hexset.onnx_record.RECORD_FIELDS` by `tests/test_onnx_record.py`, torch-free.
See `docs/engine-divergence-2026-09-02.md`, "Defect 1 in detail".
"""

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

NUM_HEXES, NUM_VERTICES, NUM_EDGES = 19, 54, 72
PLAYERS, SPACE = 4, 456
RESOURCES, DEV = 5, 5
FIXED_BATCH = "--fixed-batch" in sys.argv
B = 1 if FIXED_BATCH else "B"

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
    ("ledger_known", TP.INT64, [B, PLAYERS, RESOURCES]),
    ("ledger_unknown", TP.INT64, [B, PLAYERS]),
    ("action_mask", TP.BOOL, [B, SPACE]),
]

# The `contract` metadata number is `hexset.export_onnx._CONTRACT_VERSION`'s.
LEDGER_FIELDS = ("ledger_known", "ledger_unknown")

CONTRACT = sys.argv[1] if len(sys.argv) > 1 else "6"
if CONTRACT != "6":
    raise SystemExit("contract must be 6")
PARTIAL = "--partial" in sys.argv
VALUED = "--valued" in sys.argv
if PARTIAL:
    INPUTS = [row for row in INPUTS if row[0] not in LEDGER_FIELDS]

OUTPUTS = [
    ("action_index", TP.INT64, [B]),
    ("prior", TP.FLOAT, [B, SPACE]),
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
initializers = [one, zero, axis1]

normalise("action_mask", "prior", "action_index", "act")

if VALUED:
    # Five distinct, fixed weights -- not all equal -- so five different
    # imagined-successor deltas (`NetworkBot.accepts`/`accepts_many`) come
    # out different from one another, and a plain sum (which a broadcast
    # bug could still pass) would not. `players_ones` broadcasts the per-hand scalar to
    # every seat's column: this stub has no notion of *whose* row is whose,
    # only that all four should move together, which is enough to exercise
    # the batching and the arithmetic without pretending to be a trained
    # network.
    weight = numpy_helper.from_array(
        np.array([0.006, -0.011, 0.004, 0.013, -0.008], dtype=np.float32),
        "resource_weight",
    )
    players_ones = numpy_helper.from_array(
        np.ones(PLAYERS, dtype=np.float32), "players_ones"
    )
    initializers += [weight, players_ones]
    nodes.append(helper.make_node("Cast", ["own_hand"], ["hand_f"], to=TP.FLOAT))
    nodes.append(helper.make_node("Mul", ["hand_f", "resource_weight"], ["weighted"]))
    nodes.append(
        helper.make_node(
            "ReduceSum", ["weighted", "act_axis"], ["hand_value"], keepdims=1
        )
    )
    nodes.append(helper.make_node("Mul", ["hand_value", "players_ones"], ["value"]))
else:
    # Value rides the batch axis off an input so the shape follows B, and is
    # multiplied to zero: a stub must not look like it has an opinion.
    nodes.append(helper.make_node("Cast", ["hand_totals"], ["ht_f"], to=TP.FLOAT))
    nodes.append(helper.make_node("Mul", ["ht_f", "zero"], ["value"]))

tag = "".join(
    suffix
    for flag, suffix in ((PARTIAL, "-partial"), (VALUED, "-valued"), (FIXED_BATCH, "-batch1"))
    if flag
)

graph = helper.make_graph(
    nodes,
    f"hexset-contract-{CONTRACT}{tag}-stub",
    [helper.make_tensor_value_info(n, t, s) for n, t, s in INPUTS],
    [helper.make_tensor_value_info(n, t, s) for n, t, s in OUTPUTS],
    initializer=initializers,
)

model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
)
model.doc_string = (
    f"Contract-{CONTRACT} STUB. No learned parameters. Uniform over the legal "
    "mask, first legal slot as the action, "
    + ("a fixed linear read of own_hand" if VALUED else "zero")
    + " value. For building and testing the record-contract loader path "
    "only -- never for play or measurement."
)

meta = {
    "contract": CONTRACT,
    "players": str(PLAYERS),
    "num_hexes": str(NUM_HEXES),
    "num_vertices": str(NUM_VERTICES),
    "num_edges": str(NUM_EDGES),
    "max_trades": "",
    "iteration": "0",
    "search": "none",
    "stub": "linear-own-hand" if VALUED else "uniform-over-legal",
}
for k, v in meta.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = k, v

onnx.checker.check_model(model)
name = f"stub-contract{CONTRACT}{tag}.onnx"
out = pathlib.Path(__file__).with_name(name)
onnx.save(model, str(out))
print(f"written {out.name}: {len(INPUTS)} inputs")
