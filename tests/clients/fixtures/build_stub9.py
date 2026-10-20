"""Build a contract-9 stub graph: the inputs a decision graph may declare and the
answers HexSet reads, with constant scores and no weights.

    python build_stub9.py                 # -> stub-contract9.onnx

`build(path, ...)` writes variants: one that declares `double_rows` (the
double turn asked as one decision), one without `prior`, one that asks for an
input HexSet does not supply, one of another observation width.

What it answers is deliberately simple and deliberately legal. Every flat
action has a fixed score (a city, a settlement, a development card, the roll,
a road, ending the turn, in that order, everything else below), every setup
turn, double turn, Road Building set and discard scores zero, and the graph
masks and chooses as a decision graph must: the best legal option over the
blocks in `DecisionKind` order, a tie to the earlier block and then the lower
index, so the first legal setup turn, pair, road set or discard. `value` is
`softmax(2, 1, 0, -1)` in every row's own frame (the seat itself first, the
others in turn order), so a runner that turns it to board-seat order the
wrong way round is caught; `prior` the softmax of the flat scores over the
row's `flat` mask; `points` `(4, 3, 2, 1)`, which HexSet must not read.

NOT a genuine export: names, shapes and dtypes follow
`hexset.clients.onnxbot.DECISION_INPUTS` and the action contract
`hexset.clients.onnxbot.DecisionPolicy` reads, checked by
`tests/clients/test_contract_dispatch.py`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

NUM_HEXES, NUM_VERTICES, NUM_EDGES, PLAYERS = 19, 54, 72, 4
WIDTH = 1029          # `hexset.contract9.layout(4, 19, 54, 72).width`
SPACE = 456
INCIDENCES = 144
NEG = float("-inf")
# `hexset.actions.build_space(54, 72, 19, 4)`'s blocks: (offset, size).
BLOCKS = {
    "ROLL": (0, 1), "END_TURN": (1, 1), "BUY_DEV_CARD": (2, 1), "BUILD_ROAD": (130, 72),
    "BUILD_SETTLEMENT": (202, 54), "BUILD_CITY": (256, 54),
}
PREFERENCE = {"BUILD_CITY": 5.0, "BUILD_SETTLEMENT": 4.0, "BUY_DEV_CARD": 3.0, "ROLL": 2.0, "BUILD_ROAD": 1.0,
              "END_TURN": 0.0}


def _flat_scores() -> np.ndarray:
    scores = np.full(SPACE, -1.0, dtype=np.float32)
    for name, (offset, size) in BLOCKS.items():
        scores[offset:offset + size] = PREFERENCE[name]
    return scores


def _double_pairs() -> np.ndarray:
    """`hexset.decisions.double_pairs` on the standard board, from HexSet."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    try:
        from hexset.arena import deal_game
        from hexset.decisions import double_pairs
    finally:
        sys.path.pop(0)
    return double_pairs(deal_game(0, 0, PLAYERS)._state.board.topology)


def build(path: pathlib.Path, *, width: int = WIDTH, joint: bool = False, prior: bool = True,
          points: bool = True, extra: str | None = None, meta: dict[str, str] | None = None) -> pathlib.Path:
    """Write a contract-9 stub to `path` (see the module docstring)."""
    inputs = [
        ("observation", TP.FLOAT, ["batch", width]),
        ("flat", TP.BOOL, ["batch", SPACE]),
        ("setup", TP.BOOL, ["batch", INCIDENCES]),
        ("road_building", TP.INT64, ["road_building", 2]),
        ("road_building_row", TP.INT64, ["road_building"]),
        ("discard", TP.INT64, ["discards", 10]),
        ("discard_row", TP.INT64, ["discards"]),
    ]
    if joint:
        inputs.append(("double_rows", TP.INT64, ["doubles"]))
    if extra:
        inputs.append((extra, TP.FLOAT, ["batch", 1]))
    outputs = [("decision", TP.INT64, ["batch", 2]), ("value", TP.FLOAT, ["batch", PLAYERS])]
    if prior:
        outputs.append(("prior", TP.FLOAT, ["batch", SPACE]))
    if points:
        outputs.append(("points", TP.FLOAT, ["batch", PLAYERS]))

    logits = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    value_row = np.exp(logits) / np.exp(logits).sum()

    def const(name, array):
        return numpy_helper.from_array(np.asarray(array), name)

    initializers = [
        const("value_row", value_row[None].astype(np.float32)),
        const("points_row", np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)),
        const("flat_row", _flat_scores()[None]),
        const("zero_index", np.array([0], dtype=np.int64)),
        const("axis1", np.array([1], dtype=np.int64)),
        const("players", np.array([PLAYERS], dtype=np.int64)),
        const("space", np.array([SPACE], dtype=np.int64)),
        const("zero", np.array(0.0, dtype=np.float32)),
        const("neg", np.array(NEG, dtype=np.float32)),
    ]

    def fill(name, shape, value, dtype):
        tensor = helper.make_tensor(f"{name}_value", dtype, [1], [value])
        return helper.make_node("ConstantOfShape", [shape], [name], value=tensor)

    nodes = [
        helper.make_node("Shape", ["observation"], ["observation_shape"]),
        helper.make_node("Gather", ["observation_shape", "zero_index"], ["batch"], axis=0),
        helper.make_node("Concat", ["batch", "players"], ["by_players"], axis=0),
        helper.make_node("Concat", ["batch", "space"], ["by_space"], axis=0),
        helper.make_node("Expand", ["value_row", "by_players"], ["value"]),
        fill("batch_neg", "batch", NEG, TP.FLOAT),
        fill("batch_zero", "batch", 0.0, TP.FLOAT),
        fill("batch_first", "batch", 0, TP.INT64),
        # The flat block, masked.
        helper.make_node("Expand", ["flat_row", "by_space"], ["flat_scores"]),
        helper.make_node("Where", ["flat", "flat_scores", "neg"], ["flat_masked"]),
        helper.make_node("ReduceMax", ["flat_masked", "axis1"], ["flat_top"], keepdims=0),
        helper.make_node("ArgMax", ["flat_masked"], ["flat_at"], axis=1, keepdims=0),
    ]
    # A double turn's rows, and the single setup turn on every other row.
    if joint:
        nodes += [
            helper.make_node("Shape", ["double_rows"], ["double_shape"]),
            fill("double_ones", "double_shape", 1.0, TP.FLOAT),
            helper.make_node("ScatterElements", ["batch_zero", "double_rows", "double_ones"], ["double_count"],
                             axis=0),
            helper.make_node("Greater", ["double_count", "zero"], ["is_double"]),
        ]
    else:
        nodes.append(helper.make_node("Greater", ["batch_zero", "zero"], ["is_double"]))
    nodes += [
        helper.make_node("Not", ["is_double"], ["not_double"]),
        helper.make_node("Unsqueeze", ["not_double", "axis1"], ["not_double_column"]),
        helper.make_node("And", ["setup", "not_double_column"], ["single"]),
        helper.make_node("Where", ["single", "zero", "neg"], ["setup_masked"]),
        helper.make_node("ReduceMax", ["setup_masked", "axis1"], ["setup_top"], keepdims=0),
        helper.make_node("ArgMax", ["setup_masked"], ["setup_at"], axis=1, keepdims=0),
    ]
    if joint:
        initializers += [const("axis2", np.array([2], dtype=np.int64)), const("pairs_ok", _double_pairs()),
                         const("pair_width", np.array([INCIDENCES * INCIDENCES], dtype=np.int64))]
        nodes += [
            helper.make_node("Gather", ["setup", "double_rows"], ["double_setup"], axis=0),
            helper.make_node("Unsqueeze", ["double_setup", "axis2"], ["first_legal"]),
            helper.make_node("Unsqueeze", ["double_setup", "axis1"], ["second_legal"]),
            helper.make_node("And", ["first_legal", "second_legal"], ["both_legal"]),
            helper.make_node("And", ["both_legal", "pairs_ok"], ["pair_legal"]),
            helper.make_node("Where", ["pair_legal", "zero", "neg"], ["pair_cube"]),
            helper.make_node("Concat", ["double_shape", "pair_width"], ["pair_shape"], axis=0),
            helper.make_node("Reshape", ["pair_cube", "pair_shape"], ["pair_scores"]),
            helper.make_node("ReduceMax", ["pair_scores", "axis1"], ["pair_top"], keepdims=0),
            helper.make_node("ArgMax", ["pair_scores"], ["pair_at"], axis=1, keepdims=0),
            helper.make_node("ScatterElements", ["batch_neg", "double_rows", "pair_top"], ["double_top"], axis=0),
            helper.make_node("ScatterElements", ["batch_first", "double_rows", "pair_at"], ["double_at"], axis=0),
        ]
    else:
        nodes += [helper.make_node("Identity", ["batch_neg"], ["double_top"]),
                  helper.make_node("Identity", ["batch_first"], ["double_at"])]
    # A candidate block: zero for a row with candidates, its first chosen.
    for name in ("road_building", "discard"):
        nodes += [
            helper.make_node("Shape", [f"{name}_row"], [f"{name}_shape"]),
            fill(f"{name}_ones", f"{name}_shape", 1.0, TP.FLOAT),
            helper.make_node("ScatterElements", ["batch_zero", f"{name}_row", f"{name}_ones"], [f"{name}_count"],
                             axis=0, reduction="add"),
            helper.make_node("Greater", [f"{name}_count", "zero"], [f"{name}_has"]),
            helper.make_node("Where", [f"{name}_has", "zero", "neg"], [f"{name}_top"]),
        ]
    blocks = ("flat", "setup", "double", "road_building", "discard")
    tops = [f"{b}_top" for b in blocks]
    ats = ["flat_at", "setup_at", "double_at", "batch_first", "batch_first"]
    nodes += [helper.make_node("Unsqueeze", [t, "axis1"], [f"{t}_column"]) for t in tops]
    nodes += [helper.make_node("Unsqueeze", [a, "axis1"], [f"{a}_{i}_column"]) for i, a in enumerate(ats)]
    nodes += [
        helper.make_node("Concat", [f"{t}_column" for t in tops], ["tops"], axis=1),
        helper.make_node("Concat", [f"{a}_{i}_column" for i, a in enumerate(ats)], ["ats"], axis=1),
        helper.make_node("ArgMax", ["tops"], ["kind"], axis=1, keepdims=1),
        helper.make_node("GatherElements", ["ats", "kind"], ["index"], axis=1),
        helper.make_node("Concat", ["kind", "index"], ["decision"], axis=1),
    ]
    if prior:
        nodes += [
            helper.make_node("Softmax", ["flat_masked"], ["flat_soft"], axis=1),
            helper.make_node("Cast", ["flat"], ["flat_f"], to=TP.FLOAT),
            helper.make_node("ReduceMax", ["flat_f", "axis1"], ["any_flat"], keepdims=1),
            helper.make_node("Greater", ["any_flat", "zero"], ["has_flat"]),
            helper.make_node("Where", ["has_flat", "flat_soft", "zero"], ["prior"]),
        ]
    if points:
        nodes.append(helper.make_node("Expand", ["points_row", "by_players"], ["points"]))
    graph = helper.make_graph(
        nodes, "hexset-contract-9-stub",
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info(n, t, s) for n, t, s in outputs],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    model.doc_string = ("Contract-9 STUB. No learned parameters: constant scores, masked and chosen as a decision "
                        "graph must. For building and testing the decision-graph loader only -- never for play or "
                        "measurement.")
    props = {
        "contract": "9", "players": str(PLAYERS), "num_hexes": str(NUM_HEXES),
        "num_vertices": str(NUM_VERTICES), "num_edges": str(NUM_EDGES), "max_offers": "0", "iteration": "0",
        "stub": "constant-scores",
        **(meta or {}),
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


if __name__ == "__main__":
    out = build(pathlib.Path(__file__).parent / "stub-contract9.onnx")
    print(f"wrote {out}")
