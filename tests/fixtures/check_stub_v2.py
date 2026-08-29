"""Run the stub against real positions and prove it can only choose legal moves."""

import pathlib
import random
import sys

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from hexset_ui.actions import ActionType, apply, build_space, options_for
from hexset_ui.board.board import random_base_board
from hexset_ui.game import start, to_move

NUM_RESOURCES = 5

rng = random.Random(7)
board = random_base_board(rng)
game = start(board, 4, rng)
space = build_space(
    board.topology.num_vertices, board.topology.num_edges, board.topology.num_hexes, 4
)
session = ort.InferenceSession(str(HERE / "stub-v2.onnx"), providers=["CPUExecutionProvider"])

print("inputs :", len(session.get_inputs()), "outputs:", len(session.get_outputs()))
print("meta   :", session.get_modelmeta().custom_metadata_map)


def record(game, seat, options):
    """The v2 information-set record, built the way hexset_ui.record will."""
    st = game.state
    mask = np.zeros(space.size, dtype=bool)
    for a in options:
        mask[space.index(a)] = True
    pair = np.zeros(NUM_RESOURCES * NUM_RESOURCES, dtype=bool)
    for a in options:
        if a.type is ActionType.PROPOSE_TRADE:
            pair[a.give.index(1) * NUM_RESOURCES + a.want.index(1)] = True
    port = np.full(board.topology.num_vertices, -1, dtype=np.int64)
    for p in board.ports:
        for v in p.vertices:
            port[v] = 0 if p.resource is None else 1 + int(p.resource)
    row = {
        "terrain": np.array(board.terrain, dtype=np.int64),
        "token": np.array(board.tokens, dtype=np.int64),
        "port_code": port,
        "robber": np.int64(st.robber),
        "vertex_owner": np.array(st.vertex_owner, dtype=np.int64),
        "vertex_building": np.array(st.vertex_building, dtype=np.int64),
        "edge_owner": np.array(st.edge_owner, dtype=np.int64),
        "bank": np.array(st.bank, dtype=np.int64),
        "knights_played": np.array(st.knights_played, dtype=np.int64),
        "award_points": np.zeros(4, dtype=np.int64),
        "longest_road_holder": np.int64(st.longest_road_holder),
        "largest_army_holder": np.int64(st.largest_army_holder),
        "phase": np.int64(int(game.phase)),
        "free_roads": np.int64(game.free_roads),
        "deck_size": np.int64(len(st.deck)),
        "turns": np.int64(game.turns),
        "perspective": np.int64(seat),
        "own_hand": np.array(st.hands[seat], dtype=np.int64),
        "hand_totals": np.array([sum(h) for h in st.hands], dtype=np.int64),
        "own_dev": np.array(
            [a + b for a, b in zip(st.dev_cards[seat], st.new_dev_cards[seat])],
            dtype=np.int64,
        ),
        "dev_totals": np.array(
            [sum(d) + sum(n) for d, n in zip(st.dev_cards, st.new_dev_cards)],
            dtype=np.int64,
        ),
        "action_mask": mask,
        "pair_mask": pair,
    }
    return {k: np.expand_dims(v, 0) for k, v in row.items()}


checked = illegal = 0
for step in range(400):
    options = options_for(game)
    if not options:
        break
    seat = to_move(game)
    out = session.run(None, record(game, seat, options))
    names = [o.name for o in session.get_outputs()]
    got = dict(zip(names, out))

    index = int(got["action_index"][0])
    legal = {space.index(a) for a in options}
    if index not in legal:
        illegal += 1
    checked += 1

    prior = got["prior"][0]
    mask = record(game, seat, options)["action_mask"][0]
    assert abs(prior.sum() - 1.0) < 1e-5, f"prior sums to {prior.sum()}"
    assert not prior[~mask].any(), "prior put weight on an illegal slot"
    assert got["value"].shape == (1, 4) and not got["value"].any()

    apply(game, rng.choice(options))

print(f"positions checked: {checked}")
print(f"illegal choices  : {illegal}")

# Batch axis really is dynamic — the searches feed a whole wave.
options = options_for(game) or [None]
if options[0] is not None:
    single = record(game, to_move(game), options)
    wave = {k: np.repeat(v, 16, axis=0) for k, v in single.items()}
    out = session.run(None, wave)
    print("wave of 16 ok    :", out[0].shape, out[2].shape, out[4].shape)

print("STUB OK" if illegal == 0 else "STUB BROKEN")
