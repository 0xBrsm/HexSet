# SPDX-License-Identifier: GPL-3.0-only
"""The information-set record: the actual boundary between the rules and the
model.

`hexset.encoding.encode`/`encode_batch` take a `Game` and read its private
state directly -- fine for training, where the caller and the network are the
same process, but wrong for a served checkpoint: it means the *feature
layout* has to be reimplemented, bit-identical, in whatever language serves
the model (`onnx-contract-v2.md`'s complaint about hexset-ui's `encoding.py`).

The record below is the alternative. It states the position **in the rules'
own terms, already filtered to what the perspective seat may legally know**
-- own hand and own development cards exact, everyone else by count alone --
and hands that to the graph. Filtering stays here, on the engine side, where
`test_record_is_information_set_correct`-style tests can pin it; everything
downstream of the record (one-hot encoding, rotation, scaling) is mechanical
and can safely live inside a traced graph instead -- `hexnet.export_onnx.
RecordEncoder` is that traced half: `record -> (hexes, vertices, edges,
globals)`, the same four tensors `hexset.encoding.encode` produces and
`HexNet` consumes.

**Everything in this module is numpy, and stays that way.** This is the
contract module the gym and any other torch-free consumer build a request
from, so `import hexset.onnx_record` must succeed with torch absent --
`test_onnx_record_is_torch_free` pins it. The traced encoder that reads this
record, and everything else that needs a model in hand, lives in
`hexnet.export_onnx` instead.

`record_from_game`/`record_batch` build the record from a live `Game`, for
tests and for generating realistic export/parity samples, and this is also
what `hexset.server.api`'s `GET /api/record` and `hexset.clients.onnxbot` now
call directly -- the two once carried their own copy (`hexset_ui.record`)
because this module imported torch; it no longer does (see the module
docstring above), so that copy is gone.

Two properties of `hexset.encoding` carry over unchanged, because they are
properties of the *feature layout*, not of who computes it:

*Seat-relative.* Everything is rotated so the perspective seat is column/slot
0. `RecordEncoder` does the rotation now (mirrors `hexset.encoding._seat`);
the record itself stays in board-seat order, which is what makes
`perspective` meaningful as an input rather than baked in.

*Information-set correct, by construction upstream of here.* `own_hand` and
`own_dev` are exact only for the perspective seat; every other seat con-
tributes a total alone (`hand_totals`, `dev_totals`), plus `hexset.ledger`'s
public-knowledge reconstruction of their composition (`ledger_known`,
`ledger_unknown`) -- itself never sharper than what the public log could
have derived, whichever seat is asking. The traced encoder never sees a
hidden card -- there is nowhere in the record for one to be.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .actions import Action, ActionSpace, legal_actions
from .board.board import Board
from .board.terrain import NUM_RESOURCES
from .cards import NUM_DEV_CARDS
from .encoding import StaticGraph
from .game import Game, to_move
from .state import NO_OWNER
from .victory import award_points

# `hexset.server.modelmeta`/`onnxbot._load_cached` read this off the exported
# graph's metadata: `"1"` (absent) is the old feature-tensor-in shape; `"2"`
# the 23-input record; `"3"` contract 2 plus the four live-offer record
# fields (trading design part 1); `"4"` contract 3 plus the two
# public-knowledge ledger fields (`agents/reference/trading-design.md` §7.2)
# -- same outputs, two more inputs; `"5"` replaced the four `offer_*` fields
# with `valuations`, one vector per seat -- the one-event trade mechanic's
# public layer. `"6"` carries two independent changes that happened to land
# together: the trading redesign drops `valuations` outright
# (`agents/reference/trading-final.md`, item 1: there is no public layer any
# more, so nothing replaces it) *and* the knight two-step fix changes the
# flat `ActionSpace`'s width (`PLAY_KNIGHT` dropped its operands -- it only
# spends the card now; the robber move is a separate `MOVE_ROBBER` decision
# through the same `Phase.ROBBER` a seven enters). `RECORD_FIELDS` shrinks by
# one field and the action space shrinks independently of it, either change
# alone would have forced this bump, and contracts 1-5 are refused the same
# way. Lives here, not in `hexnet.export_onnx`, because it names *this
# record's* shape and only incidentally the graph's -- bump it again only if
# `RECORD_FIELDS`, the export's output tuple, or the action space changes.
CONTRACT_VERSION = "6"

# Every field name in the record, in the order the plan's table lists them
# (board, then position, then information set, then legality).
# `hexnet.export_onnx` reuses this order for the graph's input names.
RECORD_FIELDS: tuple[str, ...] = (
    "terrain",
    "token",
    "port_code",
    "robber",
    "vertex_owner",
    "vertex_building",
    "edge_owner",
    "bank",
    "knights_played",
    "award_points",
    "longest_road_holder",
    "largest_army_holder",
    "phase",
    "free_roads",
    "deck_size",
    "turns",
    "perspective",
    "own_hand",
    "hand_totals",
    "own_dev",
    "dev_totals",
    "ledger_known",
    "ledger_unknown",
    "action_mask",
)


def record_shapes(graph: StaticGraph, players: int, space: ActionSpace) -> dict[str, tuple]:
    """Every input field's per-row shape, i.e. `RECORD_FIELDS` minus the batch
    axis. `hexnet.export_onnx` extends this with its own output shapes for
    the sample inputs, the read-back check and the tests -- one table, not
    two, for the half both sides agree on."""
    return {
        "terrain": (graph.num_hexes,),
        "token": (graph.num_hexes,),
        "port_code": (graph.num_vertices,),
        "robber": (),
        "vertex_owner": (graph.num_vertices,),
        "vertex_building": (graph.num_vertices,),
        "edge_owner": (graph.num_edges,),
        "bank": (NUM_RESOURCES,),
        "knights_played": (players,),
        "award_points": (players,),
        "longest_road_holder": (),
        "largest_army_holder": (),
        "phase": (),
        "free_roads": (),
        "deck_size": (),
        "turns": (),
        "perspective": (),
        "own_hand": (NUM_RESOURCES,),
        "hand_totals": (players,),
        "own_dev": (NUM_DEV_CARDS,),
        "dev_totals": (players,),
        "ledger_known": (players, NUM_RESOURCES),
        "ledger_unknown": (players,),
        "action_mask": (space.size,),
    }


def _port_code(board: Board) -> np.ndarray:
    """`-1` no port, `0` generic, `1+r` a port for resource `r` -- one code
    per vertex, the static half of `hexset.encoding._template_by_value`'s port
    block, read back out of the board rather than the template."""
    codes = np.full(board.topology.num_vertices, NO_OWNER, dtype=np.int64)
    for port in board.ports:
        value = 0 if port.resource is None else 1 + int(port.resource)
        for v in port.vertices:
            codes[v] = value
    return codes


def record_from_game(
    game: Game,
    perspective: int | None,
    space: ActionSpace,
    options: Sequence[Action] | None = None,
) -> dict[str, np.ndarray]:
    """The information-set record for `perspective`, as a single (unbatched)
    row per field. `perspective` defaults to `to_move(game)`, same as
    `hexset.clients.onnxbot.Request.seat` -- the mask is a property of
    whoever is to move, and encoding from a different seat would pair a
    stranger's legal moves with someone else's hand.

    `options` is `legal_actions(game)` if not given -- pass it when the
    caller already computed the legal-option set (a gym step, a search leaf)
    so this does not recompute it."""
    state = game._state
    players = state.num_players
    if perspective is None:
        perspective = to_move(game)
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")

    if options is None:
        options = legal_actions(game)
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True

    own_dev = np.array(
        [held + fresh for held, fresh in zip(state.dev_cards[perspective], state.new_dev_cards[perspective])],
        dtype=np.int64,
    )
    hand_totals = np.array([sum(state.hands[s]) for s in range(players)], dtype=np.int64)
    dev_totals = np.array(
        [sum(state.dev_cards[s]) + sum(state.new_dev_cards[s]) for s in range(players)],
        dtype=np.int64,
    )
    award = np.array([award_points(state, s) for s in range(players)], dtype=np.int64)

    # The public-knowledge ledger (`hexset.ledger`), board-seat order like
    # every other field -- `RecordEncoder` rotates it and drops the
    # perspective seat's own entry (already exact via `own_hand` above).
    ledger_known = np.asarray(
        [game.ledger.seats[s].known for s in range(players)], dtype=np.int64
    )
    ledger_unknown = np.asarray(
        [game.ledger.seats[s].unknown for s in range(players)], dtype=np.int64
    )

    return {
        "terrain": np.asarray([int(t) for t in state.board.terrain], dtype=np.int64),
        "token": np.asarray(state.board.tokens, dtype=np.int64),
        "port_code": _port_code(state.board),
        "robber": np.int64(state.robber),
        "vertex_owner": np.asarray(state.vertex_owner, dtype=np.int64),
        "vertex_building": np.asarray(state.vertex_building, dtype=np.int64),
        "edge_owner": np.asarray(state.edge_owner, dtype=np.int64),
        "bank": np.asarray(state.bank, dtype=np.int64),
        "knights_played": np.asarray(state.knights_played, dtype=np.int64),
        "award_points": award,
        "longest_road_holder": np.int64(state.longest_road_holder),
        "largest_army_holder": np.int64(state.largest_army_holder),
        "phase": np.int64(int(game.phase)),
        "free_roads": np.int64(game.free_roads),
        "deck_size": np.int64(len(state.deck)),
        "turns": np.int64(game.turns),
        "perspective": np.int64(perspective),
        "own_hand": np.asarray(state.hands[perspective], dtype=np.int64),
        "hand_totals": hand_totals,
        "own_dev": own_dev,
        "dev_totals": dev_totals,
        "ledger_known": ledger_known,
        "ledger_unknown": ledger_unknown,
        "action_mask": mask,
    }


def record_batch(
    games_and_perspectives: Sequence[tuple[Game, int | None]], space: ActionSpace
) -> dict[str, np.ndarray]:
    """`record_from_game` for several positions at once, stacked on a leading
    batch axis -- what `torch.onnx.export`'s dummy inputs and the parity
    check both want. Every game must share `space`'s topology and player
    count, same restriction `encode_batch` places on its own callers."""
    if not games_and_perspectives:
        raise ValueError("need at least one position")
    rows = [record_from_game(game, seat, space) for game, seat in games_and_perspectives]
    return {name: np.stack([row[name] for row in rows]) for name in RECORD_FIELDS}
