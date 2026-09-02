"""The position, stated in the rules' own terms and filtered to what one seat
may legally know — the interface between the engine and any way of choosing
an action, whether that is a handcrafted search or a `.onnx` file.

**This is the one engine-adjacent module this package still carries its own
copy of**, and it should not be. The canonical definition of the record is
`hexset.onnx_record` (`RECORD_FIELDS`, `record_from_game`), with the per-field
shapes in `hexset.export_onnx._shapes` and the contract number in
`_CONTRACT_VERSION`. All three are unreachable from here: those two modules
import torch at module scope, and this package ships an onnxruntime-only
image. `docs/engine-divergence-2026-09-02.md` files that as change request R1
— split the torch-free half out of `onnx_record.py` and this file deletes.
Until then `tests/test_record_contract.py` pins every field name and shape
against dev's own definitions wherever torch happens to be installed.

The one deliberate difference from `record_from_game`, and it stays after R1:
`options` is a *parameter* here. dev's version calls `legal_actions` itself,
which is the engine's omniscient trade sample; every seat at a HexSet table
gets the honest one instead (`hexset_ui.rules.fair_legal_actions`).

See `docs/onnx-contract-v2.md` for the field-by-field contract. This module
says what is true and visible; it never says how a network reads it, so it
imports nothing model-shaped and knows no feature layout.

Filtering here is load-bearing, not incidental: own hand and dev cards are
exact; everyone else's resource composition is only ever as certain as the
public log makes it (`ledger_known`/`ledger_unknown`, see `hexset.ledger`
— counting resources is not hidden information in this game, only a steal's
identity is), and dev-card composition is a total alone. A caller that wants
more than this record describes is asking the engine to leak a hidden card.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hexset.actions import Action, ActionSpace, ActionType
from hexset.board.terrain import NUM_RESOURCES
from hexset.devcards import holdings
from hexset.game import Game
from hexset.state import NO_OWNER
from hexset.trading import responders as offer_responders
from hexset.victory import award_points

NUM_PAIRS = NUM_RESOURCES * NUM_RESOURCES


def action_mask(space: ActionSpace, options: Sequence[Action]) -> np.ndarray:
    """Mark already-enumerated actions without enumerating them again."""
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True
    return mask


def pair_index(give: Sequence[int], want: Sequence[int]) -> int:
    """The flat pair slot for a one-for-one offer's two one-hot bundles."""
    return give.index(1) * NUM_RESOURCES + want.index(1)


def pair_mask(options: Sequence[Action]) -> np.ndarray:
    """Which one-for-one offers were legal, as a flat `(NUM_PAIRS,)` bool."""
    mask = np.zeros(NUM_PAIRS, dtype=bool)
    for option in options:
        if option.type is ActionType.PROPOSE_TRADE:
            mask[pair_index(option.give, option.want)] = True
    return mask


def _port_code(board, num_vertices: int) -> np.ndarray:
    code = np.full(num_vertices, -1, dtype=np.int64)
    for port in board.ports:
        value = 0 if port.resource is None else 1 + int(port.resource)
        for v in port.vertices:
            code[v] = value
    return code


def build_record(
    game: Game, seat: int, options: Sequence[Action], space: ActionSpace
) -> dict[str, np.ndarray]:
    """`Game + seat + options -> dict[str, np.ndarray]`, per
    `docs/onnx-contract-v2.md`. `options` is normally `legal_actions(game)`
    (after any offer-budget trim), the same list `action_mask`/`pair_mask`
    are built from.
    """
    state = game.state
    board = state.board
    players = state.num_players

    own_dev = holdings(state, seat)

    # The live trade offer, filtered exactly as a viewer's own state_view is:
    # give/want and the proposer are public while an offer stands, but who
    # has answered it so far is the proposer's own information only -- a
    # responder must not condition on an earlier decline (see `webplay.py`'s
    # own comment on why `state_view` leaves `responders` out entirely).
    offer = game.offer
    offer_give = np.zeros(NUM_RESOURCES, dtype=np.int64)
    offer_want = np.zeros(NUM_RESOURCES, dtype=np.int64)
    offer_proposer = np.array(NO_OWNER, dtype=np.int64)
    offer_answered = np.zeros(players, dtype=np.int64)
    if offer is not None:
        offer_give = np.array(offer.give, dtype=np.int64)
        offer_want = np.array(offer.want, dtype=np.int64)
        offer_proposer = np.array(offer.proposer, dtype=np.int64)
        if seat == offer.proposer:
            declined = set(offer_responders(state, offer)) - set(game.pending_responders)
            for responder in declined:
                offer_answered[responder] = 1

    # The public-knowledge ledger (`hexset_ui.ledger`): each seat's own
    # reconstruction from moves that were public by the rules, never from a
    # steal's hidden identity. `own_hand`/`hand_totals` above already say
    # the same thing at seat and total granularity; this is the per-resource
    # floor a bystander's own reasoning could support, plus what it cannot
    # yet type -- see `PublicLedger`'s own docstring for the proof that this
    # is exactly (not merely approximately) common knowledge.
    ledger_known = np.array(
        [game.ledger.seats[p].known for p in range(players)], dtype=np.int64
    )
    ledger_unknown = np.array(
        [game.ledger.seats[p].unknown for p in range(players)], dtype=np.int64
    )

    return {
        # Board.
        "terrain": np.array(board.terrain, dtype=np.int64),
        "token": np.array(board.tokens, dtype=np.int64),
        "port_code": _port_code(board, board.topology.num_vertices),
        # Position.
        "robber": np.array(state.robber, dtype=np.int64),
        "vertex_owner": np.array(state.vertex_owner, dtype=np.int64),
        "vertex_building": np.array(state.vertex_building, dtype=np.int64),
        "edge_owner": np.array(state.edge_owner, dtype=np.int64),
        "bank": np.array(state.bank, dtype=np.int64),
        "knights_played": np.array(state.knights_played, dtype=np.int64),
        "award_points": np.array(
            [award_points(state, p) for p in range(players)], dtype=np.int64
        ),
        "longest_road_holder": np.array(state.longest_road_holder, dtype=np.int64),
        "largest_army_holder": np.array(state.largest_army_holder, dtype=np.int64),
        "phase": np.array(int(game.phase), dtype=np.int64),
        "free_roads": np.array(game.free_roads, dtype=np.int64),
        "deck_size": np.array(len(state.deck), dtype=np.int64),
        "turns": np.array(game.turns, dtype=np.int64),
        "perspective": np.array(seat, dtype=np.int64),
        # Information set, in board-seat order -- exact for `seat`, totals only
        # for everyone else.
        "own_hand": np.array(state.hands[seat], dtype=np.int64),
        "hand_totals": np.array(
            [sum(state.hands[p]) for p in range(players)], dtype=np.int64
        ),
        "own_dev": np.array(own_dev, dtype=np.int64),
        "dev_totals": np.array(
            [
                sum(state.dev_cards[p]) + sum(state.new_dev_cards[p])
                for p in range(players)
            ],
            dtype=np.int64,
        ),
        # The live trade offer.
        "offer_give": offer_give,
        "offer_want": offer_want,
        "offer_proposer": offer_proposer,
        "offer_answered": offer_answered,
        # The public-knowledge ledger.
        "ledger_known": ledger_known,
        "ledger_unknown": ledger_unknown,
        # Legality.
        "action_mask": action_mask(space, options),
        "pair_mask": pair_mask(options),
    }
