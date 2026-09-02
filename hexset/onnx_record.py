# SPDX-License-Identifier: GPL-3.0-only
"""The information-set record: the actual boundary between the rules and the
model, and a traceable torch encoder that reads it.

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
and can safely live inside the traced graph instead.

`record_from_game`/`record_batch` build the record from a live `Game`, for
tests and for generating realistic export/parity samples -- not for shipping
to hexset-ui, which builds its own from the rules it already has (phase 3 of
the plan lives in that repo, not this one). `RecordEncoder` is the traced
half: `record -> (hexes, vertices, edges, globals)`, the same four tensors
`hexset.encoding.encode` produces and `HexNet` consumes.

Two properties of `hexset.encoding` carry over unchanged, because they are
properties of the *feature layout*, not of who computes it:

*Seat-relative.* Everything is rotated so the perspective seat is column/slot
0. `RecordEncoder` does the rotation now (`_rotate_slot` mirrors
`hexset.encoding._seat`); the record itself stays in board-seat order, which
is what makes `perspective` meaningful as an input rather than baked in.

*Information-set correct, by construction upstream of here.* `own_hand` and
`own_dev` are exact only for the perspective seat; every other seat con-
tributes a total alone (`hand_totals`, `dev_totals`), plus `hexset.ledger`'s
public-knowledge reconstruction of their composition (`ledger_known`,
`ledger_unknown`) -- itself never sharper than what the public log could
have derived, whichever seat is asking. `RecordEncoder` never sees a hidden
card -- there is nowhere in the record for one to be.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .actions import ActionSpace, legal_actions
from .actions import pair_mask as _pair_mask_of
from .board.board import Board, pips
from .board.terrain import NUM_RESOURCES
from .cards import DECK_SIZE
from .encoding import (
    BANK_SCALE,
    HAND_SCALE,
    NUM_BUILDINGS,
    NUM_PHASES,
    NUM_TERRAIN,
    TURN_SCALE,
    StaticGraph,
)
from .game import Game, to_move
from .state import NO_OWNER
from .trading import responders as offer_responders
from .victory import award_points

# Every field name in the record, in the order the plan's table lists them
# (board, then position, then information set, then legality). `_shapes` in
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
    "offer_give",
    "offer_want",
    "offer_proposer",
    "offer_answered",
    "ledger_known",
    "ledger_unknown",
    "action_mask",
    "pair_mask",
)


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
    game: Game, perspective: int | None, space: ActionSpace
) -> dict[str, np.ndarray]:
    """The information-set record for `perspective`, as a single (unbatched)
    row per field. `perspective` defaults to `to_move(game)`, same as
    `hexset_ui.onnxbot.Request.seat` -- masks and the offer budget are
    properties of whoever is to move, and encoding from a different seat
    would pair a stranger's legal moves with someone else's hand."""
    state = game.state
    players = state.num_players
    if perspective is None:
        perspective = to_move(game)
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")

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

    # The live trade offer, filtered exactly as `encoding._offer_parts` filters
    # it: give/want and the proposer are public while an offer stands; who has
    # answered is the proposer's information only (part 3 of the trading design
    # approximates simultaneous responses, so a responder must not condition on
    # earlier declines). Board-seat order here, like every other field — the
    # rotation to seat-relative happens inside `RecordEncoder`.
    offer = game.offer
    offer_give = np.zeros(NUM_RESOURCES, dtype=np.int64)
    offer_want = np.zeros(NUM_RESOURCES, dtype=np.int64)
    offer_proposer = np.int64(NO_OWNER)
    offer_answered = np.zeros(players, dtype=np.int64)
    if offer is not None:
        offer_give = np.asarray(offer.give, dtype=np.int64)
        offer_want = np.asarray(offer.want, dtype=np.int64)
        offer_proposer = np.int64(offer.proposer)
        if perspective == offer.proposer:
            declined = set(offer_responders(state, offer)) - set(game.pending_responders)
            for seat in declined:
                offer_answered[seat] = 1

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
        "offer_give": offer_give,
        "offer_want": offer_want,
        "offer_proposer": offer_proposer,
        "offer_answered": offer_answered,
        "ledger_known": ledger_known,
        "ledger_unknown": ledger_unknown,
        "action_mask": mask,
        "pair_mask": _pair_mask_of(options),
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


def _onehot(index: Tensor, num_classes: int) -> Tensor:
    """`(..., num_classes)` one-hot from an equality broadcast rather than
    `F.one_hot`, so it traces to a plain `Equal` the exporter has never
    wobbled on, whatever the batch shape."""
    classes = torch.arange(num_classes, device=index.device, dtype=index.dtype)
    return (index.unsqueeze(-1) == classes).to(torch.float32)


def _rotate_slot(owner: Tensor, perspective: Tensor, players: int) -> Tensor:
    """Owner-as-stored -> seat-relative column, `players` for `NO_OWNER`.

    Mirrors `hexset.encoding._seat` (`(seat - perspective) % players`) plus the
    "`NO_OWNER` sits last" convention `_slots` encodes as a lookup table --
    written here as `where`/`remainder` because a `Tensor` cannot walk a
    Python list.
    """
    if perspective.dim() < owner.dim():
        perspective = perspective.unsqueeze(-1)
    slot = torch.remainder(owner - perspective, players)
    return torch.where(owner == NO_OWNER, torch.full_like(slot, players), slot)


# Ways to roll each token with two dice, scaled and cast to float32 exactly
# once -- as a plain Python computation, not a tensor op -- so this table is
# bit-identical to `hexset.encoding._template_by_value`'s
# `pips(token) / MAX_TOKEN_PIPS` for every entry. Reusing `hexset.board.board.pips`
# rather than re-deriving `6 - abs(7 - token)` here keeps the formula in one
# place, per the standing rule against a second definition of a rules number.
_MAX_TOKEN = 12
MAX_TOKEN_PIPS = 5


def _pips_table() -> list[float]:
    return [pips(token) / MAX_TOKEN_PIPS for token in range(_MAX_TOKEN + 1)]


class RecordEncoder(nn.Module):
    """`record -> (hexes, vertices, edges, globals)`: the traceable half of
    `hexset.encoding.encode`/`encode_batch`.

    No data-dependent control flow -- every branch below is a `where` or a
    comparison evaluated on every row, never a Python `if` on a tensor value
    -- which is exactly what tracing requires and what the original numpy
    encoder already had none of.

    Board adjacency (`StaticGraph`) plays no part here: that is baked into
    `HexNet`'s own buffers, unchanged by this work, and only `num_hexes` is
    needed, to size the robber one-hot.
    """

    def __init__(self, graph: StaticGraph, players: int) -> None:
        super().__init__()
        self.players = players
        self.num_hexes = graph.num_hexes
        self.register_buffer(
            "pips_table", torch.tensor(_pips_table(), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        terrain: Tensor,
        token: Tensor,
        port_code: Tensor,
        robber: Tensor,
        vertex_owner: Tensor,
        vertex_building: Tensor,
        edge_owner: Tensor,
        bank: Tensor,
        knights_played: Tensor,
        award_points_: Tensor,
        longest_road_holder: Tensor,
        largest_army_holder: Tensor,
        phase: Tensor,
        free_roads: Tensor,
        deck_size: Tensor,
        turns: Tensor,
        perspective: Tensor,
        own_hand: Tensor,
        hand_totals: Tensor,
        own_dev: Tensor,
        dev_totals: Tensor,
        offer_give: Tensor,
        offer_want: Tensor,
        offer_proposer: Tensor,
        offer_answered: Tensor,
        ledger_known: Tensor,
        ledger_unknown: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        players = self.players

        # --- hexes: terrain one-hot, has-token, pips/5, is-robber ---
        terrain_onehot = _onehot(terrain, NUM_TERRAIN)
        has_token = (token != 0).to(torch.float32)
        pips_scaled = self.pips_table[token]
        is_robber = _onehot(robber, self.num_hexes)
        hexes = torch.cat(
            [
                terrain_onehot,
                has_token.unsqueeze(-1),
                pips_scaled.unsqueeze(-1),
                is_robber.unsqueeze(-1),
            ],
            dim=-1,
        )

        # --- vertices: building one-hot, owner slot one-hot, port block ---
        building_onehot = _onehot(vertex_building, NUM_BUILDINGS)
        owner_slot = _rotate_slot(vertex_owner, perspective, players)
        owner_onehot = _onehot(owner_slot, players + 1)
        is_generic_port = (port_code == 0).to(torch.float32)
        specific_resource = torch.clamp(port_code - 1, min=0)
        has_specific_port = (port_code >= 1).to(torch.float32)
        specific_port_onehot = _onehot(specific_resource, NUM_RESOURCES) * has_specific_port.unsqueeze(-1)
        vertices = torch.cat(
            [
                building_onehot,
                owner_onehot,
                is_generic_port.unsqueeze(-1),
                specific_port_onehot,
            ],
            dim=-1,
        )

        # --- edges: owner slot one-hot ---
        edge_slot = _rotate_slot(edge_owner, perspective, players)
        edges = _onehot(edge_slot, players + 1)

        # --- building points, already in seat-relative slot order because
        # `owner_onehot`'s columns already are: mirrors
        # `encoding._building_points`, a contraction of the vertex block
        # rather than a second board walk.
        building_value = torch.arange(NUM_BUILDINGS, dtype=torch.float32, device=vertices.device)
        per_vertex_value = building_onehot @ building_value
        building_points = torch.einsum("bv,bvp->bp", per_vertex_value, owner_onehot[..., :players])

        # --- globals, in exactly `encoding._encode_globals`'s part order ---
        seats = [torch.remainder(perspective + i, players) for i in range(players)]

        def gather_seat(values: Tensor, seat: Tensor) -> Tensor:
            return values.gather(1, seat.unsqueeze(1)).squeeze(1)

        def gather_seat_vec(values: Tensor, seat: Tensor) -> Tensor:
            """`gather_seat` for a `(B, players, width)` tensor -- the
            per-seat `width`-wide row instead of a scalar."""
            width = values.shape[-1]
            index = seat.view(-1, 1, 1).expand(-1, 1, width)
            return values.gather(1, index).squeeze(1)

        parts: list[Tensor] = [own_hand.to(torch.float32) / HAND_SCALE]
        parts.append(
            torch.stack(
                [gather_seat(hand_totals, seats[i]).to(torch.float32) / HAND_SCALE for i in range(1, players)],
                dim=1,
            )
        )
        parts.append(bank.to(torch.float32) / BANK_SCALE)
        parts.append(own_dev.to(torch.float32) / 5.0)
        parts.append(
            torch.stack(
                [gather_seat(dev_totals, seats[i]).to(torch.float32) / 5.0 for i in range(1, players)],
                dim=1,
            )
        )
        parts.append(
            torch.stack(
                [gather_seat(knights_played, seats[i]).to(torch.float32) / 5.0 for i in range(players)],
                dim=1,
            )
        )
        award_seat = torch.stack(
            [gather_seat(award_points_, seats[i]).to(torch.float32) for i in range(players)], dim=1
        )
        parts.append((building_points + award_seat) / 10.0)

        for holder in (longest_road_holder, largest_army_holder):
            parts.append(_onehot(_rotate_slot(holder, perspective, players), players + 1))

        parts.append(_onehot(phase, NUM_PHASES))
        parts.append((free_roads.to(torch.float32) / 2.0).unsqueeze(-1))
        parts.append((deck_size.to(torch.float32) / DECK_SIZE).unsqueeze(-1))
        parts.append(torch.clamp(turns.to(torch.float32) / TURN_SCALE, max=1.0).unsqueeze(-1))

        # --- live trade offer, in exactly `encoding._offer_parts`'s order:
        # give, want, proposer one-hot, answered. `_rotate_slot` maps a
        # NO_OWNER proposer (no offer standing) to slot `players`, which a
        # `players`-wide one-hot renders as all zeros — the same all-zero
        # block the numpy path writes.
        parts.append(offer_give.to(torch.float32) / HAND_SCALE)
        parts.append(offer_want.to(torch.float32) / HAND_SCALE)
        parts.append(_onehot(_rotate_slot(offer_proposer, perspective, players), players))
        parts.append(
            torch.stack(
                [gather_seat(offer_answered, seats[i]).to(torch.float32) for i in range(players)],
                dim=1,
            )
        )

        # --- the public-knowledge ledger, in exactly `encoding._ledger_parts`'s
        # order: each opponent's known[5] then unknown, seat-relative, own
        # seat excluded (own hand is already exact via `own_hand` above).
        for i in range(1, players):
            parts.append(gather_seat_vec(ledger_known, seats[i]).to(torch.float32) / HAND_SCALE)
            parts.append(
                (gather_seat(ledger_unknown, seats[i]).to(torch.float32) / HAND_SCALE).unsqueeze(-1)
            )

        globals_ = torch.cat(parts, dim=-1)

        return hexes, vertices, edges, globals_
