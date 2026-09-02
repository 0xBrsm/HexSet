from __future__ import annotations

import random

import numpy as np

from hexset.actions import build_space, within_offer_budget
from hexset.server.rules import options_for
from hexset.board.board import random_base_board
from hexset.game import start, to_move
from hexset.onnx_record import record_from_game

from conftest import step_randomly


def _redistribute(hand: list[int]) -> None:
    """Move one unit between two resources, in place, total unchanged."""
    src = next(i for i, n in enumerate(hand) if n > 0)
    dst = (src + 1) % len(hand)
    hand[src] -= 1
    hand[dst] += 1


def test_record_hides_an_opponents_exact_hand_and_dev_composition():
    """Two states differing only in *how* an opponent's cards are made up --
    never in the totals -- must produce byte-identical records. If any array
    here changed, the record would be exposing exactly the hidden card
    identity `encoding.py`'s information-set guarantee exists to keep out.
    """
    board = random_base_board(random.Random(0))
    rng = random.Random(1)
    game = start(board, 3, rng)

    for _ in range(80):
        if game.won_by is not None:
            break
        step_randomly(game, rng)
    assert game.won_by is None

    state = game.state
    seat = to_move(game)
    opponent = next(p for p in range(state.num_players) if p != seat)
    assert sum(state.hands[opponent]) > 0, "nothing to redistribute -- pick a longer playout"

    topology = state.board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, state.num_players
    )
    options = tuple(within_offer_budget(game, options_for(game), None))
    before = record_from_game(game, seat, space, options)

    _redistribute(state.hands[opponent])
    held = state.dev_cards[opponent]
    if any(held):
        _redistribute(held)

    after = record_from_game(game, seat, space, options)

    assert before.keys() == after.keys()
    for key in before:
        assert np.array_equal(before[key], after[key]), key
