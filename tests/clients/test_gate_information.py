"""A network gate prices trades from its own seat's information set, through
the real record path (`V2Policy` over `record_from_game`).

The graph is a stand-in whose value is a hash of every input it is fed, so
any field that moves moves the value. The probe redeals the *other* seats'
hidden cards among themselves, composition-preserving: the gate's seat sees
a bit-identical position, so nothing it answers may move -- not its gains,
not its estimates of the counterparty, not a value row of its own -- whether
it is the seat to move or a seat answering another's offer.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import ActionSpace, build_space, options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.clients.netbot import bot_for  # noqa: E402
from hexset.clients.onnxbot import V2Policy  # noqa: E402
from hexset.game import Phase, imagine, is_over, observe, start, to_move  # noqa: E402
from hexset.onnx_record import NO_MOVE, RECORD_FIELDS, record_from_game  # noqa: E402
from hexset.seat import Seat  # noqa: E402
from hexset.state import copy_state  # noqa: E402
from hexset.trading import ENUMERATION_CARDS, UNLIMITED, Offer  # noqa: E402
from hexset.trading._engine import _belief_candidates  # noqa: E402
from hexset.view import View  # noqa: E402
from conftest import step_randomly  # noqa: E402

PLAYERS = 4


class HashGraph:
    """An `onnxruntime.InferenceSession` stand-in: declares every record
    field with a dynamic batch axis and answers `value` as a hash of each
    row's inputs, one number in `[0, 1)` per seat."""

    def get_inputs(self):
        return [SimpleNamespace(name=name, shape=["batch"]) for name in RECORD_FIELDS]

    def run(self, outputs, inputs):
        assert outputs == ["value"], outputs
        rows = len(inputs["action_mask"])
        value = np.empty((rows, PLAYERS))
        for i in range(rows):
            digest = hashlib.sha256()
            for name in sorted(inputs):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(inputs[name][i]).tobytes())
            value[i] = [b / 256 for b in digest.digest()[:PLAYERS]]
        return [value]


@dataclass(frozen=True)
class HashCheckpoint:
    policy: V2Policy
    space: ActionSpace
    players: int = PLAYERS
    trade_params: object = UNLIMITED


def checkpoint(board) -> HashCheckpoint:
    space = build_space(
        board.topology.num_vertices, board.topology.num_edges, board.topology.num_hexes, PLAYERS
    )
    return HashCheckpoint(policy=V2Policy(HashGraph(), space), space=space)


def positions(wanted=3, seed=5):
    """Main-phase positions with several untyped cards in the other seats'
    hands, so a redeal has something to move."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), PLAYERS, rng)
    found = []
    for _ in range(3000):
        if is_over(game) or len(found) == wanted:
            break
        if game.phase is Phase.MAIN and len(options_for(game)) > 3:
            mover = to_move(game)
            if sum(View.from_game(game, mover).unknown) >= 4:
                found.append(imagine(game, random.Random(1), randomize_deck=False))
        step_randomly(game, rng)
    assert len(found) == wanted, f"only {len(found)} usable positions"
    return found


def permuted(game, seat, rng):
    """`game` with every seat but `seat` redealt its hidden resource and
    development cards among the others, each pile's size preserved."""
    twin = imagine(game, random.Random(1), randomize_deck=False)
    state = copy_state(game.state(0, hidden=False))
    seats = [s for s in range(PLAYERS) if s != seat]

    def redeal(piles):
        pool = [c for s in seats for c, n in enumerate(piles[s]) for _ in range(n)]
        rng.shuffle(pool)
        cursor = 0
        for s in seats:
            size = sum(piles[s])
            fresh = [0] * len(piles[s])
            for card in pool[cursor : cursor + size]:
                fresh[card] += 1
            cursor += size
            piles[s] = fresh

    redeal(state.hands)
    redeal(state.dev_cards)
    twin.set_state(state)
    return twin


def answers(board, game, seat, counterparties):
    """Everything the gate at `seat` answers about `game`: its value row, and
    its gains and estimates over every candidate with each counterparty."""
    bot = bot_for(checkpoint(board), rng=random.Random(0))
    bot.seat_at(game)
    view = game.state(seat)
    candidates = [
        (them, bundle)
        for them in counterparties
        for bundle in _belief_candidates(view, seat, them, ENUMERATION_CARDS)
    ][:40]
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]
    return (
        bot.policy.value_rows([(game, seat)]),
        bot.gains_many(view, received, thems),
        bot.estimate_many(view, candidates),
    )


@pytest.mark.parametrize("role", ["mover", "responder"])
def test_redealing_hidden_cards_does_not_move_what_the_gate_answers(role):
    moved = []
    for truth in positions():
        board = truth.state(0, hidden=False).board
        mover = to_move(truth)
        if role == "mover":
            seat, counterparties = mover, [s for s in range(PLAYERS) if s != mover]
        else:
            seat, counterparties = (mover + 1) % PLAYERS, [mover]
        twin = permuted(truth, seat, random.Random(11))
        assert View.from_game(truth, seat) == View.from_game(twin, seat)
        here = answers(board, truth, seat, counterparties)
        there = answers(board, twin, seat, counterparties)
        assert any(g != -1.0 for g in here[1]), "no candidate was priced at all"
        parts = [name for name, a, b in zip(("value", "gains", "estimates"), here, there) if a != b]
        if parts:
            moved.append((seat, parts))
    assert not moved, f"the gate's answers moved with the other seats' hidden cards: {moved}"


def test_a_record_for_a_seat_not_to_move_masks_no_other_seats_moves():
    """The fixed mask, not the mover's: it is all the graph needs to
    normalise over, and it says nothing about anybody's cards."""
    truth = positions(wanted=1)[0]
    board = truth.state(0, hidden=False).board
    space = checkpoint(board).space
    idle = (to_move(truth) + 1) % PLAYERS
    record = record_from_game(truth, idle, space)
    assert np.flatnonzero(record["action_mask"]).tolist() == [space.index(NO_MOVE)]
    mover = to_move(truth)
    own = record_from_game(truth, mover, space)
    assert own["action_mask"].sum() == len(options_for(truth))


@pytest.mark.parametrize("role", ["mover", "responder"])
def test_a_network_gate_prices_an_offer_at_a_hosted_seat(role):
    """A hosted seat's game holds every other seat's cards as hidden piles,
    so any read of their identity raises `HiddenRead`."""
    truth = positions(wanted=1)[0]
    board = truth.state(0, hidden=False).board
    mover = to_move(truth)
    seat = mover if role == "mover" else (mover + 1) % PLAYERS
    bot = bot_for(checkpoint(board), rng=random.Random(0))
    hosted = Seat(observe(truth, seat), seat, bot)
    # The seat seats its gate itself, before any move it is asked for.
    assert bot._seated is hosted.game

    if role == "mover":
        others = [s for s in range(PLAYERS) if s != seat]
        bundles = [
            b for them in others
            for b in _belief_candidates(hosted.game.state(seat), seat, them, ENUMERATION_CARDS)
        ][:20]
        gains = hosted.gains(bundles, others[0])
        assert len(gains) == len(bundles)
        hosted.offer()
    else:
        mine = _belief_candidates(hosted.game.state(seat), seat, mover, ENUMERATION_CARDS)
        assert hosted.gains(mine[:20], mover)
        # An offer towards the mover is the negation of one towards this seat.
        hosted.answer(Offer(mover, tuple(-n for n in mine[0])))
