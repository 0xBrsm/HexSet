# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.mcts.Search(whole=True)`: a node that opens a decision of several
moves is searched over the policy's whole options, each edge applying the
whole decision; `choose` plays the chosen one a move at a time, answering the
asks inside it from the rest without searching; and with `whole` off nothing
changes."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from helpers import clear_hand, give

from hexset import decisions as D
from hexset.actions import ActionType, apply, legal_actions
from hexset.arena import _mcts
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.clients.netbot import LeafEvaluator
from hexset.game import Phase, roll_dice, start, to_move
from hexset.mcts import Search


def _score(option) -> float:
    """A fixed score per whole option, the same in every process."""
    return (hash(D.key(option)) % 1_000_003) / 1_000_003 * 6.0


class Whole:
    """Lists whole options scored by `_score` (`option_rows`); `score_rows` is
    their softmax marginalised onto first moves; values are each seat's hand
    size over 10."""

    has_options = True

    def _listed(self, game, seat):
        listed = tuple(D.options(game, seat, double_turn=False))
        scores = np.array([_score(o) for o in listed])
        probs = np.exp(scores - scores.max())
        return listed, probs / probs.sum()

    def option_rows(self, rows):
        return [self._listed(game, seat) for game, seat in rows]

    def score_rows(self, rows):
        out = []
        for game, seat, options in rows:
            listed, probs = self._listed(game, seat)
            mass = {}
            for option, p in zip(listed, probs):
                first = D.first_move(option)
                mass[first] = mass.get(first, 0.0) + p
            prior = np.array([mass.get(o, 0.0) for o in options]) + 1e-9
            out.append((prior / prior.sum(), self.values(game)))
        return out

    def value_rows(self, rows):
        return [self.values(game) for game, _ in rows]

    @staticmethod
    def values(game):
        return tuple(sum(hand) / 10 for hand in game.state(0, hidden=False).hands)


def opening(seed: int = 4):
    return start(random_base_board(random.Random(seed)), 4, random.Random(seed))


def discarding(seed: int = 0):
    """Seat 0 owing half of ten cards to a seven, nobody else owing."""
    game = opening(seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    for seat in range(4):
        clear_hand(game._state, seat)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    game.phase, game.current_player = Phase.ROLL, 0
    roll_dice(game, 7)
    assert game.phase is Phase.DISCARD and game.discard_quota[0] == 5
    return game


def search(whole: bool, simulations: int = 48) -> Search:
    return Search(LeafEvaluator(policy=Whole()), simulations=simulations, wave=8, whole=whole,
                  rng=random.Random(5))


@pytest.mark.parametrize("make, kind", [(opening, D.CompoundKind.SETUP), (discarding, D.CompoundKind.DISCARD)],
                         ids=["setup turn", "discard"])
def test_a_decision_of_several_moves_is_searched_over_its_whole_options(make, kind):
    game = make()
    root, options, visits = search(True).run(game)
    assert options and all(isinstance(o, D.Compound) and o.kind is kind for o in options)
    assert set(map(D.key, options)) == {D.key(o) for o in D.options(game, to_move(game), double_turn=False)}
    assert visits.sum() == pytest.approx(48)
    # Each edge searched lands past the whole decision, not inside it.
    for option, child in zip(root.options, root.children):
        if child is not None:
            assert not D.inside(child.game)
            assert child.game.phase is not Phase.SETUP_ROAD


def test_with_whole_off_every_node_searches_atomic_moves():
    game = opening()
    _, options, _ = search(False).run(game)
    assert all(not isinstance(o, D.Compound) for o in options)
    assert all(o.type is ActionType.SETUP_SETTLEMENT for o in options)


def test_choose_plays_a_whole_option_a_move_at_a_time():
    game = opening()
    searcher = search(True)
    first = searcher.choose(game)
    assert first.type is ActionType.SETUP_SETTLEMENT
    decision = searcher.decision_log[-1]
    chosen = decision.options[decision.chosen]
    assert isinstance(chosen, D.Compound) and chosen.parts[0] == first
    apply(game, first)
    road = searcher.choose(game)
    assert road == chosen.parts[1]
    assert len(searcher.decision_log) == 1      # the road came from the plan, unsearched
    apply(game, road)
    searcher.choose(game)                       # the next seat's turn is searched again
    assert len(searcher.decision_log) == 2


def test_a_discard_is_chosen_whole_and_finished_from_the_plan():
    game = discarding()
    searcher = search(True)
    decision_parts = None
    while game.phase is Phase.DISCARD:
        move = searcher.choose(game)
        if decision_parts is None:
            decision = searcher.decision_log[-1]
            decision_parts = decision.options[decision.chosen].parts
        assert move.type is ActionType.DISCARD
        apply(game, move, 0)
    assert len(searcher.decision_log) == 1
    assert sum(game._state.hands[0]) == 5


def test_a_whole_decision_logs_its_options_as_json():
    game = opening()
    searcher = search(True)
    searcher.choose(game)
    logged = json.loads(json.dumps(searcher.decision_log[-1].to_json()))
    assert logged["searched"]
    assert all(entry[0] == "SETUP" and isinstance(entry[1], list) for entry in logged["actions"])


def test_a_whole_search_needs_an_evaluator_that_lists_options():
    class Atomic(Whole):
        has_options = False

    with pytest.raises(ValueError, match="whole options"):
        Search(LeafEvaluator(policy=Atomic()), whole=True)


def test_the_mcts_spec_reads_whole():
    entrant = _mcts("mcts:net.onnx@64:whole=1")
    assert entrant.option("whole") is True and entrant.name == "mcts64whole"
    assert _mcts("mcts:net.onnx@64").option("whole", False) is False
    with pytest.raises(ValueError, match="whole=1"):
        _mcts("mcts:net.onnx@64:whole=2")
