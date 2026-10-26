from __future__ import annotations

import random
from pathlib import Path

import pytest

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.clients._modelmeta import finish_decisions_config
from hexset.clients.finishing import Finishing, finished, inside
from hexset.game import Phase, start, to_move

FIXTURE_V2 = Path(__file__).parent / "fixtures" / "stub-contract6.onnx"


class PhasePolicy:
    """A policy whose value says only whether a setup road is owed (seat 0
    reads 1.0 there, 0.0 elsewhere) and whose prior favours its options'
    first; it counts the rows it is asked."""

    has_prior = True

    def __init__(self) -> None:
        self.scored = 0

    def _value(self, game):
        return (1.0 if game.phase is Phase.SETUP_ROAD else 0.0, 0.0, 0.0, 0.0)

    def score_rows(self, rows):
        self.scored += len(rows)
        out = []
        for game, _seat, options in rows:
            prior = [0.0] * len(options)
            prior[0] = 1.0
            out.append((prior, self._value(game)))
        return out

    def value_rows(self, rows):
        return [self._value(game) for game, _seat in rows]

    def label(self) -> str:
        return "phase"


def _after_settlement():
    rng = random.Random(3)
    game = start(random_base_board(rng), 4, rng)
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT))
    assert game.phase is Phase.SETUP_ROAD
    return game


def test_a_position_inside_a_setup_turn_is_read_where_the_turn_is_finished():
    game = _after_settlement()
    seat = to_move(game)
    policy = PhasePolicy()
    wrapped = Finishing(policy)
    assert inside(game)
    assert policy.value_rows([(game, seat)])[0][0] == 1.0
    assert wrapped.value_rows([(game, seat)])[0][0] == 0.0
    assert game.phase is Phase.SETUP_ROAD  # the position itself is untouched


def test_the_prior_is_read_at_the_position_itself():
    game = _after_settlement()
    seat = to_move(game)
    options = tuple(legal_actions(game))
    ((prior, value),) = Finishing(PhasePolicy()).score_rows([(game, seat, options)])
    assert len(prior) == len(options) and prior[0] == 1.0
    assert value[0] == 0.0


def test_finishing_plays_the_policys_own_most_probable_road():
    game = _after_settlement()
    first_road = tuple(legal_actions(game))[0]
    done = finished(PhasePolicy(), game)
    assert not inside(done)
    assert done._state.edge_owner[first_road.a] == to_move(game)


def test_a_position_outside_a_decision_passes_through():
    rng = random.Random(5)
    game = start(random_base_board(rng), 4, rng)
    assert not inside(game)
    policy = PhasePolicy()
    wrapped = Finishing(policy)
    options = tuple(legal_actions(game))
    assert wrapped.score_rows([(game, to_move(game), options)]) == policy.score_rows([(game, to_move(game), options)])
    assert wrapped.label() == "phase"


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("0", False), ("", False)])
def test_the_metadata_key_reads_as_a_flag(value, expected):
    assert finish_decisions_config({"finish_decisions": value}) is expected
    assert finish_decisions_config({}) is False


def test_a_file_asking_for_it_is_loaded_finishing(tmp_path):
    pytest.importorskip("onnxruntime")
    from hexset.clients.onnxbot import _load_cached, load
    from hexset.clients.onnxmeta import with_metadata

    path = tmp_path / "finishing.onnx"
    path.write_bytes(with_metadata(FIXTURE_V2.read_bytes(), {"finish_decisions": "1"}))
    board = random_base_board(random.Random(0))
    try:
        assert isinstance(load(str(path), board.topology).policy, Finishing)
        assert not isinstance(load(str(FIXTURE_V2), board.topology).policy, Finishing)
    finally:
        _load_cached.cache_clear()
