import pytest

from hexset.bots.evaluate import Survey
from hexset.bots.heximax.port_progress import port_aware_progress


def survey(*ratios: int) -> Survey:
    return Survey(0, 0, 0, 0, 0, 0, 0, 0, 1, tuple(ratios))


def test_specific_port_conversion_completes_purchase():
    # Resource 0 has a 2:1 port.  The extra copy converts into the missing
    # resource 3, taking settlement progress from 3/4 to a complete purchase.
    assert port_aware_progress(
        [3, 1, 1, 0, 0], survey(2, 4, 4, 4, 4), deck_left=20,
        bank=[0, 0, 0, 1, 0],
    ) == 1.0


def test_bank_availability_is_enforced():
    # Without a received card in the bank, the apparent port conversion is
    # illegal and raw progress is preserved.
    assert port_aware_progress(
        [3, 1, 1, 0, 0], survey(2, 4, 4, 4, 4), deck_left=20,
        bank=[0, 0, 0, 0, 0],
    ) == 0.75


def test_conversion_search_terminates_without_arbitrage():
    # Every trade loses at least one card, so a hand cannot loop by converting
    # back and forth.  This also exercises multiple successive conversions.
    value = port_aware_progress(
        [8, 0, 0, 0, 0], survey(2, 3, 3, 3, 3), deck_left=20,
        bank=[0, 8, 8, 8, 8],
    )
    assert 0.0 <= value <= 1.0


def test_port_aware_scalar_and_batch_scoring_match():
    import numpy as np
    import random
    from hexset.board.board import random_base_board
    from hexset.bots.heximax.evaluate import HonestEvaluator
    from hexset.state import Building, new_game

    board = random_base_board(random.Random(0))
    state = new_game(board, 3, random.Random(1))
    vertex = board.ports[0].vertices[0]
    state.vertex_owner[vertex] = 0
    state.vertex_building[vertex] = Building.SETTLEMENT
    state.hands[0] = [0, 1, 1, 4, 0]
    evaluator = HonestEvaluator(board, port_aware=True)
    hands = np.asarray([state.hands, [[0] * 5 for _ in range(3)]], dtype=float)

    batch = evaluator.score_many(state, 0, hands)
    for row in range(len(hands)):
        for seat in range(state.num_players):
            scalar = evaluator.score(
                state, seat, hands[row, seat], knower=0,
            )
            assert batch[row, seat] == pytest.approx(scalar, abs=1e-12)


def test_port_aware_evaluation_cache_tracks_bank_changes():
    import random
    from hexset.board.board import random_base_board
    from hexset.bots.heximax.evaluate import HonestEvaluator
    from hexset.ledger import PublicLedger
    from hexset.state import Building, new_game

    board = random_base_board(random.Random(0))
    state = new_game(board, 3, random.Random(1))
    vertex = board.ports[0].vertices[0]
    state.vertex_owner[vertex] = 0
    state.vertex_building[vertex] = Building.SETTLEMENT
    state.hands[0] = [0, 1, 1, 4, 0]
    evaluator = HonestEvaluator(board, port_aware=True)
    belief = evaluator.belief_for(state, PublicLedger.new(3), 0)
    before = evaluator.evaluate(state, 0, belief)
    state.bank[4] = 0
    after = evaluator.evaluate(state, 0, belief)
    assert after[0] < before[0]
