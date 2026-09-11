# SPDX-License-Identifier: GPL-3.0-only
"""The study must measure an interaction, not a main effect or seat bias."""
from copy import deepcopy

import pytest

from hexset.bench.trade_curve_study import GRID, analyze


def block():
    return [dict(
        alpha=a, initiate=r, respond=r, gate_floor=0, unfinished=0,
        focal_trades_by_game=[0] * 4, table_trades_by_game=[0] * 4,
        experiment=dict(settings=dict(seed=10, games=4), outcomes=[
            dict(index=i, board_index=i // 2, seating=[0, 1, 2, 3],
                 winner=1, points=[5, 10, 3, 3]) for i in range(4)]),
    ) for r, a in GRID]


def set_wins(cells, key, wins):
    cell = next(c for c in cells if (c['initiate'], c['alpha']) == key)
    for row, win in zip(cell['experiment']['outcomes'], wins):
        row['winner'] = 0 if win else 1


def test_primary_is_change_in_weight_advantage_with_positive_sign():
    cells = block()
    set_wins(cells, (1, 1), [1, 1, 1, 1])
    set_wins(cells, (0, 1), [0, 0, 1, 1])
    result = analyze([cells])
    estimate = result['primary_interaction']['win_rate']
    assert estimate['mean'] == 0.5
    assert estimate['samples'] == 2
    assert estimate['lower'] < 0 < estimate['upper']


def test_constant_weight_advantage_cancels_out_of_interaction():
    cells = block()
    for rate in (0, 1):
        set_wins(cells, (rate, 1), [1, 1, 0, 0])
    result = analyze([cells])
    assert result['primary_interaction']['win_rate']['mean'] == 0
    assert result['within_regime'][0]['win_rate']['mean'] == -0.5


def test_complete_blocks_accumulate_independent_board_samples():
    first = block()
    second = deepcopy(first)
    for cell in second:
        cell['experiment']['settings']['seed'] += 1
    result = analyze([first, second])
    assert result['boards'] == 4
    assert all(c['games'] == 8 for c in result['cells'])


def test_partial_grid_cannot_create_unpaired_inference():
    with pytest.raises(ValueError, match='complete grid'):
        analyze([block()[:-1]])


@pytest.mark.parametrize('change', ['seed', 'floor', 'seating'])
def test_mismatched_cells_are_rejected(change):
    cells = block()
    if change == 'seed':
        cells[-1]['experiment']['settings']['seed'] += 1
    elif change == 'floor':
        cells[-1]['gate_floor'] = 0.0197
    else:
        cells[-1]['experiment']['outcomes'][0]['seating'] = [1, 0, 2, 3]
    with pytest.raises(ValueError, match='paired'):
        analyze([cells])



def test_holdout_does_not_reselect_using_validation_winners():
    from hexset.bench.trade_curve_holdout import validate
    train, held = block(), block()
    for cell in held:
        cell['experiment']['settings']['seed'] = 11
    for rate in (0, 0.5, 1):
        set_wins(train, (rate, 1 if rate == 0 else 0), [1] * 4)
        set_wins(held, (rate, 0 if rate == 0 else 1), [1] * 4)
    result = validate([train, held], split=1)
    assert result['selected_static_alpha'] == 0
    assert result['selected_adaptive_alphas'] == {'0.0': 1, '0.5': 0, '1.0': 0}
    assert result['holdout_wins'] == {'trading_baseline': 8, 'static': 4, 'adaptive': 0}
    assert result['comparisons']['adaptive_minus_static']['mean'] == pytest.approx(-1 / 3)
    assert result['comparisons']['adaptive_minus_static']['samples'] == 2


def test_holdout_cannot_reuse_training_seed():
    from hexset.bench.trade_curve_holdout import validate
    with pytest.raises(ValueError, match='distinct seeds'):
        validate([block(), block()], split=1)
