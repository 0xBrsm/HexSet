"""What a checkpoint declares about itself, and what is done with it. These
bounds are all that stands between a typo'd export and a hung seat, and
`hexset.clients._modelmeta` imports without a runtime wheel.
"""

from __future__ import annotations

import pytest

from hexset.clients._modelmeta import (
    MAX_SIMULATIONS,
    MAX_WAVE,
    gate_config,
    search_config,
    trader_config,
    trader_of,
)
from hexset.trading import UNLIMITED
from hexset.trading._params import MAX_GATE_PLIES, MAX_TRADE_FLOOR


def test_a_checkpoint_asking_for_search_gets_it_with_its_own_budget():
    config = search_config({"search": "mcts", "simulations": "256", "wave": "32"})
    assert config.searches
    assert config.simulations == 256
    assert config.wave == 32


def test_search_settings_are_ignored_unless_the_file_asks_to_be_searched():
    config = search_config({"simulations": "256", "wave": "32"})
    assert not config.searches
    assert config.simulations == 0


def test_an_absurd_budget_is_clamped_rather_than_honoured():
    """A bot is spawned synchronously inside a request, so an absurd budget must
    not be able to hang the seat it is dealt to.
    """
    config = search_config({"search": "mcts", "simulations": "10000000", "wave": "10000000"})
    assert config.simulations <= MAX_SIMULATIONS
    assert config.wave <= MAX_WAVE


@pytest.mark.parametrize("value", ["not-a-number", "-5"])
def test_an_unreadable_budget_falls_back_instead_of_failing_the_load(value):
    config = search_config({"search": "mcts", "simulations": value})
    assert config.searches
    assert config.simulations == 128


def test_a_checkpoint_that_declares_no_gate_is_read_as_unmeasured():
    """An unmeasured value head has no resolution to express, so strict
    positivity is the whole gate (`hexset.trading.clears_floor`).
    """
    assert gate_config({}) == UNLIMITED
    assert gate_config({}).trade_floor == 0.0
    assert gate_config({}).gate_plies == 0


def test_a_checkpoint_carries_its_own_measured_floor():
    config = gate_config({"trade_floor": "0.0197"})
    assert config.trade_floor == pytest.approx(0.0197)


def test_an_absurd_gate_plies_is_clamped_rather_than_honoured():
    assert gate_config({"gate_plies": "10000"}).gate_plies == MAX_GATE_PLIES


def test_an_unreadable_gate_plies_falls_back_instead_of_failing_the_load():
    assert gate_config({"gate_plies": "not-a-number"}).gate_plies == 0


@pytest.mark.parametrize(
    "meta, floor",
    [
        ({"trade_floor": "9.5"}, MAX_TRADE_FLOOR),
        ({"trade_floor": "-0.5"}, 0.0),
    ],
)
def test_an_absurd_gate_setting_is_clamped_rather_than_honoured(meta, floor):
    """`trade_floor_of` refuses a negative floor outright, so it is pulled to
    zero here rather than raised on at the first trade event. Gains are win
    probabilities, so no floor above 1.0 is clearable.
    """
    config = gate_config(meta)
    assert config.trade_floor == floor


@pytest.mark.parametrize("value", ["not-a-number", "nan"])
def test_an_unreadable_floor_falls_back_instead_of_failing_the_load(value):
    """`nan` in particular would compare false against every gain, silently
    muting the seat's trading.
    """
    assert gate_config({"trade_floor": value}).trade_floor == 0.0


def test_a_checkpoint_names_its_trader_or_trades_for_itself():
    assert trader_config({"trader": "some-bot"}) == "some-bot"
    assert trader_config({}) is None and trader_config({"trader": ""}) is None

    class Old:
        pass

    assert trader_of(Old()) is None
