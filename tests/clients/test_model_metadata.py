"""What a checkpoint declares about itself, and what is done with it. These
bounds are all that stands between a typo'd export and a hung seat, and
`hexset.clients._modelmeta` imports without a runtime wheel.
"""

from __future__ import annotations

import pytest

from hexset.clients._modelmeta import (
    MAX_SIMULATIONS,
    MAX_TOP,
    MAX_WAVE,
    MAX_WORLDS,
    RolloutConfig,
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


def test_a_rollout_file_gets_its_worlds_candidates_horizon_and_decisive_prior():
    config = search_config(
        {"search": "rollout", "worlds": "32", "top": "6", "decisive": "0.97", "horizon": "8"}
    )
    assert config.searches
    assert config.simulations == 0
    assert config.rollout == RolloutConfig(worlds=32, top=6, horizon=8)
    assert config.decisive == 0.97


def test_a_rollout_file_that_says_nothing_more_gets_the_rollouts_own_defaults():
    """`hexset.rollout.PairedRollouts`: 32 worlds, the top 3 moves, every
    root searched, played to the end."""
    config = search_config({"search": "rollout"})
    assert config.rollout == RolloutConfig(worlds=32, top=3, horizon=None)
    assert config.decisive is None
    assert search_config({"search": "rollout", "horizon": ""}).rollout.horizon is None


def test_rollout_settings_are_ignored_unless_the_file_asks_for_rollouts():
    keys = {"worlds": "32", "top": "6", "horizon": "8"}
    assert not search_config(keys).searches
    assert search_config(keys).rollout is None
    mcts = search_config({**keys, "search": "mcts", "simulations": "64"})
    assert mcts.rollout is None and mcts.simulations == 64


def test_an_absurd_rollout_budget_is_clamped_and_an_unreadable_one_falls_back():
    config = search_config({"search": "rollout", "worlds": "10000000", "top": "10000000"})
    assert config.rollout.worlds == MAX_WORLDS
    assert config.rollout.top == MAX_TOP
    unreadable = search_config({"search": "rollout", "worlds": "many", "top": "-5"})
    assert unreadable.rollout == RolloutConfig(worlds=32, top=3)
    assert search_config({"search": "rollout", "top": "1"}).rollout.top == 2


@pytest.mark.parametrize("value", ["0", "-1", "eight", "8.5"])
def test_an_unreadable_horizon_fails_the_load(value):
    """A horizon changes what the search is: read wrong, it would play every
    rollout to the end, at several times the cost."""
    with pytest.raises(ValueError, match="horizon"):
        search_config({"search": "rollout", "horizon": value})


def test_a_rollout_file_can_ask_for_its_opening_searched_in_full():
    assert search_config({"search": "rollout", "setup": "full"}).rollout.full_setup
    assert not search_config({"search": "rollout"}).rollout.full_setup
    assert not search_config({"search": "rollout", "setup": ""}).rollout.full_setup
    with pytest.raises(ValueError, match="setup"):
        search_config({"search": "rollout", "setup": "half"})


def test_a_rollout_file_reads_decisive_as_a_search_does_and_refuses_a_calibration():
    with pytest.raises(ValueError, match="decisive"):
        search_config({"search": "rollout", "decisive": "1.2"})
    with pytest.raises(ValueError, match="own_calibration"):
        search_config({"search": "rollout", "own_calibration": "0.32,0.81"})


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



def test_a_searched_file_can_ask_for_its_own_seat_recalibrated():
    assert search_config({"search": "mcts", "own_calibration": "0.32,0.81"}).own_calibration == (0.32, 0.81)
    assert search_config({"search": "mcts"}).own_calibration is None
    assert search_config({"own_calibration": "0.32,0.81"}).own_calibration is None   # not searched


@pytest.mark.parametrize("value", ["0.3", "a,b", "0.3,0", "0.3,-1", "nan,1"])
def test_an_unreadable_calibration_fails_the_load(value):
    with pytest.raises(ValueError):
        search_config({"search": "mcts", "own_calibration": value})


def test_a_searched_file_can_ask_to_play_its_confident_roots_unsearched():
    assert search_config({"search": "mcts", "decisive": "0.95"}).decisive == 0.95
    assert search_config({"search": "mcts"}).decisive is None
    assert search_config({"decisive": "0.95"}).decisive is None   # not searched
    for bad in ("0", "1.2", "sure"):
        with pytest.raises(ValueError, match="decisive"):
            search_config({"search": "mcts", "decisive": bad})
