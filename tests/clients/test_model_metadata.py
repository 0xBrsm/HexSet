"""What a checkpoint declares about itself, and what is done with it.

These bounds are the only thing standing between a typo'd export and a hung
seat, and `hexset.clients.modelmeta` is importable without a runtime wheel so they
can be checked on a machine that cannot load a session at all — which is the
usual development machine here.
"""

from __future__ import annotations

import pytest

from hexset.clients.modelmeta import (
    MAX_GATE_PLIES,
    MAX_SIMULATIONS,
    MAX_TRADE_FLOOR,
    MAX_WAVE,
    GateConfig,
    SearchConfig,
    gate_config,
    search_config,
)


def test_a_checkpoint_that_says_nothing_is_played_as_a_single_forward():
    """The default has to be the cheap one: every checkpoint exported before
    this key existed says nothing, and none of them wanted a search."""
    assert search_config({}) == SearchConfig()
    assert not search_config({}).searches


def test_a_checkpoint_asking_for_search_gets_it_with_its_own_budget():
    config = search_config({"search": "mcts", "simulations": "256", "wave": "32"})
    assert config.searches
    assert config.simulations == 256
    assert config.wave == 32


def test_search_settings_are_ignored_unless_the_file_asks_to_be_searched():
    """Otherwise a stale `simulations` left in an export would silently turn a
    policy checkpoint into a search."""
    config = search_config({"simulations": "256", "wave": "32"})
    assert not config.searches
    assert config.simulations == 0


def test_a_searching_checkpoint_that_names_no_budget_takes_the_default():
    config = search_config({"search": "mcts"})
    assert config.simulations == 128
    assert config.wave == 16


@pytest.mark.parametrize(
    "meta",
    [
        {"search": "mcts", "simulations": "10000000"},
        {"search": "mcts", "wave": "10000000"},
    ],
)
def test_an_absurd_budget_is_clamped_rather_than_honoured(meta):
    """A bot is spawned synchronously inside a request. A file asking for ten
    million simulations must not be able to hang the seat it is dealt to."""
    config = search_config(meta)
    assert config.simulations <= MAX_SIMULATIONS
    assert config.wave <= MAX_WAVE


@pytest.mark.parametrize("value", ["", "not-a-number", "-5", "3.5"])
def test_an_unreadable_budget_falls_back_instead_of_failing_the_load(value):
    """A checkpoint is a model first. A bad hint costs the hint, not the
    opponent — the file still plays, at the default budget."""
    config = search_config({"search": "mcts", "simulations": value})
    assert config.searches
    assert config.simulations == 128


def test_a_checkpoint_that_declares_no_gate_is_read_as_unmeasured():
    """The floor is a property of the value head that was exported, and an
    unmeasured one has no resolution to express: strict positivity is then
    the whole gate (`hexset.trading.clears_floor`). Likewise `gate_plies`:
    a file that says nothing rolls no continuation, one forward over the
    exchanged hand, the training-collection default."""
    assert gate_config({}) == GateConfig()
    assert gate_config({}).trade_floor == 0.0
    assert gate_config({}).plies == 0


def test_a_checkpoint_carries_its_own_measured_floor():
    config = gate_config({"trade_floor": "0.0197"})
    assert config.trade_floor == pytest.approx(0.0197)


def test_a_checkpoint_carries_its_own_gate_plies():
    """The rollout is search on the trade decision, same footing as
    `simulations`: a served file asks for it in its own metadata."""
    config = gate_config({"gate_plies": "8"})
    assert config.plies == 8


@pytest.mark.parametrize(
    "meta, plies",
    [
        ({"gate_plies": "10000"}, MAX_GATE_PLIES),
        ({"gate_plies": "-5"}, 0),
    ],
)
def test_an_absurd_gate_plies_is_clamped_rather_than_honoured(meta, plies):
    """The rollout runs synchronously inside a trade ask; an unbounded
    budget would hang the seat rather than gate it."""
    assert gate_config(meta).plies == plies


@pytest.mark.parametrize("value", ["", "not-a-number", "3.5"])
def test_an_unreadable_gate_plies_falls_back_instead_of_failing_the_load(value):
    assert gate_config({"gate_plies": value}).plies == 0


def test_a_stale_row_bound_is_read_without_complaint_and_ignored():
    """`gate_rows` named the most candidates one batched forward scored --
    gone with the continuation rollout it used to bound, since every
    coverable candidate is filtered arithmetically now. A file exported
    before that carries the key regardless; it is simply never read."""
    config = gate_config({"trade_floor": "0.0197", "gate_rows": "64"})
    assert config.trade_floor == pytest.approx(0.0197)


@pytest.mark.parametrize(
    "meta, floor",
    [
        ({"trade_floor": "9.5"}, MAX_TRADE_FLOOR),
        ({"trade_floor": "-0.5"}, 0.0),
    ],
)
def test_an_absurd_gate_setting_is_clamped_rather_than_honoured(meta, floor):
    """A negative floor is refused outright by `hexset.trading.trade_floor_of`,
    so it is pulled to zero here rather than loaded and raised on at the first
    trade event; gains are win probabilities, so nothing above 1.0 is a floor
    any gain could clear."""
    config = gate_config(meta)
    assert config.trade_floor == floor


@pytest.mark.parametrize("value", ["", "not-a-number", "nan"])
def test_an_unreadable_floor_falls_back_instead_of_failing_the_load(value):
    """Same bargain the search budget strikes -- and `nan` in particular would
    compare false against every gain, silently muting the seat's trading."""
    assert gate_config({"trade_floor": value}).trade_floor == 0.0
