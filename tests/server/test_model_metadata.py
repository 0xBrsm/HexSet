"""What a checkpoint declares about itself, and what is done with it.

These bounds are the only thing standing between a typo'd export and a hung
seat, and `hexset.server.modelmeta` is importable without a runtime wheel so they
can be checked on a machine that cannot load a session at all — which is the
usual development machine here.
"""

from __future__ import annotations

import pytest

from hexset.server.modelmeta import MAX_SIMULATIONS, MAX_WAVE, SearchConfig, search_config


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
