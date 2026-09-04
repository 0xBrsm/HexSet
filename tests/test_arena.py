# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle
import random

import pytest

from hexset.arena import (
    Entrant,
    compete,
    entrant_from_name,
    lineup_from_names,
    spawn,
)
from hexset.board.board import random_base_board


def test_a_lineup_the_rotation_cannot_balance_is_refused():
    with pytest.raises(ValueError, match="divide evenly"):
        compete(lineup_from_names(["random"] * 4), 6)


def test_a_tournament_needs_opponents():
    with pytest.raises(ValueError, match="at least two"):
        compete(lineup_from_names(["random"]), 4)


def test_repeated_bots_are_numbered_and_unknown_ones_rejected():
    named = [e.name for e in lineup_from_names(["greedy", "random", "greedy"])]
    assert named == ["greedy#0", "random", "greedy#1"]
    with pytest.raises(ValueError, match="unknown bots: mcts"):
        lineup_from_names(["mcts", "random"])


def test_a_checkpoint_path_names_an_entrant_wherever_a_preset_would():
    lineup = lineup_from_names(
        ["network:/tmp/latest.pt", "network:/tmp/latest.pt", "greedy", "greedy"]
    )
    assert [e.name for e in lineup] == [
        "network#0",
        "network#1",
        "greedy#0",
        "greedy#1",
    ]
    assert lineup[0].kind == "network"
    assert lineup[0].weights == "/tmp/latest.pt"
    # The whole reason entrants are descriptions: this has to reach a worker.
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_a_network_entrant_trades_unless_the_spec_says_otherwise():
    """`None` is the engine's default: trading on. `@0` is the off switch."""
    assert entrant_from_name("network:/tmp/latest.pt").max_trades is None
    assert entrant_from_name("network:/tmp/latest.pt@0").max_trades == 0


def test_an_mcts_entrant_names_its_simulation_and_wave_budgets():
    plain = entrant_from_name("mcts:/tmp/x.pt")
    assert (plain.kind, plain.weights, plain.simulations, plain.wave) == (
        "mcts",
        "/tmp/x.pt",
        128,
        16,
    )

    sized = entrant_from_name("mcts:/tmp/x.pt@32")
    assert (sized.name, sized.simulations, sized.wave) == ("mcts32", 32, 16)

    batched = entrant_from_name("mcts:/tmp/x.pt@256w64")
    assert (batched.name, batched.simulations, batched.wave) == (
        "mcts256w64",
        256,
        64,
    )


def test_an_unknown_bot_kind_is_refused():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="unknown bot kind"):
        spawn(Entrant("bogus", kind="oracle"), board, random.Random(0))


def test_entrants_are_picklable_so_they_can_cross_a_process():
    """The reason entrants are data and not closures."""
    lineup = lineup_from_names(["greedy", "search2"])
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_a_network_spec_can_self_impose_an_offer_budget():
    """`network:<path>@<offers>`, mirroring `mcts:<path>@<simulations>`.

    The switch exists so a duel can price trading against itself; without a
    suffix a network entrant trades (`max_trades=None`), and `@0` is the one
    meaningful value -- the engine has no trade budget to tune, only an off
    switch.
    """
    plain = entrant_from_name("network:/tmp/x.pt")
    assert plain.max_trades is None
    assert plain.name == "network"
    assert plain.weights == "/tmp/x.pt"

    capped = entrant_from_name("network:/tmp/x.pt@0")
    assert capped.max_trades == 0
    assert capped.name == "network-trades0"
    assert capped.weights == "/tmp/x.pt"
