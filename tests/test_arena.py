# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pickle

import pytest

from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.arena import (
    compete,
    entrant_from_name,
    lineup_from_names,
)


def test_a_lineup_the_rotation_cannot_balance_is_refused():
    with pytest.raises(ValueError, match="divide evenly"):
        compete(lineup_from_names(["random"] * 4), 6)


def test_a_checkpoint_path_names_an_entrant_wherever_a_preset_would():
    lineup = lineup_from_names(
        ["network:/tmp/latest.pt", "network:/tmp/latest.pt", "random", "random"]
    )
    assert [e.name for e in lineup] == [
        "network#0",
        "network#1",
        "random#0",
        "random#1",
    ]
    assert lineup[0].kind == "network"
    assert lineup[0].weights == "/tmp/latest.pt"
    # Entrants are descriptions because they have to reach a worker.
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_an_mcts_entrant_names_its_determinized_worlds():
    """`k` is `Entrant.k`: worlds sampled from the mover's belief and searched
    per decision."""
    worlds = entrant_from_name("mcts:/tmp/x.pt@256w64:k=4")
    assert (worlds.name, worlds.simulations, worlds.wave, worlds.k) == (
        "mcts256w64k4",
        256,
        64,
        4,
    )
    alone = entrant_from_name("mcts:/tmp/x.pt@:k=8")
    assert (alone.name, alone.simulations, alone.k) == ("mctsk8", 128, 8)


def test_a_network_spec_can_switch_its_own_trading_off():
    """`@0` is the only meaningful value: an off switch, not a budget. A
    budget is a thing a checkpoint declares in its own metadata."""
    plain = entrant_from_name("network:/tmp/x.pt")
    assert plain.trade is None
    assert plain.name == "network"
    assert plain.weights == "/tmp/x.pt"

    capped = entrant_from_name("network:/tmp/x.pt@0")
    assert capped.trade.max_offers == 0
    assert capped.name == "network-notrade"
    assert capped.weights == "/tmp/x.pt"

    with pytest.raises(ValueError, match="no-trade switch"):
        entrant_from_name("network:/tmp/x.pt@2")


def test_a_runtime_registers_its_bots_in_a_fresh_interpreter():
    """A bot hexset does not ship is unknown until its runtime is loaded, and
    nameable everywhere a preset is once it is."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    here = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-c",
         "from hexset.arena import entrant_from_name, load_runtime\n"
         "try:\n"
         "    entrant_from_name('test-trader')\n"
         "except ValueError as error:\n"
         "    assert 'runtime' in str(error)\n"
         "else:\n"
         "    raise AssertionError('named before its runtime was loaded')\n"
         "load_runtime('traders')\n"
         "assert entrant_from_name('test-trader').kind == 'test-trader'\n"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(here.parent), str(here)])},
    )
    assert result.returncode == 0, result.stderr


def test_a_registered_spec_parses_its_own_names():
    from hexset.arena import Entrant, register_spec, unregister_spec

    def parse(name):
        return Entrant(name, kind="test-trader", options=(("depth", int(name.split(":")[1])),))

    register_spec("deep:", parse)
    try:
        entrant = pickle.loads(pickle.dumps(entrant_from_name("deep:3")))
        assert (entrant.kind, entrant.option("depth"), entrant.option("width", 6)) == (
            "test-trader", 3, 6,
        )
        assert [e.name for e in lineup_from_names(["deep:3", "deep:3~test-trader"])] == [
            "deep:3", "deep:3~test-trader",
        ]
        with pytest.raises(ValueError, match="unknown bots"):
            lineup_from_names(["shallow:3"])
    finally:
        unregister_spec("deep:")
    with pytest.raises(ValueError, match="unknown bot"):
        entrant_from_name("deep:3")


def test_a_name_is_registered_once_and_shipped_names_never():
    """A second, different registration under a name raises rather than
    silently changing what a lineup means; the same one again is harmless."""
    from hexset.arena import (
        Entrant, register_entrant_kind, register_preset, register_spec,
        unregister_entrant_kind, unregister_preset, unregister_spec,
    )

    def factory(entrant, board, rng):
        raise AssertionError("never built")

    def other(entrant, board, rng):
        raise AssertionError("never built")

    def parse(name):
        return Entrant(name, kind="test-once")

    register_entrant_kind("test-once", factory)
    register_preset("test-once", Entrant("test-once", kind="test-once"))
    register_spec("test-once:", parse)
    try:
        register_entrant_kind("test-once", factory)
        register_preset("test-once", Entrant("test-once", kind="test-once"))
        register_spec("test-once:", parse)
        with pytest.raises(ValueError, match="already registered"):
            register_entrant_kind("test-once", other)
        with pytest.raises(ValueError, match="already registered"):
            register_preset("test-once", Entrant("test-once", kind="test-once", depth=3))
        with pytest.raises(ValueError, match="already registered"):
            register_spec("test-once:", lambda name: Entrant(name, kind="test-once"))
    finally:
        unregister_entrant_kind("test-once")
        unregister_preset("test-once")
        unregister_spec("test-once:")
    register_entrant_kind("test-once", other)  # free again once unregistered
    unregister_entrant_kind("test-once")

    for name in ("random", "retired", "catanatron"):
        with pytest.raises(ValueError, match="ships"):
            register_entrant_kind(name, factory)
        with pytest.raises(ValueError, match="ships"):
            register_preset(name, Entrant(name, kind="test-once"))
        with pytest.raises(ValueError, match="ships"):
            unregister_preset(name)
    for prefix in ("network:", "mcts:", "catanatron:", "network:x", "net"):
        with pytest.raises(ValueError, match="parses itself"):
            register_spec(prefix, parse)
    with pytest.raises(ValueError, match="parses itself"):
        unregister_spec("mcts:")
    assert entrant_from_name("random").kind == "random"


def test_the_checkpoint_kinds_take_whichever_loader_was_installed_last():
    """`network`/`mcts` build from the process's checkpoint loader, and a
    runtime installing another replaces it rather than colliding."""
    from hexset.arena import Entrant, register_entrant_kind, spawn, unregister_entrant_kind
    from hexset.board.board import random_base_board
    import random

    board = random_base_board(random.Random(0))
    try:
        register_entrant_kind("network", lambda entrant, board, rng: "first")
        register_entrant_kind("network", lambda entrant, board, rng: "second")
        assert spawn(Entrant("n", kind="network", weights="x"), board, random.Random(0)) == "second"
    finally:
        unregister_entrant_kind("network")
    with pytest.raises(ValueError, match="register_entrants"):
        spawn(Entrant("n", kind="network", weights="x"), board, random.Random(0))


def test_the_longest_registered_prefix_parses_and_a_preset_comes_first():
    from hexset.arena import Entrant, register_spec, unregister_spec

    register_spec("test-p:", lambda name: Entrant(name, kind="test-trader", depth=1))
    register_spec("test-p:deep:", lambda name: Entrant(name, kind="test-trader", depth=2))
    register_spec("test-tr", lambda name: Entrant(name, kind="test-trader", depth=3))
    try:
        assert entrant_from_name("test-p:x").depth == 1
        assert entrant_from_name("test-p:deep:x").depth == 2
        # `test-trader` is a preset, so the spec never sees it.
        assert entrant_from_name("test-trader").depth == 2
        assert entrant_from_name("test-tr:x").depth == 3
    finally:
        for prefix in ("test-p:", "test-p:deep:", "test-tr"):
            unregister_spec(prefix)


def test_an_entrants_options_are_one_setting_whatever_their_order():
    """Two entrants naming the same settings are equal, hash equal and so
    journal as one run, however the settings were spelled."""
    from hexset.arena import Entrant

    pairs = Entrant("x", kind="x", options=(("b", 1), ("a", 2)))
    mapping = Entrant("x", kind="x", options={"a": 2, "b": 1})
    assert pairs == mapping and hash(pairs) == hash(mapping)
    assert pairs.options == (("a", 2), ("b", 1))
    assert repr(pairs) == repr(mapping)
    assert pickle.loads(pickle.dumps(pairs)) == pairs
    with pytest.raises(ValueError, match="more than once"):
        Entrant("x", kind="x", options=(("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="pair"):
        Entrant("x", kind="x", options=(("a", 1, 2),))


def test_the_public_surface_is_what_the_module_names():
    import hexset.arena as arena

    assert all(hasattr(arena, name) for name in arena.__all__)
    assert not [name for name in arena.__all__ if name.startswith("_")]


def test_the_runs_bargaining_settings_reach_every_game(monkeypatch):
    """A policy is only read back in the environment it was trained against."""
    import hexset.arena as arena

    seen = []
    original = arena.play_game

    def spy(game, bots, **kwargs):
        seen.append(kwargs.get("trade_mode"))
        return original(game, bots, **kwargs)

    monkeypatch.setattr(arena, "play_game", spy)
    compete(lineup_from_names(["random"] * 2), 2, workers=1, action_cap=40, trade_mode="auto")
    assert seen == ["auto"] * 2
    seen.clear()
    compete(lineup_from_names(["random"] * 2), 2, workers=1, action_cap=40)
    assert seen == ["round"] * 2, "the mechanism is all a run names"


def test_compete_reports_from_the_calling_process_with_workers():
    seen = []
    tournament = compete(
        lineup_from_names(["random"] * 2), 2, workers=2, turn_cap=UNSTRUCTURED_TURN_CAP,
        progress=lambda done, games, outcome: seen.append((done, outcome.winner)),
    )
    assert [done for done, _ in seen] == [1, 2]
    assert [winner for _, winner in seen] == list(tournament.winners)


# --- who opens, and the antithetic complement ---------------------------------

OPENERS: list[str] = []


class _Opener:
    """Takes the first legal action and notes who it is: under `action_cap=1`
    the only seat a game asks is the one that opens."""

    def __init__(self, name):
        self.name = name

    def choose(self, game):
        from hexset.actions import options_for

        OPENERS.append(self.name)
        return options_for(game)[0]


@pytest.fixture
def opener_kind():
    from hexset.arena import register_entrant_kind, unregister_entrant_kind

    register_entrant_kind("test-opener", _opener)
    OPENERS.clear()
    try:
        yield
    finally:
        unregister_entrant_kind("test-opener")


def _opener(entrant, board, rng):
    return _Opener(entrant.name)


def _sides(geometry):
    from hexset.arena import PRESETS, Entrant

    return [
        Entrant(slot, kind="test-opener") if slot in "ab" else PRESETS[slot]
        for slot in geometry.split(",")
    ]


@pytest.mark.parametrize("geometry,complement", [
    ("a,b,retired,retired", (1, 0, 2, 3)),
    ("a,retired,b,retired", None),
])
@pytest.mark.parametrize("antithetic", [True, False])
def test_every_playing_entrant_opens_equally_often(opener_kind, geometry, complement, antithetic):
    """A retired seat 0 used to pass the first turn to the lowest playing
    seat, giving side A of `a,b,retired,retired` three openings in four."""
    compete(
        _sides(geometry), 16, workers=1, action_cap=1, antithetic=antithetic,
        complement=complement if antithetic else None,
    )
    assert sorted(OPENERS) == ["a"] * 8 + ["b"] * 8


def test_a_full_table_opens_on_seat_zero_under_the_old_rotation():
    """Nobody retired: the seating and the opener are exactly what they were
    before the opener rotated, so such a run's games are unchanged."""
    from hexset.arena import PRESETS, deal_seats, half_turn

    for seats in (2, 3, 4):
        lineup = [PRESETS["random"]] * seats
        for index in range(4 * seats):
            pair, half = divmod(index, 2)
            rotation = pair + half * (seats // 2)
            old = tuple((e + rotation) % seats for e in range(seats))
            assert deal_seats(lineup, index, half_turn(seats)) == (pair, old, 0)
            plain = tuple((e + index) % seats for e in range(seats))
            assert deal_seats(lineup, index) == (index, plain, 0)


def test_the_default_complement_is_refused_where_it_moves_a_retired_seat():
    with pytest.raises(ValueError, match="retired entrant onto a playing seat"):
        compete(lineup_from_names(["random", "random", "retired", "retired"]), 4, action_cap=1)
    with pytest.raises(ValueError, match="not a permutation"):
        compete(lineup_from_names(["random"] * 2), 2, action_cap=1, complement=(0, 0))
    with pytest.raises(ValueError, match="antithetic=True"):
        compete(lineup_from_names(["random"] * 2), 2, action_cap=1, antithetic=False,
                complement=(1, 0))


@pytest.mark.parametrize("geometry,complement", [
    ("abab", (1, 0, 3, 2)),
    ("abrr", (1, 0, 2, 3)),
])
def test_the_second_game_of_a_pair_trades_the_two_sides_seats(geometry, complement):
    """`abab` under the half-turn shift maps each side onto itself, and
    `a,b,random,random` moves side A onto a third party's seat; the duel's
    side swap exchanges exactly the two sides."""
    from hexset.arena import PRESETS, deal_seats

    lineup = [PRESETS["random"]] * 4
    a = [e for e, slot in enumerate(geometry) if slot == "a"]
    b = [e for e, slot in enumerate(geometry) if slot == "b"]
    others = [e for e, slot in enumerate(geometry) if slot not in "ab"]
    for pair in range(4):
        _, one, _ = deal_seats(lineup, 2 * pair, complement)
        _, two, _ = deal_seats(lineup, 2 * pair + 1, complement)
        assert {two[e] for e in a} == {one[e] for e in b}
        assert {two[e] for e in b} == {one[e] for e in a}
        assert [two[e] for e in others] == [one[e] for e in others]
