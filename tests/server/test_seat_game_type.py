"""A bot is seated knowing the table's game type (`Entrant.game_type`), and a
preset registered `listed=False` is left out of the picker but still seats.
"""

from __future__ import annotations

import random

import pytest

from conftest import new_tables
from traders import ScarcityTrader, TRADE

from hexset.arena import (
    Entrant, entrant_from_name, register_entrant_kind, register_preset,
    registered_presets, unregister_entrant_kind, unregister_preset,
)
from hexset.server.api import listed_models

SEATED: list[str | None] = []


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def seated():
    """`test-seated` (listed) and `test-hidden` (unlisted), both recording the
    game type their entrant was built for."""
    def build(entrant, board, rng):
        SEATED.append(entrant.game_type)
        return ScarcityTrader(rng, TRADE)

    SEATED.clear()
    register_entrant_kind("test-seated", build)
    register_preset("test-seated", Entrant("test-seated", kind="test-seated"))
    register_preset("test-hidden", Entrant("test-hidden", kind="test-seated"), listed=False)
    yield
    unregister_preset("test-seated")
    unregister_preset("test-hidden")
    unregister_entrant_kind("test-seated")


def deal(registry, **body):
    data = registry.handle("POST", "/api/games", body, None)
    return data, registry.get(data["code"])


def test_an_unlisted_preset_is_not_offered_but_still_resolves(seated):
    assert "test-seated" in registered_presets() and "test-seated" in listed_models()
    assert "test-hidden" not in registered_presets() and "test-hidden" not in listed_models()
    assert entrant_from_name("test-hidden").kind == "test-seated"
    unregister_preset("test-hidden")
    assert "test-hidden" not in registered_presets()
    register_preset("test-hidden", Entrant("test-hidden", kind="test-seated"))
    assert "test-hidden" in registered_presets(), "unregistering forgets that it was unlisted"
    unregister_preset("test-hidden")
    register_preset("test-hidden", Entrant("test-hidden", kind="test-seated"), listed=False)


@pytest.mark.parametrize("game_type", ["standard", "duel-variant"])
def test_a_dealt_bot_is_built_for_the_tables_game_type(seated, game_type):
    deal(new_tables(), game_type=game_type, bots=["test-seated"])
    assert SEATED == [game_type]


def test_a_bot_seated_later_is_built_for_the_tables_game_type(seated):
    registry = new_tables()
    data, _ = deal(registry, game_type="standard", bots=[])
    registry.handle("POST", "/api/bot", {"seat": 1, "model": "test-seated"}, data["token"])
    assert SEATED == ["standard"]


def test_a_resumed_table_builds_its_bots_for_the_game_type_it_was_dealt(seated, tmp_path):
    data, _ = deal(new_tables(games_dir=str(tmp_path), seed=5), game_type="duel-variant", bots=["test-seated"])
    assert SEATED == ["duel-variant"]
    new_tables(games_dir=str(tmp_path), seed=5).get(data["code"])
    assert SEATED == ["duel-variant", "duel-variant"]
