"""A bot's `label`: the name the server shows for it, from its preset's
profile or its model file's metadata. Clients, rankings and journals still
name it by its entry."""

from __future__ import annotations

import json
import random
import sys
import types

import pytest

from conftest import new_tables
from hexset.clients.onnxmeta import with_metadata
from hexset.server import api


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def labelled_file(monkeypatch, tmp_path):
    """A model file offered as `mynet`, labelled `MyNet`, whose bot plays at
    random and whose value head rates every seat at a half."""
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "f.onnx").write_bytes(with_metadata(b"\x3a\x00", {"name": "mynet", "label": "MyNet"}))
    monkeypatch.setattr(api, "MODELS_DIR", directory)
    monkeypatch.setattr(api, "_warned", set())

    def spawn(path, board, *, rng=None, device="cpu", players=None, game_type=None, workers=0, pools=None):
        from hexset.bots import RandomBot

        return RandomBot(random.Random(0))

    class Policy:
        def value_rows(self, rows):
            return [tuple(0.5 for _ in range(game.num_players)) for game, _ in rows]

    fake = types.ModuleType("hexset.clients.onnxbot")
    fake.spawn = spawn
    fake.load = lambda path, topology, device="cpu", threads=1: types.SimpleNamespace(policy=Policy())
    monkeypatch.setitem(sys.modules, "hexset.clients.onnxbot", fake)
    return "mynet"


@pytest.fixture
def labelled_preset():
    from hexset.arena import PRESETS, register_preset, unregister_preset

    register_preset("lowbot", PRESETS["test-trader"], profile={"label": "LowBot"})
    yield "lowbot"
    unregister_preset("lowbot")


def test_a_model_files_label_is_listed_and_shown_for_its_seat(labelled_file):
    tables = new_tables()
    listed = tables.handle("GET", "/api/models", {}, None)["game_types"]["standard"]["mynet"]
    assert listed["name"] == "mynet" and listed["label"] == "MyNet"

    dealt = tables.handle("POST", "/api/games", {"bots": ["mynet"]}, None)
    table = tables.get(dealt["code"])
    table.stop_runners()
    [seat] = [s for s in dealt["seats"] if s["kind"] == "bot"]
    assert (seat["name"], seat["label"]) == ("mynet", "MyNet"), "asked for by name, shown by label"
    assert dealt["players"][seat["seat"]]["name"] == "MyNet"
    assert dealt["players"][seat["seat"]]["bot"] == "mynet"
    assert tables.handle("GET", "/api/chance", {}, dealt["token"])["rater"] == "MyNet"


def test_a_seat_swapped_to_a_labelled_preset_is_shown_by_its_label(labelled_preset):
    tables = new_tables()
    dealt = tables.handle("POST", "/api/games", {}, None)
    seat = next(s["seat"] for s in dealt["seats"] if s["kind"] == "empty")
    view = tables.handle("POST", "/api/bot", {"seat": seat, "model": labelled_preset}, dealt["token"])
    tables.get(dealt["code"]).stop_runners()
    assert view["seats"][seat]["label"] == "LowBot"
    assert view["players"][seat]["name"] == "LowBot"


def test_a_bot_without_a_label_carries_none(tmp_path):
    tables = new_tables()
    dealt = tables.handle("POST", "/api/games", {"bots": ["test-trader"]}, None)
    tables.get(dealt["code"]).stop_runners()
    [seat] = [s for s in dealt["seats"] if s["kind"] == "bot"]
    assert "label" not in seat
    assert dealt["players"][seat["seat"]]["name"] == "test-trader"


def test_a_rankings_row_is_named_by_its_bots_label_and_journalled_by_its_entry(tmp_path, labelled_preset):
    def rank(games):
        rows = [{"name": "lowbot", "id": f"bot:{labelled_preset}", "games": 12}]
        return {"boards": [{"key": "all", "title": "All", "columns": [], "rows": rows}]}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank, ranked_bot=labelled_preset)
    tables.deal_ranked("duel-variant", [{"id": "a" * 64, "kind": "api"}], labelled_preset, 1, [None])
    header = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    assert [bot["name"] for bot in header["bots"].values()] == ["lowbot"]
    tables.rankings.wait()
    [row] = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"]
    assert (row["name"], row["id"]) == ("LowBot", "bot:lowbot")


def test_the_shipped_catanatron_preset_is_shown_as_catanatron():
    pytest.importorskip("catanatron.game")
    from hexset.catanatron import bot  # noqa: F401  registers the preset

    assert api.bot_label("catanatron") == "Catanatron"
    assert api.bot_label("catanatron:depth=3") == "catanatron:depth=3"
