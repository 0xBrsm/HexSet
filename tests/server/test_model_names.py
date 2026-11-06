"""The picker names a model file by the `name` it declares, never by its
filename; its `version` is its profile's. One name is one bot, which may be
one file at some game types and another at the rest, each file offered only
at the game types it declares."""

from __future__ import annotations

import random
import sys
import types

import pytest

from conftest import new_tables
from hexset.clients.onnxmeta import with_metadata
from hexset.server import api
from hexset.server.api import ApiError, listed_models, model_catalogue, model_options


def named(**meta: str) -> bytes:
    """An empty graph field and `meta`: enough for the listing to read."""
    return with_metadata(b"\x3a\x00", meta)


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def models(monkeypatch, tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setattr(api, "MODELS_DIR", directory)
    monkeypatch.setattr(api, "_warned", set())
    return directory


@pytest.fixture
def spawned(monkeypatch):
    """Every ONNX spawn the server asks for, as `(path, game_type)`."""
    asked = []

    def spawn(path, board, *, rng=None, device="cpu", players=None, game_type=None, workers=0, pools=None):
        asked.append((path, game_type))
        from hexset.bots import RandomBot

        return RandomBot(random.Random(0))

    fake = types.ModuleType("hexset.clients.onnxbot")
    fake.spawn = spawn
    monkeypatch.setitem(sys.modules, "hexset.clients.onnxbot", fake)
    return asked


def test_a_file_is_offered_by_the_name_it_declares_not_its_filename(models):
    (models / "whatever-file.onnx").write_bytes(named(name="mynet", version="1.0", commit="abc"))
    (models / "unversioned.onnx").write_bytes(named(name="plain"))
    assert model_options()["mynet"] == str(models / "whatever-file.onnx")
    assert model_options()["plain"] == str(models / "unversioned.onnx")
    assert "whatever-file" not in listed_models() and "mynet-1.0" not in listed_models()
    view = new_tables().handle("GET", "/api/models", {}, None)
    assert "mynet" in view["models"]
    assert view["game_types"]["standard"]["mynet"] == {"name": "mynet", "version": "1.0", "commit": "abc"}
    assert "mynet" not in view["game_types"]["duel-variant"], "no modes is standard only"


def test_files_that_name_nothing_are_not_offered(models, capsys):
    (models / "anonymous.onnx").write_bytes(named(version="1.0"))
    (models / "garbage.onnx").write_bytes(b"not a model")
    (models / "mybot-ish.onnx").write_bytes(named(name="test-trader"))
    catalogue = model_catalogue()
    assert set(catalogue["standard"]) == set(catalogue["duel-variant"]) == set(api.registered_presets())
    assert catalogue["standard"]["test-trader"].spec == "test-trader", "a preset keeps its name"
    model_catalogue()
    logged = capsys.readouterr().err
    assert "declares no name" in logged and "not an ONNX model" in logged and "registered bot" in logged
    assert logged.count("declares no name") == 1, "each logged once"


def test_one_name_is_one_file_at_some_game_types_and_another_at_the_rest(models, spawned):
    (models / "a.onnx").write_bytes(named(name="net", modes="standard", version="6.2", commit="x"))
    (models / "b.onnx").write_bytes(named(name="net", modes="duel-variant", version="4d.3", commit="y"))
    registry = new_tables()
    view = registry.handle("GET", "/api/models", {}, None)
    assert [m for m in view["models"] if m.startswith("net")] == ["net"]
    assert view["game_types"]["standard"]["net"]["version"] == "6.2"
    assert view["game_types"]["duel-variant"]["net"]["version"] == "4d.3"
    assert api.bot_profile("net", "standard") == {"version": "6.2"}
    assert api.bot_profile("net", "duel-variant") == {"version": "4d.3"}
    assert api.bot_profile("net", spec=str(models / "b.onnx")) == {"version": "4d.3"}

    registry.handle("POST", "/api/games", {"bots": ["net", "net"]}, None)
    registry.handle("POST", "/api/games", {"bots": ["net"], "game_type": "duel-variant"}, None)
    assert spawned == [
        (str(models / "a.onnx"), "standard"), (str(models / "a.onnx"), "standard"),
        (str(models / "b.onnx"), "duel-variant"),
    ]


def test_an_entry_not_offered_at_the_table_s_game_type_is_refused(models, spawned):
    (models / "a.onnx").write_bytes(named(name="fourseat", version="3.6"))
    registry = new_tables()
    with pytest.raises(ApiError, match="fourseat does not play duel-variant games"):
        registry.handle("POST", "/api/games", {"bots": ["fourseat"], "game_type": "duel-variant"}, None)
    with pytest.raises(ApiError, match="unknown model: fourseat-3.6"):
        registry.handle("POST", "/api/games", {"bots": ["fourseat-3.6"]}, None)

    data = registry.handle("POST", "/api/games", {"game_type": "duel-variant"}, None)
    table = registry.get(data["code"])
    seat = next(i for i in (0, 1) if table.seats[i].kind.name == "EMPTY")
    with pytest.raises(ApiError, match="does not play duel-variant"):
        registry.handle("POST", "/api/bot", {"seat": seat, "model": "fourseat"}, data["token"])
    assert spawned == []


def test_two_files_offering_one_name_at_one_game_type_are_neither_offered_there(models, capsys):
    (models / "a.onnx").write_bytes(named(name="net", version="1", modes="standard,duel-variant"))
    (models / "b.onnx").write_bytes(named(name="net", version="2"))
    catalogue = model_catalogue()
    assert "net" not in catalogue["standard"], "both offer it at the standard game"
    assert catalogue["duel-variant"]["net"].spec == str(models / "a.onnx"), "only one offers it in duels"
    assert "none is offered there" in capsys.readouterr().err


def test_a_replaced_file_is_read_again(models):
    path = models / "a.onnx"
    path.write_bytes(named(name="old"))
    assert "old" in model_options()
    path.write_bytes(named(name="newer", version="2"))
    assert "newer" in model_options() and "old" not in model_options()


def test_a_file_may_declare_a_profile_held_to_a_clients_rules(models, capsys):
    (models / "a.onnx").write_bytes(named(name="mynet", version="1.0", commit="abc", owner="octocat",
                                          url="https://example.org/mynet", description="A small net."))
    (models / "b.onnx").write_bytes(named(name="other", version="1", owner="octocat", url="ftp://x",
                                          description="x" * 101))
    listed = new_tables().handle("GET", "/api/models", {}, None)["game_types"]["standard"]
    assert listed["mynet"] == {"name": "mynet", "version": "1.0", "commit": "abc", "owner": "octocat",
                                   "url": "https://example.org/mynet", "description": "A small net."}
    assert listed["other"] == {"name": "other", "version": "1", "owner": "octocat"}
    logged = capsys.readouterr().err
    assert "declares a url it may not" in logged and "declares a description it may not" in logged
    assert api.bot_profile("mynet") == {"version": "1.0", "owner": "octocat",
                                            "url": "https://example.org/mynet", "description": "A small net."}
    assert api.bot_profile("test-trader") == {} and api.bot_profile("nothing") == {}


def test_a_preset_is_offered_only_at_the_game_types_preset_modes_names(models, monkeypatch):
    monkeypatch.setattr(api, "PRESET_MODES", {"test-trader": ("standard",)})
    catalogue = model_catalogue()
    assert "test-trader" in catalogue["standard"]
    assert "test-trader" not in catalogue["duel-variant"]
    others = set(api.registered_presets()) - {"test-trader"}
    assert others <= set(catalogue["duel-variant"]), "a preset not named is offered everywhere"
    assert "test-trader" in listed_models()
    registry = new_tables()
    with pytest.raises(ApiError, match="test-trader does not play duel-variant games"):
        registry.handle("POST", "/api/games", {"bots": ["test-trader"], "game_type": "duel-variant"}, None)


def test_preset_modes_are_held_to_listed_presets_and_dealt_game_types():
    import argparse

    from hexset.server.web import preset_modes

    parser = argparse.ArgumentParser()
    held = preset_modes(parser, ["test-trader=standard, standard"])
    assert held == {"test-trader": ("standard",)}
    for bad in ("test-trader", "test-trader=", "no-such-bot=standard", "test-trader=fiveseat"):
        with pytest.raises(SystemExit):
            preset_modes(parser, [bad])
    with pytest.raises(SystemExit):
        preset_modes(parser, ["test-trader=standard", "test-trader=duel-variant"])
