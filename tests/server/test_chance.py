"""`GET /api/chance`: a seat's chance of winning as the table's model rates
it, read from that seat's own side, once per table version, and never at a
ranked table."""

from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest

from conftest import new_tables
from hexset.clients.onnxmeta import with_metadata
from hexset.actions import Action, ActionType, legal_actions
from hexset.game import Phase, to_move
from hexset.server.wire import action_to_wire
from hexset.server import api
from hexset.server.api import ApiError

FIXTURES = Path(__file__).parent.parent / "clients" / "fixtures"


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
def stub_model(models, monkeypatch):
    """A model file `rater` whose value head answers seat `s` with `0.1 * (s
    + 1)`, every read it is asked for kept as `(path, seat)`."""
    (models / "rater.onnx").write_bytes(with_metadata(b"\x3a\x00", {"name": "rater"}))
    asked = []

    class Policy:
        def value_rows(self, rows):
            out = []
            for game, seat in rows:
                asked.append(seat)
                out.append(tuple(0.1 * (s + 1) for s in range(game.num_players)))
            return out

    def spawn(path, board, *, rng=None, device="cpu", players=None, game_type=None, workers=0, pools=None):
        from hexset.bots import RandomBot

        return RandomBot(random.Random(0))

    def load(path, topology, device="cpu", threads=1):
        return types.SimpleNamespace(policy=Policy())

    fake = types.ModuleType("hexset.clients.onnxbot")
    fake.spawn, fake.load = spawn, load
    monkeypatch.setitem(sys.modules, "hexset.clients.onnxbot", fake)
    return asked


def _dealt(registry, bots):
    data = registry.handle("POST", "/api/games", {"bots": bots}, None)
    table = registry.get(data["code"])
    table.stop_runners()
    return table, data["token"], data["seat"]


def test_the_seat_is_rated_from_its_own_side(stub_model):
    registry = new_tables()
    table, token, seat = _dealt(registry, ["test-trader", "rater"])
    got = registry.handle("GET", "/api/chance", {}, token)
    assert got == {"version": table.version, "rater": "rater", "chance": round(0.1 * (seat + 1), 4)}
    assert stub_model == [seat], "one read, from the asking seat"


def test_an_unchanged_table_is_not_read_again(stub_model):
    registry = new_tables()
    table, token, _ = _dealt(registry, ["rater"])
    first = registry.handle("GET", "/api/chance", {}, token)
    assert registry.handle("GET", "/api/chance", {}, token) == first
    assert len(stub_model) == 1
    table.bump()
    assert registry.handle("GET", "/api/chance", {}, token)["version"] == first["version"] + 1
    assert len(stub_model) == 2


def test_a_table_without_a_model_has_no_rater(stub_model):
    registry = new_tables()
    _, token, _ = _dealt(registry, ["test-trader", "test-trader"])
    assert registry.handle("GET", "/api/chance", {}, token)["rater"] is None
    assert stub_model == []


def _played(tmp_path, moves: int = 120):
    """A table with a `rater` seat, `moves` random moves into its game, every
    one journalled under `tmp_path`."""
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    table, token, seat = _dealt(registry, ["test-trader", "test-trader", "rater"])
    session, rng = table.session, random.Random(4)
    for _ in range(moves):
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        session.submit(to_move(session.game), action_to_wire(rng.choice(legal_actions(session.game))))
    table.bump()
    return registry, table, token, seat


def _ends_at(monkeypatch, table):
    """Make `table`'s game read as over where it stands, and every position
    before it as still in play."""
    turns = table.session.game.turns
    monkeypatch.setattr(api, "is_over", lambda game: game.turns >= turns)
    return turns


def test_a_finished_game_reads_its_last_position_still_in_play(stub_model, monkeypatch, tmp_path):
    registry, table, token, seat = _played(tmp_path)
    turns = _ends_at(monkeypatch, table)
    seen = []
    real = api.win_chance
    monkeypatch.setattr(api, "win_chance", lambda spec, game, s, config: seen.append(game.turns) or real(spec, game, s, config))
    got = registry.handle("GET", "/api/chance", {}, token)
    assert got == {"version": table.version, "rater": "rater", "chance": round(0.1 * (seat + 1), 4)}
    assert seen and seen[0] < turns, "read before the end, not at it"
    assert table.session.game.turns == turns, "the live game is untouched"


def test_a_seat_reads_its_chance_at_a_replayed_round_and_a_spectator_none(stub_model, monkeypatch, tmp_path):
    registry, table, token, seat = _played(tmp_path)
    code = table.code
    past = registry.handle("GET", f"/api/table/{code}/replay?round=1", {}, token)
    assert past["chance"] == {"rater": "rater", "chance": round(0.1 * (seat + 1), 4)}
    assert "chance" not in registry.handle("GET", f"/api/table/{code}/replay?round=1", {}, None)
    # At the round the game ended, the read before the end.
    _ends_at(monkeypatch, table)
    last = registry.handle("GET", f"/api/table/{code}/replay?round=99999", {}, token)
    assert last["chance"] == {"rater": "rater", "chance": round(0.1 * (seat + 1), 4)}


def test_a_ranked_table_refuses_it(stub_model):
    registry = new_tables()
    table, token, _ = _dealt(registry, ["rater"])
    table.session.ranked = True
    with pytest.raises(ApiError) as refused:
        registry.handle("GET", "/api/chance", {}, token)
    assert refused.value.status == 409
    assert stub_model == []


def test_a_real_checkpoint_rates_the_seat(models):
    """The route on an actual ONNX file, past setup: a win chance in [0, 1]."""
    pytest.importorskip("onnxruntime")
    from hexset.clients.onnxmeta import stamp

    stamp(FIXTURES / "stub-contract6-valued.onnx", models / "stub.onnx", name="valued",
          modes=["standard"], version="1", commit="0")
    registry = new_tables()
    table, token, seat = _dealt(registry, ["valued"])
    table.session.game.phase = Phase.MAIN
    table.bump()
    got = registry.handle("GET", "/api/chance", {}, token)
    assert got["rater"] == "valued"
    assert 0.0 <= got["chance"] <= 1.0


def test_the_servers_chance_model_rates_a_table_it_does_not_sit_at(stub_model):
    registry = new_tables(chance_model="rater")
    table, token, seat = _dealt(registry, ["test-trader", "test-trader"])
    got = registry.handle("GET", "/api/chance", {}, token)
    assert got == {"version": table.version, "rater": "rater", "chance": round(0.1 * (seat + 1), 4)}
    assert stub_model == [seat]


def test_the_servers_chance_model_rates_over_a_seated_one(stub_model, models, monkeypatch):
    (models / "other.onnx").write_bytes(with_metadata(b"\x3a\x00", {"name": "other"}))
    specs = []
    real = api.win_chance
    monkeypatch.setattr(api, "win_chance", lambda spec, game, s, config: specs.append(spec) or real(spec, game, s, config))
    registry = new_tables(chance_model="rater")
    _, token, _ = _dealt(registry, ["other"])
    assert registry.handle("GET", "/api/chance", {}, token)["rater"] == "rater"
    assert specs == [str(models / "rater.onnx")]


def test_a_chance_model_not_offered_at_the_game_type_leaves_it_to_the_seats(stub_model):
    registry = new_tables(chance_model="absent")
    _, token, _ = _dealt(registry, ["test-trader", "rater"])
    assert registry.handle("GET", "/api/chance", {}, token)["rater"] == "rater"
