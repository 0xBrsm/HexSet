"""What `hexset.server.api` refuses, and what a refusal leaves behind: a bot
that will not spawn, a malformed request body, the token-free reads, and the
locks a read or a reopen takes.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import pytest

from hexset.game import is_over
from hexset.server import _journal as journal, api
from hexset.server.api import (
    ApiError,
    Config,
    Seat,
    SeatKind,
    build_session,
)

from conftest import new_tables

SOLO = ["test-trader", "test-trader", "test-trader"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def broken_model(monkeypatch, tmp_path):
    """A picker entry, `broken`, whose file is not a model at all."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "broken.onnx").write_bytes(b"not a model")
    monkeypatch.setattr(api, "MODELS_DIR", models)
    return models / "broken.onnx"


def _hidden(view: dict) -> set[int]:
    """The seats whose hand `view` does not spell out."""
    return {p["seat"] for p in view["players"] if "hand" not in p}


# --- the models directory ----------------------------------------------------


def test_the_default_models_directory_is_models_beside_the_package():
    root = Path(api.__file__).resolve().parents[2]
    assert (root / "hexset" / "server" / "api.py").is_file()
    assert (root / "pyproject.toml").is_file(), "the source tree's root holds pyproject.toml"
    assert api.default_models_dir() == root / "models"


def test_an_installed_package_looks_for_models_under_the_working_directory(monkeypatch, tmp_path):
    installed = tmp_path / "site-packages" / "hexset" / "server" / "api.py"
    monkeypatch.setattr(api, "__file__", str(installed))
    monkeypatch.chdir(tmp_path)
    assert api.default_models_dir() == tmp_path / "models"


# --- a bot that will not spawn leaves nothing behind ----------------------------


def test_a_game_whose_bot_will_not_spawn_is_refused_and_never_dealt(broken_model, tmp_path):
    games = tmp_path / "games"
    registry = new_tables(games_dir=str(games))

    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/games", {"bots": ["broken"]}, None)

    assert refused.value.status == 400
    assert "cannot seat broken" in str(refused.value)
    assert registry._tables == {}
    assert not games.exists() or not list(games.glob("*.jsonl"))


def test_a_checkpoint_is_spawned_for_the_table_s_player_count(monkeypatch, tmp_path):
    """`onnxbot.spawn` is told how many seats the table has, so a checkpoint
    trained for another count is refused before the table exists."""
    import sys
    import types

    models = tmp_path / "models"
    models.mkdir()
    (models / "net.onnx").write_bytes(b"")
    monkeypatch.setattr(api, "MODELS_DIR", models)
    asked = []

    def spawn(path, board, *, rng=None, device="cpu", players=None):
        asked.append(players)
        raise ValueError(f"{path} was trained for 2 players, not this table's {players}")

    fake = types.ModuleType("hexset.clients.onnxbot")
    fake.spawn = spawn
    monkeypatch.setitem(sys.modules, "hexset.clients.onnxbot", fake)
    registry = new_tables()

    with pytest.raises(ApiError, match="trained for 2 players"):
        registry.handle("POST", "/api/games", {"bots": ["net", "net", "net"]}, None)

    assert asked == [4] and registry._tables == {}


def test_seating_a_bot_that_will_not_spawn_leaves_the_seat_as_it_was(broken_model, tmp_path):
    games = tmp_path / "games"
    registry = new_tables(games_dir=str(games))
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    token = data["token"]
    table = registry.get(data["code"])
    path = next(games.glob("*.jsonl"))
    before = path.read_bytes()

    with pytest.raises(ApiError) as refused:
        registry.handle("POST", "/api/bot", {"seat": 1, "model": "broken"}, token)

    assert refused.value.status == 400
    assert table.seats[1].kind is SeatKind.EMPTY
    assert table.runners == []
    assert path.read_bytes() == before, "nothing journalled for a seat that never changed"


def test_a_closed_seat_given_a_bot_that_will_not_spawn_stays_closed(broken_model):
    registry = new_tables()
    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    token = data["token"]
    registry.handle("POST", "/api/close", {"seat": 1}, token)

    with pytest.raises(ApiError):
        registry.handle("POST", "/api/bot", {"seat": 1, "model": "broken"}, token)

    assert registry.handle("GET", "/api/state", {}, token)["locked"] == [1]


def test_a_journalled_game_whose_bot_will_not_spawn_does_not_reopen(broken_model, tmp_path):
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada"),
        Seat(kind=SeatKind.BOT, name="broken", spec=str(broken_model)),
        Seat(),
        Seat(),
    ]
    build_session("abc234", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
    path = next(tmp_path.glob("*.jsonl"))
    before = path.read_bytes()

    registry = new_tables(games_dir=str(tmp_path))
    with pytest.raises(ApiError) as refused:
        registry.get("abc234")

    assert refused.value.status == 400
    assert registry._tables == {}
    assert path.read_bytes() == before, "no `reopened` line for a game that did not reopen"


# --- a token-free read -------------------------------------------------------


def test_a_spectator_sees_every_hand_and_a_seat_still_sees_only_its_own():
    registry = new_tables()
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]

    watched = registry.handle("GET", f"/api/table/{code}", {}, None)
    seated = registry.handle("GET", "/api/state", {}, token)

    assert _hidden(watched) == set()
    assert all("dev_cards" in p for p in watched["players"])
    assert _hidden(seated) == set(range(api.MAX_SEATS)) - {mine}


def test_a_seat_replaying_its_own_game_by_an_uppercase_code_is_still_that_seat(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    code, token = dealt["code"], dealt["token"]
    mine = registry.by_token(token)[1]

    past = registry.handle("GET", f"/api/table/{code.upper()}/replay?round=0", {}, token)

    assert _hidden(past) == set(range(api.MAX_SEATS)) - {mine}


def _play_out(session, rng: random.Random) -> None:
    """Every seat's moves at random until the game ends."""
    from hexset.actions import Action, ActionType, legal_actions
    from hexset.game import to_move
    from hexset.server.wire import action_to_wire

    for _ in range(3000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def test_once_the_game_is_over_a_spectator_sees_every_hand(tmp_path):
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada")] + [
        Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader") for _ in range(3)
    ]
    session = build_session("abc234", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
    _play_out(session, random.Random(4))
    assert is_over(session.game)

    watched = new_tables(games_dir=str(tmp_path)).handle("GET", "/api/table/abc234", {}, None)

    assert _hidden(watched) == set()


# --- locks ---------------------------------------------------------------------


def test_every_token_free_view_is_built_under_the_tables_lock(monkeypatch):
    """The spectator read, the create reply and the reclaim reply, each of
    which once built its view with the table's lock released."""
    held: list[bool] = []
    real = api.Table.view

    def view(self, *args, **kwargs):
        held.append(self.lock.locked())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(api.Table, "view", view)
    registry = new_tables()
    secret = "a secret of the creator's own"
    import hashlib

    client = {"id": hashlib.sha256(secret.encode()).hexdigest(), "kind": "api"}
    code = registry.handle("POST", "/api/games", {"bots": [], "client": client}, None)["code"]
    registry.handle("GET", f"/api/table/{code}", {}, None)
    registry.handle("POST", "/api/reclaim", {"code": code, "secret": secret}, None)

    assert held == [True, True, True]


def test_a_reopen_runs_with_the_registry_unlocked(monkeypatch, tmp_path):
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada"), Seat(), Seat(), Seat()]
    build_session("abc234", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
    registry = new_tables(games_dir=str(tmp_path))
    held: list[bool] = []
    real = api.reopen_session

    def reopen(*args, **kwargs):
        held.append(registry._registry_lock.locked())
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "reopen_session", reopen)
    assert registry.get("abc234").code == "abc234"
    assert registry.get("abc234").code == "abc234"
    assert held == [False], "reopened once, and with the registry free"


def test_a_code_with_no_game_is_looked_for_once_in_a_burst(monkeypatch, tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    looked: list[str] = []
    real = journal.most_recent

    def most_recent(directory, code):
        looked.append(code)
        return real(directory, code)

    monkeypatch.setattr(journal, "most_recent", most_recent)
    for _ in range(3):
        with pytest.raises(ApiError) as missing:
            registry.get("zzzzzz")
        assert missing.value.status == 404
    assert looked == ["zzzzzz"]


def test_two_reads_of_one_lost_code_reopen_it_once(monkeypatch, tmp_path):
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada"), Seat(), Seat(), Seat()]
    build_session("abc234", seats, Config(games_dir=str(tmp_path), seed=99), first=0)
    registry = new_tables(games_dir=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    real = api.reopen_session

    def reopen(*args, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "reopen_session", reopen)
    got: list[object] = []
    readers = [threading.Thread(target=lambda: got.append(registry.get("abc234"))) for _ in range(2)]
    readers[0].start()
    assert entered.wait(5)
    readers[1].start()
    # Another code is served meanwhile: the registry is not held by the reopen.
    other = registry.handle("POST", "/api/games", {"bots": []}, None)["code"]
    assert other != "abc234"
    release.set()
    for reader in readers:
        reader.join(5)
    assert len(calls) == 1
    assert len(got) == 2 and got[0] is got[1]


def test_swapping_a_parked_bot_does_not_wait_out_its_long_poll():
    registry = new_tables()
    dealt = registry.handle("POST", "/api/games", {"bots": SOLO}, None)
    token = dealt["token"]
    table = registry.get(dealt["code"])
    time.sleep(0.3)  # the runners park: the creator, at seat 0, is to move

    started = time.monotonic()
    registry.handle("POST", "/api/bot", {"seat": 1, "model": "test-trader"}, token)

    assert time.monotonic() - started < 1.5
    assert sorted(runner.seat for runner, _ in table.runners) == [1, 2, 3]


# --- malformed request bodies are a 400 ------------------------------------------


@pytest.mark.parametrize(
    "path, payload",
    [
        ("/api/bot", {"seat": "1", "model": "test-trader"}),
        ("/api/bot", {"seat": [1], "model": "test-trader"}),
        ("/api/open", {"seat": None}),
        ("/api/close", {"seat": 1.5}),
        ("/api/name", {"name": ["Ada"]}),
        ("/api/action", {"action": ["END_TURN"]}),
        ("/api/action", {"action": {"type": "END_TURN"}, "version": "7"}),
    ],
)
def test_a_wrongly_typed_field_is_a_400(path, payload):
    registry = new_tables()
    token = registry.handle("POST", "/api/games", {"bots": []}, None)["token"]
    for seat in (1, 2, 3):  # nothing else stands between the request and its checks
        registry.handle("POST", "/api/close", {"seat": seat}, token)
    with pytest.raises(ApiError) as refused:
        registry.handle("POST", path, payload, token)
    assert refused.value.status == 400


@pytest.mark.parametrize("payload", [{"bots": "test-trader"}, {"bots": [1]}, {"name": 7}])
def test_a_wrongly_typed_deal_is_a_400(payload):
    with pytest.raises(ApiError) as refused:
        new_tables().handle("POST", "/api/games", payload, None)
    assert refused.value.status == 400


# --- names ---------------------------------------------------------------------


def test_a_rename_survives_a_reopen(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    dealt = registry.handle("POST", "/api/games", {"bots": [], "name": "Ada"}, None)
    code, token = dealt["code"], dealt["token"]
    seat = registry.by_token(token)[1]
    registry.handle("POST", "/api/name", {"name": "Grace"}, token)

    events = journal.read(next(tmp_path.glob("*.jsonl")))
    assert journal.players(events)[seat] == "Grace"
    reopened = new_tables(games_dir=str(tmp_path)).get(code)
    assert reopened.seats[seat].name == reopened.session.player_names[seat] == "Grace"


def test_an_empty_name_falls_back_to_the_seats_default():
    registry = new_tables()
    dealt = registry.handle("POST", "/api/games", {"bots": [], "name": "Ada"}, None)
    token = dealt["token"]
    seat = registry.by_token(token)[1]

    view = registry.handle("POST", "/api/name", {"name": "   "}, token)

    assert view["seats"][seat]["name"] == api.default_seat_name("api")
