"""Where a request came from: journalled with the seat it claims, read from
the configured forwarding header, and limiting how many games one address may
deal."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import new_tables
from hexset.server.api import LIVE_IDLE_SECONDS, ApiError, ApiKey, Origin, load_api_keys, stamp_client
from hexset.server.web import HexSetServer, forwarded_ip

WEB = {"id": "a" * 64, "kind": "web"}


def _clients(games_dir: Path) -> list[dict]:
    """Every client the journals in `games_dir` recorded, header and seated."""
    found = []
    for path in sorted(games_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("kind") == "game":
                found.extend(event.get("clients", {}).values())
            elif event.get("kind") == "seated" and event.get("client"):
                found.append(event["client"])
    return found


@pytest.fixture
def serve(tmp_path):
    started = []

    def start(forwarded_header=None, **config):
        server = HexSetServer(
            ("127.0.0.1", 0), new_tables(games_dir=str(tmp_path), **config), forwarded_header
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        started.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server, thread in started:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _deal(base: str, headers: dict | None = None, body: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + "/api/games",
        data=json.dumps(body or {"client": WEB}).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_journal_keeps_the_address_route_and_agent_a_seat_was_claimed_from(serve, tmp_path):
    base = serve()
    status, _ = _deal(base, {"User-Agent": "probe/1.0"})
    assert status == 200
    [client] = _clients(tmp_path)
    assert client == {**WEB, "via": "http", "ip": "127.0.0.1", "ua": "probe/1.0"}


def test_only_the_named_forwarding_header_is_read(serve, tmp_path):
    both = {"CF-Connecting-IP": "203.0.113.9", "X-Forwarded-For": "198.51.100.1, 203.0.113.7"}
    _deal(serve(), both)
    _deal(serve(forwarded_header="CF-Connecting-IP"), both)
    _deal(serve(forwarded_header="X-Forwarded-For"), both)
    _deal(serve(forwarded_header="CF-Connecting-IP"), {"X-Forwarded-For": "203.0.113.7"})
    ips = sorted(c["ip"] for c in _clients(tmp_path))
    assert ips == ["127.0.0.1", "127.0.0.1", "203.0.113.7", "203.0.113.9"]


def test_forwarded_ip_takes_the_last_entry_and_rejects_garbage():
    assert forwarded_ip("2001:db8::1") == "2001:db8::1"
    assert forwarded_ip("a, 203.0.113.1") == "203.0.113.1"
    assert forwarded_ip("203.0.113.1, junk") is None
    assert forwarded_ip(None) is None


def test_a_seats_kind_follows_the_route_it_came_in_on():
    over_mcp = Origin(ip="203.0.113.9", via="mcp")
    over_http = Origin(ip="203.0.113.9", via="http", user_agent="x" * 500)
    assert stamp_client(WEB, over_mcp)["kind"] == "mcp"
    assert stamp_client({"id": None, "kind": "mcp"}, over_http)["kind"] == "api"
    assert stamp_client(WEB, over_http)["kind"] == "web"
    assert len(stamp_client(WEB, over_http)["ua"]) == 200
    assert stamp_client(WEB, None) == WEB


def _limited(**config):
    config.setdefault("limit_exempt", ())
    return new_tables(**config)


def _deal_from(tables, ip: str | None, key: str | None = None, **payload):
    return tables.handle("POST", "/api/games", {"client": WEB, **payload}, None, Origin(ip=ip, key=key))


def test_an_address_may_deal_only_so_many_games_a_period():
    tables = _limited(games_per_period=2, live_tables_per_ip=0)
    _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError):
        _deal_from(tables, "203.0.113.9", bots=["no-such-bot"])
    _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables, "203.0.113.9")
    assert refused.value.status == 429
    _deal_from(tables, "198.51.100.1")
    tables.handle("POST", "/api/games", {}, None)  # in-process: never limited


def test_an_address_may_have_only_so_many_unfinished_games_in_play():
    tables = _limited(games_per_period=0, live_tables_per_ip=1)
    first = _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables, "203.0.113.9")
    assert refused.value.status == 429
    tables.get(first["code"]).last_seen = time.monotonic() - LIVE_IDLE_SECONDS - 1
    _deal_from(tables, "203.0.113.9")


def test_an_exempt_network_is_never_limited():
    tables = new_tables(games_per_period=1, live_tables_per_ip=1, limit_exempt=("10.0.0.0/8",))
    for _ in range(3):
        _deal_from(tables, "10.1.2.3")
    _deal_from(tables, "127.0.0.1")
    with pytest.raises(ApiError):
        _deal_from(tables, "127.0.0.1")


def test_the_period_count_survives_a_restart_and_forgets_old_deals(tmp_path, monkeypatch):
    def tables():
        return _limited(games_dir=str(tmp_path), games_per_period=2, live_tables_per_ip=0)

    _deal_from(tables(), "203.0.113.9")
    _deal_from(tables(), "203.0.113.9")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables(), "203.0.113.9")
    assert refused.value.status == 429
    _deal_from(tables(), "198.51.100.1")

    later = time.time() + 31 * 86400
    monkeypatch.setattr(time, "time", lambda: later)
    _deal_from(tables(), "203.0.113.9")


KEYS = {"k-alice": ApiKey(name="alice", games_per_period=3)}


def test_an_api_key_deals_on_its_own_allowance_not_its_address():
    tables = _limited(games_per_period=1, live_tables_per_ip=0, api_keys=KEYS)
    _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError):
        _deal_from(tables, "203.0.113.9")
    for _ in range(3):
        _deal_from(tables, "203.0.113.9", key="k-alice")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables, "198.51.100.1", key="k-alice")
    assert refused.value.status == 429
    assert "API key" in str(refused.value)
    with pytest.raises(ApiError) as unknown:
        _deal_from(tables, "198.51.100.1", key="k-mallory")
    assert unknown.value.status == 401
    _deal_from(tables, "198.51.100.1")  # the unknown key cost that address nothing


def test_a_keyed_deal_journals_the_holder_and_is_recounted_after_a_restart(tmp_path):
    def tables():
        return _limited(games_dir=str(tmp_path), games_per_period=1, live_tables_per_ip=0, api_keys=KEYS)

    for _ in range(3):
        _deal_from(tables(), "203.0.113.9", key="k-alice")
    with pytest.raises(ApiError):
        _deal_from(tables(), "203.0.113.9", key="k-alice")
    _deal_from(tables(), "203.0.113.9")
    clients = _clients(tmp_path)
    assert [c.get("key") for c in clients].count("alice") == 3
    assert "k-alice" not in json.dumps(clients)


def test_the_key_travels_on_its_header(serve, tmp_path):
    base = serve(api_keys=KEYS, limit_exempt=())
    assert _deal(base, {"X-HexSet-Key": "k-alice"})[0] == 200
    assert _deal(base, {"X-HexSet-Key": "nope"})[0] == 401
    assert [c.get("key") for c in _clients(tmp_path)] == ["alice"]


def test_the_keys_file_names_each_holder_once(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"a": {"name": "alice", "games_per_period": 1000}}))
    assert load_api_keys(str(path)) == {"a": ApiKey("alice", 1000)}
    path.write_text(json.dumps({"a": {"name": "alice", "games_per_period": 1}, "b": {"name": "alice", "games_per_period": 2}}))
    with pytest.raises(ValueError):
        load_api_keys(str(path))
