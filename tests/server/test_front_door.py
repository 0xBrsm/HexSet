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
from hexset.server.api import LIVE_IDLE_SECONDS, ApiError, ApiKey, Origin, load_api_keys, stamp_client, PARALLEL_NOTICE
from hexset.server.web import HexSetServer, forwarded_ip

# A client that says what it is, as the HTTP API asks of any but the page.
CLIENT = {"id": "a" * 64, "name": "probe", "version": "1", "owner": "octocat",
          "description": "A test client that deals and stops."}


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
        data=json.dumps(body if body is not None else {"client": CLIENT}).encode("utf-8"),
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
    status, _ = _deal(base, {"User-Agent": "probe/1.0", "Sec-Fetch-Site": "same-origin"})
    assert status == 200
    [client] = _clients(tmp_path)
    assert client == {**CLIENT, "kind": "web", "via": "http", "ip": "127.0.0.1", "ua": "probe/1.0"}


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


def test_a_seats_kind_is_the_servers_call_from_where_it_came_from():
    claimed = {"id": None, "kind": "web"}
    over_mcp = Origin(ip="203.0.113.9", via="mcp")
    over_http = Origin(ip="203.0.113.9", via="http", user_agent="x" * 500)
    from_page = Origin(ip="203.0.113.9", via="http", page=True)
    assert stamp_client(claimed, over_mcp)["kind"] == "mcp"
    assert stamp_client(claimed, over_http)["kind"] == "api"
    assert stamp_client({"id": None}, from_page)["kind"] == "web"
    assert len(stamp_client(claimed, over_http)["ua"]) == 200
    assert stamp_client({"id": None}, None) == {"id": None, "kind": "api"}


@pytest.mark.parametrize("headers, kind", [
    ({"Sec-Fetch-Site": "same-origin"}, "web"),
    ({"Sec-Fetch-Site": "cross-site", "Origin": "http://127.0.0.1"}, "api"),
    ({}, "api"),
])
def test_a_seat_is_web_only_when_the_browser_marks_the_request_as_the_pages(serve, tmp_path, headers, kind):
    base = serve()
    # What the client says it is plays no part.
    status, _ = _deal(base, headers, {"client": {**CLIENT, "kind": "web"}})
    assert status == 200
    assert [c["kind"] for c in _clients(tmp_path)] == [kind]


@pytest.mark.parametrize("client", [None, {"id": "a" * 64}, {**CLIENT, "owner": "  "}])
def test_an_api_client_that_does_not_say_what_it_is_may_not_play(serve, tmp_path, client):
    base = serve()
    body = {} if client is None else {"client": client}
    status, refused = _deal(base, None, body)
    assert status == 400 and "client." in refused["error"]
    assert not list(tmp_path.glob("*.jsonl"))
    # The page is not asked; nor is a script, once it says what it is.
    assert _deal(base, {"Sec-Fetch-Site": "same-origin"}, body)[0] == 200
    assert _deal(base, None, {"client": CLIENT})[0] == 200


def test_a_browser_without_fetch_metadata_is_known_by_an_origin_naming_the_host(serve, tmp_path):
    base = serve()
    _deal(base, {"Origin": base})
    _deal(base, {"Origin": "http://elsewhere.example"})
    assert sorted(c["kind"] for c in _clients(tmp_path)) == ["api", "web"]


def _limited(**config):
    config.setdefault("limit_exempt", ())
    return new_tables(**config)


def _deal_from(tables, ip: str | None, key: str | None = None, **payload):
    return tables.handle("POST", "/api/games", {"client": CLIENT, **payload}, None, Origin(ip=ip, key=key))


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


def test_a_game_limit_refusal_names_the_next_slot_and_the_contact():
    tables = _limited(games_per_period=1, live_tables_per_ip=0, contact="ask@example.org")
    _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables, "203.0.113.9")
    error = refused.value
    assert error.status == 429
    assert str(error).startswith("game limit: 1 per 30 days for this address; next slot ")
    assert str(error).endswith("; contact: ask@example.org")
    assert error.fields["contact"] == "ask@example.org"
    assert error.fields["next_slot"] in str(error)
    assert 0 < error.retry_after <= 30 * 86400


def test_a_refusal_without_a_configured_contact_names_none():
    tables = _limited(games_per_period=1, live_tables_per_ip=0)
    _deal_from(tables, "203.0.113.9")
    with pytest.raises(ApiError) as refused:
        _deal_from(tables, "203.0.113.9")
    assert "contact:" not in str(refused.value) and "contact" not in refused.value.fields


def test_a_new_game_reports_the_allowance_left_and_warns_of_parallel_play():
    tables = _limited(games_per_period=12, live_tables_per_ip=5, contact="ask@example.org")
    first = _deal_from(tables, "203.0.113.9")
    assert first["allowance"] == {
        "games_left": 11, "games_per_period": 12, "period_days": 30.0,
        "live_tables": 1, "live_tables_limit": 5,
    }
    assert "notice" not in first
    second = _deal_from(tables, "203.0.113.9")
    assert second["allowance"]["games_left"] == 10
    assert second["allowance"]["contact"] == "ask@example.org"
    assert second["allowance"]["live_tables"] == 2
    assert second["notice"] == PARALLEL_NOTICE
    assert tables.handle("GET", "/api/limits", {}, None, Origin(ip="203.0.113.9")) == second["allowance"]
    assert "allowance" not in tables.handle("POST", "/api/games", {}, None)  # in-process: no limits


def test_over_http_a_game_limit_refusal_carries_retry_after_and_its_fields(serve):
    base = serve(games_per_period=1, live_tables_per_ip=0, limit_exempt=(), contact="ask@example.org")
    assert _deal(base)[0] == 200
    request = urllib.request.Request(
        base + "/api/games", data=json.dumps({"client": CLIENT}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    body = json.loads(refused.value.read())
    assert refused.value.code == 429
    assert int(refused.value.headers["Retry-After"]) > 0
    assert body["contact"] == "ask@example.org" and body["next_slot"] in body["error"]


def test_over_http_the_bare_api_route_answers_rather_than_404s(serve):
    from hexset.server.api import API_DOCS

    base = serve()
    for path in ("/api", "/api/", "/api?x=1"):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            assert response.status == 200
            assert json.loads(response.read())["docs"] == API_DOCS
