"""HTTP-layer tests for `hexset.server.web`: only what the transport adds --
status codes, the static file, the token header, and that a refusal
`api.py` raises arrives as the status it carries. Torch-free: every
opponent is `test-trader` (`tests/traders.py`).
"""

from __future__ import annotations

import json

import random

import threading

import urllib.error

import urllib.request

import pytest

from hexset.actions import ActionType, legal_actions

from conftest import new_tables

from hexset.server.web import TOKEN_HEADER, HexSetServer

from hexset.server.wire import action_to_wire

SOLO = ["test-trader", "test-trader", "test-trader"]


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    """Pin the creator to seat 0: a randomly seated creator can leave a bot seat
    on move first, whose runner thread then races this test's own requests.
    """
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), new_tables())
    # A short `poll_interval` makes `shutdown()` return immediately in
    # teardown instead of waiting on `serve_forever`'s default half-second.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class Client:

    def __init__(self, base: str) -> None:
        self.base = base
        self.token: str | None = None

    def _send(self, request: urllib.request.Request):
        if self.token is not None:
            request.add_header(TOKEN_HEADER, self.token)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def get(self, path: str):
        return self._send(urllib.request.Request(self.base + path))

    def post(self, path: str, payload: dict):
        status, data = self._send(
            urllib.request.Request(
                self.base + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        if isinstance(data, dict) and data.get("token"):
            self.token = data["token"]
        return status, data


def seated(base: str, **kwargs) -> tuple[Client, dict]:
    client = Client(base)
    kwargs.setdefault("bots", SOLO)
    status, data = client.post("/api/games", kwargs)
    assert status == 200, data
    return client, data


def test_a_game_is_dealt_and_played_over_http(live_server):
    server, base = live_server
    client, table = seated(base)
    assert table["code"]
    assert table["phase"] == "SETUP_SETTLEMENT"  # already playable, no separate start

    session = server.tables.get(table["code"]).session
    settlement = action_to_wire(
        next(a for a in legal_actions(session.game) if a.type is ActionType.SETUP_SETTLEMENT)
    )
    status, data = client.post("/api/action", {"action": settlement})
    assert status == 200
    assert data["phase"] == "SETUP_ROAD"
    assert len(data["log"]) >= 1


def test_the_entry_point_serves_the_page():
    """`python -m hexset.server.web` is what the container runs; the page tests
    serve in-process, so this is the one check that `main` still comes up."""
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(port), "--games-dir", "",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as page:
                    assert page.status == 200
                    assert b"<html" in page.read().lower()
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            raise AssertionError("hexset.server.web did not come up in time")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.parametrize("length", ["-5", "lots"])
@pytest.mark.parametrize("path", ["/api/games", "/mcp"])
def test_a_content_length_that_is_not_a_byte_count_is_a_400(live_server, path, length):
    import http.client

    _, base = live_server
    connection = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    connection.putrequest("POST", path)
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", length)
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 400
    assert "Content-Length" in json.loads(response.read())["error"]
    connection.close()
