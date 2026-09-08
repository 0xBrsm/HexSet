"""A real browser's client identity, end to end -- `index.html`'s `clientInfo()`
(the page's own half of `api.parse_client`) sends `{"id": <sha256>, "kind":
"web"}` on the deal and on a join, and the server names an unnamed one
`"human"` (`api.default_seat_name`). Everything else in this suite drives the
API/MCP surface directly; this is the one check that the browser itself
still sends what those layers assume, per `agents/notes/deploy-proof-
browser.md`'s lesson that an API-only proof missed a broken page once
before.

Skipped outright if Playwright's Python package or its Chromium browser
isn't available, rather than failing the suite for an optional dependency
(see `pyproject.toml` -- Playwright is not one of this project's own extras).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

PORT = 8791
BASE_URL = f"http://127.0.0.1:{PORT}"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def running_server():
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", "",
        ],
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{BASE_URL}/api/models", timeout=1)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            raise RuntimeError("hexset.server.web did not come up in time")
        yield BASE_URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _capture(page, url_suffix: str):
    """Navigates a fresh page to `url_suffix` (relative to `BASE_URL`) and
    returns the (request body, response body) of whichever of `/api/games`
    or `/api/join` the page's own identity code fires while dealing or
    joining -- the network request itself, since the page never exposes the
    server's `Tables` object to reach into directly."""
    with page.expect_response(
        lambda r: r.request.method == "POST" and ("/api/games" in r.url or "/api/join" in r.url)
    ) as response_info:
        page.goto(f"{BASE_URL}{url_suffix}", wait_until="load")
    response = response_info.value
    request_body = json.loads(response.request.post_data)
    response_body = response.json()
    return request_body, response_body


def _own_seat(response_body: dict) -> dict:
    seat = next(s for s in response_body["seats"] if s["seat"] == response_body["seat"])
    return seat


def test_the_page_sends_a_hashed_web_identity_on_deal_and_on_join(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            creator_context = browser.new_context()
            creator_page = creator_context.new_page()
            request_body, response_body = _capture(creator_page, "/")

            client = request_body["client"]
            assert client["kind"] == "web"
            assert HEX64.match(client["id"]), client["id"]
            creator_seat = _own_seat(response_body)
            assert creator_seat["kind"] == "player"
            assert creator_seat["name"] == "human"
            code = response_body["code"]

            # A second, unrelated browser (its own context -- no shared
            # localStorage/token) joins by the code the first page dealt.
            joiner_context = browser.new_context()
            joiner_page = joiner_context.new_page()
            join_request, join_response = _capture(joiner_page, f"/{code}")

            join_client = join_request["client"]
            assert join_client["kind"] == "web"
            assert HEX64.match(join_client["id"]), join_client["id"]
            joiner_seat = _own_seat(join_response)
            assert joiner_seat["kind"] == "player"
            assert joiner_seat["name"] == "human"

            joiner_context.close()
            creator_context.close()
        finally:
            browser.close()
