"""A real browser's client identity, end to end: `index.html`'s
`clientInfo()` sends `{"id": <sha256>, "kind": "web"}` on the deal and on a
join, and the server names an unnamed one `"human"`. The rest of the suite
drives the API/MCP surface directly; this is the one check that the browser
sends what those layers assume. Skipped without Playwright, which is not
one of this project's extras.
"""

from __future__ import annotations

import json
import re

import pytest

from _page_server import launch, serving


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
    with serving(PORT) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _capture(page, url_suffix: str):
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
        browser = launch(playwright)
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

            # A second, unrelated browser (own context, no shared
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
