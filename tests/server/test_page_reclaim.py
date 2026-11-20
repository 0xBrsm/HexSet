# SPDX-License-Identifier: GPL-3.0-only
"""A person's seat at a game reopened from its journal, in a real browser.
The seat comes back claimed and with no token, so the page gets back in by
its client secret (`POST /api/reclaim`): on arrival whether or not it kept a
token, and under an open page whose token stops working. A stranger's page
asks the same question as a probe and gets a plain no.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request

import pytest

from hexset.server.api import Config, Seat, SeatKind, build_session

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

PORT = 8831
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "rcl123"
SECRET = "the browser that sat at seat 0"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}
STALE = "minted before the restart"


@pytest.fixture(scope="module")
def running_server(tmp_path_factory):
    # Seat 0 is first to place, so the bots wait on it and the game stands
    # still between tests.
    directory = tmp_path_factory.mktemp("reclaim")
    bot = lambda: Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")  # noqa: E731
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT), bot(), bot(), bot()]
    build_session(CODE.upper(), seats, Config(games_dir=str(directory), seed=99), first=0)
    with serving(PORT, str(directory)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = launch(playwright)
        yield browser
        browser.close()


def _context(browser, secret: str, token: str | None = None):
    script = f"localStorage.setItem('hexset.secret', {json.dumps(secret)});"
    if token is not None:
        script += f"if (!sessionStorage.getItem('seeded')) {{ localStorage.setItem('hexset.token.{CODE}', {json.dumps(token)}); sessionStorage.setItem('seeded', '1'); }}"
    context = browser.new_context()
    context.add_init_script(f"try {{ {script} }} catch (e) {{}}")
    return context


def _token(page):
    return page.evaluate(f"localStorage.getItem('hexset.token.{CODE}')")


def _seat(page):
    return page.evaluate("state.seat")


def _refusals(page) -> list[str]:
    refused: list[str] = []
    # The test server keeps no rankings, so that read is a 404 on every page.
    page.on(
        "response",
        lambda r: refused.append(f"{r.status} {r.url}")
        if r.status >= 400 and not r.url.endswith("/api/rankings")
        else None,
    )
    return refused


def test_a_returning_browser_with_no_saved_token_sits_back_down(running_server, browser):
    context = _context(browser, SECRET)
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.goto(f"{BASE_URL}/{CODE}", wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    page.wait_for_selector(".player-row")
    assert _seat(page) == 0
    assert _token(page) == reclaimed.value.json()["token"]
    context.close()


def test_a_stranger_is_asked_without_a_refused_request(running_server, browser):
    context = _context(browser, "somebody else's secret")
    page = context.new_page()
    refused = _refusals(page)
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.goto(f"{BASE_URL}/{CODE}", wait_until="load")
    assert reclaimed.value.status == 200
    assert reclaimed.value.json() == {"token": None}
    page.wait_for_selector(".player-row")
    assert _seat(page) is None
    assert _token(page) is None
    assert refused == []
    context.close()


def test_a_reclaim_that_cannot_be_asked_keeps_the_token(running_server, browser):
    """The server is unreachable mid-restart: the page watches, but keeps
    its stale token until a reclaim is answered, and the next visit sits."""
    context = _context(browser, SECRET, STALE)
    page = context.new_page()
    page.route("**/api/reclaim", lambda route: route.abort())
    page.goto(f"{BASE_URL}/{CODE}", wait_until="load")
    page.wait_for_selector(".player-row")
    assert _seat(page) is None
    assert _token(page) == STALE

    page.unroute("**/api/reclaim")
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.reload(wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    page.wait_for_selector(".player-row")
    assert _seat(page) == 0
    context.close()


def _reclaim_elsewhere() -> str:
    """Another reclaim mints a new token, as a restart leaves none: the
    page's own stops working under it."""
    request = urllib.request.Request(
        f"{BASE_URL}/api/reclaim",
        data=json.dumps({"code": CODE, "secret": SECRET}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["token"]


def test_an_open_page_whose_token_goes_stale_still_plays(running_server, browser):
    """Seat 0 is to move, so the page is not polling: the first request
    after the token goes stale is refused, and the page reclaims and sends
    it again."""
    context = _context(browser, SECRET)
    page = context.new_page()
    page.goto(f"{BASE_URL}/{CODE}", wait_until="load")
    page.wait_for_selector(".player-row")
    page.wait_for_function("state.seat === 0")
    first = _token(page)
    elsewhere = _reclaim_elsewhere()

    statuses: list[int] = []
    page.on("response", lambda r: statuses.append(r.status) if r.url.endswith("/api/name") else None)
    assert page.evaluate("postJSON('/api/name', {name: 'Ada'}).then(r => r.ok)") is True
    assert statuses == [403, 200]
    assert _token(page) not in (None, first, elsewhere)
    assert _seat(page) == 0
    context.close()
