# SPDX-License-Identifier: GPL-3.0-only
"""The rankings in a real browser: `/rankings` draws the boards the
ranker made, one tab each, leaves out the rows under the minimum games, marks the visitor's own row by the client
id the game page sends, and says so where the server keeps none; the game page's
New game modal links it, at the front door and at a seat, only where there
is one. Skipped without Playwright, which is not one of this project's
extras.
"""

from __future__ import annotations

import hashlib

import pytest

from _page_server import launch, serving, front_door


try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

PORT = 8812
BARE_PORT = 8813
BASE_URL = f"http://127.0.0.1:{PORT}"
BARE_URL = f"http://127.0.0.1:{BARE_PORT}"

# The browser's secret, as index.html keeps it; the row carrying its sha256
# is this visitor's.
SECRET = "a visitor's own secret"
MINE = hashlib.sha256(SECRET.encode("utf-8")).hexdigest()


def rank(games: list[dict]) -> dict:
    columns = [{"key": "rating", "label": "Rating"}, {"key": "games", "label": "Games"}]
    return {"boards": [
        {"key": "duel", "title": "Duel", "note": "Two seats.", "columns": columns, "rows": [
            {"name": "Ada", "id": "a" * 64, "rating": 1612.25, "games": 40,
             "version": "5.5", "owner": "octocat", "url": "https://example.org/ada",
             "description": "A language model playing through MCP."},
            {"name": "rehex", "id": None, "rating": 1550, "games": 30, "pinned": True},
            {"name": "Me", "id": MINE, "rating": 1500, "games": 12},
            {"name": "Newcomer", "id": None, "rating": 1490, "games": 9},
        ]},
        {"key": "four", "title": "Four seats", "columns": columns, "rows": [
            {"name": "Bea", "id": None, "rating": None, "games": 11},
        ]},
    ]}


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's servers and their
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def servers():
    with serving(PORT, ranker=rank), serving(BARE_PORT):
        yield


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = launch(playwright)
        try:
            yield chromium
        finally:
            chromium.close()


def rows(page) -> list[list[str]]:
    return page.eval_on_selector_all(
        "#board tbody tr", "(trs) => trs.map((tr) => [...tr.cells].map((td) => td.innerText))"
    )


def test_the_page_draws_each_board_and_marks_the_visitors_row(servers, browser):
    context = browser.new_context()
    context.add_init_script(f"localStorage.setItem('hexset.secret', {SECRET!r})")
    page = context.new_page()
    try:
        page.goto(f"{BASE_URL}/rankings", wait_until="load")
        page.wait_for_selector("#board tbody tr")
        assert page.eval_on_selector_all("#tabs button", "(bs) => bs.map((b) => b.textContent)") == [
            "Duel", "Four seats",
        ]
        assert page.eval_on_selector_all("#board th", "(ths) => ths.map((th) => th.textContent)") == [
            "Name", "Rating", "Games",
        ]
        assert rows(page) == [["Ada", "1612.25", "40"], ["rehex", "1550", "30"], ["Me", "1500", "12"]]
        # A rating the ranker holds fixed carries a pin, beside the rating.
        assert page.eval_on_selector_all(
            "#board svg.pin", "(ps) => ps.map((p) => [p.closest('tr').cells[0].innerText, p.closest('td').cellIndex])",
        ) == [["rehex", 1]]
        assert page.text_content("#board svg.pin title") == "Rating pinned"
        # A profile shows only once its name is clicked, in a row under it.
        assert page.eval_on_selector_all("#board button.who", "(bs) => bs.map((b) => b.textContent)") == ["Ada"]
        assert not page.query_selector("#board tr.profile")
        page.click("#board button.who")
        assert page.get_attribute("#board button.who", "aria-expanded") == "true"
        assert page.eval_on_selector_all(
            "#board tr.profile dt, #board tr.profile dd", "(els) => els.map((e) => e.textContent)"
        ) == ["Version", "5.5", "Owner", "octocat", "Link", "https://example.org/ada",
              "About", "A language model playing through MCP."]
        assert page.get_attribute("#board tr.profile a", "href") == "https://example.org/ada"
        assert page.get_attribute("#board tr.profile a", "rel") == "noopener noreferrer nofollow"
        page.click("#board button.who")
        assert not page.query_selector("#board tr.profile")
        # The page shows the boards alone: no note, no link away.
        assert not page.query_selector("#board .note") and not page.query_selector("header a")
        assert page.eval_on_selector("#board tr.you td", "(td) => getComputedStyle(td).backgroundColor") == "rgb(34, 94, 140)"
        assert page.eval_on_selector_all("#board tr.you", "(trs) => trs.map((tr) => tr.cells[0].textContent)") == ["Me"]
        assert "10 or more games" in page.text_content("#status")
        page.click('#tabs button[data-board="four"]')
        page.wait_for_selector('#tabs button[data-board="four"][aria-pressed="true"]')
        assert rows(page) == [["Bea", "–", "11"]]
        assert page.text_content("#status") == "Only players with 10 or more games are shown."
    finally:
        context.close()


def test_a_server_without_a_ranker_says_it_keeps_none(servers, browser):
    page = browser.new_page()
    try:
        page.goto(f"{BARE_URL}/rankings", wait_until="load")
        page.wait_for_function("document.getElementById('status').textContent.includes('keeps no rankings')")
        assert not page.query_selector("#board table")
    finally:
        page.close()


def test_the_new_game_modal_links_it_at_the_front_door_and_at_a_seat(servers, browser):
    page = browser.new_context().new_page()
    try:
        page.goto(f"{BASE_URL}/", wait_until="load")
        page.wait_for_selector('#modal.show a.modal-link[href="/rankings"]')
        front_door(page, BASE_URL)
        page.wait_for_selector("#players .player-row")
        page.click("#new-game")
        page.wait_for_selector('#modal.show a.modal-link[href="/rankings"]')
    finally:
        page.close()


def test_without_rankings_the_modal_links_nothing(servers, browser):
    page = browser.new_context().new_page()
    try:
        front_door(page, BARE_URL)
        # Reloaded to see the page's one ask for the rankings answered.
        with page.expect_response(lambda r: r.url.endswith("/api/rankings")) as asked:
            page.reload(wait_until="load")
        assert asked.value.status == 404
        page.wait_for_selector("#players .player-row")
        page.click("#new-game")
        page.wait_for_selector("#modal.show #deal-new-game")
        assert not page.query_selector("#modal a.modal-link")
    finally:
        page.close()
