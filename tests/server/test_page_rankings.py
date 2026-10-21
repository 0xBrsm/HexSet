# SPDX-License-Identifier: GPL-3.0-only
"""The rankings in a real browser: `/rankings` draws the boards the
ranker made, one tab each, leaves out the rows under the minimum games, marks
the visitor's own row by the client id the game page sends, shows two checked
rows head to head, and says so where the server keeps none; the game page's
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
    columns = [{"key": "rating", "label": "Rating"}, {"key": "games", "label": "Games"},
               {"key": "wins", "label": "Wins"}]
    return {"boards": [
        {"key": "duel", "title": "Duel", "note": "Two seats.", "columns": columns, "rows": [
            {"name": "Ada", "id": "a" * 64, "rating": 1612.25, "games": 40, "wins": "63%",
             "version": "5.5", "owner": "octocat", "url": "https://example.org/ada",
             "description": "A language model playing through MCP."},
            {"name": "rehex", "id": None, "rating": 1550, "games": 30, "pinned": True, "wins": "40%"},
            {"name": "Me", "id": MINE, "rating": 1500, "games": 12, "wins": "50%"},
            {"name": "Newcomer", "id": None, "rating": 1490, "games": 9},
        ], "pairs": [{"a": "a" * 64, "b": MINE, "games": 5, "a_wins": 3, "b_wins": 2, "expected": 0.62,
                    "expected_low": 0.4, "expected_high": 0.8}]},
        {"key": "four", "title": "Four seats", "columns": columns, "rows": [
            {"name": "Bea", "id": None, "rating": None, "games": 11},
        ]},
        {"key": "never", "title": "Never met", "columns": columns, "rows": [
            {"name": "Cy", "id": "c" * 64, "rating": 1600, "games": 20},
            {"name": "Di", "id": "d" * 64, "rating": 1500, "games": 20},
        ], "pairs": [{"a": "c" * 64, "b": "d" * 64, "games": 0, "a_wins": 0, "b_wins": 0, "expected": 0.7}]},
        {"key": "half", "title": "Half", "columns": columns, "rows": [
            {"name": "Ed", "id": "e" * 64, "rating": 1700, "games": 200},
            {"name": "Flo", "id": "f" * 64, "rating": 1500, "games": 200},
        ], "pairs": [{"a": "e" * 64, "b": "f" * 64, "games": 200, "a_wins": 159, "b_wins": 41, "expected": 0.795}]},
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
            "Duel", "Four seats", "Never met", "Half",
        ]
        assert page.eval_on_selector_all("#board th", "(ths) => ths.map((th) => th.textContent)") == [
            "Name", "Rating", "Games", "Wins",
        ]
        assert rows(page) == [["Ada", "1612.25", "40", "63%"], ["rehex", "1550", "30", "40%"],
                              ["Me", "1500", "12", "50%"]]
        # A column of percentages is aligned as the numbers are.
        assert page.eval_on_selector_all(
            "#board th, #board tbody tr:first-child td", "(cs) => cs.map((c) => c.className)"
        ) == ["", "number", "number", "number", "", "number", "number", "number"]
        # A rating the ranker holds fixed carries a pin, beside the rating.
        assert page.eval_on_selector_all(
            "#board svg.pin", "(ps) => ps.map((p) => [p.closest('tr').cells[0].innerText, p.closest('td').cellIndex])",
        ) == [["rehex", 1]]
        assert page.text_content("#board svg.pin title") == "Rating pinned"
        # A row's details show in a card above the table once it is clicked.
        assert not page.query_selector("#versus")
        page.click('#board tbody tr.pickable:has(td:first-child:text-is("Ada"))')
        page.wait_for_selector("#versus .score.totals")
        # The one card: the pick on the left where it stays, an empty slot on
        # the right, its other columns where the score will be, and below, its profile.
        assert page.text_content("#versus .head h2") == "Head to head"
        texts = lambda sel: page.eval_on_selector_all(sel, "(els) => els.map((e) => e.innerText)")  # noqa: E731
        assert texts("#versus .side.a > div") == ["Ada", "Rating 1612.25", "5.5 · octocat"]
        assert texts("#versus .side.b.empty") == [""]
        assert texts("#versus .score.totals .total") == ["40", "63%"]
        assert texts("#versus .score.totals .games") == ["games", "wins"]
        assert texts("#versus .details p") == ["A language model playing through MCP.", "https://example.org/ada"]
        assert page.get_attribute("#versus a", "href") == "https://example.org/ada"
        assert page.get_attribute("#versus a", "rel") == "noopener noreferrer nofollow"
        page.click("#versus .close")
        assert not page.query_selector("#versus")
        # The page shows the boards alone: no note, no link away.
        assert not page.query_selector("#board .note") and not page.query_selector("header a")
        assert page.eval_on_selector("#board tr.you td", "(td) => getComputedStyle(td).backgroundColor") == "rgb(34, 94, 140)"
        assert page.eval_on_selector_all("#board tr.you", "(trs) => trs.map((tr) => tr.cells[0].textContent)") == ["Me"]
        assert "10 or more games" in page.text_content("#status")
        page.click('#tabs button[data-board="four"]')
        page.wait_for_selector('#tabs button[data-board="four"][aria-pressed="true"]')
        assert rows(page) == [["Bea", "–", "11", "–"]]
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


def test_two_rows_clicked_show_their_head_to_head_record(servers, browser):
    page = browser.new_page()
    try:
        page.goto(f"{BASE_URL}/rankings", wait_until="load")
        page.wait_for_selector("#board tbody tr")

        def row(name):
            return f'#board tbody tr.pickable:has(td:first-child:text-is("{name}"))'

        def sides():
            return page.eval_on_selector_all(
                "#board tbody tr", "(trs) => trs.map((tr) => [tr.cells[0].innerText, tr.className.match(/side-[ab]/)?.[0] || ''])")

        # only rows with an id can be picked: rehex here has none
        assert page.eval_on_selector_all("#board tbody tr.pickable", "(trs) => trs.map((tr) => tr.cells[0].innerText)") == [
            "Ada", "Me"]
        assert page.text_content("#hint") == "Click a row for its details, or two to compare them head to head."
        page.click(row("Ada"))
        assert page.query_selector("#versus .side.b.empty")
        assert not page.query_selector("#hint")
        assert ["Ada", "side-a"] in sides()
        page.click(row("Me"))
        page.wait_for_selector("#versus")

        def panel():
            text = lambda sel: page.eval_on_selector_all(sel, "(els) => els.map((e) => e.innerText)")  # noqa: E731
            return (text("#versus .side .name"), text("#versus .side .sub"), text("#versus .wins span"),
                    text("#versus .games"), text("#versus .shares > span"))
        assert panel() == (["Ada", "Me"], ["Rating 1612.25", "Rating 1500"], ["3", "–", "2"], ["5 games"],
                           ["60%", "40%"])
        # each side's version and owner, its description on hover
        assert page.eval_on_selector_all("#versus .meta", "(els) => els.map((e) => [e.innerText, e.title])") == [
            ["5.5 · octocat", "A language model playing through MCP."]]
        # one bar, what each won at its ends: solid to the expected range's ends, lighter
        # across it, split at the expected share
        assert page.eval_on_selector_all(
            "#versus .shares .bar > div", "(ds) => ds.map((d) => [d.className, Math.round(parseFloat(d.style.width))])"
        ) == [["fill-a", 40], ["band-a", 22], ["band-b", 18], ["fill-b", 20]]
        assert sides()[:3] == [["Ada", "side-a"], ["rehex", ""], ["Me", "side-b"]]
        # what the second click added fades in; the first pick stays put
        entering = "(els) => els.filter((e) => e.classList.contains('enter')).map((e) => e.className)"
        assert page.eval_on_selector_all("#versus, #versus *", entering) == ["score enter", "side b enter", "shares enter"]
        # the picks outlive the refresh, which redraws the table without fading anything
        page.evaluate("load()")
        page.wait_for_selector("#versus")
        assert panel()[0] == ["Ada", "Me"]
        assert page.eval_on_selector_all("#versus, #versus *", entering) == []
        # a picked row clicked again is let go; picked anew, it reads second
        page.click(row("Ada"))
        assert page.text_content("#versus .side.a .name") == "Me"
        page.click(row("Ada"))
        assert panel()[:3] == (["Me", "Ada"], ["Rating 1500", "Rating 1612.25"], ["2", "–", "3"])
        assert panel()[4] == ["40%", "60%"]
        # keyboard: Enter on a focused row picks it; Escape lets both go
        page.keyboard.press("Escape")
        assert not page.query_selector("#versus") and [s for _, s in sides()] == ["", "", ""]
        page.focus(row("Me"))
        page.keyboard.press("Enter")
        assert page.text_content("#versus .side.a .name") == "Me"
        page.focus(row("Ada"))
        page.keyboard.press(" ")
        page.wait_for_selector("#versus")
        page.click("#versus .close")
        assert not page.query_selector("#versus")
    finally:
        page.close()


def test_two_who_never_met_show_the_expected_share_alone(servers, browser):
    page = browser.new_page()
    try:
        page.goto(f"{BASE_URL}/rankings#never", wait_until="load")
        page.wait_for_selector("#board tbody tr")
        for name in ("Cy", "Di"):
            page.click(f'#board tbody tr.pickable:has(td:first-child:text-is("{name}"))')
        page.wait_for_selector("#versus")
        assert page.text_content("#versus .games") == "Not met yet"
        assert not page.query_selector("#versus .wins")
        # no wins to show: the expected shares at the bar's ends, muted
        assert page.get_attribute("#versus .shares", "class") == "shares unmet enter"
        assert page.eval_on_selector_all("#versus .shares > span", "(els) => els.map((e) => e.innerText)") == ["70%", "30%"]
    finally:
        page.close()


def test_either_order_of_picking_shows_the_same_shares(servers, browser):
    """A share that rounds at a half: 159 of 200 reads 80 to 20 picked one way
    and 20 to 80 the other, never 21 to 79."""
    page = browser.new_page()
    try:
        page.goto(f"{BASE_URL}/rankings#half", wait_until="load")
        page.wait_for_selector("#board tbody tr")
        row = lambda name: f'#board tbody tr.pickable:has(td:first-child:text-is("{name}"))'  # noqa: E731
        ends = lambda: page.eval_on_selector_all("#versus .shares > span", "(els) => els.map((e) => e.innerText)")  # noqa: E731
        for first, second, want in (("Ed", "Flo", ["80%", "20%"]), ("Flo", "Ed", ["20%", "80%"])):
            page.keyboard.press("Escape")
            page.click(row(first))
            page.click(row(second))
            page.wait_for_selector("#versus .shares")
            assert ends() == want
    finally:
        page.close()
