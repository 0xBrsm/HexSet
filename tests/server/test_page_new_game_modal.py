# SPDX-License-Identifier: GPL-3.0-only
"""The header's New game button asks what the next table is dealt as before
dealing it: a board (spiral, the default, or random) and rules (standard, the
default, or duel). A duel table is played at two seats (or one, practising),
its other two closed for good, and neither the seated page nor a watching one
offers them back.
Skipped without Playwright, which is not one of this project's extras.
"""

from __future__ import annotations

import json
import re
import urllib.request

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

PORT = 8802
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def server():
    with serving(PORT) as url:
        yield url


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = launch(playwright)
        try:
            yield chromium
        finally:
            chromium.close()


def table(code: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/api/table/{code}", timeout=10) as response:
        return json.load(response)


def code_of(page) -> str:
    return page.url.rstrip("/").rsplit("/", 1)[-1]


def at_a_game(browser):
    page = browser.new_context().new_page()
    front_door(page, BASE_URL)
    page.wait_for_selector("#players .player-row")
    return page


def pressed(page, choice: str) -> list[str]:
    return page.eval_on_selector_all(
        f'#modal-body button[data-choice="{choice}"][aria-pressed="true"]',
        "(buttons) => buttons.map((b) => b.dataset.value)",
    )


def deal_from_modal(page, **choices) -> str:
    """Opens the modal, picks `choices` (data-choice -> data-value), deals,
    and returns the new table's code once the page is at it."""
    old = code_of(page)
    page.click("#new-game")
    page.wait_for_selector("#modal.show #deal-new-game")
    for choice, value in choices.items():
        page.click(f'#modal-body button[data-choice="{choice}"][data-value="{value}"]')
    page.click("#deal-new-game")
    page.wait_for_url(re.compile(rf"/(?!{old}$)[a-z0-9]{{6}}$"))
    page.wait_for_selector("#players .player-row")
    return code_of(page)


def test_new_game_opens_the_modal_with_spiral_and_duel_off_chosen(server, browser):
    page = at_a_game(browser)
    try:
        before = code_of(page)
        page.click("#new-game")
        page.wait_for_selector("#modal.show")
        assert page.text_content("#modal-title") == "New game"
        assert pressed(page, "board") == ["spiral"] and pressed(page, "duel") == ["false"]
        # Nothing is dealt until asked: the page is still at its own table.
        assert code_of(page) == before
        page.keyboard.press("Escape")
        assert not page.is_visible("#modal")
        assert code_of(page) == before
    finally:
        page.context.close()


def test_the_front_door_asks_what_to_deal_and_deals_nothing_until_asked(server, browser):
    context = browser.new_context()
    page = context.new_page()
    try:
        dealt = []
        page.on("request", lambda r: dealt.append(r.url) if r.method == "POST" and "/api/games" in r.url else None)
        page.goto(f"{BASE_URL}/", wait_until="load")
        page.wait_for_selector("#modal.show #deal-new-game")
        assert page.text_content("#modal-title") == "New game"
        assert pressed(page, "board") == ["spiral"] and pressed(page, "duel") == ["false"]
        # Nothing behind it to go back to: no Cancel, and Escape and a click
        # on the backdrop leave it up.
        assert not page.is_visible("#modal-close")
        page.keyboard.press("Escape")
        page.mouse.click(2, 2)
        assert page.is_visible("#modal #deal-new-game")
        assert dealt == []
        page.click('#modal-body button[data-choice="duel"][data-value="true"]')
        page.click("#deal-new-game")
        page.wait_for_url(re.compile(r"/[a-z0-9]{6}$"))
        page.wait_for_selector("#players .player-row")
        assert table(code_of(page))["game_type"] == "duel-variant"
    finally:
        context.close()


def test_the_front_door_resumes_the_last_game_without_asking(server, browser):
    page = at_a_game(browser)
    try:
        code = code_of(page)
        page.goto(f"{BASE_URL}/", wait_until="load")
        page.wait_for_selector("#players .player-row")
        assert code_of(page) == code
        assert not page.is_visible("#modal")
    finally:
        page.context.close()


def test_dealing_with_the_defaults_is_a_standard_spiral_game_at_four_seats(server, browser):
    page = at_a_game(browser)
    try:
        view = table(deal_from_modal(page))
        assert (view["board_mode"], view["game_type"], view["locked"]) == ("spiral", "standard", [])
        assert len(page.query_selector_all("#players .player-row")) == 4
        assert page.query_selector_all('#players option[value="__close__"]')
    finally:
        page.context.close()


def test_random_deals_a_random_board(server, browser):
    page = at_a_game(browser)
    try:
        page.click("#new-game")
        page.click('#modal-body button[data-choice="board"][data-value="random"]')
        assert pressed(page, "board") == ["random"]
        page.keyboard.press("Escape")
        view = table(deal_from_modal(page, board="random"))
        assert (view["board_mode"], view["game_type"]) == ("random", "standard")
    finally:
        page.context.close()


def test_a_chosen_option_stays_filled_under_the_pointer(server, browser):
    page = at_a_game(browser)
    try:
        page.click("#new-game")
        page.wait_for_selector("#modal.show #deal-new-game")
        page.click('#modal-body button[data-choice="duel"][data-value="true"]')
        on = '#modal-body button[data-choice="duel"][data-value="true"]'
        page.hover(on)
        fill = page.eval_on_selector(on, "(b) => getComputedStyle(b).backgroundColor")
        accent = page.evaluate("""() => {
            const probe = document.createElement("div");
            probe.style.background = "var(--accent)";
            document.body.appendChild(probe);
            const colour = getComputedStyle(probe).backgroundColor;
            probe.remove();
            return colour;
        }""")
        assert fill == accent
        page.keyboard.press("Escape")
    finally:
        page.context.close()


def test_duel_on_deals_a_two_seat_table_under_the_duel_rules(server, browser):
    page = at_a_game(browser)
    try:
        code = deal_from_modal(page, duel="true")
        view = table(code)
        assert (view["game_type"], view["winning_points"], view["locked"]) == ("duel-variant", 15, [2, 3])
        assert view["board_mode"] == "spiral" and view["seats_fixed"] is True
        # Seated: our row and the one open seat, which can take a bot or be
        # closed to practise alone; the two closed seats are not on the
        # roster at all.
        rows = page.query_selector_all("#players .player-row")
        assert len(rows) == 2
        assert page.query_selector_all("#players select")
        assert page.query_selector_all('#players option[value="__close__"]')
        assert "15 to win" in page.get_attribute("#players .stat[title^='Victory points']", "title")

        # The bot the picker seats plays the duel.
        page.select_option("#players select", "test-trader")
        page.wait_for_function("() => state.seats[0].kind !== 'empty' && state.seats[1].kind !== 'empty'")
        assert table(code)["waiting_for"] == []

        # Watching the full table from elsewhere: the same two rows.
        watcher = browser.new_context().new_page()
        try:
            watcher.goto(f"{BASE_URL}/{code}", wait_until="load")
            watcher.wait_for_selector("#players .player-row")
            assert watcher.evaluate("() => state.seat") is None
            assert len(watcher.query_selector_all("#players .player-row")) == 2
        finally:
            watcher.context.close()
    finally:
        page.context.close()
