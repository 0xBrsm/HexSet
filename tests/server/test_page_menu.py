# SPDX-License-Identifier: GPL-3.0-only
"""The game page's menu in a real browser: the header's hamburger opens it on
the Log tab, with this game's code and a button that copies its link; the
History tab lists every game this browser has played, newest first, each with
its result once the table says it, and opens one on a tap. A game played
before the browser kept records is read from its table, and a code with
nothing behind it is forgotten. Skipped without Playwright, which is not one
of this project's extras.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire

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

PORT = 8816
BASE_URL = f"http://127.0.0.1:{PORT}"
# A game this browser played before it kept records, journalled and over.
OLD = "old123"
# A code this browser holds a token for that the server has nothing behind.
GONE = "gone99"
# A duel this browser won, as its record keeps it: over, so never read again.
WON = "won111"
WON_RECORD = {"type": "duel-variant", "seat": 1, "names": ["Bo", "Ada"], "over": True, "winner": 1}
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _play_out(session, rng: random.Random) -> None:
    for _ in range(2000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))
    raise AssertionError("the seeded game did not finish")


@pytest.fixture(scope="module")
def games_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("menu")
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT)]
    seats += [Seat(kind=SeatKind.BOT, name=f"bot-{i}", spec="test-trader") for i in (1, 2, 3)]
    session = build_session(OLD, seats, Config(games_dir=str(directory), seed=99), first=0)
    _play_out(session, random.Random(4))
    return directory


@pytest.fixture(scope="module")
def running_server(games_dir):
    with serving(PORT, str(games_dir)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = launch(playwright)
        try:
            yield chromium
        finally:
            chromium.close()


def _old_view() -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/api/table/{OLD}", timeout=5) as response:
        return json.loads(response.read())


def _history(page) -> list[list[str]]:
    return page.eval_on_selector_all(
        "#history .history-row",
        "(rows) => rows.map((r) => [r.querySelector('b').textContent,"
        " r.querySelector('.history-result').textContent, r.tagName])",
    )


def test_the_menu_hands_out_this_game_and_lists_the_games_played(running_server, browser):
    # Random play to the turn limit: over, with nobody at 10.
    view = _old_view()
    assert view["game_over"] and view["winner"] is None
    context = browser.new_context()
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=BASE_URL)
    context.add_init_script(
        "try {"
        "  if (!localStorage.getItem('hexset.secret')) {"
        f"    localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"    localStorage.setItem('hexset.token.{OLD}', 'minted before the restart');"
        f"    localStorage.setItem('hexset.token.{GONE}', 'minted before the restart');"
        f"    localStorage.setItem('hexset.game.{WON}',"
        f"      JSON.stringify({{...{json.dumps(WON_RECORD)}, at: Date.now() - 86400000}}));"
        "  }"
        "} catch (e) {}"
    )
    page = context.new_page()
    try:
        _, deal = front_door(page, BASE_URL)
        code = deal["code"]
        page.wait_for_selector("#players .player-row")
        assert not page.is_visible("#log-sheet")

        # Opened on the Log tab, with this game's code and its link.
        page.click("#open-log")
        page.wait_for_selector("#log-sheet.show")
        assert page.text_content("#this-game-code") == code
        assert page.get_attribute('#menu-tabs [data-tab="log"]', "aria-pressed") == "true"
        assert page.is_visible("#log-full") and not page.is_visible("#history")
        page.click("#copy-link")
        page.wait_for_function("document.getElementById('copy-link').textContent === 'Copied'")
        assert page.evaluate("navigator.clipboard.readText()") == f"{BASE_URL}/{code}"
        # This server keeps no rankings, so the menu links none.
        assert not page.is_visible("#rankings-link")

        # History, newest first: this game, kept as it was dealt; the duel
        # won yesterday, from its record alone; the old game read from its
        # table, whose public view cannot say which seat was ours, and with
        # no date, which nothing kept; the code with nothing behind it gone
        # from the list and from storage.
        page.click('#menu-tabs [data-tab="history"]')
        assert page.is_visible("#history") and not page.is_visible("#log-full")
        page.wait_for_function(
            f"document.querySelector('#history a[href=\"/{OLD}\"] .history-result')?.textContent === 'No winner'"
            " && document.querySelectorAll('#history .history-row').length === 3"
        )
        assert _history(page) == [[code, "In progress", "DIV"], [WON, "Won", "A"], [OLD, "No winner", "A"]]
        assert "this game" in page.text_content("#history .history-row.current")
        won = f'#history a[href="/{WON}"]'
        assert page.text_content(f"{won} .history-who") == "vs Bo"
        assert page.text_content(f"{won} .history-what").startswith(f"{WON} · Duel · ")
        assert page.get_attribute(f"{won} .history-result", "class") == "history-result won"
        assert page.text_content(f'#history a[href="/{OLD}"] .history-what') == f"{OLD} · Standard"
        assert page.evaluate(f"localStorage.getItem('hexset.token.{GONE}')") is None
        old = json.loads(page.evaluate(f"localStorage.getItem('hexset.game.{OLD}')"))
        assert old["over"] and old["seat"] is None and old["at"] is None

        # A tap opens the old game, where this browser proves its seat; the
        # record then knows the seat was ours, and drops it from the opponents.
        with page.expect_response(lambda r: "/api/reclaim" in r.url):
            page.click(f'#history a.history-row[href="/{OLD}"]')
        page.wait_for_url(f"{BASE_URL}/{OLD}")
        page.wait_for_selector("#players .player-row")
        page.wait_for_function(f"JSON.parse(localStorage.getItem('hexset.game.{OLD}')).seat === 0")
        page.click("#open-log")
        page.wait_for_selector("#log-sheet.show")
        # A fresh page opens the menu on the Log tab again.
        assert page.get_attribute('#menu-tabs [data-tab="log"]', "aria-pressed") == "true"
        page.click('#menu-tabs [data-tab="history"]')
        page.wait_for_selector("#history .history-row.current")
        assert _history(page) == [[code, "In progress", "A"], [WON, "Won", "A"], [OLD, "No winner", "DIV"]]
        assert page.text_content("#history .history-row.current .history-who") == "vs bot-1, bot-2, bot-3"
        assert page.text_content("#history .history-row.current .history-what") == f"{OLD} · Standard · this game"
        page.keyboard.press("Escape")
        assert not page.is_visible("#log-sheet")
    finally:
        context.close()
