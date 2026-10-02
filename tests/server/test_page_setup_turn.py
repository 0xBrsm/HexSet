# SPDX-License-Identifier: GPL-3.0-only
"""The board, mid-setup, at a seat holding its turn open.

A setup road hands the snake on inside the placement itself, so a browser
seat keeps the turn until it ends it (`awaiting_confirm`), which is what
makes a setup placement undoable at all.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.game import to_move
from hexset.server.api import Config, Seat, SeatKind, build_session

from _page_server import serving


try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

# Its own port, so this module can run alongside test_page_finished_game.py.
PORT = 8793
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "SETUP1"
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session.apply_action(seat, action)


@pytest.fixture(scope="module")
def held_game(tmp_path_factory):
    """Stops where the hold begins. Journalled because the server rebuilds the
    table from this file, which also proves a reopen restores the hold.
    """
    directory = tmp_path_factory.mktemp("held")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat(),
        _bot_seat(),
        _bot_seat(),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    assert to_move(session.game) == 0
    _place(session, 0, ActionType.SETUP_SETTLEMENT)
    _place(session, 0, ActionType.SETUP_ROAD)
    assert session.awaiting_confirm == 0, "the fixture is meant to stop mid-hold"
    assert to_move(session.game) != 0, "the engine's snake has moved on"
    return directory


@pytest.fixture(scope="module")
def running_server(held_game):
    with serving(PORT, str(held_game)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _returning_context(browser):
    context = browser.new_context()
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    return context


def _seated_page(browser):
    context = _returning_context(browser)
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    page.wait_for_selector(".player-row")
    return context, page


def _shown(page, fab_id: str) -> bool:
    """`showFab` toggles a class rather than adding the node."""
    return "show" in (page.locator(f"#{fab_id}").get_attribute("class") or "")


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


def test_the_held_seat_sees_its_own_turn_not_the_next_players(running_server, browser):
    context, page = _seated_page(browser)

    # Named for the hold, not the phase: the engine is already in the next
    # seat's SETUP_SETTLEMENT.
    assert page.locator("#phase").inner_text().strip() == "END TURN"

    rows = page.locator(".player-row")
    assert "active" in (rows.nth(0).get_attribute("class") or "")
    for other in (1, 2, 3):
        assert "active" not in (rows.nth(other).get_attribute("class") or "")

    context.close()


def test_the_held_seat_is_offered_end_turn_and_undo(running_server, browser):
    context, page = _seated_page(browser)

    assert _shown(page, "end-turn"), "no way to end the held turn"
    assert _shown(page, "undo-build"), "the take-back the hold exists for"
    assert not _shown(page, "roll-dice")
    assert not _shown(page, "buy-dev-card")

    context.close()


def test_ending_the_held_turn_releases_the_table(running_server, browser):
    context, page = _seated_page(browser)
    assert page.locator("#phase").inner_text().strip() == "END TURN"

    page.click("#end-turn")

    page.wait_for_function(
        "document.getElementById('phase').innerText.trim() !== 'END TURN'",
        timeout=10_000,
    )
    assert not _shown(page, "end-turn")

    context.close()
