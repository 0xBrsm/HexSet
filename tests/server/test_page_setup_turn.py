# SPDX-License-Identifier: GPL-3.0-only
"""The board, mid-setup, at a seat holding its turn open.

A setup road hands the snake on inside the placement itself, so a browser
seat keeps the turn until it ends it (`webplay.GameSession.awaiting_confirm`)
-- which is what makes a setup placement undoable at all. Every part of that
a player can actually see lives in `index.html`, and the Python suite passes
straight through all of it: the hold shipped with three page-visible defects
in a row (the buttons knew about it while the banner and the roster row did
not, then the banner named the next seat's phase), each one found by hand.
This is the test that would have caught them.

Same shape as `test_page_finished_game.py` -- a journal, a server started
against it, a returning browser that reclaims its seat -- and skips the same
way where playwright is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.webplay import action_to_wire

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

# Its own port, so this module can run alongside test_page_finished_game.py.
PORT = 8793
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "SETUP1"
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="heximax", spec="heximax")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session._apply(seat, action)


@pytest.fixture(scope="module")
def held_game(tmp_path_factory):
    """A journal that stops exactly where the hold begins: seat 0 (this
    browser) has placed its settlement and its road, the engine's snake has
    moved on, and the table is waiting for the seat to say it is done.

    Played through the session rather than over HTTP so the fixture can stop
    on that precise step -- and journalled, because the server the test starts
    has nothing in memory and rebuilds the table from this file, which is also
    what proves a reopen restores the hold rather than dropping it.
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
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(held_game),
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


def _returning_context(browser):
    """A browser that sat at this game before the restart: its own secret, and
    a token that died with the process that minted it. See
    `test_page_finished_game.py`'s copy, which this mirrors."""
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
    """A board-corner button is offered. `showFab` toggles a class rather than
    adding or removing the node, so presence proves nothing."""
    return "show" in (page.locator(f"#{fab_id}").get_attribute("class") or "")


def test_the_held_seat_sees_its_own_turn_not_the_next_players(running_server):
    """The defect that shipped: the buttons knew about the hold and nothing
    else did, so the banner and the highlighted row both announced the next
    player's turn at a table that was waiting on this one."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)

            # Named for the hold, not for the phase -- the engine is already
            # in the *next* seat's SETUP_SETTLEMENT, which is what had the
            # board telling a seat with nothing to place to place a
            # settlement.
            assert page.locator("#phase").inner_text().strip() == "END YOUR TURN"

            # And the active row is still ours.
            rows = page.locator(".player-row")
            assert "active" in (rows.nth(0).get_attribute("class") or "")
            for other in (1, 2, 3):
                assert "active" not in (rows.nth(other).get_attribute("class") or "")

            context.close()
        finally:
            browser.close()


def test_the_held_seat_is_offered_end_turn_and_undo(running_server):
    """The whole point of holding the turn: the take-back is reachable. Undo
    was previously wiped by the next seat moving, which with peer-client bots
    happened in milliseconds."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)

            assert _shown(page, "end-turn"), "no way to end the held turn"
            assert _shown(page, "undo-build"), "the take-back the hold exists for"
            # Neither of these belongs to a seat that has already placed.
            assert not _shown(page, "roll-dice")
            assert not _shown(page, "buy-dev-card")

            context.close()
        finally:
            browser.close()


def test_ending_the_held_turn_releases_the_table(running_server):
    """The other half: the hold ends, and it ends by the ordinary End Turn
    button rather than anything setup-specific."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)
            assert page.locator("#phase").inner_text().strip() == "END YOUR TURN"

            page.click("#end-turn")

            # The bots were held behind us and take their placements now, so
            # what the banner says next is theirs -- either way it is no
            # longer ours, and the button that got us here is gone.
            page.wait_for_function(
                "document.getElementById('phase').innerText.trim() !== 'END YOUR TURN'",
                timeout=10_000,
            )
            assert not _shown(page, "end-turn")

            context.close()
        finally:
            browser.close()
