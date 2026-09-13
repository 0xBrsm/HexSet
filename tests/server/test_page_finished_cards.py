# SPDX-License-Identifier: GPL-3.0-only
"""A spectator's click-to-reveal at a finished game (`index.html`'s
`renderHand`/`renderPlayers`), for the one thing `test_page_finished_game.py`
never checked: what happens to the *rest* of the page when a row is opened.

Every hand is revealed once a game is over (`webplay`: `reveal = over or
omniscient or p == viewer`), and clicking an open-able row (`.player-row-
openable`, see `renderPlayers`) is how a spectator or a seat's own returning
holder reads one. Before this fix, `renderHand` had exactly two shapes: a
one-line `.hand-hint` ("Click a player to see their cards.") when nobody was
picked, or the full two-column card grid once somebody was — nothing in
between, and nothing naming whose cards the grid was even showing. Swapping
one for the other changes `#hand`'s own height by however much a header row
and a row of cards cost over a single line of muted text, and `#hand` sits
in `#play-area`, which is sized to its own content (see the CSS comment on
`#play-area`) with `#log` taking whatever's left below it — so every click
that opened or closed a row shoved `#log` up or down by that same amount.

This drives a finished game the same way `test_page_finished_game.py` does
(a session played out to `game_over`, no server involved yet), opens it as a
plain spectator (no seat, no reclaim — the defect never needed one), and
checks the geometry of the page's own regions across a run of opens and
closes, not just what `#hand` ends up containing.
"""

from __future__ import annotations

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
from hexset.game import is_over, to_move
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

# Its own port and code, so this module can run alongside the other page
# suites without colliding.
PORT = 8796
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "FINCRD"

# The regions a click-to-reveal has no business moving. #hand is deliberately
# not one of them -- its own content is exactly what a click changes -- but
# everything around it is.
STABLE_REGIONS = ["#board-pane", "#log", "#players"]


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="heximax", spec="heximax")


def _play_out(session, rng: random.Random) -> None:
    """Random legal play until somebody wins. Same seed and shape as
    `test_page_finished_game.py`'s own fixture, which is known to converge
    well inside this bound."""
    for _ in range(2000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(
                session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN))
            )
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))
    raise AssertionError("the seeded game did not finish")


@pytest.fixture(scope="module")
def finished_game(tmp_path_factory):
    """A played-out game's journal: four bots, nobody's browser seated --
    the defect is a spectator's, not a returning seat's, so there is no
    reclaim to arrange here."""
    directory = tmp_path_factory.mktemp("finished-cards")
    seats = [_bot_seat(), _bot_seat(), _bot_seat(), _bot_seat()]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    _play_out(session, random.Random(4))
    return directory


@pytest.fixture(scope="module")
def running_server(finished_game):
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(finished_game),
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


def _boxes(page) -> dict:
    """Bounding boxes of the regions a click-to-reveal must never move."""
    return {
        sel: page.eval_on_selector(
            sel,
            "e => { const r = e.getBoundingClientRect(); "
            "return {x: r.x, y: r.y, w: r.width, h: r.height}; }",
        )
        for sel in STABLE_REGIONS
    }


def _assert_boxes_unchanged(before: dict, after: dict) -> None:
    for sel in STABLE_REGIONS:
        b, a = before[sel], after[sel]
        assert a == b, f"{sel} moved: {b} -> {a}"


def test_clicking_a_player_reveals_cards_in_place(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            # A plain spectator: no seat, no localStorage, nothing to reclaim.
            # The table is public and the game is over, so this is enough to
            # see every hand (webplay's reveal = over or omniscient or
            # p == viewer).
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
            page.wait_for_selector(".player-row")

            initial = _boxes(page)
            # Nobody picked yet: the hint, and no actual cards -- whether the
            # pane also carries the two (empty) columns at this point is the
            # fix's own business, not something this test pins down.
            hint = page.locator("#hand .hand-hint")
            # inner_text is the rendered text, so the all-caps requirement is checked as such.
            assert hint.inner_text() == "CLICK A PLAYER TO SEE THEIR CARDS"
            # Centred in the pane, not a line above the columns.
            pane_box, hint_box = page.locator("#hand").bounding_box(), hint.bounding_box()
            assert abs((hint_box["x"] + hint_box["width"] / 2) - (pane_box["x"] + pane_box["width"] / 2)) < 2
            assert abs((hint_box["y"] + hint_box["height"] / 2) - (pane_box["y"] + pane_box["height"] / 2)) < 2
            assert page.locator("#hand .card").count() == 0
            assert page.locator("#hand .hand-banner").count() == 0

            rows = page.locator(".player-row")
            assert rows.count() == 4

            # Open seat 1's cards.
            rows.nth(1).click()
            page.wait_for_selector(".player-row-open")
            after_open = _boxes(page)
            _assert_boxes_unchanged(initial, after_open)
            # The cards land in the same container -- and with the same
            # section structure -- the viewer's own hand uses during play.
            hand = page.locator("#hand")
            assert hand.locator(".hand-col").count() == 2
            headers = hand.locator(".hand-col-header").all_inner_texts()
            assert any("Resource Cards" in h for h in headers)
            assert any("Development Cards" in h for h in headers)
            # No title over the columns -- the highlighted roster row says whose
            # they are -- and no hint either.
            assert hand.locator(".hand-banner").count() == 0
            assert hand.locator(".hand-hint").count() == 0
            assert hand.locator(".hand-cols-hidden").count() == 0

            # Close it again -- back to the hint, and back to the original
            # geometry, not just close to it.
            rows.nth(1).click()
            assert page.locator(".player-row-open").count() == 0
            after_close = _boxes(page)
            _assert_boxes_unchanged(initial, after_close)
            assert page.locator("#hand .hand-hint").count() == 1

            # A different row now.
            rows.nth(2).click()
            page.wait_for_selector(".player-row-open")
            after_second_open = _boxes(page)
            _assert_boxes_unchanged(initial, after_second_open)
            assert hand.locator(".hand-col").count() == 2

            # Off again.
            rows.nth(2).click()
            assert page.locator(".player-row-open").count() == 0
            after_second_close = _boxes(page)
            _assert_boxes_unchanged(initial, after_second_close)

            context.close()
        finally:
            browser.close()
