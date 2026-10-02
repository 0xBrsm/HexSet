# SPDX-License-Identifier: GPL-3.0-only
"""A spectator's click-to-reveal at a finished game: opening or closing a
revealed row must not move the rest of the page. `#hand` sits in
`#play-area`, sized to its own content with `#log` taking what is left, so
any change in `#hand`'s height shoves `#log` up or down.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire

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

PORT = 8796
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "FINCRD"

# The regions a click-to-reveal must not move. `#hand` is not one: its own
# content is exactly what a click changes.
STABLE_REGIONS = ["#board-pane", "#log", "#players"]


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _play_out(session, rng: random.Random) -> None:
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
    directory = tmp_path_factory.mktemp("finished-cards")
    seats = [_bot_seat(), _bot_seat(), _bot_seat(), _bot_seat()]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    _play_out(session, random.Random(4))
    return directory


@pytest.fixture(scope="module")
def running_server(finished_game):
    with serving(PORT, str(finished_game)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _boxes(page) -> dict:
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
        browser = launch(playwright)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
            page.wait_for_selector(".player-row")

            initial = _boxes(page)
            hint = page.locator("#hand .hand-hint")
            # inner_text is the rendered text, so all-caps is checked as such.
            assert hint.inner_text() == "CLICK A PLAYER TO SEE THEIR CARDS"
            pane_box, hint_box = page.locator("#hand").bounding_box(), hint.bounding_box()
            assert abs((hint_box["x"] + hint_box["width"] / 2) - (pane_box["x"] + pane_box["width"] / 2)) < 2
            assert abs((hint_box["y"] + hint_box["height"] / 2) - (pane_box["y"] + pane_box["height"] / 2)) < 2
            assert page.locator("#hand .card").count() == 0
            assert page.locator("#hand .hand-banner").count() == 0

            rows = page.locator(".player-row")
            assert rows.count() == 4

            rows.nth(1).click()
            page.wait_for_selector(".player-row-open")
            after_open = _boxes(page)
            _assert_boxes_unchanged(initial, after_open)
            hand = page.locator("#hand")
            assert hand.locator(".hand-col").count() == 2
            headers = hand.locator(".hand-col-header").all_inner_texts()
            assert any("Resource Cards" in h for h in headers)
            assert any("Development Cards" in h for h in headers)
            assert hand.locator(".hand-banner").count() == 0
            assert hand.locator(".hand-hint").count() == 0
            assert hand.locator(".hand-cols-hidden").count() == 0

            rows.nth(1).click()
            assert page.locator(".player-row-open").count() == 0
            after_close = _boxes(page)
            _assert_boxes_unchanged(initial, after_close)
            assert page.locator("#hand .hand-hint").count() == 1

            rows.nth(2).click()
            page.wait_for_selector(".player-row-open")
            after_second_open = _boxes(page)
            _assert_boxes_unchanged(initial, after_second_open)
            assert hand.locator(".hand-col").count() == 2

            rows.nth(2).click()
            assert page.locator(".player-row-open").count() == 0
            after_second_close = _boxes(page)
            _assert_boxes_unchanged(initial, after_second_close)

            context.close()
        finally:
            browser.close()
