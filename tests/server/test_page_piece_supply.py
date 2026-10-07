# SPDX-License-Identifier: GPL-3.0-only
"""The piece-supply corner follows the seat being read. Playing, it is our
own seat's, with nothing to click. Reading the table -- a finished game, or
a round of one stepped back to -- it is the seat whose row is open, as the
hand area is, nobody's until one is, and counted off the round on screen.
"""

from __future__ import annotations

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

PORT = 8800
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "SUPPLY"


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
    directory = tmp_path_factory.mktemp("piece-supply")
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


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}") as response:
        return json.load(response)


def _left(view: dict, supply: dict, seat: int) -> dict:
    """What `seat` has left to place on the position `view` holds."""
    owned = [v for v, owner in enumerate(view["vertex_owner"]) if owner == seat]
    return {
        "settlement": supply["settlement"]
        - sum(1 for v in owned if view["vertex_building"][v] == 1),
        "city": supply["city"] - sum(1 for v in owned if view["vertex_building"][v] == 2),
        "road": supply["road"] - sum(1 for owner in view["edge_owner"] if owner == seat),
    }


def _shown(page) -> dict:
    return {
        kind: int(page.inner_text(f"#piece-supply .supply-{kind} .count"))
        for kind in ("settlement", "city", "road")
    }


def _tiles(page) -> int:
    return page.locator("#piece-supply .supply-tile").count()


def test_reading_a_finished_game_shows_the_open_seats_pieces(running_server):
    code = CODE.lower()
    supply = _get(f"/api/table/{code}/board")["piece_supply"]
    end = _get(f"/api/table/{code}")
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = browser.new_page()
            page.goto(f"{BASE_URL}/{code}", wait_until="load")
            page.wait_for_selector(".player-row")
            rows = page.locator(".player-row")
            assert _tiles(page) == 0

            rows.nth(1).click()
            page.wait_for_selector(".player-row-open")
            assert _tiles(page) == 3
            assert _shown(page) == _left(end, supply, 1)

            rows.nth(2).click()
            assert _shown(page) == _left(end, supply, 2)

            rows.nth(2).click()
            assert page.locator(".player-row-open").count() == 0
            assert _tiles(page) == 0
        finally:
            browser.close()


def test_a_replayed_round_shows_the_open_seats_pieces_as_they_stood(running_server):
    code = CODE.lower()
    supply = _get(f"/api/table/{code}/board")["piece_supply"]
    opening = _get(f"/api/table/{code}/replay?round=0")
    first = _get(f"/api/table/{code}/replay?round=1")
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = browser.new_page()
            page.goto(f"{BASE_URL}/{code}", wait_until="load")
            page.wait_for_selector(".player-row")
            page.locator(".player-row").nth(3).click()
            page.wait_for_selector(".player-row-open")

            label = page.locator("#replay-label")
            page.click("#replay-first")
            page.wait_for_function(
                "() => document.getElementById('replay-label').textContent.startsWith('ROUND 0 /')"
            )
            # The open row survives the step back, and so does its supply.
            assert page.locator(".player-row-open").count() == 1
            assert _shown(page) == _left(opening, supply, 3)
            # Two settlements and two roads each, once the opening is done.
            assert _shown(page) == {
                "settlement": supply["settlement"] - 2,
                "city": supply["city"],
                "road": supply["road"] - 2,
            }

            page.click("#replay-next")
            page.wait_for_function(
                "() => document.getElementById('replay-label').textContent.startsWith('ROUND 1 /')"
            )
            assert label.inner_text().startswith("ROUND 1 /")
            assert _shown(page) == _left(first, supply, 3)

            page.locator(".player-row").nth(3).click()
            assert _tiles(page) == 0
        finally:
            browser.close()


def test_a_seated_player_sees_their_own_pieces_without_a_click(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = browser.new_page()
            code = front_door(page, BASE_URL)[1]["code"]
            page.wait_for_selector("#piece-supply .supply-tile")
            supply = _get(f"/api/table/{code}/board")["piece_supply"]
            assert _tiles(page) == 3
            assert _shown(page) == supply
        finally:
            browser.close()
