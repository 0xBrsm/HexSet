# SPDX-License-Identifier: GPL-3.0-only
"""The trade-acceptance pane's check buttons: accent for the row that takes
the deal exactly as offered, gray for a counter, none for an answer the hand
cannot cover.

`acceptancePane` is a function of `state.trade_round` and nothing else, so
the page is driven to a real server-issued `state` and a synthetic
`trade_round` swapped in client-side before re-rendering, rather than
threading a real accept and counter through `hexset.trading`'s gates.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.game import to_move
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

PORT = 8795
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "TRADCL"
SECRET = "a returning browser's own secret, trade-colors edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat(name: str) -> Seat:
    return Seat(kind=SeatKind.BOT, name=name, spec="test-trader")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session.apply_action(seat, action)


@pytest.fixture(scope="module")
def seated_game(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tradecolors")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat("Bez"),
        _bot_seat("Cy"),
        _bot_seat("Del"),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=41), first=0)
    assert to_move(session.game) == 0
    _place(session, 0, ActionType.SETUP_SETTLEMENT)
    _place(session, 0, ActionType.SETUP_ROAD)
    return directory


@pytest.fixture(scope="module")
def running_server(seated_game):
    with serving(PORT, str(seated_game)) as base:
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


def test_a_counters_check_is_gray_and_an_accepts_is_colored(running_server):
    """Seat 1 accepted the offer exactly as sent, seat 2 countered, seat 3
    hasn't answered. Only seat 1's check should carry `.primary` -- the
    accent color -- afterward."""
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            context, page = _seated_page(browser)

            page.evaluate(
                """() => {
                    state.pending = [];
                    state.players.find((p) => p.seat === 0).hand = {Wood: 1, Brick: 1, Sheep: 0, Wheat: 0, Ore: 0};
                    state.trade_round = {
                        offer: {actor: 0, bundle: [-1, 1, 0, 0, 0]},
                        responses: [
                            {seat: 1, kind: "accept", bundle: [-1, 1, 0, 0, 0]},
                            {seat: 2, kind: "counter", bundle: [0, -1, 1, 0, 0]},
                        ],
                        awaiting: [3],
                    };
                    render();
                }"""
            )
            page.wait_for_selector("#modal.show .pane-row")

            rows = page.locator("#modal .pane-row")
            assert rows.count() == 4, "our own row, plus one each for seats 1/2/3"

            def check_is_primary(row_index: int) -> bool:
                btn = rows.nth(row_index).locator("button.modal-btn")
                classes = (btn.get_attribute("class") or "").split()
                return "primary" in classes

            # Row 0 is our own offer; rows 1-3 are seats 1-3 in seat order.
            assert rows.nth(1).inner_text().strip() == "Accepted"
            assert check_is_primary(1), "the accept is the deal to take -- it stays colored"
            assert not check_is_primary(2), "a counter is a different deal, not the one on offer"

            context.close()
        finally:
            browser.close()


def test_an_answer_the_hand_cannot_cover_is_shaded_and_has_no_check(running_server):
    """Our open offer is any card for a Brick. Seat 1 counters with a Wood
    for an Ore this hand doesn't hold; seat 2 counters with a Wood for the
    Sheep it does. Only seat 2's row can be taken, and seat 1's Ore is
    shaded as on an offer received."""
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            context, page = _seated_page(browser)

            page.evaluate(
                """() => {
                    state.pending = [];
                    state.players.find((p) => p.seat === 0).hand = {Wood: 1, Brick: 0, Sheep: 1, Wheat: 3, Ore: 0};
                    state.trade_round = {
                        offer: {actor: 0, bundle: [0, 1, 0, 0, 0], any: -1},
                        responses: [
                            {seat: 1, kind: "counter", bundle: [1, 0, 0, 0, -1]},
                            {seat: 2, kind: "counter", bundle: [1, 0, -1, 0, 0]},
                            {seat: 3, kind: "pass", bundle: null},
                        ],
                        awaiting: [],
                    };
                    render();
                }"""
            )
            page.wait_for_selector("#modal.show .pane-row")
            rows = page.locator("#modal .pane-row")
            assert rows.count() == 4

            short = rows.nth(1)
            assert short.locator("button.modal-btn").count() == 0, "a counter this hand can't pay has no check"
            assert short.locator(".card.shortfall").count() == 1, "the Ore this hand lacks is shaded"
            payable = rows.nth(2)
            assert payable.locator("button.modal-btn").count() == 1
            assert payable.locator(".card.shortfall").count() == 0

            context.close()
        finally:
            browser.close()
