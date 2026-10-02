# SPDX-License-Identifier: GPL-3.0-only
"""Open offers in the trade modal: the "any" card ("?") on either side of
an offer being composed, an incoming open offer that can only be countered,
and our own open offer in the acceptance pane.

Driven like the acceptance-pane test: a real server-issued `state`, with the
round swapped in client-side before re-rendering; the one POST that matters
(the offer itself) is intercepted, so its body is read rather than inferred.
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

PORT = 8799
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "TRADAN"
SECRET = "a returning browser's own secret, trade-any edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat(name: str) -> Seat:
    return Seat(kind=SeatKind.BOT, name=name, spec="test-trader")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session.apply_action(seat, action)


@pytest.fixture(scope="module")
def seated_game(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tradeany")
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


def test_an_incoming_open_offer_can_only_be_countered(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)
            page.evaluate(
                """() => {
                    state.trade_round = null;
                    state.pending = [{actor: 1, bundle: [0, 0, 0, 0, 1], any: -1}];
                    render();
                }"""
            )
            page.wait_for_selector("#modal.show .counter-row")
            quoted = page.locator("#modal .counter-row").first
            assert quoted.locator(".card.any-card").count() == 1, "their any card, drawn as ?"
            assert quoted.locator(".card.any-card").inner_text().strip() == "?"
            assert quoted.locator("button.modal-btn").is_disabled(), \
                "an open offer can't be taken as it stands"
            # The counter composer offers no any card: a counter names its cards.
            assert page.locator("#modal .trade-row .any-card").count() == 0
            context.close()
        finally:
            browser.close()


def test_our_open_offer_shows_its_any_card_in_the_acceptance_pane(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)
            page.evaluate(
                """() => {
                    state.pending = [];
                    state.trade_round = {
                        offer: {actor: 0, bundle: [0, 0, 0, 0, 1], any: -1},
                        responses: [{seat: 1, kind: "counter", bundle: [-1, 0, 0, 0, 1]}],
                        awaiting: [2, 3],
                    };
                    render();
                }"""
            )
            page.wait_for_selector("#modal.show .pane-row")
            ours = page.locator("#modal .pane-row").first
            assert ours.locator(".card.any-card").count() == 1
            countered = page.locator("#modal .pane-row").nth(1)
            assert countered.locator(".card.any-card").count() == 0, "a counter is concrete"
            context.close()
        finally:
            browser.close()


def test_composing_an_offer_for_any_card_posts_want_any(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)
            posted = []

            def capture(route):
                posted.append(json.loads(route.request.post_data or "{}"))
                route.fulfill(status=400, content_type="application/json",
                              body=json.dumps({"error": "captured"}))

            page.route("**/trade/round", capture)
            page.evaluate(
                """() => {
                    state.pending = [];
                    state.trade_round = null;
                    state.phase = "MAIN";
                    state.to_move = state.seat;
                    const me = state.players.find((pl) => pl.seat === state.seat);
                    me.hand = Object.fromEntries(board.resources.map((r, i) => [r, i === 0 ? 1 : 0]));
                    openModal("trade", {give: 0});
                }"""
            )
            page.wait_for_selector("#modal.show .trade-row")
            give_any = page.locator("#modal .trade-row").nth(0).locator(".any-card")
            want_any = page.locator("#modal .trade-row").nth(1).locator(".any-card")
            assert give_any.count() == 1 and want_any.count() == 1
            want_any.click()
            classes = (page.locator("#modal .trade-row").nth(0).locator(".any-card").get_attribute("class") or "")
            assert "disabled" in classes.split(), "any cards go on one side of an offer"
            page.locator('#modal button[title="Offer this to every other player"]').click()
            page.wait_for_timeout(300)
            assert posted == [{"give": [1, 0, 0, 0, 0], "want": [0, 0, 0, 0, 0], "give_any": 0, "want_any": 1}]
            context.close()
        finally:
            browser.close()
