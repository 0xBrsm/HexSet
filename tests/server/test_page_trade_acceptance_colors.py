# SPDX-License-Identifier: GPL-3.0-only
"""The trade-acceptance pane's check buttons: colored (accent) for the row
that takes the deal exactly as offered, gray for a counter -- so the one
thing worth pressing without reading the row first is the accept, not
whichever button happens to be closest.

Reported back after the "just say Accepted, don't restate the trade" fix
(`index.html`'s `acceptancePane`) also left every row's check button
colored regardless of what it answered. That part of the pane is pure
rendering -- `acceptancePane` is a function of `state.trade_round` and
nothing else -- so this test drives a page to a real, server-issued
`state` (same fixture shape as `test_page_setup_turn.py`, just enough to
get seat 0 seated with real seats 1-3 to draw rows for) and then swaps in
a synthetic `trade_round` client-side before re-rendering, rather than
threading a real accept and a real counter through the trading protocol's
own gates and floors (`hexset.trading`, already covered end to end by
`tests/server/test_trade_round_api.py`) just to get two rows on screen.
The seam this crosses -- state in, DOM out -- is exactly `acceptancePane`'s
own contract, so exercising it this way tests the real function, not a
stand-in for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.game import to_move
from hexset.server.api import Config, Seat, SeatKind, build_session

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

PORT = 8795
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "TRADCL"
SECRET = "a returning browser's own secret, trade-colors edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat(name: str) -> Seat:
    return Seat(kind=SeatKind.BOT, name=name, spec="heximax")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session._apply(seat, action)


@pytest.fixture(scope="module")
def seated_game(tmp_path_factory):
    """Just enough of a real table for `state.players` to have four real
    rows to draw -- who is sitting where matters to this test (the pane's
    seat pips and labels), the actual game state past that doesn't. Same
    shape as `test_page_setup_turn.py::held_game`, minus caring where the
    hold leaves seat 0: any state the browser can load is enough of a
    canvas for the injected `trade_round`.
    """
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
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(seated_game),
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
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)

            page.evaluate(
                """() => {
                    state.pending = [];
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

            # Row 0 is our own offer (no pip, no check). Rows 1-3 are seats
            # 1-3 in seat order.
            assert rows.nth(1).inner_text().strip() == "Accepted"
            assert check_is_primary(1), "the accept is the deal to take -- it stays colored"
            assert not check_is_primary(2), "a counter is a different deal, not the one on offer"

            context.close()
        finally:
            browser.close()
