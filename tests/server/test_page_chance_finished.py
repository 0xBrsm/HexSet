# SPDX-License-Identifier: GPL-3.0-only
"""A finished game and its past rounds, in a real browser, show no win
chance: the bar is the guide's, and only while a game is live. The corner
keeps the sound toggle once the game is over. Skipped without Playwright or
onnxruntime.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire

from _page_server import launch, serving

pytest.importorskip("onnxruntime")

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

PORT = 8824
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "CHN123"
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}
VALUED = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract6-valued.onnx"


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server outlives each
    test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def finished_game(tmp_path_factory):
    """A game played out at random, a model file at seat 1, journalled."""
    from hexset.clients.onnxmeta import stamp

    directory = tmp_path_factory.mktemp("finished")
    model = tmp_path_factory.mktemp("models") / "valued.onnx"
    stamp(VALUED, model, name="valued", modes=["standard"], version="1", commit="0",
          profile={"label": "Valued"})
    trader = Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        Seat(kind=SeatKind.BOT, name="valued", spec=str(model)),
        trader,
        Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader"),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    rng = random.Random(4)
    for _ in range(4000):
        if is_over(session.game):
            break
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        session.submit(to_move(session.game), action_to_wire(rng.choice(legal_actions(session.game))))
    assert is_over(session.game), "the seeded game did not finish"
    return directory


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = launch(playwright)
        try:
            yield chromium
        finally:
            chromium.close()


def test_a_finished_game_and_its_past_rounds_show_no_chance(finished_game, browser):
    with serving(PORT, str(finished_game)):
        context = browser.new_context()
        context.add_init_script(
            "try {"
            f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
            f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
            "} catch (e) {}"
        )
        page = context.new_page()
        try:
            with page.expect_response(lambda r: "/api/reclaim" in r.url):
                page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
            page.wait_for_function("() => state && state.game_over === true")
            assert page.inner_text("#replay-pct") == ""
            # Over, the corner keeps sound: it is set between games too.
            assert page.is_visible("#sound-toggle")
            assert page.query_selector("#chance-toggle") is None

            # A swipe right across the board steps back a move; the LIVE pill
            # brings the game back. No buttons in the replay bar.
            # One pill each side of the strip: the win chance (none without the
            # guide) and the round, which returns to the game once stepped back.
            assert page.query_selector("#replay-bar button:not(#replay-pct):not(#replay-round)") is None
            assert not page.is_visible("#replay-pct"), "no win chance without the guide"
            assert page.query_selector("#replay-round.back") is None
            box = page.query_selector("#board-pane").bounding_box()
            y = box["y"] + box["height"] / 2
            page.mouse.move(box["x"] + box["width"] * 0.3, y)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.7, y, steps=5)
            page.mouse.up()
            page.wait_for_function("() => state.replay && replayStep !== null")
            page.wait_for_timeout(300)
            assert "%" not in page.inner_text("#replay-strip"), "a past round shows no chance either"
            assert page.is_visible("#replay-round.back")
            assert not page.is_visible("#undo-build"), "no guide shown: nothing to play on with"
            page.click("#replay-round")
            page.wait_for_function("() => replayStep === null")
            assert page.query_selector("#replay-round.back") is None
        finally:
            context.close()
