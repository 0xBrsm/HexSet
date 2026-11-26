# SPDX-License-Identifier: GPL-3.0-only
"""The win chance is the guide's alone: a seat at a table where a model sits
but no guide is on is shown none, nor is a spectator, and the board's corner
holds no win chance toggle. The guide's own bar is checked in
test_page_guide.py. Skipped without Playwright or onnxruntime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _page_server import front_door, launch, serving
from hexset.server import api

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

PORT = 8823
BASE_URL = f"http://127.0.0.1:{PORT}"
VALUED = Path(__file__).parent.parent / "clients" / "fixtures" / "stub-contract6-valued.onnx"


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from hexset.clients.onnxmeta import stamp

    models = tmp_path_factory.mktemp("models")
    stamp(VALUED, models / "valued.onnx", name="valued", modes=["standard"], version="1", commit="0",
          profile={"label": "Valued"})
    patch = pytest.MonkeyPatch()
    patch.setattr(api, "MODELS_DIR", models)
    try:
        with serving(PORT) as url:
            yield url
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chromium = launch(playwright)
        try:
            yield chromium
        finally:
            chromium.close()


def test_without_the_guide_no_seat_or_spectator_sees_a_chance(server, browser):
    page = browser.new_context().new_page()
    try:
        front_door(page, BASE_URL)
        page.wait_for_selector("#players .bot-pick")
        assert page.query_selector("#chance-toggle") is None, "the corner holds no win chance toggle"
        page.click("#players .bot-pick")
        page.wait_for_selector("#modal.show .bot-card")
        page.click('.bot-card[data-model="valued"]')
        page.wait_for_selector("#modal.show", state="hidden")
        assert "Valued" in page.inner_text("#players")

        # Every seat filled, so the game plays and the link watches rather than joins.
        while page.query_selector("#players .bot-pick.unset"):
            page.click("#players .bot-pick.unset")
            page.wait_for_selector("#modal.show .bot-card")
            page.click('.bot-card[data-model="test-trader"]')
            page.wait_for_selector("#modal.show", state="hidden")
            page.wait_for_timeout(200)
        page.wait_for_function("() => state.waiting_for.length === 0")
        page.wait_for_timeout(500)
        assert page.inner_text("#replay-pct") == "", "a model at the table shows no chance without the guide"

        watcher = browser.new_context().new_page()
        try:
            watcher.goto(page.url, wait_until="load")
            watcher.wait_for_selector("#players .player-row")
            assert watcher.evaluate("() => state.seat") is None
            watcher.wait_for_timeout(500)
            assert watcher.inner_text("#replay-pct") == "", "a spectator is shown no seat's chance"
        finally:
            watcher.context.close()
    finally:
        page.context.close()
