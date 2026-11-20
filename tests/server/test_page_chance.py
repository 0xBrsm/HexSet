# SPDX-License-Identifier: GPL-3.0-only
"""The seat's chance of winning as the table's model rates it, under the
player list: shown to the seat once a model sits at the table, never to a
spectator. Skipped without Playwright or onnxruntime.
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


def test_a_seat_sees_its_chance_once_a_model_sits_and_a_spectator_never(server, browser):
    page = browser.new_context().new_page()
    try:
        front_door(page, BASE_URL)
        page.wait_for_selector("#players .bot-pick")
        # The line's row is held from the start, empty until a model sits.
        page.wait_for_selector("#chance.held")
        held = page.eval_on_selector("#log", "(el) => el.getBoundingClientRect().top")
        assert not page.is_visible("#chance .chance-label"), "no model at the table, no chance"
        assert not page.is_visible("#chance-toggle")

        page.click("#players .bot-pick")
        page.wait_for_selector("#modal.show .bot-card")
        assert page.inner_text('.bot-card[data-model="valued"] .bot-card-name') == "Valued"
        page.click('.bot-card[data-model="valued"]')
        page.wait_for_selector("#chance.show")
        assert page.inner_text("#chance .chance-who") == "WIN CHANCE"
        assert "Valued" in page.inner_text("#players")
        assert page.eval_on_selector("#log", "(el) => el.getBoundingClientRect().top") == held, \
            "a chance appearing moves nothing"
        # The corner: sound in it, the percent sign under it.
        percent_sign = page.eval_on_selector("#chance-toggle", "(el) => el.getBoundingClientRect().toJSON()")
        sound = page.eval_on_selector("#sound-toggle", "(el) => el.getBoundingClientRect().toJSON()")
        assert percent_sign["top"] > sound["bottom"] and percent_sign["left"] == sound["left"]
        percent = page.inner_text("#chance .chance-pct")
        assert percent.endswith("%") and 0 <= int(percent[:-1]) <= 100
        # The bar is drawn filled to the chance, not only styled to it: at a
        # read of one half, the fill covers half the bar.
        page.evaluate("() => { chanceRead.chance = 0.5; renderChance(); }")
        assert page.inner_text("#chance .chance-pct") == "50%"
        page.wait_for_timeout(600)
        drawn = page.eval_on_selector(
            "#chance", "(el) => el.querySelector('.chance-fill').getBoundingClientRect().width"
                       " / el.querySelector('.chance-bar').getBoundingClientRect().width")
        assert abs(drawn - 0.5) < 0.02, drawn

        # Every seat filled, so the link watches rather than joins.
        while page.query_selector("#players .bot-pick.unset"):
            page.click("#players .bot-pick.unset")
            page.wait_for_selector("#modal.show .bot-card")
            page.click('.bot-card[data-model="test-trader"]')
            page.wait_for_selector("#modal.show", state="hidden")
            page.wait_for_timeout(200)
        page.wait_for_function("() => state.waiting_for.length === 0")
        assert page.is_visible("#chance.show")

        # A click opens the note, never a hover; a click elsewhere closes it.
        assert not page.is_visible("#chance .chance-note")
        page.hover("#chance")
        assert not page.is_visible("#chance .chance-note")
        page.dispatch_event("#chance", "click")
        assert page.is_visible("#chance .chance-note")
        assert page.inner_text("#chance .chance-note") == "Valued's estimate of your chance of winning, read from your view."
        page.dispatch_event("#log", "click")
        assert not page.is_visible("#chance .chance-note")

        # The corner toggle hides it, for this browser, across a reload, and
        # moves nothing either way.
        top = page.eval_on_selector("#log", "(el) => el.getBoundingClientRect().top")
        assert page.get_attribute("#chance-toggle", "aria-pressed") == "true"
        page.click("#chance-toggle")
        assert not page.is_visible("#chance .chance-label")
        assert page.get_attribute("#chance-toggle", "aria-pressed") == "false"
        assert page.eval_on_selector("#log", "(el) => el.getBoundingClientRect().top") == top
        page.reload(wait_until="load")
        page.wait_for_selector("#chance-toggle:not([hidden])")
        assert not page.is_visible("#chance .chance-label")
        page.click("#chance-toggle")
        page.wait_for_selector("#chance.show")

        watcher = browser.new_context().new_page()
        try:
            watcher.goto(page.url, wait_until="load")
            watcher.wait_for_selector("#players .player-row")
            assert watcher.evaluate("() => state.seat") is None
            watcher.wait_for_timeout(500)
            assert not watcher.is_visible("#chance"), "a spectator is shown no seat's chance, nor its row"
            assert not watcher.is_visible("#chance-toggle")
        finally:
            watcher.context.close()
    finally:
        page.context.close()
