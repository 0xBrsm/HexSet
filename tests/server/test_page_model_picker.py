# SPDX-License-Identifier: GPL-3.0-only
"""A seat's model picker offers what the table's game type seats: a model
file as the name and version it declares, at the game types it declares.
Skipped without Playwright, which is not one of this project's extras.
"""

from __future__ import annotations

import pytest

from _page_server import launch, serving, front_door
from hexset.clients.onnxmeta import with_metadata
from hexset.server import api

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

PORT = 8811
BASE_URL = f"http://127.0.0.1:{PORT}"


def named(**meta: str) -> bytes:
    return with_metadata(b"\x3a\x00", meta)


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    (models / "a.onnx").write_bytes(named(name="net", modes="standard", version="4.1", commit="x"))
    (models / "b.onnx").write_bytes(named(name="net", modes="duel-variant", version="4d.0", commit="y"))
    (models / "c.onnx").write_bytes(named(name="fourseat", version="3.6", commit="z"))
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


def offered(page) -> list[str]:
    """The options of the first empty seat's picker, past "(empty)" and "(closed)"."""
    page.wait_for_selector("#players select")
    return page.eval_on_selector(
        "#players select",
        "(select) => [...select.options].map((o) => o.textContent).filter((t) => !t.startsWith('('))",
    )


def test_a_standard_table_offers_the_standard_file_with_its_version(server, browser):
    page = browser.new_context().new_page()
    try:
        front_door(page, BASE_URL)
        names = offered(page)
        assert "net-4.1" in names and "fourseat-3.6" in names
        assert "net-4d.0" not in names
    finally:
        page.context.close()


def test_a_duel_table_offers_only_what_declares_the_duel(server, browser):
    page = browser.new_context().new_page()
    try:
        page.goto(f"{BASE_URL}/", wait_until="load")
        page.wait_for_selector("#modal.show #deal-new-game")
        page.click('#modal-body button[data-choice="duel"][data-value="true"]')
        page.click("#deal-new-game")
        page.wait_for_selector("#players .player-row")
        names = offered(page)
        assert "net-4d.0" in names
        assert "net-4.1" not in names and not any(n.startswith("fourseat") for n in names)

        # A watcher's page reads the same table and offers the same.
        watcher = page.context.browser.new_context().new_page()
        try:
            watcher.goto(page.url, wait_until="load")
            watcher.wait_for_selector("#players .player-row")
            if watcher.query_selector("#players select"):
                assert offered(watcher) == names
        finally:
            watcher.context.close()
    finally:
        page.context.close()
