"""The board's viewBox is fitted to what is drawn: every port marker and
vertex sits inside it, the land is centred left to right, and the same
small margin of water is added on every side. Skipped without Playwright, which
is not one of this project's extras.
"""

from __future__ import annotations

import pytest

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

PORT = 8801
BASE_URL = f"http://127.0.0.1:{PORT}"
# index.html's VERTEX_REACH and PORT_REACH, and BOARD_MARGIN.
REACH = 20
MARGIN = 24


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def test_the_viewbox_is_fitted_to_the_vertices_and_port_markers():
    with serving(PORT), sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = browser.new_page()
            front_door(page, BASE_URL)
            page.wait_for_selector("#board-svg .port-marker", state="attached")
            drawn = page.evaluate(
                """() => {
                    const svg = document.getElementById("board-svg");
                    const box = svg.viewBox.baseVal;
                    const centres = (selector) => [...svg.querySelectorAll(selector)].map(
                        (c) => [c.cx.baseVal.value, c.cy.baseVal.value]);
                    return {
                        box: [box.x, box.y, box.width, box.height],
                        ports: centres(".port-marker"),
                        vertices: centres(".vertex-dot"),
                    };
                }"""
            )
        finally:
            browser.close()

    x, y, width, height = drawn["box"]
    assert len(drawn["ports"]) == 9 and len(drawn["vertices"]) == 54
    points = drawn["ports"] + drawn["vertices"]
    xs = [px for px, _ in drawn["vertices"]]
    middle = (min(xs) + max(xs)) / 2
    half = max(middle - (min(px for px, _ in points) - REACH), max(px for px, _ in points) + REACH - middle)
    left, right = middle - half - MARGIN, middle + half + MARGIN
    top = min(py for _, py in points) - REACH - MARGIN
    bottom = max(py for _, py in points) + REACH + MARGIN
    assert x == pytest.approx(left, abs=0.01)
    assert y == pytest.approx(top, abs=0.01)
    assert x + width == pytest.approx(right, abs=0.01)
    assert y + height == pytest.approx(bottom, abs=0.01)
    assert x + width / 2 == pytest.approx(middle, abs=0.01)
