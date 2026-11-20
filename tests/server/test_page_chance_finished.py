# SPDX-License-Identifier: GPL-3.0-only
"""The win chance at a finished game and at a past round, in a real browser:
the seat's line keeps its row and shows the read at the game's last position
still in play, or at the round being read; the corner keeps only the win
chance's toggle once the game is over. Skipped without Playwright or
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


def _top(page, selector: str) -> float:
    return page.eval_on_selector(selector, "(el) => el.getBoundingClientRect().top")


# The colour the chance bar is filled in, beside the seat's own.
_BAR_AND_SEAT = """
() => {
  const probe = document.createElement("i");
  probe.style.color = SEAT_COLORS[liveView().seat].fill;
  document.body.appendChild(probe);
  const seat = getComputedStyle(probe).color;
  probe.remove();
  return [getComputedStyle(document.querySelector("#chance .chance-fill")).backgroundColor, seat];
}
"""


def test_a_finished_game_and_its_past_rounds_keep_the_seats_chance(finished_game, browser):
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
            page.wait_for_selector("#chance.show")
            assert page.evaluate("() => state.game_over") is True
            final = page.inner_text("#chance .chance-pct")
            assert final.endswith("%") and 0 <= int(final[:-1]) <= 100
            log_top = _top(page, "#log")
            # The bar is the seat's own colour, at the end and at every round.
            bar, seat = page.evaluate(_BAR_AND_SEAT)
            assert bar == seat

            # Over, the corner keeps both: sound is set between games too.
            assert page.is_visible("#chance-toggle")
            assert page.is_visible("#sound-toggle")

            # A past round shows that round's read, in the same row.
            page.click("#replay-prev")
            page.wait_for_function("() => state.replay && state.chance && typeof state.chance.chance === 'number'")
            page.wait_for_selector("#chance.show")
            shown = page.inner_text("#chance .chance-pct")
            assert shown == f"{round(page.evaluate('() => state.chance.chance') * 100)}%"
            assert page.evaluate(_BAR_AND_SEAT) == [seat, seat], "a past round keeps the bar's colour"
            assert _top(page, "#log") == log_top, "stepping back moves nothing"
            page.click("#replay-first")
            page.wait_for_function("() => state.replay && state.replay.round === 0")
            page.wait_for_selector("#chance.show")
            assert page.evaluate(_BAR_AND_SEAT) == [seat, seat]
            assert _top(page, "#log") == log_top
            assert page.is_visible("#chance-toggle") and page.is_visible("#sound-toggle")
        finally:
            context.close()
