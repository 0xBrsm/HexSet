"""Which game a visit to `/` opens, in a real browser: the most recently
active game this browser sits at that is still going (`activeCode`), not
whichever game it opened last. Stepping Back to a finished game used to make
that game the one `/` resumed, and the game still being played was reachable
only by its link.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

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

PORT = 8815
BASE_URL = f"http://127.0.0.1:{PORT}"
FINISHED = "fin123"
# The browser's secret, so it can reclaim its seat at the finished game.
SECRET = "a browser that played the finished game"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}
CODE_URL = re.compile(r"/[a-z0-9]{6}$")


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
def running_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("default-game")
    bot = lambda: Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")  # noqa: E731
    seats = [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT), bot(), bot(), bot()]
    session = build_session(FINISHED.upper(), seats, Config(games_dir=str(directory), seed=99), first=0)
    _play_out(session, random.Random(4))
    with serving(PORT, str(directory)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _code(page) -> str:
    return page.url.rsplit("/", 1)[1]


def _stamp(page, code: str):
    return page.evaluate(f"JSON.parse(localStorage.getItem('hexset.active.{code}'))")


def test_a_visit_to_the_front_door_opens_the_most_recently_active_unfinished_game(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            context = browser.new_context()
            # A seat at the finished game, held by a token from before a
            # restart, which the page trades for a fresh one (reclaim).
            context.add_init_script(
                "try {"
                f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
                f"  if (!localStorage.getItem('hexset.token.{FINISHED}'))"
                f"    localStorage.setItem('hexset.token.{FINISHED}', 'minted before the restart');"
                "} catch (e) {}"
            )
            page = context.new_page()

            # Two games of our own, both still going: the second dealt from
            # the first with New game.
            front_door(page, BASE_URL)
            first = _code(page)
            page.wait_for_function(f"localStorage.getItem('hexset.active.{first}') !== null")
            page.click("#new-game")
            page.wait_for_selector("#modal.show #deal-new-game")
            page.click("#deal-new-game")
            page.wait_for_url(lambda url: CODE_URL.search(url) and not url.endswith(first))
            second = _code(page)
            page.wait_for_function(f"localStorage.getItem('hexset.active.{second}') !== null")
            assert _stamp(page, second)["at"] >= _stamp(page, first)["at"]

            # Back to an old game, here one already over: it becomes the last
            # game opened, but it is no game to resume and carries no stamp.
            with page.expect_response(lambda r: "/api/reclaim" in r.url):
                page.goto(f"{BASE_URL}/{FINISHED}", wait_until="load")
            page.wait_for_selector(".player-row")
            assert page.evaluate("localStorage.getItem('hexset.last')") == FINISHED
            assert _stamp(page, FINISHED) is None

            page.goto(f"{BASE_URL}/", wait_until="load")
            page.wait_for_url(CODE_URL)
            assert _code(page) == second

            # Whichever moved last wins, not whichever was dealt last.
            page.evaluate(
                f"localStorage.setItem('hexset.active.{first}',"
                f" JSON.stringify({{...JSON.parse(localStorage.getItem('hexset.active.{first}')), at: Date.now() + 60000}}))"
            )
            page.goto(f"{BASE_URL}/", wait_until="load")
            page.wait_for_url(CODE_URL)
            assert _code(page) == first

            # Opening a game is not activity: viewing the second again leaves
            # the first as the default.
            page.goto(f"{BASE_URL}/{second}", wait_until="load")
            page.wait_for_selector(".player-row")
            page.goto(f"{BASE_URL}/", wait_until="load")
            page.wait_for_url(CODE_URL)
            assert _code(page) == first

            context.close()
        finally:
            browser.close()
