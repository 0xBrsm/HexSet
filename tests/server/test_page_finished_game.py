"""A seat's own page at a game that is already over, in a real browser: no
live controls (`frozen()`), and a hand area that reads the table the way a
spectator's does (`reading()`).

The arrangement is deliberately a browser that *holds* the seat, at a game
reopened from its journal: a spectator's page was never broken, and an
API-only check says nothing about what a browser renders.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request

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

PORT = 8792
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "ABC123"
# The browser's half of its own identity: the secret stays in localStorage
# and only its sha256 is sent, so the journalled seat carries the digest.
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


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
def finished_game(tmp_path_factory):
    directory = tmp_path_factory.mktemp("finished")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat(),
        _bot_seat(),
        _bot_seat(),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    _play_out(session, random.Random(4))
    return directory


@pytest.fixture(scope="module")
def running_server(finished_game):
    with serving(PORT, str(finished_game)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _table_view() -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/api/table/{CODE.lower()}", timeout=5) as response:
        return json.loads(response.read())


def _returning_context(browser):
    """Its own secret, and a token that died with the process that minted it:
    the dead token is what sends the page down the reclaim path.
    """
    context = browser.new_context()
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    return context


def _cards_shown(page) -> int:
    return sum(int(text) for text in page.locator("#hand .hand-col").first.locator(".count").all_inner_texts())


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = launch(playwright)
        yield browser
        browser.close()


def test_a_finished_game_reads_as_a_spectator_view_even_for_a_seat_it_held(running_server, browser):
    view = _table_view()
    assert view["game_over"] is True
    hands = {p["seat"]: sum(p["hand"].values()) for p in view["players"]}

    context = _returning_context(browser)
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url):
        page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
    page.wait_for_selector(".player-row")

    assert page.locator("#hand .hand-hint").count() == 1
    assert _cards_shown(page) == 0

    rows = page.locator(".player-row")
    rows.nth(0).click()
    page.wait_for_selector(".player-row-open")
    assert _cards_shown(page) == hands[0]

    rows.nth(1).click()
    assert _cards_shown(page) == hands[1]

    rows.nth(1).click()
    assert page.locator("#hand .hand-hint").count() == 1
    assert _cards_shown(page) == 0

    context.close()


def _row_heights(page) -> list[float]:
    return page.eval_on_selector_all(
        "#players .player-row",
        "els => els.map(e => e.getBoundingClientRect().height)",
    )


def test_the_roster_keeps_its_height_when_a_game_ends(running_server, browser):
    """Read-only rows swap every `<select>`/`<input>` for a text node, which
    carries no padding or border, so the roster collapses and the board below
    it jumps. The seat's own finished page is the one with no live controls
    at all (`frozen()`), next to a live game that still offers them.
    """
    live = browser.new_context()  # nobody: this page deals its own game
    live_page = live.new_page()
    front_door(live_page, BASE_URL)
    live_page.wait_for_selector(".player-row")
    live_heights = _row_heights(live_page)

    over = _returning_context(browser)
    over_page = over.new_page()
    with over_page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        over_page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    over_page.wait_for_selector(".player-row")
    over_heights = _row_heights(over_page)

    assert live_page.locator("#players select").count() > 0
    assert live_page.locator("#players input").count() == 1
    assert over_page.locator("#players select, #players input").count() == 0

    assert live_heights and over_heights
    every = live_heights + over_heights
    assert max(every) - min(every) < 1.0, (
        f"roster rows disagree on height: live={live_heights} "
        f"finished={over_heights}"
    )

    live.close()
    over.close()
