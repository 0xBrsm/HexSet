"""A seat's own page at a game that is already over, in a real browser.

The server has been right about this for a while: `api._not_over` refuses
every seat-changing request past `is_over`, and `webplay`'s wire state says
`game_over` and reveals every hand to everyone once a game ends. The page
did not read either. `renderPlayers` asked `watching()` — *do I hold a seat
here* — and a seat at a finished game is still a seat, so whoever played the
game went on being offered a live name box on their own row and a live model
picker on every bot's, each one a POST the server then answered 409. A
spectator at the same table, holding no seat, got the read-only list all
along, which is why an API-only proof and a spectator's eyes both missed it.

So this drives the one arrangement that was broken: a browser that holds the
seat, at a game that is over, reopened from its journal the way a restart
reopens one (`api.Tables._reopen`). Skipped without Playwright, the same as
`test_page_identity.py` and for the same reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hexset.actions import legal_actions
from hexset.game import is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.webplay import action_to_wire

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

PORT = 8792
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "ABC123"
# The browser's half of its own identity (`index.html`'s `clientSecret`): the
# secret stays in localStorage and only its sha256 is ever sent, so the seat
# in the journal below is recorded with the digest and the page proves the
# seat is its own by presenting the secret to `POST /api/reclaim`.
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="heximax", spec="heximax")


def _play_out(session, rng: random.Random) -> None:
    """Random legal play until somebody wins. Seeded exactly as
    `test_api.py`'s `finished_game` is, which is known to converge well
    inside this bound; played out rather than forced, since a `won_by` set by
    hand leaves nothing in the journal for a reopen to replay to."""
    for _ in range(2000):
        if is_over(session.game):
            return
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))
    raise AssertionError("the seeded game did not finish")


@pytest.fixture(scope="module")
def finished_game(tmp_path_factory):
    """A played-out game's journal, in a directory of its own: Ada (this
    browser, `SECRET`) at seat 0 against three bots. No server involved —
    the point is that the server the test starts has nothing in memory and
    must rebuild the table from this file, which is what a restart does."""
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
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(finished_game),
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


def _table_view() -> dict:
    """What a spectator is told about the table — used here only to check the
    page against the same numbers the server sent it."""
    with urllib.request.urlopen(f"{BASE_URL}/api/table/{CODE.lower()}", timeout=5) as response:
        return json.loads(response.read())


def _returning_context(browser):
    """A browser that already sat at this game before the restart: its own
    secret, and a token that died with the process that minted it. That dead
    token is what sends the page down the reclaim path (`resumeGame`), which
    is the only way back into a seat whose token never touched disk."""
    context = browser.new_context()
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    return context


def _cards_shown(page) -> int:
    """Resource cards the hand area is currently showing. Each tile carries
    its own count, so this is the total in whatever hand is open."""
    return sum(int(text) for text in page.locator("#hand .hand-col").first.locator(".count").all_inner_texts())


def test_a_seat_at_a_finished_game_gets_no_live_controls(running_server):
    """The defect itself: every row read-only, for the seat's holder too."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = _returning_context(browser)
            page = context.new_page()
            # The reclaim is the page proving the seat is still Ada's; without
            # it this would be an ordinary spectator, which was never broken.
            with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
                page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
            assert reclaimed.value.json()["seat"] == 0
            page.wait_for_selector(".player-row")

            # Nothing on the table is ours to change any more. A <select> here
            # is `swapBot` (409 "the game is already over"); an <input> is
            # `renameSelf` on our own row, refused the same way.
            assert page.locator("#players select").count() == 0
            assert page.locator("#players input").count() == 0
            # Still ours, though — the row keeps the name we played under
            # rather than falling back to the generic "human".
            assert "Ada" in page.locator(".player-row").first.inner_text()

            context.close()
        finally:
            browser.close()


def test_a_seat_at_a_finished_game_can_read_every_hand(running_server):
    """The other half: with the pickers gone, a row does what a spectator's
    row does — it opens. Every hand is revealed once a game is over
    (`webplay`: `reveal = over or omniscient or p == viewer`), so this is the
    seat's holder finally being able to see what beat them."""
    view = _table_view()
    assert view["game_over"] is True
    hands = {p["seat"]: sum(p["hand"].values()) for p in view["players"]}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = _returning_context(browser)
            page = context.new_page()
            with page.expect_response(lambda r: "/api/reclaim" in r.url):
                page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
            page.wait_for_selector(".player-row")

            # Our own hand is what shows until we ask for another, the same as
            # it did all game.
            assert _cards_shown(page) == hands[0]

            # A bot's row, which used to be that bot's model picker.
            rows = page.locator(".player-row")
            rows.nth(1).click()
            page.wait_for_selector(".player-row-open")
            assert _cards_shown(page) == hands[1]

            # Clicking the open row again comes back to our own hand rather
            # than to nobody's -- we have a seat, so there is always one to
            # fall back to.
            rows.nth(1).click()
            assert _cards_shown(page) == hands[0]

            context.close()
        finally:
            browser.close()


def test_a_live_game_still_offers_the_seat_controls(running_server):
    """The guard on the fix: read-only is `game_over`'s doing, not something
    a seat now gets for free. A fresh deal is live, and its player list is
    still the only place this table says who else is playing."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = browser.new_context()  # nobody: this page deals its own game
            page = context.new_page()
            with page.expect_response(
                lambda r: r.request.method == "POST" and "/api/games" in r.url
            ):
                page.goto(f"{BASE_URL}/", wait_until="load")
            page.wait_for_selector(".player-row")

            assert page.locator("#players select").count() > 0  # a bot seat's picker
            assert page.locator("#players input").count() == 1  # our own name box

            context.close()
        finally:
            browser.close()
