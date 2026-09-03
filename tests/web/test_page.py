"""The page, in a real browser.

Every check the web UI had before this file drove the API with a script or
read `static/index.html` as text (`tests/server/test_ui_smoke.py` says so in
its own docstring). Nothing ever loaded the page, so a frontend that threw on
its first render, polled a route it had no token for, or rebuilt an open
`<select>` out from under the cursor passed the suite and reached a player
broken. This runs Chromium against a live `HexSetServer` and drives the page
the way somebody at it would.

Marked `slow`, so the default run is unaffected, and skipped outright where
Playwright's browser isn't installed. To run it::

    pip install playwright && python -m playwright install chromium
    pytest -m slow tests/web/test_page.py

`search2` rather than the shipped `heximax` default at every bot seat: this is
a test of the page, and search2 is the cheaper opponent to have thinking in
the background while the browser works. The page's own default lineup is what
`currentBotModels` holds, so overriding that one global before dealing is the
whole difference.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("playwright", reason="pip install playwright")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from conftest import new_tables  # noqa: E402
from hexset.server.web import HexSetServer  # noqa: E402
from hexset.trading import Trade  # noqa: E402

pytestmark = pytest.mark.slow

# The page polls every 1.5 s while it isn't the reader's move (`index.html`'s
# pollWhileWaiting). "Across two polls" below means this, twice, plus slack.
POLL_SECONDS = 1.5

WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


@pytest.fixture
def live():
    """A real server on a real port, plus the registry behind it.

    `seat_grace=0.0` retires an empty seat on the second touch instead of
    after two minutes, which is what makes a game with open seats playable
    inside a test at all.
    """
    tables = new_tables()
    server = HexSetServer(("127.0.0.1", 0), tables)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield tables, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def browser():
    with sync_playwright() as play:
        try:
            engine = play.chromium.launch()
        except PlaywrightError as error:  # pragma: no cover - environment
            pytest.skip(f"chromium is not installed: {error}")
        try:
            yield engine
        finally:
            engine.close()


class Page:
    """One browser page, with everything it said to the console kept.

    A console error is a failed render, and a failed render on this page is
    not cosmetic: `render()` runs inside the poll timer's own callback, so an
    exception there stops the polling that is the only thing keeping the board
    current. Which is why every test here ends by asserting the log is empty.
    """

    def __init__(self, page):
        self.page = page
        self.console: list[str] = []
        self.polls = 0
        page.on(
            "response",
            lambda r: setattr(self, "polls", self.polls + 1)
            if "/api/state" in r.url or "/api/table/" in r.url
            else None,
        )
        page.on(
            "console",
            lambda m: self.console.append(f"{m.type}: {m.text}")
            if m.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))

    def state(self, expr: str = ""):
        return self.page.evaluate(f"() => state{expr}")

    def wait_for(self, expr: str, timeout: float = 30.0) -> None:
        self.page.wait_for_function(f"() => {expr}", timeout=timeout * 1000)


def open_page(browser, url: str) -> Page:
    page = Page(browser.new_page(viewport={"width": 1400, "height": 950}))
    page.page.goto(url, wait_until="networkidle")
    return page


def deal(page: Page, bots: int) -> str:
    """The front page's own deal button, with `bots` local opponents."""
    page.page.evaluate("() => { currentBotModels = ['search2', 'search2', 'search2']; }")
    page.page.select_option("#landing-bots", str(bots))
    page.page.click("#landing-deal")
    page.page.wait_for_url(lambda url: len(url.rstrip("/").rsplit("/", 1)[-1]) == 6)
    page.wait_for("typeof state !== 'undefined' && state && state.seats")
    # `state` is set a moment before the first render; wait for the render.
    page.page.wait_for_selector("#players .player-row")
    return page.page.url.rstrip("/").rsplit("/", 1)[-1]


def place_one_turn(page: Page) -> None:
    """This seat's settlement and road for the turn the snake is on now."""
    page.wait_for("state.to_move === state.seat", timeout=60)
    for selector in (".clickable-vertex", ".clickable-edge"):
        page.page.locator(selector).first.dispatch_event("click")
        page.page.wait_for_timeout(600)


def place_setup(page: Page, timeout: float = 60.0) -> None:
    """Click through this seat's own settlements and roads until setup ends."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.state(".phase") not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            return
        if page.state(".to_move") == page.state(".seat"):
            for selector in (".clickable-vertex", ".clickable-edge"):
                target = page.page.locator(selector)
                if target.count():
                    # dispatch_event, not click: a legal-road <line> can have a
                    # zero-width bounding box, which Playwright reads as
                    # invisible even though the page routes the click fine.
                    target.first.dispatch_event("click")
                    break
        page.page.wait_for_timeout(400)
    raise AssertionError(f"setup did not finish: phase={page.state('.phase')}")


def reach_main(page: Page) -> None:
    """Roll this seat's dice and answer the robber if the roll was a seven.

    Nobody owes a discard this early — every hand is a settlement or two's
    worth — so a seven goes straight to placing the robber, and the steal
    modal only opens when the chosen hex has more than one victim on it.
    """
    page.wait_for("state.to_move === state.seat && state.phase === 'ROLL'", timeout=90)
    page.page.click("#roll-dice")
    page.page.wait_for_timeout(800)
    if page.state(".phase") == "ROBBER":
        page.page.locator(".clickable-hex").first.dispatch_event("click")
        page.page.wait_for_timeout(800)
        victims = page.page.locator(".steal-option")
        if victims.count():
            victims.first.click()
    page.wait_for("state.phase === 'MAIN' && state.to_move === state.seat", timeout=30)


def test_a_fresh_page_deals_a_game_and_says_nothing_to_the_console(live, browser):
    _, url = live
    page = open_page(browser, url)
    assert page.console == []

    code = deal(page, 3)
    assert len(code) == 6
    assert page.state(".seat") is not None
    assert page.console == []


def test_new_game_deals_another_table_after_a_bot_has_been_swapped(live, browser):
    """The reported symptom, in order: change a bot, then ask for a new game.

    Swapping used to write the lineup slot the *renderer* had counted, and the
    renderer gave every seat a picker — so a four-seat table wrote a fourth
    lineup entry, and `POST /api/games` then asked for five seats and was
    refused. The button did nothing, visibly.
    """
    _, url = live
    page = open_page(browser, url)
    first = deal(page, 3)

    pickers = page.page.locator("#players select")
    assert pickers.count() == 3, "only the bot seats are model pickers"
    options = pickers.last.evaluate("e => Array.from(e.options).map(o => o.value)")
    other = next(name for name in options if name != pickers.last.input_value())
    pickers.last.select_option(other)
    page.page.wait_for_timeout(1000)
    assert page.page.evaluate("() => currentBotModels.length") == 3

    page.page.click("#new-game")
    page.page.wait_for_url(lambda u: u.rstrip("/").rsplit("/", 1)[-1] != first)
    page.wait_for("typeof state !== 'undefined' && state && state.seats")
    assert page.page.inner_text("#notice") == ""
    assert page.console == []


def test_a_swap_answers_the_asking_seat_not_the_swapped_one(live, browser):
    """A dropdown must not hand the page somebody else's view of the game."""
    _, url = live
    page = open_page(browser, url)
    deal(page, 3)
    mine = page.state(".seat")

    picker = page.page.locator("#players select").last
    options = picker.evaluate("e => Array.from(e.options).map(o => o.value)")
    picker.select_option(next(n for n in options if n != picker.input_value()))
    page.page.wait_for_timeout(1000)

    assert page.state(".seat") == mine
    revealed = [p["seat"] for p in page.state(".players") if "hand" in p]
    assert revealed == [mine]
    assert page.console == []


def test_an_open_picker_survives_the_polls_that_used_to_close_it(live, browser):
    _, url = live
    page = open_page(browser, url)
    deal(page, 3)
    # The poll only runs while it isn't our move, which is the case the picker
    # was unusable in: place this seat's own two pieces and the snake moves on
    # to the bots, with the page polling every 1.5 s until it comes back.
    place_one_turn(page)
    page.wait_for("state.to_move !== state.seat", timeout=60)

    picker = page.page.locator("#players select").first
    picker.evaluate("e => { e.dataset.probe = 'kept'; e.focus(); }")
    polls_before = page.polls
    page.page.wait_for_timeout(int((2 * POLL_SECONDS + 1.0) * 1000))

    assert page.polls - polls_before >= 2, "the page was not polling, so this proves nothing"
    assert page.page.evaluate(
        "() => { const e = document.querySelector('#players select');"
        " return !!(e && e.dataset.probe === 'kept' && document.activeElement === e); }"
    ), "the poll replaced the open picker"
    assert page.console == []


def test_no_seat_reads_as_locked_while_somebody_is_in_it(live, browser):
    """Every seat's line, against the server's own answer for that seat.

    A game dealt with open seats is the one that produces locked seats at all:
    the setup snake reaches each one, waits it out, and retires it. What the
    panel must never do is say that about a seat somebody is sitting in — nor,
    as it did, draw all four identically and say nothing about any of them.
    """
    tables, url = live
    page = open_page(browser, url)
    code = deal(page, 1)
    place_setup(page)

    api = tables.get(code).view(page.state(".seat"))
    assert any(s["kind"] == "empty" for s in api["seats"]), "no open seat to retire"
    assert api["locked"], "the snake retired nobody, so there is nothing to check"

    rows = page.page.locator("#players .player-row")
    assert rows.count() == len(api["seats"])
    for seat, row in enumerate(api["seats"]):
        line = rows.nth(seat).inner_text().lower()
        occupied = row["kind"] != "empty"
        assert ("locked" in line) == (seat in api["locked"]), (
            f"seat {seat} ({row['kind']}) reads {line!r}, locked={api['locked']}"
        )
        if occupied:
            assert "locked" not in line and "open seat" not in line
    assert page.console == []


def _seed_a_clearing_position(tables, code, mine: int) -> int:
    """Give this seat wood to spare and a bot next to it that wants wood.

    Hands and the counterparty's advertisement are set here rather than played
    into existence: what is under test is the panel, and a bundle that clears
    has to exist before the panel can be asked to compose one. `publish` is
    the same call `PUT .../valuation` makes, so the bot's gate is the ordinary
    `PostedValuation` a seat that never opted into confirm mode gets.
    """
    table = tables.get(code)
    with table.lock:
        game = table.session.game
        theirs = next(
            i
            for i, seat in enumerate(table.seats)
            if seat.kind.value == "bot"
        )
        game._state.hands[mine] = [4, 0, 0, 0, 0]
        game._state.hands[theirs] = [0, 4, 0, 0, 0]
        # They want wood and would part with brick; I want brick and would
        # part with wood. One wood for one brick clears for both of us.
        table.session.publish(theirs, [1.0, -1.0, 0.0, 0.0, 0.0])
    return theirs


def _advertise(page: Page, wants: int, gives: int) -> None:
    """Set this seat's own vector through the five-toggle panel."""
    rows = page.page.locator("#trading .val-row")
    rows.nth(wants).locator("button").nth(2).click()  # "+"
    page.page.wait_for_timeout(400)
    rows.nth(gives).locator("button").nth(0).click()  # "−"
    page.page.wait_for_timeout(400)


def test_the_panel_composes_and_proposes_a_trade(live, browser):
    tables, url = live
    page = open_page(browser, url)
    code = deal(page, 1)
    place_setup(page)
    reach_main(page)

    mine = page.state(".seat")
    theirs = _seed_a_clearing_position(tables, code, mine)
    _advertise(page, wants=BRICK, gives=WOOD)

    panel = page.page.locator(".negotiate-panel").first
    panel.wait_for(timeout=10_000)
    # Their "wants" chip is wood, so clicking it offers one wood; their
    # "gives" chip is brick, so clicking it asks for one brick.
    panel.locator(".negotiate-chip.chip-want").first.click()
    panel.locator(".negotiate-chip.chip-give").first.click()
    assert panel.locator(".negotiate-indicator.clears").count() == 1

    before = list(tables.get(code).session.game._state.hands[mine])
    panel.locator(".negotiate-submit").click()
    page.wait_for("state.trades.length > 0", timeout=15)

    after = list(tables.get(code).session.game._state.hands[mine])
    assert after[WOOD] == before[WOOD] - 1
    assert after[BRICK] == before[BRICK] + 1
    assert page.page.inner_text("#notice") == ""
    assert theirs != mine
    assert page.console == []


def test_a_pending_offer_can_be_accepted_from_the_panel(live, browser):
    """The confirm-mode human's half of the mechanic.

    A human seat's gate is `PendingGate` by default now, so a bot's trade
    event never clears against a person — it records the candidate and the
    page has to be what answers it. The offer is placed on `game.pending`
    directly, exactly as `PendingGate.accepts` places one, so the test does
    not depend on which bundle a particular opponent's event happens to find.
    """
    tables, url = live
    page = open_page(browser, url)
    code = deal(page, 1)
    place_setup(page)
    reach_main(page)

    mine = page.state(".seat")
    theirs = _seed_a_clearing_position(tables, code, mine)
    table = tables.get(code)
    with table.lock:
        # Signed towards `a`: one brick in, one wood out.
        table.session.game.pending.append(Trade(a=mine, b=theirs, received=(-1, 1, 0, 0, 0)))
    _advertise(page, wants=BRICK, gives=WOOD)

    accept = page.page.locator(".pending-offer .pending-accept").first
    accept.wait_for(timeout=10_000)
    before = list(table.session.game._state.hands[mine])
    accept.click()
    page.wait_for("state.pending.length === 0", timeout=15)

    after = list(table.session.game._state.hands[mine])
    assert after[WOOD] == before[WOOD] - 1
    assert after[BRICK] == before[BRICK] + 1
    assert page.page.inner_text("#notice") == ""
    assert page.console == []


def test_an_observer_without_a_seat_still_gets_a_board(live, browser):
    """The code-sharing flow: a full game, opened by somebody with no seat.

    `/api/board` and `/api/state` are both seat-gated, and an observer holds
    no token for either, so the page used to 401 on its own geometry and die
    on the first `board.vertices` it reached — a blank screen for every reader
    a shared code was meant to reach.
    """
    _, url = live
    player = open_page(browser, url)
    code = deal(player, 3)

    watcher = open_page(browser, f"{url}/{code}")
    watcher.wait_for("typeof state !== 'undefined' && state && state.players")
    watcher.page.wait_for_timeout((2 * POLL_SECONDS + 1.0) * 1000)

    assert watcher.state(".seat") is None
    assert watcher.page.locator("#players .player-row").count() == 4
    assert watcher.page.locator("#board-svg polygon").count() > 0
    assert watcher.page.inner_text("#phase") != "Loading..."
    # The join it tries first is answered 409 on a full table, by design; the
    # browser logs any 4xx as a console error, so that one is expected here
    # and nothing else is.
    assert [line for line in watcher.console if "409" not in line] == []
