"""The owner's page, in a real browser.

Every check the web UI had before this file drove the API with a script or
read `static/index.html` as text. Nothing ever loaded the page, so a frontend
that threw on its first render, polled a route it had no token for, or rebuilt
an open `<select>` out from under the cursor passed the suite and reached a
player broken. This runs Chromium against a live `HexSetServer` and drives the
page the way somebody at it would.

What it drives is `feat/ui`'s board: the address is the game, there is no
front page and no lobby, every game is public, watching is omniscient, and a
person's own row is their name. There is no landing screen to deal from --
opening `/` deals -- and no bot-count picker, because seating three bots fills
a table and a full table is one nobody can join. Opponents are chosen from the
player list, which is where everything else about a seat already lives.

There is also nothing here about trading, and that is the point of
`test_a_person_is_offered_no_way_to_trade`: the human's half of the
negotiation interface is withheld from the page (owner, 2026-09-03), the API
keeps it, and the bots go on dealing with each other. The panel tests this
file used to carry went with the panel.

Marked `slow`, so the default run is unaffected, and skipped outright where
Playwright's browser isn't installed. To run it::

    pip install playwright && python -m playwright install chromium
    pytest -m slow tests/web/test_page.py

`search2` rather than the shipped `heximax` default at every bot seat: this is
a test of the page, and search2 is the cheaper opponent to have thinking in
the background while the browser works. The same drive against three
`heximax` is what the owner's own sanity check ran.
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
from hexset.server.webplay import PendingGate  # noqa: E402
from hexset.trading import NO_VALUATION  # noqa: E402

pytestmark = pytest.mark.slow

# The page polls every 1.5 s while it isn't the reader's move (`index.html`'s
# pollWhileWaiting). "Across two polls" below means this, twice, plus slack.
POLL_SECONDS = 1.5

BOT = "search2"


@pytest.fixture
def live():
    """A real server on a real port, plus the registry behind it.

    No `seat_grace`: there is no grace window any more. An empty seat the
    setup snake reaches is retired there and then, which is what makes a
    table with open seats playable inside a test at all -- and playable for
    somebody waiting on a friend who never came, which is what it is
    actually for.
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

    def click_first(self, selector: str) -> bool:
        # dispatch_event, not click: a legal-road <line> can have a zero-width
        # bounding box, which Playwright reads as invisible even though the
        # page routes the click fine.
        target = self.page.locator(selector)
        if not target.count():
            return False
        target.first.dispatch_event("click")
        return True


def open_page(browser, url: str) -> Page:
    """A fresh browser at `url`. Opening `/` deals a game and moves to its
    address; opening a code goes to that game."""
    page = Page(browser.new_page(viewport={"width": 1400, "height": 950}))
    page.page.goto(url, wait_until="networkidle")
    page.wait_for("typeof state !== 'undefined' && state && state.seats")
    page.page.wait_for_selector("#players .player-row")
    return page


def code_of(page: Page) -> str:
    return page.page.url.rstrip("/").rsplit("/", 1)[-1]


def seat_bots(page: Page, model: str = BOT) -> None:
    """Fill every open seat from the player list, which is the only place a
    table chooses its opponents now that there is no lobby.

    Driven off `state.seats`, not off the pickers: a bot seat keeps its
    picker (that is how a bot is swapped), so "no pickers left" is never
    true and counting them would loop forever.
    """
    for seat, entry in enumerate(page.state(".seats")):
        if entry["kind"] != "empty":
            continue
        row = page.page.locator("#players .player-row").nth(seat)
        row.locator("select").select_option(model)
        page.page.wait_for_function(
            f"() => state.seats[{seat}].kind === 'bot'", timeout=15000
        )


def answer_modal(page: Page) -> bool:
    """The modals the game opens on its own: a discard owed to a seven, and
    which seat to steal from. Anything else is dismissible and dismissed."""
    if not page.page.evaluate("() => document.getElementById('modal').classList.contains('show')"):
        return False
    mode = page.page.evaluate("() => modalMode")
    if mode == "discard":
        cards = page.page.locator("#modal-body .pick-card:not(.disabled)")
        if cards.count():
            cards.first.click()
            page.page.wait_for_timeout(300)
            return True
    elif mode == "steal":
        options = page.page.locator(".steal-option")
        if options.count():
            options.first.click()
            page.page.wait_for_timeout(300)
            return True
    page.page.keyboard.press("Escape")
    page.page.wait_for_timeout(200)
    return True


def place_setup(page: Page, timeout: float = 180.0) -> None:
    """Click through this seat's own settlements and roads until setup ends."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.state(".phase") not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            return
        if answer_modal(page):
            continue
        if page.state(".to_move") == page.state(".seat"):
            if not page.click_first(".clickable-vertex"):
                page.click_first(".clickable-edge")
        page.page.wait_for_timeout(350)
    raise AssertionError(f"setup did not finish: phase={page.state('.phase')}")


def play_turns(page: Page, turns: int, timeout: float = 900.0, each=None) -> int:
    """Play `turns` of this seat's own turns through the page's controls.

    Roll, build whatever the board is offering, end the turn -- all of it by
    clicking, never by posting to the API, which is the only way this says
    anything about the page. `each` is called on every pass, for a check that
    has to happen while a bot is on move.
    """
    played = 0
    deadline = time.monotonic() + timeout
    while played < turns and time.monotonic() < deadline:
        if page.state(".game_over"):
            return played
        if answer_modal(page):
            continue
        if page.state(".to_move") != page.state(".seat"):
            if each is not None:
                each(played)
            page.page.wait_for_timeout(400)
            continue
        phase = page.state(".phase")
        if phase == "ROLL":
            page.page.click("#roll-dice")
            page.page.wait_for_timeout(600)
        elif phase == "ROBBER":
            page.click_first(".clickable-hex")
            page.page.wait_for_timeout(600)
        elif phase == "MAIN":
            if page.click_first(".clickable-vertex") or page.click_first(".clickable-edge"):
                page.page.wait_for_timeout(500)
            end = page.page.locator("#end-turn")
            if end.count() and end.is_visible():
                end.click()
                played += 1
                page.page.wait_for_timeout(500)
            else:
                page.page.wait_for_timeout(400)
        else:
            page.page.wait_for_timeout(400)
    return played


# --- The address is the game --------------------------------------------------


def test_a_fresh_load_deals_a_game_at_its_own_address(live, browser):
    """No front page, no lobby, no code to type: `/` deals and moves there."""
    _, url = live
    page = open_page(browser, url)

    assert len(code_of(page)) == 6
    assert code_of(page) == page.state(".code")
    assert page.state(".seat") is not None
    # Every seat but the creator's starts open -- filling them is a decision
    # made at the table, not before it.
    kinds = [s["kind"] for s in page.state(".seats")]
    assert kinds.count("player") == 1
    assert kinds.count("empty") == 3
    assert page.console == []


def test_the_same_address_seats_a_second_person_at_the_same_table(live, browser):
    """The address is the whole invitation."""
    _, url = live
    first = open_page(browser, url)
    code = code_of(first)

    second = open_page(browser, f"{url}/{code}")

    assert code_of(second) == code
    assert second.state(".seat") is not None
    assert second.state(".seat") != first.state(".seat")
    assert first.console == [] and second.console == []


def test_the_player_list_is_where_a_table_picks_its_opponents(live, browser):
    """There is no bot-count picker, because seating three bots fills the
    table and a full table is one nobody can join. An open seat's picker
    seats a bot on it, which is what `Tables.seat_bot` exists for."""
    _, url = live
    page = open_page(browser, url)
    mine = page.state(".seat")

    assert page.page.locator("#players select").count() == 3
    seat_bots(page)

    kinds = [s["kind"] for s in page.state(".seats")]
    assert kinds.count("bot") == 3
    assert kinds[mine] == "player"
    assert [s["name"] for s in page.state(".seats") if s["kind"] == "bot"] == [BOT] * 3
    assert page.console == []


def test_a_seat_line_says_who_is_in_it(live, browser):
    """The panel and the status line cannot disagree, because both read the
    server's own per-seat kind. Every row used to be a bot picker -- your own
    occupied seat, an open one, and a retired one all drawn identically."""
    _, url = live
    page = open_page(browser, url)
    mine = page.state(".seat")
    page.page.locator("#players select").first.select_option(BOT)
    page.page.wait_for_timeout(600)

    rows = page.page.locator("#players .player-row")
    assert rows.count() == 4
    for seat in range(4):
        kind = page.state(f".seats[{seat}].kind")
        row = rows.nth(seat)
        if seat == mine:
            # Your own row is your name: an input, not a label and not a picker.
            assert row.locator("input").count() == 1
            assert row.locator("input").get_attribute("placeholder") == "human"
        elif kind == "bot":
            assert row.locator("select").count() == 1
            assert row.locator("select").input_value() == BOT
        else:
            assert "empty" in row.inner_text()
    assert page.console == []


def test_an_empty_seat_is_retired_on_sight_with_no_countdown(live, browser):
    """A turn only advances because the seat holding it said so, so a
    countdown could only ever retire a seat somebody was still waiting on.
    The snake reaching an empty seat retires it there and then."""
    _, url = live
    page = open_page(browser, url)
    page.page.locator("#players select").first.select_option(BOT)
    page.page.wait_for_timeout(600)
    assert page.state(".locked") == []

    started = time.monotonic()
    deadline = started + 90
    while time.monotonic() < deadline and not page.state(".locked"):
        if page.state(".to_move") == page.state(".seat") and page.state(".phase").startswith(
            "SETUP"
        ):
            if not page.click_first(".clickable-vertex"):
                page.click_first(".clickable-edge")
        page.page.wait_for_timeout(250)

    assert page.state(".locked"), "the snake reached an open seat and did not retire it"
    # Immediately: inside one pass of the snake, not two minutes later.
    assert time.monotonic() - started < 30
    retired = page.state(".locked")[0]
    row = page.page.locator("#players .player-row").nth(retired)
    assert "locked seat" in row.inner_text()
    assert row.locator("select").count() == 0
    # Nothing anywhere counts down to it.
    body = page.page.inner_text("body").lower()
    assert "locks in" not in body
    assert page.console == []


def test_new_game_deals_another_table(live, browser):
    """The reported symptom, in order: change a bot, then ask for a new game.

    Seating used to write the lineup slot the *renderer* had counted, and the
    renderer gave every seat a picker -- so a four-seat table wrote a fourth
    lineup entry, and `POST /api/games` then asked for five seats and was
    refused. The button did nothing, visibly.
    """
    _, url = live
    page = open_page(browser, url)
    first = code_of(page)
    seat_bots(page)

    page.page.click("#new-game")
    page.page.wait_for_function(
        f"() => location.pathname.replace('/','') !== '{first}'", timeout=30000
    )
    page.page.wait_for_selector("#players .player-row")

    second = code_of(page)
    assert len(second) == 6 and second != first
    assert page.state(".seat") is not None
    assert [s["kind"] for s in page.state(".seats")].count("empty") == 3
    assert page.console == []


def test_a_bot_seated_answers_the_asking_seat_not_the_seated_one(live, browser):
    """A dropdown must not hand the page somebody else's view of the game.

    `POST /api/bot` answered with the *target* seat's view, so touching a
    bot's picker handed back that bot's entire hand and left the client
    believing it sat at the bot's seat until its next poll.
    """
    _, url = live
    page = open_page(browser, url)
    mine = page.state(".seat")

    page.page.locator("#players select").first.select_option(BOT)
    page.page.wait_for_timeout(800)

    assert page.state(".seat") == mine
    revealed = [p["seat"] for p in page.state(".players") if "hand" in p]
    assert revealed == [mine]
    assert page.console == []


def test_your_own_row_is_your_name_and_you_can_change_it(live, browser):
    """The row that reads "human" is an input, for the same reason a bot
    seat's row is a <select>: the control is the player line."""
    _, url = live
    page = open_page(browser, url)
    mine = page.state(".seat")
    seat_bots(page)

    field = page.page.locator("#players .player-row").nth(mine).locator("input")
    assert field.get_attribute("placeholder") == "human"
    field.fill("brian")
    field.press("Enter")
    page.page.wait_for_function("() => state.seats[state.seat].name === 'brian'", timeout=15000)

    # And it is what everyone else's list and the log call this seat from now on.
    watcher = open_page(browser, f"{url}/{code_of(page)}")
    assert watcher.state(f".seats[{mine}].name") == "brian"

    # Blanking it puts the seat back to unnamed.
    field.fill("")
    field.press("Enter")
    page.page.wait_for_function("() => !state.seats[state.seat].name", timeout=15000)
    assert page.console == [] and watcher.console == []


def test_an_open_picker_survives_the_polls_that_used_to_close_it(live, browser):
    """`render()` rebuilt `#players` wholesale on every poll -- 1.5 s while
    it is not your move -- replacing the open `<select>` under the cursor. A
    picker you cannot use: it shut about a second after it opened, every
    time. The rebuild is skipped while the container holds focus."""
    _, url = live
    page = open_page(browser, url)
    seat_bots(page)
    place_setup(page)

    checked = []

    def while_a_bot_thinks(_played):
        if checked:
            return
        picker = page.page.locator("#players select").first
        picker.focus()
        before = page.polls
        page.page.wait_for_timeout(POLL_SECONDS * 3 * 1000)
        # Really polling, and really still the same element with focus.
        assert page.polls - before >= 2, f"only {page.polls - before} polls in the window"
        assert page.page.evaluate(
            "() => document.activeElement === document.querySelectorAll('#players select')[0]"
        )
        picker.blur()
        checked.append(True)

    play_turns(page, 3, timeout=300, each=while_a_bot_thinks)
    assert checked, "never reached a bot's turn to poll through"
    assert page.console == []


# --- No trading for a person ---------------------------------------------------


def test_a_person_is_offered_no_way_to_trade_and_the_bots_still_do(live, browser):
    """Withheld on the owner's instruction, 2026-09-03.

    Nothing on the page advertises, proposes, confirms or declines; the
    seat's own published vector stays all-zero; and a zero vector is dropped
    at `hexset.trading._best_clearing`'s ranking before any gate is asked, in
    either role -- so the human is not a counterparty and no exchange it is
    party to can clear. The bots are untouched and go on dealing with each
    other through the same engine event, which the transcript says out loud.

    Fifteen turns, because a trade is a turn-scale event and one or two would
    prove nothing either way.

    What is *not* asserted here is that the bots did in fact clear a deal.
    Whether a particular game throws up a clearing bundle belongs to the
    deal and the engine, not to the page: observed runs of this same drive
    range from seven exchanges over sixteen turns to none over forty. So the
    condition this checks is the one that is actually a property of seating
    -- the human's gate never accepts and advertises nothing, while every bot
    seat has a gate of its own and has published something to be judged on --
    and the transcript is checked for what it must never say. That the
    mechanic clears deals at all is `tests/test_trading.py`'s job, on hands
    it controls.
    """
    _, url = live
    page = open_page(browser, url)
    mine = page.state(".seat")
    seat_bots(page, "heximax")
    place_setup(page)

    def seat_to_seat_lines():
        # The transcript, not `state.trades`: that block is the *current
        # turn's* exchanges and is emptied by `end_turn`, so it says nothing
        # about a deal two turns ago. The log keeps the whole game.
        return [line for line in page.state(".log") if " to Player " in line]

    played = play_turns(page, 15, timeout=1500)
    assert played >= 15, played

    # Nothing on the page to trade with.
    assert page.page.locator("input[type=range]").count() == 0
    assert page.page.locator("#negotiation").count() == 0
    text = page.page.inner_text("body").lower()
    for word in ("valuation", "advertis", "negotiat", "counterparty", "propose"):
        assert word not in text, word

    # This seat advertises nothing and holds nothing pending.
    assert page.state(".valuations")[mine] == [0, 0, 0, 0, 0]
    assert page.state(".pending") in ([], None)

    # No exchange this seat was ever party to, over the whole transcript.
    assert not [line for line in seat_to_seat_lines() if f"Player {mine + 1} " in line]

    # The asymmetry, at the seats themselves rather than by what happened to
    # clear: the human's gate never accepts and advertises nothing; every bot
    # seat has its own gate and has published something to be judged on.
    tables, _ = live
    session = tables.get(code_of(page)).session
    assert isinstance(session.traders[mine], PendingGate)
    assert tuple(session.game.valuations[mine]) == NO_VALUATION
    bots = [s for s in range(4) if s != mine and session.game.gates[s] is not None]
    assert len(bots) == 3
    assert all(tuple(session.game.valuations[s]) != NO_VALUATION for s in bots)

    # Whatever the bots cleared, they cleared among themselves: every
    # seat-to-seat line in the transcript names two other seats. (That such a
    # line is written at all when a poll fires the event is
    # `test_webplay.py::test_a_trade_a_poll_cleared_is_told_in_the_log`, on
    # hands it controls -- not something to make a live game produce.)
    assert all(f"Player {mine + 1} " not in line for line in seat_to_seat_lines())
    assert page.console == []


# --- Watching ------------------------------------------------------------------


def test_watching_a_full_game_is_omniscient(live, browser):
    """A link to a game with every seat taken opens it to watch, and a
    spectator is outside the game, so they are shown all of it.

    The exposure this is: `GET /api/table/<code>` is not authenticated and
    cannot be, since holding the link is the whole qualification. Every route
    that *acts* still answers a token and still gets its own seat's honest
    view.
    """
    _, url = live
    player = open_page(browser, url)
    seat_bots(player)
    place_setup(player)
    play_turns(player, 2, timeout=300)
    code = code_of(player)

    watcher = Page(browser.new_context().new_page())
    watcher.page.goto(f"{url}/{code}", wait_until="networkidle")
    watcher.wait_for("typeof state !== 'undefined' && state && state.seats")
    watcher.page.wait_for_selector("#players .player-row")

    assert watcher.state(".seat") is None
    # Every hand, every dev card, every true victory-point count.
    assert {p["seat"] for p in watcher.state(".players") if "hand" in p} == {0, 1, 2, 3}
    assert {p["seat"] for p in watcher.state(".players") if "dev_cards" in p} == {0, 1, 2, 3}
    # And a seat's own view is still only its own.
    assert {p["seat"] for p in player.state(".players") if "hand" in p} == {player.state(".seat")}

    # Nothing here is actionable, so the parts that only mean something to a
    # seat are absent.
    assert watcher.page.locator("#players select").count() == 0
    assert watcher.page.locator("#players input").count() == 0
    assert watcher.page.locator("#piece-supply").inner_text().strip() == ""
    assert watcher.state(".legal_actions") == []

    # A player row opens that seat's cards below, and closes again.
    assert "Click a player" in watcher.page.inner_text("#hand")
    watcher.page.locator("#players .player-row").first.click()
    watcher.page.wait_for_timeout(400)
    assert "Resource Cards" in watcher.page.inner_text("#hand")
    watcher.page.locator("#players .player-row").first.click()
    watcher.page.wait_for_timeout(400)
    assert "Click a player" in watcher.page.inner_text("#hand")

    assert watcher.console == [], watcher.console
    assert player.console == []


def test_a_code_with_no_game_behind_it_says_so(live, browser):
    _, url = live
    page = Page(browser.new_page(viewport={"width": 1400, "height": 950}))
    page.page.goto(f"{url}/zzzzzz", wait_until="networkidle")
    page.page.wait_for_function(
        "() => document.getElementById('phase').textContent.trim() !== 'Loading...'",
        timeout=15000,
    )
    assert "zzzzzz" in page.page.url
    assert page.page.locator("#players .player-row").count() == 0
