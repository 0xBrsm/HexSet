# SPDX-License-Identifier: GPL-3.0-only
"""The board, mid-setup, at a seat holding its turn open.

A setup road hands the snake on inside the placement itself, so a browser
seat keeps the turn until it ends it (`awaiting_confirm`), which is what
makes a setup placement undoable at all.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.game import to_move
from hexset.server.api import Config, Seat, SeatKind, build_session

from _page_server import launch, serving


try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

# Its own port, so this module can run alongside test_page_finished_game.py.
PORT = 8793
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "SETUP1"
SECRET = "a returning browser's own secret"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _place(session, seat: int, kind: ActionType) -> None:
    action = next(a for a in legal_actions(session.game) if a.type is kind)
    session.apply_action(seat, action)


@pytest.fixture(scope="module")
def held_game(tmp_path_factory):
    """Stops where the hold begins. Journalled because the server rebuilds the
    table from this file, which also proves a reopen restores the hold.
    """
    directory = tmp_path_factory.mktemp("held")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat(),
        _bot_seat(),
        _bot_seat(),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=99), first=0)
    assert to_move(session.game) == 0
    _place(session, 0, ActionType.SETUP_SETTLEMENT)
    _place(session, 0, ActionType.SETUP_ROAD)
    assert session.awaiting_confirm == 0, "the fixture is meant to stop mid-hold"
    assert to_move(session.game) != 0, "the engine's snake has moved on"
    return directory


@pytest.fixture(scope="module")
def running_server(held_game):
    with serving(PORT, str(held_game)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def _returning_context(browser):
    context = browser.new_context()
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    return context


def _seated_page(browser):
    context = _returning_context(browser)
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    page.wait_for_selector(".player-row")
    return context, page


def _shown(page, fab_id: str) -> bool:
    """`showFab` toggles a class rather than adding the node."""
    return "show" in (page.locator(f"#{fab_id}").get_attribute("class") or "")


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = launch(playwright)
        yield browser
        browser.close()


def test_the_held_seat_sees_its_own_turn_not_the_next_players(running_server, browser):
    context, page = _seated_page(browser)

    # Named for the hold, not the phase: the engine is already in the next
    # seat's SETUP_SETTLEMENT.
    assert page.locator("#phase").inner_text().strip() == "END TURN"

    rows = page.locator(".player-row")
    assert "active" in (rows.nth(0).get_attribute("class") or "")
    for other in (1, 2, 3):
        assert "active" not in (rows.nth(other).get_attribute("class") or "")

    context.close()


def test_the_held_seat_is_offered_end_turn_and_undo(running_server, browser):
    context, page = _seated_page(browser)

    assert _shown(page, "end-turn"), "no way to end the held turn"
    assert _shown(page, "undo-build"), "the take-back the hold exists for"
    assert not _shown(page, "roll-dice")
    assert not _shown(page, "buy-dev-card")
    # The deck stays up, faded and disabled, with the cards it has left.
    deck = page.locator("#buy-dev-card")
    assert deck.is_visible() and deck.is_disabled()
    left = page.evaluate("() => state.dev_cards_remaining")
    assert left > 0
    assert deck.locator(".deck-count").inner_text() == str(left)
    assert deck.get_attribute("title") == f"Development cards: {left} left"
    # The piece counts' hovers read the same way.
    titles = page.eval_on_selector_all("#piece-supply .supply-tile", "(tiles) => tiles.map((t) => t.title)")
    assert sorted(t.split(":")[0] for t in titles) == ["Cities", "Roads", "Settlements"]
    assert all(t.endswith(" left") for t in titles)
    assert float(deck.evaluate("(el) => getComputedStyle(el).opacity")) < 0.5
    assert not deck.locator(".board-btn-label").is_visible()
    # The bank's five piles across the top, each with what the bank holds.
    piles = page.locator("#bank .bank-pile")
    assert piles.count() == 5 and page.is_visible("#bank")
    bank = page.evaluate("() => state.bank")
    for i in range(5):
        pile = piles.nth(i)
        name = pile.get_attribute("data-resource")
        assert pile.locator(".bank-count").inner_text() == str(bank[name])
        assert ("empty" in (pile.get_attribute("class") or "")) == (bank[name] == 0)

    context.close()


def test_ending_the_held_turn_releases_the_table(running_server, browser):
    context, page = _seated_page(browser)
    assert page.locator("#phase").inner_text().strip() == "END TURN"

    page.click("#end-turn")

    page.wait_for_function(
        "document.getElementById('phase').innerText.trim() !== 'END TURN'",
        timeout=10_000,
    )
    # End turn keeps its corner through the other seat's turn, greyed out.
    assert page.is_disabled("#end-turn")

    context.close()


def test_a_rows_stat_opens_the_glossary_with_its_icon_lit(running_server, browser):
    context, page = _seated_page(browser)
    glossary = page.locator("#glossary")
    assert not glossary.is_visible()

    page.locator('.player-row .stat[data-icon="card"]').first.click()
    assert glossary.is_visible()
    lines = glossary.locator(".glossary-line")
    assert lines.count() == 6
    assert glossary.locator(".glossary-line.lit").get_attribute("data-icon") == "card"
    assert "Resource cards" in glossary.inner_text()
    # Inside the window, under or over the row.
    box = glossary.bounding_box()
    width = page.evaluate("() => window.innerWidth")
    assert box["x"] >= 0 and box["x"] + box["width"] <= width

    # Another stat lights its own line; the same one again closes it.
    page.locator('.player-row .stat[data-icon="victory-point"]').nth(1).click()
    assert glossary.locator(".glossary-line.lit").get_attribute("data-icon") == "victory-point"
    page.locator('.player-row .stat[data-icon="victory-point"]').nth(1).click()
    assert not glossary.is_visible()

    # A click elsewhere and Escape each close it.
    page.locator('.player-row .stat[data-icon="road"]').first.click()
    assert glossary.is_visible()
    page.mouse.click(5, 5)
    assert not glossary.is_visible()
    page.locator('.player-row .stat[data-icon="road"]').first.click()
    page.keyboard.press("Escape")
    assert not glossary.is_visible()

    context.close()


def test_a_road_ending_alone_at_another_seats_road_stops_short_of_the_corner(running_server, browser):
    context, page = _seated_page(browser)
    gaps = page.evaluate("""() => {
      const at = new Map();
      for (const e of board.edges) for (const v of [e.v0, e.v1]) { if (!at.has(v)) at.set(v, []); at.get(v).push(e); }
      for (const [v, edges] of at) {
        if (edges.length !== 3 || state.vertex_owner[v] >= 0) continue;
        if (edges.some((e) => state.edge_owner[e.id] >= 0)) continue;
        const owners = state.edge_owner;
        const read = (mine) => {
          state.edge_owner = owners.slice();
          mine.forEach((o, i) => { if (o !== null) state.edge_owner[edges[i].id] = o; });
          const gap = roadEndGaps();
          return edges.map((e) => (state.edge_owner[e.id] >= 0 ? gap(e, v) : null));
        };
        const out = {through: read([0, 0, 1]), meeting: read([0, 1, null]), alone: read([0, null, null]),
                    own: read([0, 0, null])};
        state.edge_owner = owners;
        return out;
      }
      return null;
    }""")
    assert gaps is not None, "no empty three-road corner on this board"
    gap = page.evaluate("() => ROAD_END_GAP")
    # Two red roads carry on through the corner; the green one ends there alone.
    assert gaps["through"] == [0, 0, gap]
    # Two seats' roads ending at the same corner both stop short of it.
    assert gaps["meeting"] == [gap, gap, None]
    # Alone, or meeting only its own seat's road, a road runs to the corner.
    assert gaps["alone"] == [0, None, None]
    assert gaps["own"] == [0, 0, None]

    context.close()


def test_the_log_is_dressed_with_rounds_named_seats_glyphs_and_tokens(running_server, browser):
    context, page = _seated_page(browser)
    read = page.evaluate("""() => {
      const box = document.createElement("div");
      box.id = "log-probe";
      document.body.appendChild(box);
      drawLog("log-probe", [
        "3\\tGreen (human) rolled 6. Red (Rehex) collects 2 Brick, 1 Sheep.",
        "3\\tRed (Rehex) moved the robber to 8 Wood and stole a card from Blue (B).",
        "4\\tPurple (<img src=x onerror=window.hacked=1>) built a road.",
        "4\\tBlue (B) traded 2 Ore for 1 Wheat with the bank.",
        "9\\tGreen (human) wins with 10 points.",
      ]);
      const lines = [...box.querySelectorAll(".log-line")];
      const out = {
        rounds: [...box.querySelectorAll(".log-round")].map((r) => r.textContent),
        seats: lines.map((l) => [...l.querySelectorAll(".log-seat")].map((s) => [s.textContent, s.style.color])),
        chips: lines.map((l) => l.querySelector(".log-chip use")?.getAttribute("href") || null),
        chipColor: lines.map((l) => l.querySelector(".log-chip").style.getPropertyValue("--seat")),
        glyphs: lines.map((l) => [...l.querySelectorAll(".log-glyph")].map((g) => g.title)),
        rolls: lines.map((l) => [...l.querySelectorAll(".log-roll")].map((r) => [r.textContent, r.className])),
        first: lines[0].querySelector(".log-text").textContent,
        subs: [...lines[0].querySelectorAll(".log-sub")].map((s) => s.textContent),
        img: box.querySelectorAll("img").length,
        win: lines[4].classList.contains("win"),
      };
      box.remove();
      return out;
    }""")
    assert read["rounds"] == ["ROUND 3", "ROUND 4", "ROUND 9"]
    # The log's colours are the seats' inks: each fill taken part way to grey.
    colors = page.evaluate("() => [0, 1, 2, 3].map(seatInk)")
    assert colors[0] == "#c45e64"
    rgb = lambda hex_: "rgb({}, {}, {})".format(*(int(hex_[i:i + 2], 16) for i in (1, 3, 5)))
    assert read["chipColor"][0] == colors[1] and read["chipColor"][1] == colors[0]
    assert read["chips"] == ["#icon-dice", "#icon-robber", "#icon-hammer", "#icon-ship", "#icon-victory-point"]
    # A seat shows by what it is called alone, in its colour.
    assert read["seats"][0] == [["human", rgb(colors[1])], ["Rehex", rgb(colors[0])]]
    assert read["seats"][1] == [["Rehex", rgb(colors[0])], ["B", rgb(colors[3])]]
    assert read["glyphs"][0] == ["Brick", "Sheep"] and read["glyphs"][1] == ["Wood"]
    # The comma between two cards goes; the count stays with its glyph.
    assert "," not in read["first"] and "2 " in read["first"]
    # The takings stand on their own line under the roll.
    assert read["first"].startswith("human rolled 6.")
    assert read["subs"] == ["Rehex collects 2  1 ."]
    assert read["rolls"][0] == [["6", "log-roll hot"]] and read["rolls"][1][0][0] == "8"
    # A name is text, never markup.
    assert read["img"] == 0 and page.evaluate("() => window.hacked") is None
    assert read["win"]

    # The heading colours the seat's whole possessive, and only the name in a win.
    heading = page.evaluate("""() => {
      const el = document.createElement("div");
      setSeatText(el, 2, "'S", " TURN");
      const turn = [el.querySelector(".seat-name").textContent, el.textContent];
      setSeatText(el, 2, "", " WINS");
      return [turn, [el.querySelector(".seat-name").textContent, el.textContent]];
    }""")
    assert heading == [["PURPLE'S", "PURPLE'S TURN"], ["PURPLE", "PURPLE WINS"]]

    context.close()
