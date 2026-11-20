# SPDX-License-Identifier: GPL-3.0-only
"""The page plays each action back as it happens (`pump`): a caption naming
it, the dice, a piece dropping in, and what it did to each seat's cards on
that seat's row. A spectator watches four bots play; a seat rolls its own
dice, ends its turn and takes in the bots' moves, and sets its sound from the
board's corner.

A roll shows what it produced where it came from (`playProduction`): its
numbers light, each building that collects rings, and its cards cross to
that seat's row, while the dice leave for the roller's row. Once an entry
has played, the rows, the hand and the board show the table as it stood
after it (`advance`), before the rest of its view's entries play.

The board they play on is drawn with depth (a sand base and its shadow under
rounded, grained tiles, on water with a fine chop of waves), a road has a
shadow and a lit side, and the robber stands at the shoulder of the number
it stops.
A place a road may go is a short bar clear of the corners at its ends.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import Phase, is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire

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

PORT = 8797
BASE_URL = f"http://127.0.0.1:{PORT}"
# CODE_LENGTH (6) characters from api.CODE_ALPHABET -- no 0/1/i/l/o.
WATCHED = "WTCH4B"
SEATED = "SEAT3B"
OPENING = "PENS4B"
SECRET = "a browser that watches its turns play back"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}

# Counts what the page puts on screen while it plays a turn back.
_WATCH_PLAYBACK = """
window.__seen = {dice: 0, captions: 0, badges: 0, pieces: 0, ghosts: 0, rings: 0};
new MutationObserver((records) => {
  for (const r of records) {
    if (r.type === "attributes" && r.target.id === "dice-tray" && !r.target.hidden) window.__seen.dice++;
    if (r.type === "childList" && r.target.id === "captions")
      for (const n of r.addedNodes) if (n.classList && n.classList.contains("caption")) window.__seen.captions++;
    for (const n of r.addedNodes || []) {
      if (!n.classList) continue;
      if (n.classList.contains("row-badge")) window.__seen.badges++;
      if (n.classList.contains("dice-ghost")) window.__seen.ghosts++;
      if (n.classList.contains("roll-ring")) window.__seen.rings++;
      if (n.querySelector && (n.classList.contains("anim-piece") || n.querySelector(".anim-piece"))) window.__seen.pieces++;
    }
  }
}).observe(document, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden"]});
"""


def _bot() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _to_seat_zero_rolling(session, rng: random.Random) -> None:
    """Past the opening, to seat 0's roll, with no dev card played (so no
    turn is held open on a choice)."""
    game = session.game
    for seat in range(4):
        session.claimed_seats.add(seat)
    for _ in range(4000):
        if is_over(game):
            raise RuntimeError("the game ended first")
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(game)
        if seat == 0 and game.phase is Phase.ROLL and session.round >= 2:
            return
        options = [a for a in legal_actions(game) if not a.type.name.startswith("PLAY_")]
        session.submit(seat, action_to_wire(rng.choice(options)))
    raise RuntimeError("did not reach seat 0's roll")


@pytest.fixture(scope="module")
def running_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("playback")
    build_session(WATCHED, [_bot(), _bot(), _bot(), _bot()],
                  Config(games_dir=str(directory), seed=11), first=0)
    seated = build_session(
        SEATED, [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT), _bot(), _bot(), _bot()],
        Config(games_dir=str(directory), seed=12), first=0)
    _to_seat_zero_rolling(seated, random.Random(5))
    # A table nobody has moved at: its seat places the first settlement.
    build_session(
        OPENING, [Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT), _bot(), _bot(), _bot()],
        Config(games_dir=str(directory), seed=13), first=0)
    with serving(PORT, str(directory)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


def test_a_spectator_sees_the_bots_turns_played_back_one_action_at_a_time(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = browser.new_page()
            page.add_init_script(_WATCH_PLAYBACK)
            page.goto(f"{BASE_URL}/{WATCHED.lower()}", wait_until="load")
            page.wait_for_selector(".player-row")
            page.wait_for_function(
                "window.__seen.dice >= 2 && window.__seen.pieces >= 2 && window.__seen.captions >= 4"
                " && window.__seen.badges >= 1 && window.__seen.ghosts >= 2 && window.__seen.rings >= 1",
                timeout=90_000)
            seen = page.evaluate("window.__seen")
            assert seen["dice"] >= 2 and seen["pieces"] >= 2 and seen["badges"] >= 1
            # Each roll's dice left for the roller's row, and a roll that
            # was not a seven ringed its numbers.
            assert seen["ghosts"] >= 2 and seen["rings"] >= 1
            # Every entry the page took in is numbered past where it began.
            assert page.evaluate("anim.last") > 0
            assert page.evaluate("document.querySelectorAll('.player-row[data-seat]').length") == 4
        finally:
            browser.close()


def _seated(browser, code=SEATED, ready="#roll-dice.show", **context):
    """A table's page as its seat, once `ready` shows: by default the seated
    table's, waiting on that seat's roll."""
    context = browser.new_context(**context)
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{code.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url):
        page.goto(f"{BASE_URL}/{code.lower()}", wait_until="load")
    page.wait_for_selector(ready)
    return page


# Plays one roll's production on the board as it stands and counts what it
# drew: `gains` 1 gives one card to the owner of one building on a numbered
# hex the robber is not on, 0 gives nobody anything. Returns the counts
# beside what the roll should have drawn.
_PLAY_PRODUCTION = """
async (gains) => {
  const owned = (h) => h.vertex_ids.find((v) => state.vertex_owner[v] >= 0);
  const hex = board.hexes.find((h) => h.token && h.id !== state.robber && owned(h) !== undefined);
  const seat = state.vertex_owner[owned(hex)];
  const resource = TERRAIN_RESOURCE[hex.terrain];
  const seen = {rings: 0, lights: 0, cards: 0};
  const count = (records) => {
    for (const r of records) for (const n of r.addedNodes) {
      if (!n.classList) continue;
      if (n.classList.contains("roll-ring")) seen.rings++;
      if (n.classList.contains("roll-light")) seen.lights++;
      if (n.classList.contains("fly-card")) seen.cards++;
    }
  };
  const observer = new MutationObserver(count);
  observer.observe(document.getElementById("board-svg"), {subtree: true, childList: true});
  observer.observe(document.getElementById("fly-layer"), {childList: true});
  await playProduction(hex.token, state,
    gains ? {[seat]: board.resources.map((r) => (r === resource ? gains : 0))} : {});
  count(observer.takeRecords());
  observer.disconnect();
  return {
    seen,
    numbered: board.hexes.filter((h) => h.token === hex.token).length,
    left: document.getElementById("fly-layer").childElementCount
      + document.querySelectorAll("#anim-layer .roll-ring, #anim-layer .roll-light").length,
  };
}
"""


def test_the_board_is_drawn_with_depth_and_the_robber_stands_beside_its_number(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = _seated(browser)
            page.wait_for_selector('#board-pane[data-water="painted"]')
            drawn = page.evaluate(
                """() => {
                    const count = (selector) => document.querySelectorAll(selector).length;
                    const water = getComputedStyle(document.getElementById("water"));
                    const blocked = () => [...document.querySelectorAll(".hex-token-g.blocked")].map((g) => g.id);
                    const robber = document.querySelector("#robber-layer .robber");
                    const numbered = board.hexes.find((h) => h.token && h.id !== state.robber);
                    const before = blocked();
                    markBlocked(numbered.id);
                    const marked = blocked();
                    render();
                    return {
                        land: board.hexes.filter((h) => h.terrain !== "SEA").length,
                        tiles: count("#static-layer path.tile"),
                        shades: count("#static-layer .tile-shade"),
                        bases: count("#static-layer .land-base"),
                        shadows: count("#static-layer .land-shadow"),
                        steps: LAND_SHADOW.steps,
                        lines: count("#static-layer .edge-line"),
                        casts: count("#static-layer .tile-cast"),
                        grains: count("#static-layer .tile-grain"),
                        specks: document.querySelector("#static-layer .tile-grain").getAttribute("d").split("M").length - 1,
                        bevels: count("#static-layer .tile-bevel"),
                        // The pane's background, top layer first, and each layer's size.
                        water: water.backgroundImage.split(/,\s*(?=url|radial)/).map((layer) => layer.slice(0, 9)),
                        waterSizes: water.backgroundSize.split(/,\s*/),
                        chop: WATER.tile / Math.min(2, Math.max(1, devicePixelRatio)),
                        filters: count("#board-svg filter, #board-svg [filter]"),
                        buildings: state.vertex_owner.filter((seat) => seat >= 0).length,
                        edges: count("#buildings-layer .piece-edge"),
                        roads: state.edge_owner.filter((seat) => seat >= 0).length,
                        roadCasts: count("#roads-layer .road-casts > line"),
                        roadLights: count("#roads-layer > .road-light"),
                        roadShades: count("#roads-layer > .road-shade"),
                        // What is drawn of a road lies under every building.
                        roadsUnder: Boolean(document.getElementById("roads-layer").compareDocumentPosition(
                          document.getElementById("buildings-layer")) & Node.DOCUMENT_POSITION_FOLLOWING),
                        // The robber's figure sits in the middle of its square.
                        robber: ["x", "y"].map((k) => Number(robber.getAttribute(k)) + ROBBER_SIZE / 2),
                        spot: robberSpot(hexById[state.robber]),
                        beside: robberSpot({x: 0, y: 0, token: 8}),
                        bare: robberSpot({x: 0, y: 0, token: null}),
                        expected: hexById[state.robber].token ? [`hex-token-${state.robber}`] : [],
                        before, marked, after: blocked(), numbered: `hex-token-${numbered.id}`,
                    };
                }"""
            )
        finally:
            browser.close()

    # One sand base under the land with its shadow's lines, and on it one
    # rounded tile and one shade a land hex; no line is drawn along an edge.
    assert drawn["bases"] == 1 and drawn["shadows"] == drawn["steps"] > 0
    assert drawn["tiles"] == drawn["shades"] == drawn["land"] and drawn["lines"] == 0
    # The tiles cast one shadow on the sand between them, and each has its
    # grain in two tones and a line inside its outline. None of it is an
    # SVG filter.
    assert drawn["casts"] == 1 and drawn["bevels"] == drawn["land"]
    assert drawn["grains"] == 2 * drawn["land"] and drawn["specks"] > 20
    assert drawn["filters"] == 0
    # The water is two pictures over the pane's gradient: the chop, a tile
    # repeated at its own size, and the patches, stretched over the pane.
    assert drawn["water"] == ['url("blob', 'url("blob', "radial-gr"]
    assert drawn["waterSizes"][:2] == [f"{drawn['chop']:g}px {drawn['chop']:g}px", "100% 100%"]
    assert drawn["buildings"] >= 8 and drawn["edges"] == drawn["buildings"]
    # Every road has its shadow, its lit side and its shaded side.
    assert drawn["roads"] >= 8 and drawn["roadsUnder"]
    assert drawn["roadCasts"] == drawn["roadLights"] == drawn["roadShades"] == drawn["roads"]
    assert drawn["robber"] == pytest.approx([drawn["spot"]["x"], drawn["spot"]["y"]])
    # Beside a number: to its left and level with it, over the edge of its
    # disc (radius 15) and well inside the hex (52 from its middle to a
    # side). In the middle of a hex with none.
    assert -30 < drawn["beside"]["x"] < -15 and abs(drawn["beside"]["y"]) < 12
    assert drawn["bare"]["x"] == 0 and abs(drawn["bare"]["y"]) < 6
    # The number under the robber is dimmed, and no other.
    assert drawn["before"] == drawn["expected"] == drawn["after"]
    assert drawn["marked"] == [drawn["numbered"]]


def test_a_roll_lights_its_numbers_and_sends_each_collecting_seat_its_cards(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = _seated(browser)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            one = page.evaluate(_PLAY_PRODUCTION, 1)
            none = page.evaluate(_PLAY_PRODUCTION, 0)
        finally:
            browser.close()

    assert not errors
    # Every hex with the number takes one light and one ring, gold or red;
    # the one building that collects takes a ring more and sends its card.
    assert one["seen"] == {"lights": one["numbered"], "rings": one["numbered"] + 1, "cards": 1}
    # A roll that gave nobody anything lights its numbers and sends nothing.
    assert none["seen"] == {"lights": none["numbered"], "rings": none["numbered"], "cards": 0}
    assert one["left"] == 0 and none["left"] == 0


# Takes in a view in which another seat rolled, this seat collected a Wood,
# and the roller then built a road, and records what the page shows each
# time an entry has played, before the view itself is drawn.
_PLAY_A_ROLL_THEN_A_ROAD = """
async () => {
  const me = state.players.find((p) => p.seat === state.seat);
  const other = state.players.find((p) => p.seat !== state.seat);
  const edge = board.edges.find((e) => state.edge_owner[e.id] < 0);
  const wood = () => {
    const count = document.querySelector('#hand .card[title^="Wood:"] .count');
    return count ? Number(count.textContent) : 0;
  };
  const shown = () => ({
    wood: wood(),
    cards: Number(seatRow(me.seat).querySelector(".stat.cards").textContent),
    theirs: Number(seatRow(other.seat).querySelector(".stat.cards").textContent),
    roads: document.querySelectorAll("#roads-layer .road-casts > line").length,
    dropped: document.querySelectorAll(".anim-piece").length,
    running: anim.running, version: state.version,
  });
  const before = shown();
  const view = JSON.parse(JSON.stringify(state));
  view.version += 1;
  const hand = {...me.hand, Wood: me.hand.Wood + 1};
  Object.assign(view.players.find((p) => p.seat === me.seat), {hand, hand_size: me.hand_size + 1});
  view.players.find((p) => p.seat === other.seat).hand_size = other.hand_size + 2;
  view.edge_owner[edge.id] = other.seat;
  const entry = (n, rest) => ({n: anim.last + n, round: view.round, actor: other.seat, ...rest});
  view.feed.events = [
    entry(1, {kind: "roll", value: 7, gains: {[me.seat]: board.resources.map((r) => (r === "Wood" ? 1 : 0))},
              after: {[me.seat]: {hand, hand_size: me.hand_size + 1}, [other.seat]: {hand_size: other.hand_size + 4}}}),
    entry(2, {kind: "build", piece: "road", at: edge.id, setup: false,
              after: {[other.seat]: {hand_size: other.hand_size + 2}}}),
    entry(3, {kind: "end"}),
  ];
  const played = [];
  const advanceWas = advance;
  advance = (step) => { advanceWas(step); played.push({kind: step.kind, ...shown()}); };
  takeView(view);
  while (anim.running) await new Promise((done) => setTimeout(done, 30));
  advance = advanceWas;
  return {before, played, after: shown(), landed: view.version, their: other.hand_size};
}
"""


def test_the_hand_the_rows_and_the_board_move_on_with_each_entry_as_it_plays(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = _seated(browser)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            seen = page.evaluate(_PLAY_A_ROLL_THEN_A_ROAD)
        finally:
            browser.close()

    assert not errors
    before, (rolled, built, ended), after = seen["before"], seen["played"], seen["after"]
    assert [rolled["kind"], built["kind"], ended["kind"]] == ["roll", "build", "end"]
    # The roll has played and the road has not: the hand and both rows show
    # the cards as the roll left them, while the view is still to come.
    assert rolled["running"] and rolled["version"] == before["version"] < seen["landed"]
    assert rolled["wood"] == before["wood"] + 1 and rolled["cards"] == before["cards"] + 1
    assert rolled["theirs"] == seen["their"] + 4 and rolled["roads"] == before["roads"]
    # The road has played: it lies with the board's own roads, paid for, and
    # the piece that dropped in is gone.
    assert built["running"] and built["version"] == before["version"]
    assert built["roads"] == before["roads"] + 1 and built["dropped"] == 0
    assert built["theirs"] == seen["their"] + 2 and built["wood"] == rolled["wood"]
    # The view lands on what the entries left.
    assert after["version"] == seen["landed"] and not after["running"]
    assert {k: after[k] for k in ("wood", "cards", "theirs", "roads")} == {
        k: built[k] for k in ("wood", "cards", "theirs", "roads")}


# The first road target on the board: where its bar and its edge end, and
# how far the bar's ends are from the corners.
_ROAD_TARGET = """
() => {
  const target = document.querySelector("#clickable-edge-layer .clickable-edge");
  const ends = (line) => ["x1", "y1", "x2", "y2"].map((k) => Number(line.getAttribute(k)));
  const [x1, y1, x2, y2] = ends(target.querySelector(".edge-mark"));
  const length = Math.hypot(x2 - x1, y2 - y1);
  const corner = (x, y) => Math.min(...board.vertices.map((v) => Math.hypot(v.x - x, v.y - y)));
  const style = (selector) => getComputedStyle(target.querySelector(selector));
  const middle = onScreen({x: (x1 + x2) / 2, y: (y1 + y2) / 2});
  return {
    at: [middle.x, middle.y],
    targets: document.querySelectorAll("#clickable-edge-layer .clickable-edge").length,
    length, clear: Math.min(corner(x1, y1), corner(x2, y2)),
    bar: parseFloat(style(".edge-mark-rim").strokeWidth), hit: parseFloat(style(".edge-hit").strokeWidth),
    hitDrawn: parseFloat(style(".edge-hit").strokeOpacity),
    under: Boolean(document.getElementById("clickable-edge-layer").compareDocumentPosition(
      document.getElementById("buildings-layer")) & Node.DOCUMENT_POSITION_FOLLOWING),
    roads: state.edge_owner.filter((seat) => seat >= 0).length,
  };
}
"""


def test_a_place_for_a_road_is_a_short_bar_clear_of_its_corners_and_a_click_builds_there(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = _seated(browser, OPENING, "#clickable-layer .clickable-vertex")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.locator("#clickable-layer .clickable-vertex").first.click()
            # Attached, not visible: a bar that runs straight up the page has
            # a box of no width.
            page.wait_for_selector("#clickable-edge-layer .clickable-edge", state="attached")
            shown = page.evaluate(_ROAD_TARGET)
            page.mouse.click(*shown["at"])
            page.wait_for_function("state.edge_owner.filter((seat) => seat >= 0).length === 1", timeout=10_000)
            built = page.evaluate(
                "({lines: document.querySelectorAll('#roads-layer .road-casts > line').length,"
                " targets: document.querySelectorAll('#clickable-edge-layer .clickable-edge').length})")
        finally:
            browser.close()

    assert not errors
    # One bar for each edge at the settlement just placed, each well short of
    # its edge (about 60 long) and clear of a piece on either corner (a
    # settlement's radius is 12), drawn under the buildings.
    assert 2 <= shown["targets"] <= 3 and shown["roads"] == 0
    assert 20 < shown["length"] < 35 and shown["clear"] - shown["bar"] / 2 > 12
    assert shown["under"]
    # The click is taken by a wider line that is not drawn.
    assert shown["hit"] > shown["bar"] and shown["hitDrawn"] == 0
    assert built == {"lines": 1, "targets": 0}


def test_with_reduced_motion_a_roll_lights_its_numbers_and_sends_no_cards(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            page = _seated(browser, reduced_motion="reduce")
            assert page.evaluate("REDUCED_MOTION")
            one = page.evaluate(_PLAY_PRODUCTION, 1)
        finally:
            browser.close()

    assert one["seen"] == {"lights": one["numbered"], "rings": one["numbered"] + 1, "cards": 0}
    assert one["left"] == 0


def test_a_seat_sees_its_own_roll_and_sets_its_sound_on_the_board(running_server):
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            context = browser.new_context()
            context.add_init_script(
                "try {"
                f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
                f"  localStorage.setItem('hexset.token.{SEATED.lower()}', 'minted before the restart');"
                "} catch (e) {}"
            )
            context.add_init_script(_WATCH_PLAYBACK)
            page = context.new_page()
            with page.expect_response(lambda r: "/api/reclaim" in r.url):
                page.goto(f"{BASE_URL}/{SEATED.lower()}", wait_until="load")
            page.wait_for_selector("#roll-dice.show")

            page.click("#roll-dice")
            page.wait_for_function("window.__seen.dice >= 1", timeout=10_000)
            # A seat's own move is drawn at once: the rolled position's controls
            # come while its dice are still on the board, and nothing queues.
            page.wait_for_function(
                "document.getElementById('phase').innerText.trim() !== 'ROLL'", timeout=10_000)
            assert not page.evaluate("anim.running")
            page.wait_for_function("document.getElementById('dice-tray').hidden", timeout=5_000)
            # The dice left the board for the roller's row, and nothing that
            # crossed with them is left over it.
            assert page.evaluate("window.__seen.ghosts") == 1
            page.wait_for_function(
                "document.getElementById('fly-layer').childElementCount === 0", timeout=5_000)

            # The board's top-left corner sets this browser's sound, off until
            # turned on; it has no speed to set.
            assert page.locator("#speed-toggle").count() == 0
            assert page.locator("#sound-toggle").get_attribute("aria-pressed") == "false"
            page.click("#sound-toggle")
            assert page.evaluate("localStorage.getItem('hexset.sound')") == "on"
            assert page.locator("#sound-toggle").get_attribute("aria-pressed") == "true"
            # Turning it on loads and decodes every clip the page plays.
            page.wait_for_function(
                "soundBuffers.size === new Set(Object.values(SOUNDS).flatMap((k) => k.files)).size",
                timeout=10_000)
            # Each is measured as it loads, so it plays at its kind's level.
            assert page.evaluate("[...soundBuffers.keys()].every((f) => soundLoudness.get(f) > 1e-4)")

            # Ending the turn plays our own move back while the table moves on:
            # the page has to keep watching and take in the bots' moves.
            page.wait_for_selector("#end-turn.show")
            # Taken before the click: instant bots can finish their turns
            # before anything read after it.
            before = page.evaluate("anim.last")
            page.click("#end-turn")
            page.wait_for_function(
                f"newestView().feed.events.some((e) => e.actor !== 0 && e.n > {before})", timeout=20_000)
            context.close()
        finally:
            browser.close()
