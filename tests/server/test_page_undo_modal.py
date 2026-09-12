# SPDX-License-Identifier: GPL-3.0-only
"""The `#undo-build` corner button (see `index.html`'s CSS comment on it and
`state.can_undo`) painted over an open modal instead of under it.

`#undo-build` carried a flat `z-index: 60`, above `#modal`'s `z-index: 55`,
for one specific reason: a robber hex with more than one victim opens the
"Steal from" modal on top of the board, and a self-played Knight stays
undoable through that forced move (`_UNDOABLE_BUILDS`), so undo has to reach
through that one modal to still be clickable. The flat number gave it that
for every other modal too -- trade, discard, Monopoly, Year of Plenty --
none of which have any reason to let a board control punch through them, and
all of which now render Undo on top of their own backdrop, unreachable
underneath it but visibly (and clickably) sitting above.

Reached the ordinary way: a seat builds a road on its own Main-phase turn
(leaving `can_undo` true), then opens the trade modal by clicking a
resource card in hand -- reachable with no dev card in play, unlike
Monopoly/Year of Plenty. Same journal-and-restart shape as the other
`test_page_*.py` modules, but built by playing real legal actions throughout
(bots included) rather than by poking `game._state` directly: this table is
handed to a freshly started server, which reopens it by replaying the
journal from scratch (`api.Tables._reopen`) -- a hand written in by hand
never happened as far as that replay is concerned, and the very first
build past it would come back "not legal in MAIN".
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

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import Phase, is_over, to_move
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

# Its own port and code, so this module can run alongside the other page
# suites without colliding.
PORT = 8797
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "UNDOMD"
SECRET = "a returning browser's own secret, undo-modal edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}

# Found by search (scratch script, not kept): the walk below reaches a
# buildable, undoable road for seat 0 well inside the step bound at this
# seed, with resources left over afterward to click for a trade.
PLAYOUT_SEED = 0


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="heximax", spec="heximax")


def _play_to_an_undoable_road(session, seed: int, max_steps: int = 3000) -> None:
    """Drive every seat's actions directly (bots included, same as
    `test_page_robber_undo.py`'s own driver) until seat 0 has just built a
    road on its own Main-phase turn -- `can_undo` true, and (checked before
    the build) resources left over afterward so there is still a card in
    hand to click and open the trade modal with."""
    game = session.game
    for seat in range(4):
        session.claimed_seats.add(seat)
    rng = random.Random(seed * 7919 + 3)
    for _ in range(max_steps):
        if is_over(game):
            raise RuntimeError("the game ended before seat 0 built an undoable road")
        if session.awaiting_confirm is not None:
            session.submit(
                session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN))
            )
            continue
        seat = to_move(game)
        options = legal_actions(game, seat)
        if not options:
            raise RuntimeError(f"seat {seat} has no legal action")
        if seat == 0 and game.phase is Phase.MAIN:
            road_options = [a for a in options if a.type is ActionType.BUILD_ROAD]
            if road_options:
                # BUILD_ROAD costs one Wood and one Brick (board.terrain's
                # Resource indices 0 and 1) -- simulated here rather than
                # read back off the session, since the point is to stop
                # with at least one other card still in hand.
                hand_after = game._state.hands[0][:]
                hand_after[0] -= 1
                hand_after[1] -= 1
                if min(hand_after) >= 0 and sum(hand_after) > 0:
                    session.submit(0, action_to_wire(road_options[0]))
                    return
        session.submit(seat, action_to_wire(rng.choice(options)))
    raise RuntimeError(f"did not reach an undoable road for seat 0 in {max_steps} steps")


@pytest.fixture(scope="module")
def undoable_road_game(tmp_path_factory):
    directory = tmp_path_factory.mktemp("undo-modal")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat(), _bot_seat(), _bot_seat(),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=PLAYOUT_SEED), first=0)
    _play_to_an_undoable_road(session, PLAYOUT_SEED)
    assert session._undo is not None and session._undo.actor == 0
    return directory


@pytest.fixture(scope="module")
def running_server(undoable_road_game):
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(undoable_road_game),
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


def _seated_page(browser):
    context = browser.new_context()
    context.add_init_script(
        "try {"
        f"  localStorage.setItem('hexset.secret', {json.dumps(SECRET)});"
        f"  localStorage.setItem('hexset.token.{CODE.lower()}', 'minted before the restart');"
        "} catch (e) {}"
    )
    page = context.new_page()
    with page.expect_response(lambda r: "/api/reclaim" in r.url) as reclaimed:
        page.goto(f"{BASE_URL}/{CODE.lower()}", wait_until="load")
    assert reclaimed.value.json()["seat"] == 0
    page.wait_for_selector(".player-row")
    return context, page


def test_undo_sits_beneath_an_open_modal(running_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)

            page.wait_for_function(
                "document.getElementById('undo-build').classList.contains('show')",
                timeout=10_000,
            )
            center = page.eval_on_selector(
                "#undo-build",
                "e => { const r = e.getBoundingClientRect(); "
                "return {x: r.x + r.width / 2, y: r.y + r.height / 2}; }",
            )

            def topmost_is_undo() -> bool:
                return page.evaluate(
                    "([x, y]) => !!document.elementFromPoint(x, y)?.closest('#undo-build')",
                    [center["x"], center["y"]],
                )

            # Guard the guard: with no modal up, Undo is reachable at its own
            # spot -- otherwise the two checks below would trivially agree.
            assert topmost_is_undo(), "undo should be clickable with nothing open over it"

            # A trade offer: reachable with no dev card drawn, unlike
            # Monopoly/Year of Plenty, and not one of the auto modes
            # (discard/steal) syncModal would put up on its own.
            page.click(".hand-card-clickable")
            page.wait_for_selector("#modal.show")

            assert not topmost_is_undo(), (
                "undo is still the topmost element at its own center with a "
                "trade modal open over it"
            )
            topmost = page.evaluate(
                "([x, y]) => { const e = document.elementFromPoint(x, y); "
                "return e ? (e.id || e.className || e.tagName) : null; }",
                [center["x"], center["y"]],
            )
            assert topmost == "modal", f"expected the modal backdrop on top, got {topmost!r}"

            context.close()
        finally:
            browser.close()
