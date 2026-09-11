# SPDX-License-Identifier: GPL-3.0-only
"""The board-level Knight undo (see webplay._UNDOABLE_BUILDS and the
`#undo-build` corner button in index.html).

Commit 3034778 ("play a knight, then move the robber -- two steps, not one")
dropped the client-side arm/cancel path a Knight used to have, on the
reasoning that it now resolves through the same forced robber phase a seven
does the instant it is played -- and left the fix's own test noting a
browser pass was still owed (`test_a_knight_resolves_through_the_session_
like_a_seven`). Reported back as a regression: the take-back was expected to
still exist, just reachable a different way. It's the ordinary undo point
every build/Road-Building play already gets (see `_UNDOABLE_BUILDS` and
`tests/server/test_webplay.py::test_playing_a_knight_is_undoable_until_the_
robber_actually_moves` for the session-level half of this); this is the
page-visible half those tests can't reach, in the spirit of
`test_page_setup_turn.py`'s own docstring about page-visible defects hiding
behind a green Python suite.

Same shape as `test_page_setup_turn.py`: a journal built by driving a
session directly, a server started against it, a returning browser that
reclaims its seat -- and skips the same way where playwright is absent.
Reaching "seat 0's own turn, holding an unplayed Knight" needs an actual
played-out game (a dev card is drawn from a real shuffled deck, not handed
out), so the fixture drives one forward with biased-random legal actions
until that precondition holds, rather than scripting a handful of
placements the way the setup-turn fixture does.
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

from hexset.actions import Action, ActionType, options_for
from hexset.cards import DevCard
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
PORT = 8794
BASE_URL = f"http://127.0.0.1:{PORT}"
# Must be CODE_LENGTH (6) characters from api.CODE_ALPHABET -- no 0/1/i/l/o.
CODE = "KNGT8X"
SECRET = "a returning browser's own secret, knight edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}

# Fixed so the playout -- and the exact table it lands on (seat 0 holding
# one Knight, one ore, phase ROLL, robber on hex 17) -- is reproducible.
# Found by search (scratch script, not kept): converges in 762 steps.
PLAYOUT_SEED = 0


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="heximax", spec="heximax")


def _play_to_a_held_knight(session, seed: int, max_steps: int = 6000) -> None:
    """Drive every seat's actions directly (bots included -- nothing here
    runs real bot policy, it's just legal-action selection) until it is
    seat 0's own turn, before it has rolled or played anything, holding an
    unplayed Knight. Biased rather than uniform: BUY_DEV_CARD is taken
    whenever it's legal (the fastest way to circulate a Knight into some
    hand), and playing a dev card is never chosen for any seat -- a Knight
    that showed up early has to survive in the hand it landed in for this to
    terminate on seat 0 specifically holding one.
    """
    game = session.game
    for seat in range(4):
        session.claimed_seats.add(seat)
    rng = random.Random(seed * 7919 + 1)
    play_types = {
        ActionType.PLAY_KNIGHT,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_MONOPOLY,
        ActionType.PLAY_YEAR_OF_PLENTY,
    }
    for _ in range(max_steps):
        if is_over(game):
            raise RuntimeError("game ended before seat 0 held an unplayed Knight")
        if session.awaiting_confirm is not None:
            session.submit(
                session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN))
            )
            continue
        seat = to_move(game)
        if (
            seat == 0
            and game.phase in (Phase.ROLL, Phase.MAIN)
            and not game.dev_card_played
            and game._state.dev_cards[0][DevCard.KNIGHT] > 0
        ):
            return
        options = options_for(game)
        if not options:
            raise RuntimeError(f"seat {seat} has no legal action")
        buyable = [a for a in options if a.type is ActionType.BUY_DEV_CARD]
        unplayed = [a for a in options if a.type not in play_types]
        pool = buyable or unplayed or options
        session.submit(seat, action_to_wire(rng.choice(pool)))
    raise RuntimeError(f"did not reach a held Knight for seat 0 in {max_steps} steps")


@pytest.fixture(scope="module")
def held_knight_game(tmp_path_factory):
    """A journal that stops the instant seat 0 (this browser) can play its
    Knight: nothing about this turn has happened yet (phase ROLL, one ore in
    hand, dev_card_played False), so every assertion below is about what
    playing the card -- and then undoing it -- actually does, not about
    picking through whatever else a played-out game left lying around.
    """
    directory = tmp_path_factory.mktemp("knight")
    seats = [
        Seat(kind=SeatKind.PLAYER, name="Ada", token="t-Ada", client=WEB_CLIENT),
        _bot_seat(),
        _bot_seat(),
        _bot_seat(),
    ]
    session = build_session(CODE, seats, Config(games_dir=str(directory), seed=PLAYOUT_SEED), first=0)
    _play_to_a_held_knight(session, PLAYOUT_SEED)
    game = session.game
    assert to_move(game) == 0
    assert game.phase is Phase.ROLL
    assert game._state.dev_cards[0][DevCard.KNIGHT] == 1
    assert not game.dev_card_played
    return directory


@pytest.fixture(scope="module")
def running_server(held_knight_game):
    process = subprocess.Popen(
        [
            sys.executable, "-m", "hexset.server.web",
            "--no-browser", "--port", str(PORT), "--games-dir", str(held_knight_game),
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
    return "show" in (page.locator(f"#{fab_id}").get_attribute("class") or "")


def test_playing_then_undoing_a_knight_offers_and_uses_the_board_cancel(running_server):
    """The regression itself, played and taken back in one pass -- the two
    halves share one live table (module-scoped `running_server`), so the
    play has to be undone by the end for the state it started from to still
    mean anything to a second test.

    A self-played Knight used to leave a CANCEL corner button up the whole
    time the board was waiting for a hex. It doesn't need to be that
    particular button back -- but it does need some way out, and
    #undo-build (state.can_undo) is what carries it now: shown the instant
    the card is played, gone the instant the take-back lands the seat
    exactly where it was."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context, page = _seated_page(browser)

            assert page.locator("#phase").inner_text().strip() == "ROLL"
            assert not _shown(page, "undo-build"), "nothing played yet to undo"

            page.click('.card[title="Play Knight"]')
            page.wait_for_function(
                "document.getElementById('phase').innerText.trim() === 'PLACE ROBBER'",
                timeout=10_000,
            )
            assert _shown(page, "undo-build"), "a self-played knight has to stay cancellable"
            # Nothing else on the corners competes with it in Phase.ROBBER.
            assert not _shown(page, "roll-dice")
            assert not _shown(page, "end-turn")
            assert not _shown(page, "buy-dev-card")

            page.click("#undo-build")
            page.wait_for_function(
                "document.getElementById('phase').innerText.trim() === 'ROLL'",
                timeout=10_000,
            )
            assert not _shown(page, "undo-build"), "nothing left to take back"
            assert _shown(page, "roll-dice"), "back on this seat's own un-rolled turn"
            assert page.locator('.card[title="Play Knight"]').count() == 1, (
                "the card itself has to be back, not just the phase"
            )

            context.close()
        finally:
            browser.close()
