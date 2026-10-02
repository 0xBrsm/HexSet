# SPDX-License-Identifier: GPL-3.0-only
"""The board-level Knight undo (`webplay._UNDOABLE_BUILDS` and `#undo-build`):
a self-played Knight must stay takeable-back from the page while the board
waits for a robber hex. The session half is in `test_webplay.py`.

Reaching "seat 0's own turn, holding an unplayed Knight" needs a real
played-out game, since a dev card is drawn from a shuffled deck.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from hexset.actions import Action, ActionType, options_for
from hexset.cards import DevCard
from hexset.game import Phase, is_over, to_move
from hexset.server.api import Config, Seat, SeatKind, build_session
from hexset.server.wire import action_to_wire

from _page_server import serving


try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001 -- any import-time failure means "skip"
    sync_playwright = None
    _PLAYWRIGHT_IMPORT_ERROR = error

pytestmark = pytest.mark.skipif(
    sync_playwright is None, reason=f"playwright not available: {_PLAYWRIGHT_IMPORT_ERROR}"
)

PORT = 8794
BASE_URL = f"http://127.0.0.1:{PORT}"
# Must be CODE_LENGTH (6) characters from api.CODE_ALPHABET -- no 0/1/i/l/o.
CODE = "KNGT8X"
SECRET = "a returning browser's own secret, knight edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}

# Fixed so the playout, and the exact table it lands on, is reproducible.
PLAYOUT_SEED = 0


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _play_to_a_held_knight(session, seed: int, max_steps: int = 6000) -> None:
    """Biased: BUY_DEV_CARD whenever legal, and no seat ever plays a dev card,
    so a Knight survives in the hand it landed in.
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
    with serving(PORT, str(held_knight_game)) as base:
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
    return "show" in (page.locator(f"#{fab_id}").get_attribute("class") or "")


def test_playing_then_undoing_a_knight_offers_and_uses_the_board_cancel(running_server):
    """The halves share one live table (module-scoped `running_server`), so the
    play must be undone by the end.
    """
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
