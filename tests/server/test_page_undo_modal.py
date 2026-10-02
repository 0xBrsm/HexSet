# SPDX-License-Identifier: GPL-3.0-only
"""The `#undo-build` corner button must sit *under* an open modal. It needs to
punch through exactly one: a robber hex with several victims opens the
"Steal from" modal over the board, and a self-played Knight stays undoable
through that forced move.

The table is built by playing real legal actions rather than poking
`game._state`, because the server reopens it by replaying the journal.
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

PORT = 8797
BASE_URL = f"http://127.0.0.1:{PORT}"
CODE = "UNDOMD"
SECRET = "a returning browser's own secret, undo-modal edition"
WEB_CLIENT = {"id": hashlib.sha256(SECRET.encode("utf-8")).hexdigest(), "kind": "web"}

# Reaches a buildable, undoable road for seat 0 inside the step bound,
# with resources left over to click for a trade.
PLAYOUT_SEED = 0


def _bot_seat() -> Seat:
    return Seat(kind=SeatKind.BOT, name="test-trader", spec="test-trader")


def _play_to_an_undoable_road(session, seed: int, max_steps: int = 3000) -> None:
    """Stop where seat 0 has just built a road on its own Main-phase turn:
    `can_undo` true, and a card still in hand to open the trade modal with.
    """
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
                # BUILD_ROAD costs one Wood and one Brick (Resource indices 0 and 1),
                # simulated rather than read back, to stop with a card left in hand.
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
    with serving(PORT, str(undoable_road_game)) as base:
        yield base


@pytest.fixture(autouse=True)
def stop_bot_runners():
    """Overrides conftest's per-test check: this module's server and its
    runners outlive each test, and `serving` checks for strays at the end."""
    yield


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

            # Guard the guard: without this the checks below could pass on an Undo
            # button that is never reachable at all.
            assert topmost_is_undo(), "undo should be clickable with nothing open over it"

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
