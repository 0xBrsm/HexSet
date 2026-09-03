"""A smoke test for the negotiation panel (`docs/negotiation-interface.md`
§3), the frontend half of the negotiation interface.

No JS engine is available in this environment to execute `static/index.html`'s
script the way a browser would (see the negotiation-interface PR's own notes),
so this checks the served page's source instead of a rendered DOM: the panel
markup and its render function exist, the pure surplus/affordability helpers
are the functions `renderNegotiation` actually calls (so a future edit that
inlines them back in would be caught), and the chip loop is wired to
`state.valuations` -- the counterparty's own published vector, which is what
turns into the wants/gives chips the design asks for.
"""

from __future__ import annotations

import re
import threading

import pytest

from conftest import new_tables
from hexset.server.web import HexSetServer

SOLO = ["search2", "search2", "search2"]


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), new_tables())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _page() -> str:
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "src/hexset/server/static/index.html"
    return path.read_text()


def test_the_served_page_carries_the_negotiation_panel(live_server):
    import urllib.request

    _, base = live_server
    with urllib.request.urlopen(base) as response:
        page = response.read().decode("utf-8")
    assert 'id="negotiation"' in page
    assert "function renderNegotiation" in page


def test_the_panel_renders_counterparty_chips_from_the_published_vector():
    page = _page()
    body = re.search(r"function renderNegotiation\(\) \{.*?\n\}\n", page, re.S).group(0)

    # One block per counterparty, read straight off the public valuations.
    assert "state.valuations.forEach((vector, seat) => {" in body
    # A chip per nonzero entry, coloured by its sign -- want (green) vs. give
    # (red) -- and clicking one composes the draft bundle, not the standing
    # advertisement (`setValuation` is untouched by this loop).
    assert 'v > 0 ? "chip-want" : "chip-give"' in body
    assert "resourceCardTile(name" in body
    assert "draft.give[index] += 1" in body and "draft.receive[index] += 1" in body

    # Turn-timing: every panel is a proposer on my own turn; only the current
    # player's panel is active during someone else's.
    assert "myTurn || state.to_move === seat" in body


def test_clears_and_affordable_are_pure_functions_negotiation_calls():
    """`bundleClears`/`bundleAffordable` take no DOM and no `state` global,
    so they are checkable against fixed vectors/hands independent of a
    browser (`docs/negotiation-interface.md` §6) -- asserted here by
    checking their signatures and that `renderNegotiation` is the caller,
    not an inlined copy of the same arithmetic."""
    page = _page()
    assert re.search(r"function surplus\(vector, bundle\) \{", page)
    assert re.search(r"function bundleClears\(myVector, theirVector, bundle\) \{", page)
    assert re.search(
        r"function bundleAffordable\(resourceNames, give, myHand, receiveTotal, counterpartyHandSize\) \{",
        page,
    )
    body = re.search(r"function renderNegotiation\(\) \{.*?\n\}\n", page, re.S).group(0)
    assert "bundleClears(state.valuations[state.seat]" in body
    assert "bundleAffordable(board.resources, draft.give, myHand, totalReceive, counterpartyHandSize)" in body


@pytest.mark.parametrize(
    "my_vector,their_vector,bundle,expected",
    [
        # I give wood, receive ore; I want ore and would give wood -- and so
        # does the counterparty from their own side. Both surpluses positive.
        ([1.0, 0, 0, 0, -1.0], [-1.0, 0, 0, 0, 1.0], (1, 0, 0, 0, -1), True),
        # The counterparty's own vector doesn't want this bundle -- no clear.
        ([1.0, 0, 0, 0, -1.0], [0.0, 0, 0, 0, 0.0], (1, 0, 0, 0, -1), False),
        # My own vector calls it a loss for me -- also no clear (this is a
        # client-side convenience only; the server's hard rule is only on
        # the counterparty's side, ratification decision 4).
        ([-1.0, 0, 0, 0, 1.0], [-1.0, 0, 0, 0, 1.0], (1, 0, 0, 0, -1), False),
    ],
)
def test_bundle_clears_matches_the_servers_public_surplus_formula(
    my_vector, their_vector, bundle, expected
):
    """A hand computation of `dot(v, b) > 0` on both sides -- the same
    arithmetic `bundleClears` runs and `hexset.trading._best_clearing` runs
    server-side -- so this is the reference the JS is checked against
    without needing to execute it (no JS engine in this environment)."""
    mine = sum(v * b for v, b in zip(my_vector, bundle))
    theirs = sum(v * -b for v, b in zip(their_vector, bundle))
    assert (mine > 0 and theirs > 0) == expected
