"""The served page offers a person no way to trade with another seat.

Withheld on the owner's instruction (2026-09-03): "no trading for humans at
this point -- we need to build back up gradually". The API keeps every route
(`PUT /api/games/<code>/valuation`, `POST .../trade`, `.../trade/confirm`,
`.../trade/decline`, and the MCP tools) for an LLM or an API client; what
goes is the browser's half of it. See the dated note in
`docs/negotiation-interface.md`.

Source-level, and deliberately so: this runs in the default suite in
milliseconds, and its job is to catch a trading control *reappearing* on the
page, which is a thing you can see in the file. What the page does once a
browser has it -- that a human's hand never moves through a trade, that bots
go on dealing with each other -- is `tests/web/test_page.py`, in Chromium,
against a live game.
"""

from __future__ import annotations

import pathlib
import re
import threading
import urllib.request

import pytest

from conftest import new_tables
from hexset.server.web import HexSetServer


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
    path = pathlib.Path(__file__).resolve().parents[2] / "src/hexset/server/static/index.html"
    return path.read_text()


# The frontend of the negotiation interface, by the names it had: the panel
# itself, its render function, the pure helpers behind it, and the five
# advertisement sliders that stood above it.
GONE = (
    'id="negotiation"',
    "renderNegotiation",
    "bundleClears",
    "bundleAffordable",
    "setValuation",
    "renderAdvertisement",
    "renderPending",
)


def test_the_served_page_has_no_trading_surface(live_server):
    _, base = live_server
    with urllib.request.urlopen(base) as response:
        page = response.read().decode("utf-8")
    assert [name for name in GONE if name in page] == []
    # No advertisement sliders, which is what the five controls were.
    assert 'type="range"' not in page


def test_the_page_calls_no_trading_route():
    """The routes are still there; the page must simply never reach them.

    Asserted over every path the page fetches rather than by searching for
    literal strings, so a trading call reintroduced through a template or a
    variable is caught the same as a hard-coded one.
    """
    page = _page()
    paths = set(re.findall(r"""["'`](/api/[^"'`\s?]*)""", page))
    assert paths == {
        "/api/models",
        "/api/board",
        "/api/state",
        "/api/games",
        "/api/join",
        "/api/action",
        "/api/undo",
        "/api/name",
        "/api/bot",
        # The two public reads, addressed by code: the game as a spectator
        # sees it, and the layout it is drawn on.
        "/api/table/${tableCode}",
        "/api/table/${tableCode}/board",
        "/api/table/${code}",
    }, sorted(paths)
    # Nothing in that set is a trading route, and there is no PUT at all --
    # `PUT /api/games/<code>/valuation` is the only one the API has.
    assert not [p for p in paths if "/trade" in p or p.endswith("/valuation")]
    assert 'method: "PUT"' not in page


def test_the_only_trade_the_page_offers_is_the_bank():
    """The half of trading that survived the one-event mechanic, and the
    owner's page keeps it: a resource card opens the bank/port route. It is
    not a deal with another seat -- `BANK_TRADE` is an action against the
    bank, taken by the seat itself -- so it stays."""
    page = _page()
    assert 'openModal("trade", {give: idx})' in page
    assert 'a.type === "BANK_TRADE"' in page
    # And nothing that would address another seat with an offer.
    assert "PROPOSE_TRADE" not in page
    assert "ACCEPT_TRADE" not in page
