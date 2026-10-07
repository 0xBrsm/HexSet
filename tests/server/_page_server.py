# SPDX-License-Identifier: GPL-3.0-only
"""The page tests' server: `hexset.server.web`'s own `HexSetServer`, on a
thread in this process rather than a subprocess, so a module pays no
interpreter start-up. `Tables` is built fresh over the games directory the
way `web.main` builds it, so a journalled game is reopened from disk exactly
as after a restart.

The server outlives each test in its module, and so do its bot runners, which
conftest's per-test `stop_bot_runners` check would call a leak: a page module
overrides that fixture, and `serving` makes the same check once it has shut
its own runners down.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading

import pytest

from hexset.server.api import Config, Tables
from hexset.server.web import HexSetServer


@contextlib.contextmanager
def serving(port: int, games_dir: str = ""):
    tables = Tables(Config(games_dir=games_dir))
    server = HexSetServer(("127.0.0.1", port), tables)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        tables.close()
        server.server_close()
        live = [t.name for t in threading.enumerate() if t.name.startswith("bot-")]
        assert not live, f"bot runner threads still alive after this module: {live}"


def launch(playwright):
    """Chromium, or a skip when its binary is not installed: the `browser`
    extra brings the `playwright` package, and the browser is a separate
    download (`docs/testing.md`), so a bare `pytest` passes without either."""
    try:
        return playwright.chromium.launch()
    except Exception as error:  # noqa: BLE001 -- playwright's own Error type
        if "Executable doesn't exist" in str(error):
            pytest.skip("Chromium is not installed: python -m playwright install chromium")
        raise


#: Wraps the page's `fetch` to keep the deal's request and response in
#: sessionStorage, which survives the navigation to the new table. The page
#: moves on before a test could read the response, and intercepting it from
#: outside (`page.route`) can leave the next page's first request paused.
_KEEP_THE_DEAL = """
(() => {
  const real = window.fetch;
  window.fetch = async (input, init) => {
    const response = await real(input, init);
    const url = typeof input === "string" ? input : input.url;
    if (init && init.method === "POST" && new URL(url, location.href).pathname === "/api/games") {
      sessionStorage.setItem("deal", JSON.stringify({
        request: JSON.parse(init.body), response: await response.clone().json(),
      }));
    }
    return response;
  };
})();
"""


def front_door(page, base_url: str) -> tuple[dict, dict]:
    """Opens the page with no table to resume, which asks what to deal, and
    deals the defaults. Call it where the page used to deal on its own.
    Returns the deal's request and response bodies."""
    page.add_init_script(_KEEP_THE_DEAL)
    page.goto(f"{base_url}/", wait_until="load")
    page.wait_for_selector("#modal.show #deal-new-game")
    page.click("#deal-new-game")
    page.wait_for_url(re.compile(r"/[a-z0-9]{6}$"))
    deal = json.loads(page.evaluate("sessionStorage.getItem('deal')"))
    return deal["request"], deal["response"]
