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
import threading

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
