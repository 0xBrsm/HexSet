# SPDX-License-Identifier: GPL-3.0-only
"""The web page's guide, supplied by a runtime.

HexSet's server offers a page seat a guide (`POST /api/guide`, `POST
/api/hint`, the view's `guide`) only where a runtime has registered one:
`register_guide` names the module that provides it, imported the first time
a table needs it. Without one, no table offers a guide and the guide's
routes answer 409.

The provider module defines:

- `table_guide(table, spec, config)`: a new guide for the
  `hexset.server.api.Table` `table`, reading with the model file `spec`
  under the server's `config`, carrying over what the table's journal holds
  of an earlier guide. The object it returns has `seats`, the seats it is on
  for, and `turn_on(seat)`, `position(game, step)`, `played(seat, step,
  round, action)`, `undone(back_to)`, `hint(seat, step)`, `view(seat, step)`
  (the view's `guide`), `series(seat, steps)`, `chance_at(seat, step)` and
  `close()`;
- `page_note_of(note)`: a journalled note as the page gets it;
- `saved_chance(notes)` and `saved_series(notes, steps)`: a seat's chance,
  and its chance after each step, from journalled notes alone;
- `flags_file(journal_path)`, `card_flags(journal_path, seat)` and
  `flag_row(fields, now)`: where a game's review flags are kept, the cards a
  seat has flagged, and one flag as it is kept (`ValueError` for a
  malformed one);
- `replay_hint(path, events, seat, step, game)` and `replay_played(path,
  events, seat, step, game)`: a replayed decision's hint and the moves
  played there, with their notes;
- optionally `clear(tables, origin, table, seat, use)`: refuse a guide
  request by raising `hexset.server.api.ApiError`; `use` is "start" for a
  guided game starting (a seat's guide turned on at a table where it never
  was, or a game played on from a position), else "guide", "hint" or
  "takeback";
- optionally `stamp_page(tables, body, user_agent)`: the page as served to
  that browser, its `hexset-page` and `hexset-turnstile` meta tags filled in
  where the guide asks the page for a token or a check;
  `page_id(tables, token, user_agent)`: the page id the token a request sends
  in `X-HexSet-Page` carries, or `None`; `pages_required(tables)`: whether a
  request is the page's only with one; and `page_check(tables, origin,
  payload)`: the answer to `POST /api/page`, the page's check;
- `carry_readings(source, path, lines, game_id)` and `readings_path(path)`:
  what a game begun from another one's position keeps of its guide, and
  where.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType

__all__ = ["guide_provider", "register_guide"]

_name: str | None = None
_module: ModuleType | None = None
_lock = threading.Lock()


def register_guide(module: str) -> None:
    """Name the module that provides the page's guide. The last one named
    wins; it is imported when a table first needs it."""
    global _name, _module
    with _lock:
        if module != _name:
            _name, _module = module, None


def guide_provider() -> ModuleType | None:
    """The registered guide module, imported; `None` where no runtime
    registered one."""
    global _module
    with _lock:
        if _name is None:
            return None
        if _module is None:
            _module = importlib.import_module(_name)
        return _module
