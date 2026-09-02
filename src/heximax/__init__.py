# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated: `heximax` moved to `hexset.bots.heximax` (every heuristic bot
now lives under `hexset.bots`, so `search2` and `heximax` can share code).
This shim keeps `import heximax` working for existing callers -- tests,
`hexset.bench`/`hexset.server` (which import it for its registration side
effect), external scripts -- but new code should import `hexset.bots.heximax`
(or use the `Heximax`/`heximax`/etc. names `hexset.bots` re-exports)
directly. `import heximax` still registers the "heximax"/"heximax-omni"/
"heximax-notrade" presets and the "heximax-trading"/"heximax-notrade"
evaluator names, the same as it always has, because that registration now
happens on `hexset.bots.heximax` import and this shim imports exactly that.
"""

from hexset.bots.heximax import *  # noqa: F401,F403
from hexset.bots.heximax import __all__  # noqa: F401
