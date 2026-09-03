# SPDX-License-Identifier: GPL-3.0-only
"""The file to pass to `catanatron-play --code`.

    catanatron-play --code path/to/register.py --players=DC:search2-notrade,AB:2

`--code` loads this file standalone via `importlib.util.spec_from_file_location`
(see catanatron's `cli/play.py`), not as part of any package, so the import
below has to be absolute -- a relative import fails with "No module named
'module'", `--code`'s own placeholder module name.
"""

from catanatron.cli.cli_players import register_cli_player

from hexset.catanatron.player import DevCatanPlayer

register_cli_player("DC", DevCatanPlayer)