# SPDX-License-Identifier: GPL-3.0-only
"""The file to pass to `catanatron-play --bot`.

    catanatron-play --bot DC=path/to/register.py#DevCatanPlayer \
        --players=DC:heximax,AB:2

`--bot` loads the file via catanatron's `sources.load_class` (under a
synthetic module name, so the import below has to be absolute -- a relative
import fails) and registers the class under the given name (`DC=` here);
the module-level `REGISTRY.register` below covers the file being imported
directly instead. (`--code`, the old mechanism this file used to target,
is gone upstream.)
"""

from catanatron.registry import REGISTRY

from hexset.catanatron.player import DevCatanPlayer

REGISTRY.register("DC", DevCatanPlayer, replace=True)
