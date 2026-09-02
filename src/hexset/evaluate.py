# SPDX-License-Identifier: GPL-3.0-only
"""Compat shim: `evaluate.py` moved to `hexset.bots.evaluate` (the handcrafted
evaluation is shared by every heuristic bot in `hexset.bots`). Existing
`from hexset.evaluate import ...` callers (`hexset.tuning`, `hexset.fitting`,
`hexset.dataset`, HexNet, tests) keep working unchanged."""

from hexset.bots.evaluate import *  # noqa: F401,F403
