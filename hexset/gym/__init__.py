# SPDX-License-Identifier: GPL-3.0-only
"""The training-loop-facing gym: `LaneEnv` (many games in lockstep, one batch of
decisions per tick, a trade gate at every seat), `HexSetAEC` (one PettingZoo
agent per seat) and `HexSetEnv` (single-agent Gymnasium wrapper, registered as
`HexSet-v0`).

Only the latter two need `pettingzoo`/`gymnasium`, gated behind the `gym` extra
and checked lazily so `import hexset` stays numpy-only.
"""

from __future__ import annotations

from .lanes import BoardBots, Decision, Episode, LaneEnv, Outcome, Request

_EXTRA = "hexset.gym's PettingZoo/Gymnasium environments require the 'gym' extra: pip install \"hexset[gym]\""

try:
    import gymnasium  # noqa: F401
    import pettingzoo  # noqa: F401
except ImportError:  # pragma: no cover - exercised by the extras check
    _HAVE_EXTRA = False
else:
    _HAVE_EXTRA = True

if _HAVE_EXTRA:
    from .aec import HexSetAEC
    from .env import HexSetEnv, register

    register()
else:

    def __getattr__(name: str):
        """The extra's names resolve to the error naming the extra. Raised on
        the attribute, not the import, so `LaneEnv` stays importable from a
        plain `pip install hexset`."""
        if name in {"HexSetAEC", "HexSetEnv", "register"}:
            raise ImportError(_EXTRA)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Only what imports here: on a base install `from hexset.gym import *` must
# not reach the extra's names.
__all__ = [
    "BoardBots",
    "Decision",
    "Episode",
    "LaneEnv",
    "Outcome",
    "Request",
]
if _HAVE_EXTRA:
    __all__ += ["HexSetAEC", "HexSetEnv", "register"]
