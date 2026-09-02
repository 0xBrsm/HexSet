# SPDX-License-Identifier: GPL-3.0-only
"""A PettingZoo AEC environment, and a Gymnasium wrapper on top of it.

`import hexset` stays numpy-only -- `pettingzoo` and `gymnasium` are only
ever imported here, gated behind the `gym` extra
(`pip install "hexset[gym]"`), the same way `hexset.server`/`hexset.clients`
gate `onnxruntime` behind their own extras.

`HexSetAEC` (`hexset.gym.aec`): one agent per seat, PettingZoo's
observation/action-mask convention, never registered with a global registry
-- PettingZoo has none -- so it is imported directly.

`HexSetEnv` (`hexset.gym.env`): a single-agent wrapper, one learner seat and
the rest `hexset.arena` bots, registered with Gymnasium under
`HexSet-v0` (`hexset.gym.register()`, called once below so
`gymnasium.make("HexSet-v0")` works as soon as this module is imported).
"""

from __future__ import annotations

try:
    import gymnasium  # noqa: F401
    import pettingzoo  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by the extras check
    raise ImportError(
        "hexset.gym requires the 'gym' extra: pip install \"hexset[gym]\""
    ) from exc

from .aec import HexSetAEC
from .env import HexSetEnv, register

__all__ = ["HexSetAEC", "HexSetEnv", "register"]

register()
