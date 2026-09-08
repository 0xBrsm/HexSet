# SPDX-License-Identifier: GPL-3.0-only
"""The training-loop-facing gym: a lane environment, a PettingZoo AEC
environment, and a Gymnasium wrapper on top of the AEC one.

`import hexset` stays numpy-only -- `pettingzoo` and `gymnasium` are only ever
imported by two of the three entry points below, gated behind the `gym` extra
(`pip install "hexset[gym]"`), the same way `hexset.server`/`hexset.clients`
gate `onnxruntime` behind their own extras.

`LaneEnv` (`hexset.gym.lanes`): many games in flight, stepped in lockstep, one
batch of decisions per tick, with a trade gate at every seat. It implements no
third-party API and so needs no third-party package: importing it costs
nothing beyond the engine, which is why the extra is checked lazily here
rather than at import.

`HexSetAEC` (`hexset.gym.aec`): one agent per seat, PettingZoo's
observation/action-mask convention, never registered with a global registry
-- PettingZoo has none -- so it is imported directly.

`HexSetEnv` (`hexset.gym.env`): a single-agent wrapper, one learner seat and
the rest `hexset.arena` bots, registered with Gymnasium under
`HexSet-v0` (`hexset.gym.register()`, called below as soon as the extra is
present, so `gymnasium.make("HexSet-v0")` works once this module is imported).
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
        """The extra's names still resolve -- to the error that names the extra.

        Raised on the attribute rather than on the import so that `LaneEnv`,
        which needs neither package, is importable from a plain `pip install
        hexset`. `from hexset.gym import HexSetAEC` still fails, and still says
        why.
        """
        if name in {"HexSetAEC", "HexSetEnv", "register"}:
            raise ImportError(_EXTRA)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BoardBots",
    "Decision",
    "Episode",
    "HexSetAEC",
    "HexSetEnv",
    "LaneEnv",
    "Outcome",
    "Request",
    "register",
]
