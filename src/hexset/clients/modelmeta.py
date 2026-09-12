"""What a checkpoint declares about itself.

The model side of the boundary — the engine never reads any of this — split
out of `hexset.clients.onnxbot` because that module imports onnxruntime at
load time and this is the part with decisions in it. Keeping the two
together meant the bounds below could only be exercised on a machine with a
runtime wheel installed, which the usual development machine here is not.
Nothing here imports a runtime and nothing here is server-only, which is why
it sits beside its readers rather than under `hexset.server`: the
runtime-neutral `hexset.clients.netbot` reads the gate settings too.

A checkpoint carrying its own settings is the point: dropping `mcts256.onnx`
into `models/` should be the whole of configuring it, with no spec grammar,
no flag, and nothing outside `hexset.clients` that knows what a simulation
is.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bounded because `models/` is a drop directory and a bot is spawned
# synchronously inside a request: a file asking for ten million simulations
# would hang the seat rather than play it.
MAX_SIMULATIONS = 4096
MAX_WAVE = 256

DEFAULT_SIMULATIONS = 128
DEFAULT_WAVE = 16

# A gate that has not measured its own resolution declares no floor, and any
# strictly positive gain clears (`hexset.trading.trade_floor_of`).
DEFAULT_TRADE_FLOOR = 0.0
# A network gate's gains are differences of win probability, so no honest
# floor reaches 1.0. A file asking for more is asking never to trade, and
# gets the nearest floor that says so.
MAX_TRADE_FLOOR = 1.0


@dataclass(frozen=True)
class SearchConfig:
    """How a checkpoint asks to be played. `simulations = 0` means no search."""

    simulations: int = 0
    wave: int = DEFAULT_WAVE

    @property
    def searches(self) -> bool:
        return self.simulations > 0


def _clamp(value: str | None, default: int, ceiling: int) -> int:
    """One metadata integer, bounded.

    A missing or unreadable key takes the default rather than failing the load:
    a checkpoint is a model first, and a typo'd hint should cost the hint, not
    the whole opponent.
    """
    try:
        wanted = int(value) if value else default
    except ValueError:
        return default
    return max(0, min(wanted, ceiling))


def _clamp_float(value: str | None, default: float, ceiling: float) -> float:
    """One metadata float, bounded into `[0.0, ceiling]`. Unreadable takes the
    default, for the same reason `_clamp` does."""
    try:
        wanted = float(value) if value else default
    except ValueError:
        return default
    if wanted != wanted:  # NaN would compare false against every floor
        return default
    return max(0.0, min(wanted, ceiling))


@dataclass(frozen=True)
class GateConfig:
    """How a checkpoint asks its trade gate to be run.

    `trade_floor` is the gate's own measured clearing resolution
    (`hexset.trading.trade_floor_of`) -- a property of the exported model,
    a floor measured against one checkpoint's value head says nothing
    about another's, so it is read off the file rather than hardcoded for
    every checkpoint alike.
    """

    trade_floor: float = DEFAULT_TRADE_FLOOR


def gate_config(meta: dict[str, str]) -> GateConfig:
    """The trade gate a checkpoint's metadata asks for.

    An absent `trade_floor` gives the shipped default: `0.0`, unmeasured, so
    strict positivity is the whole gate. A `gate_rows` key left behind by an
    older export names a bound this gate no longer has -- read without
    complaint, and ignored, the same as any other key this module has never
    heard of.
    """
    return GateConfig(
        trade_floor=_clamp_float(
            meta.get("trade_floor"), DEFAULT_TRADE_FLOOR, MAX_TRADE_FLOOR
        ),
    )


def gate_config_of(checkpoint: object) -> GateConfig:
    """The gate settings `checkpoint` declares, read by name.

    Structural, not by inheritance -- the convention `hexset.trading` already
    uses to read a gate's own surface. A loader that predates this key, or
    one in another repo that has not adopted it, carries no such attribute
    and is read at the unmeasured default: the behaviour it had before the
    key existed. A checkpoint still carrying `gate_rows` is likewise read
    without complaint; the field is simply never looked at. The floor is
    taken as given; a negative one is refused where every other floor is,
    `hexset.trading.trade_floor_of`.
    """
    floor = getattr(checkpoint, "trade_floor", None)
    return GateConfig(
        trade_floor=DEFAULT_TRADE_FLOOR if floor is None else float(floor),
    )


def search_config(meta: dict[str, str]) -> SearchConfig:
    """The search a checkpoint's metadata asks for.

    Search settings are read only when the file actually asks to be searched,
    so a stale `simulations` left behind by an export cannot quietly turn a
    policy checkpoint into a search.
    """
    if meta.get("search", "none") != "mcts":
        return SearchConfig()
    return SearchConfig(
        simulations=_clamp(meta.get("simulations"), DEFAULT_SIMULATIONS, MAX_SIMULATIONS)
        or DEFAULT_SIMULATIONS,
        wave=_clamp(meta.get("wave"), DEFAULT_WAVE, MAX_WAVE) or DEFAULT_WAVE,
    )
