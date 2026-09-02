"""What a checkpoint declares about itself.

Still the model side of the boundary — the engine never reads any of this —
but split out of `hexset.clients.onnxbot` because that module imports onnxruntime
at load time and this is the part with decisions in it. Keeping the two
together meant the bounds below could only be exercised on a machine with a
runtime wheel installed, which the usual development machine here is not.

A checkpoint carrying its own settings is the point: dropping `mcts256.onnx`
into `models/` should be the whole of configuring it, with no spec grammar,
no flag, and nothing outside `hexset.clients.onnxbot` that knows what a simulation
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
