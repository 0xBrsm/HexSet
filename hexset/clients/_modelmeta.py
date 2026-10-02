"""What a checkpoint declares about itself, read off its own metadata.

The model side of the boundary: the engine never reads any of this, and
nothing here imports a model runtime.

How a checkpoint asks to *bargain* is `hexset.trading.TradeParams`, the same
object every bot that declares its bargaining carries, so a checkpoint can ask
for the same negotiation protocol rather than a cut-down one. `gate_config` is the reader
for it; the clamping lives on `TradeParams.from_meta`, because a value that
arrives as text from a file is read the same way wherever it came from.
"""
from __future__ import annotations

from dataclasses import dataclass

from hexset.trading import UNLIMITED, TradeParams

# Bounded because a bot is spawned synchronously inside a request: an
# unbounded ask would hang the seat.
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
    """One metadata integer, bounded. A missing or unreadable key takes the
    default rather than failing the load."""
    try:
        wanted = int(value) if value else default
    except ValueError:
        return default
    return max(0, min(wanted, ceiling))


def gate_config(meta: dict[str, str]) -> TradeParams:
    """The trade gate `meta` asks for.

    Every key is optional: absent `trade_floor` gives `0.0` (strict positivity
    is the whole gate), absent `gate_plies` gives `0`, and absent bargaining
    keys give `hexset.trading.UNLIMITED` -- no caps of the gate's own, nothing
    charged for a response, no offer budget asked for, no planning. A limit
    the model does not declare is a limit it does not have, so nothing bounds
    such a gate but the cards on the table, and a checkpoint that says nothing
    bargains exactly the way every checkpoint did before these keys existed.
    A checkpoint that wants the fragmented policy asks for it by name (`fragment_trades=1`), with the
    numbers it was trained or measured under. Unrecognised keys are ignored
    without complaint.
    """
    return TradeParams.from_meta(meta)


def gate_config_of(checkpoint: object) -> TradeParams:
    """The gate settings `checkpoint` declares, read by name rather than by
    inheritance.

    A loader carrying its own `trade_params` answers with it; one carrying
    only the two loose attributes is read through them; one carrying neither
    is read at the unmeasured defaults. A negative floor is refused by
    `hexset.trading.trade_floor_of`."""
    from hexset.trading import params_of

    return params_of(checkpoint, UNLIMITED)


def trader_config(meta: dict[str, str]) -> str | None:
    """The bot `meta` names to answer the checkpoint's trades, as a lineup
    names it (`hexset.arena.traded`), or `None` for its own gate. Resolved
    when the checkpoint is seated, so a name nothing can build fails there,
    loudly, rather than trading as something else."""
    return meta.get("trader") or None


def trader_of(checkpoint: object) -> str | None:
    """The trader `checkpoint` declares, read by name like `gate_config_of`:
    a loader that carries no `trader` trades through its own gate."""
    return getattr(checkpoint, "trader", None)


def search_config(meta: dict[str, str]) -> SearchConfig:
    """The search `meta` asks for, read only when the file actually asks to be
    searched, so a stale `simulations` cannot turn a policy into a search."""
    if meta.get("search", "none") != "mcts":
        return SearchConfig()
    return SearchConfig(
        simulations=_clamp(meta.get("simulations"), DEFAULT_SIMULATIONS, MAX_SIMULATIONS)
        or DEFAULT_SIMULATIONS,
        wave=_clamp(meta.get("wave"), DEFAULT_WAVE, MAX_WAVE) or DEFAULT_WAVE,
    )
