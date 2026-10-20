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

import math

from dataclasses import dataclass

from hexset.trading import UNLIMITED, TradeParams

# Bounded because a bot is spawned synchronously inside a request: an
# unbounded ask would hang the seat.
MAX_SIMULATIONS = 4096
MAX_WAVE = 256
MAX_WORLDS = 256
MAX_TOP = 16

DEFAULT_SIMULATIONS = 128
DEFAULT_WAVE = 16
# `hexset.rollout.PairedRollouts`'s own defaults.
DEFAULT_WORLDS = 32
DEFAULT_TOP = 3


@dataclass(frozen=True)
class RolloutConfig:
    """Paired rollouts of the checkpoint's own policy
    (`hexset.rollout.PairedRollouts`), as metadata `search = rollout` asks
    for them: `worlds` shared-chance worlds, the policy's `top` moves by
    prior as candidates, each playout cut `horizon` turns on (metadata
    `horizon`) or, at `None`, played to the game's end."""

    worlds: int = DEFAULT_WORLDS
    top: int = DEFAULT_TOP
    horizon: int | None = None


@dataclass(frozen=True)
class SearchConfig:
    """How a checkpoint asks to be played: one forward (the default), MCTS
    (`simulations > 0`) or paired rollouts (`rollout`)."""

    simulations: int = 0
    wave: int = DEFAULT_WAVE
    #: `(a, b)` for `hexset.mcts.calibrate_own`, or `None`: metadata
    #: `own_calibration = "a,b"`, read only for an MCTS file.
    own_calibration: tuple[float, float] | None = None
    #: The prior at which a root is played without searching
    #: (`hexset.mcts.Search`, `hexset.rollout.PairedRollouts`), or `None`:
    #: metadata `decisive`, read only for a searched file.
    decisive: float | None = None
    #: The rollouts a `search = rollout` file asks for, `None` otherwise.
    rollout: RolloutConfig | None = None

    @property
    def searches(self) -> bool:
        return self.simulations > 0 or self.rollout is not None


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
    when the checkpoint is spawned (`hexset.clients.netbot.declared_trader`),
    so a name nothing can build fails there, naming the file, rather than
    trading as something else."""
    return meta.get("trader") or None


def trader_of(checkpoint: object) -> str | None:
    """The trader `checkpoint` declares, read by name like `gate_config_of`:
    a loader that carries no `trader` trades through its own gate."""
    return getattr(checkpoint, "trader", None)


def trader_veto_config(meta: dict[str, str]) -> bool:
    """Whether `meta` asks for its own gate to veto its trader
    (`trader_veto`, `hexset.arena.traded`): each exchange the trader would
    make must also clear the checkpoint's own value gain above its declared
    `trade_floor`. Off unless the key reads as true."""
    return str(meta.get("trader_veto", "")).strip().lower() in ("1", "true", "yes", "on")


def trader_veto_of(checkpoint: object) -> bool:
    """Whether `checkpoint` vetoes its trader, read by name like `trader_of`."""
    return bool(getattr(checkpoint, "trader_veto", False))


def trader_veto_pick_config(meta: dict[str, str]) -> bool:
    """Whether `meta` asks its veto to choose which taker of its own offer
    to deal with (`trader_veto_pick = gain`, `hexset.bots.Vetoed` with
    ``picks``): of the answers its trader would take, the one its own value
    gains most from. Otherwise, and without a veto, the trader picks."""
    return str(meta.get("trader_veto_pick", "")).strip().lower() == "gain"


def trader_veto_pick_of(checkpoint: object) -> bool:
    """Whether `checkpoint`'s veto picks the taker, read by name like `trader_of`."""
    return bool(getattr(checkpoint, "trader_veto_pick", False))
def robber_tip_config(meta: dict[str, str]) -> float:
    """How far `meta` asks its robber moves to lean towards the leader
    (`robber_tip`, `hexset.clients.netbot.tip_robber`): the fraction of the
    chosen move's prior that robbing the leader must reach to be played in its
    place. `0.0`, off, when the key is absent or unreadable; held to [0, 1]."""
    try:
        tip = float(meta.get("robber_tip") or 0.0)
    except ValueError:
        return 0.0
    return min(max(tip, 0.0), 1.0) if math.isfinite(tip) else 0.0


def robber_tip_of(checkpoint: object) -> float:
    """The robber tip `checkpoint` declares, read by name like `trader_of`:
    a loader that carries none plays its robber moves untouched."""
    return float(getattr(checkpoint, "robber_tip", 0.0))


def search_config(meta: dict[str, str]) -> SearchConfig:
    """The search `meta` asks for, read only when the file actually asks to be
    searched, so a stale `simulations` cannot turn a policy into a search.

    `search = mcts` reads `simulations`, `wave`, `own_calibration` and
    `decisive`; `search = rollout` reads `worlds`, `top`, `decisive` and
    `horizon`. Budgets (`simulations`, `wave`, `worlds`, `top`) are bounded
    and an unreadable one takes its default; a setting that changes what the
    search is (`decisive`, `horizon`, `own_calibration`) fails the load
    when it does not read. Any other `search` is no search."""
    method = meta.get("search", "none")
    if method == "rollout":
        if meta.get("own_calibration", "").strip():
            raise ValueError("own_calibration recalibrates an MCTS search; a rollout file cannot ask for it")
        return SearchConfig(
            decisive=_decisive(meta.get("decisive")),
            rollout=RolloutConfig(
                worlds=_clamp(meta.get("worlds"), DEFAULT_WORLDS, MAX_WORLDS) or DEFAULT_WORLDS,
                # One candidate is no comparison: two is the least a rollout plays out.
                top=max(2, _clamp(meta.get("top"), DEFAULT_TOP, MAX_TOP) or DEFAULT_TOP),
                horizon=_horizon(meta.get("horizon")),
            ),
        )
    if method != "mcts":
        return SearchConfig()
    return SearchConfig(
        simulations=_clamp(meta.get("simulations"), DEFAULT_SIMULATIONS, MAX_SIMULATIONS)
        or DEFAULT_SIMULATIONS,
        wave=_clamp(meta.get("wave"), DEFAULT_WAVE, MAX_WAVE) or DEFAULT_WAVE,
        own_calibration=_calibration(meta.get("own_calibration")),
        decisive=_decisive(meta.get("decisive")),
    )


def _horizon(value: str | None) -> int | None:
    """Metadata `horizon`: whole turns, 1 or more, or absent for playouts
    to the game's end."""
    if value is None or not value.strip():
        return None
    try:
        horizon = int(value)
    except ValueError:
        raise ValueError(f"horizon {value!r} is not a whole number of turns") from None
    if horizon < 1:
        raise ValueError(f"horizon {value!r} is not 1 turn or more")
    return horizon


def _decisive(value: str | None) -> float | None:
    """Metadata `decisive`: a prior in (0, 1], or absent."""
    if value is None or value == "":
        return None
    try:
        decisive = float(value)
    except ValueError:
        raise ValueError(f"decisive {value!r} is not a number") from None
    if not 0.0 < decisive <= 1.0:
        raise ValueError(f"decisive {value!r} is not a prior in (0, 1]")
    return decisive


def _calibration(value: str | None) -> tuple[float, float] | None:
    """`"a,b"` as `(a, b)`; absent is `None`. A value that does not read as
    two finite numbers with `b > 0` fails the load rather than searching on
    a guess: the order of positions has to survive the recalibration."""
    if value is None or not value.strip():
        return None
    try:
        a, b = (float(part) for part in value.split(","))
    except ValueError:
        raise ValueError(f"own_calibration {value!r} is not two numbers 'a,b'") from None
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
        raise ValueError(f"own_calibration {value!r} needs finite a and b > 0")
    return (a, b)
