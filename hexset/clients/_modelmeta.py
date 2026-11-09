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

from hexset.holding import HoldModel
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
    `horizon`) or, at `None`, played to the game's end, and the opening
    searched in full under `full_setup`."""

    worlds: int = DEFAULT_WORLDS
    top: int = DEFAULT_TOP
    horizon: int | None = None
    #: Every opening root searched whatever `decisive` says (metadata
    #: `setup = full`, `PairedRollouts.full_setup`).
    full_setup: bool = False
    #: The mixed strategy's temperature over the searched candidates
    #: (metadata `mix`, `PairedRollouts.mix`); `0.0` plays the best score.
    mix: float = 0.0
    #: The exponent on the prior the mixed strategy is anchored to (metadata
    #: `anchor`, `PairedRollouts.anchor`): 1, the prior itself.
    anchor: float = 1.0
    #: Each searched decision seeded from what the seat can see (metadata
    #: `det`, `PairedRollouts.det`) and `salt` (metadata `salt`).
    det: bool = False
    salt: int = 0
    #: `(policy, encoder)`: the files that play every other seat of a playout
    #: (`hexset.clients.conditioned.Opponents`), named beside this one by
    #: metadata `opponents` and `opponent_encoder`; `None` for the file's own
    #: policy at every seat.
    opponents: tuple[str, str] | None = None


@dataclass(frozen=True)
class SearchConfig:
    """How a checkpoint asks to be played: one forward (the default), MCTS
    (`simulations > 0`) or paired rollouts (`rollout`)."""

    simulations: int = 0
    wave: int = DEFAULT_WAVE
    #: `(a, b)` for `hexset.mcts.calibrate_own`, or `None`: metadata
    #: `own_calibration = "a,b"`, read for every file. An MCTS file reads its
    #: own seat through it at every evaluated position and in its trade gate;
    #: any other file, in its trade gate alone (trades and robber deals), so
    #: a rollout's playouts and a one-forward policy choose moves as before.
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


def closer_config(meta: dict[str, str]) -> str | None:
    """The bot `meta` names to close the checkpoint's games, as a lineup
    names it (`hexset.arena.closed`, `hexset.bots.Closes`): a move it names
    is played in place of the checkpoint's own. `None` for none. Resolved
    when the checkpoint is spawned (`hexset.clients.netbot.declared_closer`)."""
    return meta.get("closer") or None


def closer_of(checkpoint: object) -> str | None:
    """The closer `checkpoint` declares, read by name like `trader_of`."""
    return getattr(checkpoint, "closer", None)


def finish_decisions_config(meta: dict[str, str]) -> bool:
    """Whether `meta` asks for its values to be read where its decisions are
    finished (`finish_decisions`, `hexset.clients.finishing.Finishing`): for
    a checkpoint whose win head was taught only at positions where a
    decision starts. Off unless the key reads as true."""
    return str(meta.get("finish_decisions", "")).strip().lower() in ("1", "true", "yes", "on")


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


def hold_config(meta: dict[str, str]) -> HoldModel | None:
    """The hold model `meta` declares for the other seats' held development
    cards (`hexset.holding.HoldModel`, written by `HoldModel.meta`):
    `hold_shape`, a positive number, and `hold_scale`, one chance in (0, 1]
    for each kind, comma separated. `None` when it declares neither, and its
    search worlds deal every unseen card alike. One without the other, or a
    value out of range, fails the load: a declared reading is never dropped
    quietly."""
    shape, scale = (meta.get("hold_shape") or "").strip(), (meta.get("hold_scale") or "").strip()
    if not shape and not scale:
        return None
    if not shape or not scale:
        raise ValueError("`hold_shape` and `hold_scale` declare one hold model: give both or neither")
    try:
        return HoldModel(shape=float(shape), scale=tuple(float(s) for s in scale.split(",")))
    except ValueError as error:
        raise ValueError(f"unreadable hold model ({shape!r}; {scale!r}): {error}") from None


def hold_of(checkpoint: object) -> HoldModel | None:
    """The hold model `checkpoint` declares, read by name like `trader_of`:
    a loader that carries none deals its worlds' unseen cards alike."""
    return getattr(checkpoint, "hold", None)


def _opponents(policy: str | None, encoder: str | None) -> tuple[str, str] | None:
    """`opponents` and `opponent_encoder`, both or neither: a file name each,
    beside the file that names them. One without the other fails the load:
    a model of the other seats with nothing to read them by, or the reverse."""
    policy, encoder = (policy or "").strip(), (encoder or "").strip()
    if not policy and not encoder:
        return None
    if not policy or not encoder:
        raise ValueError("`opponents` and `opponent_encoder` name a pair: give both or neither")
    for name in (policy, encoder):
        if "/" in name or name.startswith("."):
            raise ValueError(f"{name!r}: an opponent file is named beside the file, not by path")
    return policy, encoder


def search_config(meta: dict[str, str]) -> SearchConfig:
    """The search `meta` asks for, read only when the file actually asks to be
    searched, so a stale `simulations` cannot turn a policy into a search.

    `search = mcts` reads `simulations`, `wave` and `decisive`; `search =
    rollout` reads `worlds`, `top`, `decisive`, `horizon`, `setup`, `mix`,
    `anchor`, `det`, `salt` and the `opponents`/`opponent_encoder` pair. Every
    file reads `own_calibration` (`SearchConfig.own_calibration` says where
    it applies). Budgets (`simulations`, `wave`, `worlds`, `top`) are bounded
    and an unreadable one takes its default; a setting that changes what the
    seat does (`decisive`, `horizon`, `setup`, `mix`, `anchor`, `det`, `salt`,
    `own_calibration`) fails the
    load when it does not read. Any other `search` is no search."""
    method = meta.get("search", "none")
    calibration = _calibration(meta.get("own_calibration"))
    if method == "rollout":
        return SearchConfig(
            own_calibration=calibration,
            decisive=_decisive(meta.get("decisive")),
            rollout=RolloutConfig(
                worlds=_clamp(meta.get("worlds"), DEFAULT_WORLDS, MAX_WORLDS) or DEFAULT_WORLDS,
                # One candidate is no comparison: two is the least a rollout plays out.
                top=max(2, _clamp(meta.get("top"), DEFAULT_TOP, MAX_TOP) or DEFAULT_TOP),
                horizon=_horizon(meta.get("horizon")),
                full_setup=_full_setup(meta.get("setup")),
                mix=_mix(meta.get("mix")),
                anchor=_anchor(meta.get("anchor")),
                **_det(meta.get("det"), meta.get("salt")),
                opponents=_opponents(meta.get("opponents"), meta.get("opponent_encoder")),
            ),
        )
    if method != "mcts":
        return SearchConfig(own_calibration=calibration)
    return SearchConfig(
        simulations=_clamp(meta.get("simulations"), DEFAULT_SIMULATIONS, MAX_SIMULATIONS)
        or DEFAULT_SIMULATIONS,
        wave=_clamp(meta.get("wave"), DEFAULT_WAVE, MAX_WAVE) or DEFAULT_WAVE,
        own_calibration=calibration,
        decisive=_decisive(meta.get("decisive")),
    )


def _full_setup(value: str | None) -> bool:
    """Metadata `setup`: `full` to search every opening root, absent for the
    `decisive` skip everywhere."""
    if value is None or not value.strip():
        return False
    if value.strip() != "full":
        raise ValueError(f"setup {value!r} is not 'full'")
    return True


def _mix(value: str | None) -> float:
    """Metadata `mix`: a temperature of 0 or more in win chance
    (`hexset.rollout.mixed_strategy`), or absent for the greedy search."""
    if value is None or not value.strip():
        return 0.0
    try:
        mix = float(value)
    except ValueError:
        raise ValueError(f"mix {value!r} is not a number") from None
    if not (math.isfinite(mix) and mix >= 0.0):
        raise ValueError(f"mix {value!r} is not a temperature of 0 or more")
    return mix


def _anchor(value: str | None) -> float:
    """Metadata `anchor`: the exponent of 0 or more on the prior a mixed
    strategy is anchored to (`hexset.rollout.mixed_strategy`), or absent
    for 1, the prior itself."""
    if value is None or not value.strip():
        return 1.0
    try:
        anchor = float(value)
    except ValueError:
        raise ValueError(f"anchor {value!r} is not a number") from None
    if not (math.isfinite(anchor) and anchor >= 0.0):
        raise ValueError(f"anchor {value!r} is not an exponent of 0 or more")
    return anchor


def _det(det: str | None, salt: str | None) -> dict:
    """Metadata `det` (`1` on, `0` or absent off) and `salt` (a whole
    number, 0 or absent by default, read only with `det` on), as
    `RolloutConfig` fields."""
    det = (det or "").strip()
    if det not in ("", "0", "1"):
        raise ValueError(f"det {det!r} is not 0 or 1")
    salt = (salt or "").strip()
    if salt and not salt.isdigit():
        raise ValueError(f"salt {salt!r} is not a whole number")
    if salt and det != "1":
        raise ValueError("salt seeds a deterministic search: set det = 1 with it")
    return {"det": det == "1", "salt": int(salt) if salt else 0}


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
    two finite numbers with `b > 0` fails the load rather than playing on a
    guess: the order of positions has to survive the recalibration."""
    if value is None or not value.strip():
        return None
    try:
        a, b = (float(part) for part in value.split(","))
    except ValueError:
        raise ValueError(f"own_calibration {value!r} is not two numbers 'a,b'") from None
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
        raise ValueError(f"own_calibration {value!r} needs finite a and b > 0")
    return (a, b)
