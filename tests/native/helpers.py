# SPDX-License-Identifier: GPL-3.0-only
"""Typed full-state comparisons for native rule tests."""
import dataclasses
import enum
import json
import numpy as np

def canonical(value):
    """Typed values; no pickle aliasing or tuple-keyed-dictionary ambiguity."""
    if isinstance(value, enum.Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, canonical(value.value)]
    if dataclasses.is_dataclass(value):
        return ["dataclass", type(value).__module__, type(value).__qualname__,
                [[f.name, canonical(getattr(value, f.name))] for f in dataclasses.fields(value)]]
    if isinstance(value, np.ndarray):
        return ["ndarray", value.dtype.str, list(value.shape), value.tobytes(order="C").hex()]
    if isinstance(value, dict):
        pairs = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ["dict", sorted(pairs, key=lambda pair: json.dumps(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [canonical(v) for v in value]]
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted([canonical(v) for v in value], key=json.dumps)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if value is None or type(value) in (int, bool, str):
        return [type(value).__name__, value]
    raise TypeError(f"unsupported debug value: {type(value)!r}")


def canonical_bytes(value) -> bytes:
    return json.dumps(canonical(value), separators=(",", ":"), ensure_ascii=True).encode()


def debug_state(game) -> bytes:
    """Full referee state for equality only: never a policy input.

    Explicitly omit callable gates and RNG implementations, retain every
    rule/turn/robber/trade setting and all common AND private ledger rows.
    Recorded chance outcomes are checked, rather than interpreter RNG storage.
    """
    excluded = {"_state", "ledger", "rng", "chance", "gates"}
    control = {f.name: getattr(game, f.name) for f in dataclasses.fields(game)
               if f.name not in excluded}
    if not hasattr(game.chance, "events"):
        raise TypeError("debug equality requires the existing Recording chance source")
    return canonical_bytes((game._state, game.ledger, control, game.chance.events))
