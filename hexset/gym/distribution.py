# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic per-game rule and board draws for batched environments.

A specification has weighted fixed ``presets`` and a ``random`` component,
whose fields are independent weighted categorical draws. ``exclude`` holds
partial field mappings to reject from the random component, for held-out
combinations. Weights are relative within each field; component ``weight``s
are relative across presets and the random component. Nothing reads a global
random stream: a draw depends only on the seed and game index.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from itertools import product

from ..board.board import BOARD_MODES
from ..rules import GameType, Rules, STANDARD_GAME

__all__ = ["Deal", "Distribution"]


@dataclass(frozen=True)
class Deal:
    """One game's format, retired seats and board generator."""

    game_type: GameType = STANDARD_GAME
    locked: frozenset[int] = frozenset()
    board_mode: str = "random"


FIELDS = ("players", "winning_points", "discard_limit", "friendly_robber", "balanced_dice", "board_mode")


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"a draw weight must be finite and positive: {value!r}")
    return float(value)


def _value(name, value, players):
    if name in ("players", "winning_points", "discard_limit"):
        lower, upper = (2, players) if name == "players" else (1, None) if name == "winning_points" else (0, None)
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError(f"invalid {name}: {value!r}")
    elif name in ("friendly_robber", "balanced_dice"):
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    elif name == "board_mode" and value not in BOARD_MODES:
        raise ValueError(f"unknown board mode {value!r}")
    return value


class Distribution:
    """A validated JSON specification, callable as ``law(game_index)``.

    ``players`` is the number of dealt slots. A draw with fewer live players
    retires a uniform subset, keeping tensor shapes fixed across games.
    Specifications are copied at construction, so later edits cannot change
    a run's law. Only the random component is conditioned on ``exclude``.
    """

    def __init__(self, specification: str | dict, players: int, seed: int):
        spec = json.loads(specification) if isinstance(specification, str) else json.loads(json.dumps(specification))
        if set(spec) - {"presets", "random", "exclude"}:
            raise ValueError("unknown distribution fields")
        self.players, self.seed = players, seed
        self.components = []
        for preset in spec.get("presets", []):
            if set(preset) != {*FIELDS, "weight"}:
                raise ValueError("a preset needs weight and every format field")
            values = tuple(_value(n, preset[n], players) for n in FIELDS)
            self.components.append((_weight(preset["weight"]), (values,), (1.0,)))
        excludes = spec.get("exclude", [])
        for excluded in excludes:
            if not excluded or set(excluded) - set(FIELDS):
                raise ValueError("an exclusion needs known format fields")
            for name, value in excluded.items():
                _value(name, value, players)
        varied = spec.get("random")
        if varied is not None:
            if set(varied) != {*FIELDS, "weight"}:
                raise ValueError("a random component needs weight and every format field")
            choices = []
            for name in FIELDS:
                choices.append([(_value(name, v, players), _weight(w)) for v, w in varied[name]])
                if not choices[-1]:
                    raise ValueError(f"empty draw for {name}")
            values, weights = [], []
            for cell in product(*choices):
                value = tuple(v for v, _ in cell)
                row = dict(zip(FIELDS, value))
                if any(all(row[n] == v for n, v in ex.items()) for ex in excludes):
                    continue
                values.append(value)
                weights.append(math.prod(w for _, w in cell))
            if not values:
                raise ValueError("exclusions remove every random format")
            self.components.append((_weight(varied["weight"]), tuple(values), tuple(weights)))
        if not self.components:
            raise ValueError("a distribution needs at least one component")
        # Compile once. Per-game draws do two binary searches, regardless of
        # how many combinations the independent fields define.
        from itertools import accumulate
        self.cumulative = tuple(accumulate(w for w, _, _ in self.components))
        self.cells = tuple((vs, tuple(accumulate(ws))) for _, vs, ws in self.components)

    def __call__(self, index: int) -> Deal:
        from bisect import bisect_right
        rng = random.Random(f"{self.seed}:{index}:rules")
        component = bisect_right(self.cumulative, rng.random() * self.cumulative[-1])
        values, cumulative = self.cells[component]
        value = values[bisect_right(cumulative, rng.random() * cumulative[-1])]
        row = dict(zip(FIELDS, value))
        live = row.pop("players")
        board_mode = row.pop("board_mode")
        locked = frozenset(rng.sample(range(self.players), self.players - live))
        return Deal(GameType("custom", Rules(**row), (live,)), locked, board_mode)
