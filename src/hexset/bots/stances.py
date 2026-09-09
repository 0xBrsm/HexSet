# SPDX-License-Identifier: GPL-3.0-only
"""Convert per-seat evaluation vectors to a player's search objective."""
from __future__ import annotations

import math
from typing import Sequence

def own(vector: Sequence[float], seat: int) -> float:
    """Plain max^n: each seat wants its own score high and ignores the rest."""
    return vector[seat]


def relative(vector: Sequence[float], seat: int) -> float:
    """Own score less the average of everyone else's.

    A constant-sum reading of the vector. Catan has exactly one winner, so a
    position is only worth what it is worth *compared to* the table, and an
    action that lifts everyone equally has achieved nothing.
    """
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - sum(others) / len(others)


def paranoid(vector: Sequence[float], seat: int) -> float:
    """Own score less the best opponent's. The leader is the only rival."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - max(others)


# Fitted by maximum likelihood over 4,463 per-turn score-vector samples from
# 48 archived four-player games (stance `relative`, the shipped mechanic at fit time),
# one sample per turn at the mover's first decision, labelling the eventual
# winner, seed 40000: mean log loss 1.107 against the eventual winner, vs.
# 1.386 for the uniform baseline. `agents/reference/heximax.md`, "Registration
# 2026-09-04: the objective — a win-probability stance against the
# relative-VP stance".
WIN_TEMPERATURE = 2.476644394795811


def win_at(vector: Sequence[float], seat: int, temperature: float) -> float:
    """Softmax(vector / temperature)[seat]: the seat's win probability at a
    given exchange rate between the vector's units and win log-odds.
    Numerically stable: the max is subtracted before exponentiating."""
    scaled = [v / temperature for v in vector]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    return exps[seat] / sum(exps)


def win(vector: Sequence[float], seat: int) -> float:
    """Softmax(vector / WIN_TEMPERATURE)[seat]: the seat's win probability.

    This reads the vector as the seat's win probability, which is what the
    game actually pays, rather than a margin over the table — at a
    temperature fitted by maximum likelihood against real game outcomes (see
    `WIN_TEMPERATURE`). 
    """
    return win_at(vector, seat, WIN_TEMPERATURE)


# How a seat turns the per-seat vector into the one number it maximises. The
# evaluation is unchanged; only the reading of it differs.
STANCES = {"own": own, "relative": relative, "paranoid": paranoid, "win": win}

