# SPDX-License-Identifier: GPL-3.0-only
"""What our bot actually does in a game, in the shape human aggregates are quoted in.

Strength is one question and behaviour is another. A bot can win at the right
rate while never buying a development card, and for the half of this project
that wants an opponent worth playing against, that would be a failure nobody's
win-rate table would catch.

So this walks recorded games and asks the same questions a published summary of
Colonist.io games answers: how often is each card played, when in the game, and
what does a player who played it go on to do. That summary is third-party and
unverifiable — see `benchmarks/behaviour.py` for what it is and what it is not —
so its figures are targets to notice divergence from, not ground truth to fit to.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .actions import ActionType, apply
from .game import start, to_move
from .record import Record, actions_of, board_of

# The card plays worth counting, and the names they are reported under.
TRACKED: dict[ActionType, str] = {
    ActionType.PLAY_KNIGHT: "knight",
    ActionType.PLAY_MONOPOLY: "monopoly",
    ActionType.PLAY_ROAD_BUILDING: "road_building",
    ActionType.PLAY_YEAR_OF_PLENTY: "year_of_plenty",
    ActionType.BUY_DEV_CARD: "bought",
    ActionType.BUILD_SETTLEMENT: "settlement",
    ActionType.BUILD_CITY: "city",
    ActionType.BUILD_ROAD: "road",
    ActionType.BANK_TRADE: "bank_trade",
}


@dataclass(frozen=True)
class Play:
    seat: int
    kind: str
    progress: float


@dataclass(frozen=True)
class Walk:
    """One replayed game: who did what, when, and who won."""

    game: int
    winner: int
    players: int
    turns: int
    plays: tuple[Play, ...]

    def count(self, seat: int, kind: str) -> int:
        return sum(1 for p in self.plays if p.seat == seat and p.kind == kind)


def walk(record: Record, game: int) -> Walk | None:
    """Replay a record, noting each tracked action and who took it.

    Legality is not rechecked — `hexset.record.replay` is what verifies a record.
    This only needs the sequence of who did what.
    """
    if record.winner is None:
        return None

    live = start(board_of(record), record.num_players, random.Random(record.seed))
    total = max(1, len(record.actions))
    plays: list[Play] = []

    for step, action in enumerate(actions_of(record)):
        name = TRACKED.get(action.type)
        if name is not None:
            plays.append(Play(to_move(live), name, step / total))
        apply(live, action)

    return Walk(
        game=game,
        winner=record.winner,
        players=record.num_players,
        turns=record.turns,
        plays=tuple(plays),
    )


def walks(records: Iterable[Record]) -> list[Walk]:
    out = []
    for game, record in enumerate(records):
        found = walk(record, game)
        if found is not None:
            out.append(found)
    return out


def per_game(walked: Sequence[Walk], kind: str) -> float:
    """Mean number of times `kind` happens per player-game."""
    if not walked:
        return 0.0
    seats = sum(w.players for w in walked)
    total = sum(1 for w in walked for p in w.plays if p.kind == kind)
    return total / seats


def by_count(walked: Sequence[Walk], kind: str, cap: int = 4) -> dict[int, tuple[int, int]]:
    """Win record grouped by how many times a seat did `kind`.

    Counts at or above `cap` are pooled, since the tail is thin and the
    published human tables pool it the same way.
    """
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for w in walked:
        for seat in range(w.players):
            bucket = min(w.count(seat, kind), cap)
            grouped[bucket][0] += 1
            grouped[bucket][1] += int(seat == w.winner)
    return {k: (v[0], v[1]) for k, v in sorted(grouped.items())}


def by_timing(
    walked: Sequence[Walk], kind: str, buckets: int = 5
) -> dict[int, tuple[int, int]]:
    """Win record grouped by which fifth of the game the first `kind` fell in.

    The first one rather than all of them, so a seat is counted once and the
    groups stay disjoint.
    """
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for w in walked:
        first: dict[int, float] = {}
        for p in w.plays:
            if p.kind == kind and p.seat not in first:
                first[p.seat] = p.progress
        for seat, progress in first.items():
            bucket = min(int(progress * buckets), buckets - 1)
            grouped[bucket][0] += 1
            grouped[bucket][1] += int(seat == w.winner)
    return {k: (v[0], v[1]) for k, v in sorted(grouped.items())}


def seat_win_rates(walked: Sequence[Walk]) -> dict[int, tuple[int, int]]:
    """Wins by seat. Records are not seat-rotated, so this reads the draft directly."""
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for w in walked:
        for seat in range(w.players):
            grouped[seat][0] += 1
            grouped[seat][1] += int(seat == w.winner)
    return {k: (v[0], v[1]) for k, v in sorted(grouped.items())}
