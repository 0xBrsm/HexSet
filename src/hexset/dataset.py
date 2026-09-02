# SPDX-License-Identifier: GPL-3.0-only
"""Turn recorded games into labelled positions.

This is where the labels the hill climb never had come from. Replay a record,
stop every so often, and ask of each seat: what did the evaluation's terms look
like here, and did this seat go on to win? That is a supervised learning problem
with a differentiable loss, which the win-rate objective is not.

One sample per seat rather than per position, because the evaluation scores
seats, and a seat's own hidden cards are legitimately its own knowledge.

Positions inside a game are heavily correlated — consecutive turns differ by one
build. So samples carry the game they came from and splits have to be made by
game, never by position, or the test set is really the training set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .actions import apply
from .evaluate import Evaluator, Weights
from .game import is_over, start
from .record import Record, actions_of, board_of

DEFAULT_STRIDE = 8


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    won: int
    game: int
    seat: int
    progress: float


def samples_from(
    record: Record,
    game: int,
    *,
    stride: int = DEFAULT_STRIDE,
    weights: Weights | None = None,
) -> Iterator[Sample]:
    """Replay one record, featurising every `stride` actions.

    Undecided games yield nothing: with no winner there is no label.
    """
    if record.winner is None:
        return

    evaluator = Evaluator(board_of(record), weights)
    state_game = start(board_of(record), record.num_players, random.Random(record.seed))
    total = len(record.actions)

    for step, action in enumerate(actions_of(record)):
        if step % stride == 0 and not is_over(state_game):
            fraction = step / total
            for seat in range(record.num_players):
                yield Sample(
                    features=evaluator.terms(
                        state_game.state, seat, knower=seat
                    ),
                    won=1 if seat == record.winner else 0,
                    game=game,
                    seat=seat,
                    progress=fraction,
                )
        apply(state_game, action)


def build(
    records: Iterable[Record],
    *,
    stride: int = DEFAULT_STRIDE,
    weights: Weights | None = None,
) -> list[Sample]:
    out: list[Sample] = []
    for game, record in enumerate(records):
        out.extend(samples_from(record, game, stride=stride, weights=weights))
    return out


def split_by_game(
    samples: Sequence[Sample], *, holdout: float = 0.2, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Split train/test by game.

    Splitting by position would put two nearly identical positions from the same
    game on both sides and report a test score that means nothing.
    """
    games = sorted({s.game for s in samples})
    rng = random.Random(seed)
    rng.shuffle(games)
    cut = int(len(games) * (1.0 - holdout))
    train_games = set(games[:cut])
    train = [s for s in samples if s.game in train_games]
    test = [s for s in samples if s.game not in train_games]
    return train, test


def base_rate(samples: Sequence[Sample]) -> float:
    """Fraction of samples that won. The accuracy any model has to beat."""
    return sum(s.won for s in samples) / len(samples) if samples else 0.0
