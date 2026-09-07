# SPDX-License-Identifier: GPL-3.0-only
"""Turn recorded games into labelled choice sets.

A recorded game is replayed, stopped every `stride` actions at a main-phase
position, and read the way the bot reads it: from one seat's information, one
feature row per seat (`HonestEvaluator.rows_game`). The label is the seat
that went on to win. That is a discrete choice with four alternatives sharing
one coefficient vector -- a conditional logit (`hexset.fitting`) -- and the
coefficients *are* the evaluation weights and the temperature, so the fit
drops straight back into the bot.

Honest by default: the knower's row is exact and every other row is read
through the ledger (`View.expected_hand`, `expected_card_points`), exactly as
the deployed evaluator reads them. `omniscient=True` reads every row exact,
which measures how much the honest reading attenuates the fit.

Positions inside a game are heavily correlated -- consecutive samples differ
by one build, and every choice set in a game shares one winner -- so a
choice set carries its game and anything that splits or resamples the data
does so by game, never by position.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .bots.heximax.evaluate import HonestEvaluator
from .game import Phase, is_over, to_move
from .record import Record, advance, board_of, open_record, steps

DEFAULT_STRIDE = 8


@dataclass(frozen=True)
class ChoiceSet:
    """One position read from one seat: a feature row per seat, and who won.

    `rows` are `HonestEvaluator.terms` in `evaluate.TERM_NAMES` order, in
    seat order; `winner` indexes them. `mover` is the seat to move at the
    position, `knower` the seat whose information the rows are read from.
    `progress` is how far through the game (by action) the position sits.
    """

    rows: tuple[tuple[float, ...], ...]
    winner: int
    game: int
    knower: int
    mover: int
    turn: int
    progress: float


def samples_from(
    record: Record,
    game: int,
    *,
    stride: int = DEFAULT_STRIDE,
    omniscient: bool = False,
    knowers: Sequence[int] | None = None,
) -> Iterator[ChoiceSet]:
    """Replay one record, reading every `stride`-th main-phase position.

    One choice set per knower at each sampled position (every seat by
    default, `knowers` to restrict). Undecided games yield nothing: with no
    winner there is no label.
    """
    if record.winner is None:
        return

    evaluator = HonestEvaluator(board_of(record), omniscient=omniscient)
    state_game = open_record(record)
    total = len(record.actions)
    seats = range(record.num_players) if knowers is None else knowers

    for step, (action, trades) in enumerate(steps(record)):
        if (
            step % stride == 0
            and state_game.phase is Phase.MAIN
            and not is_over(state_game)
        ):
            mover = to_move(state_game)
            for knower in seats:
                yield ChoiceSet(
                    rows=tuple(evaluator.rows_game(state_game, knower)),
                    winner=record.winner,
                    game=game,
                    knower=knower,
                    mover=mover,
                    turn=state_game.turns,
                    progress=step / total,
                )
        advance(state_game, action, trades)


def build(
    records: Iterable[Record],
    *,
    stride: int = DEFAULT_STRIDE,
    omniscient: bool = False,
) -> list[ChoiceSet]:
    out: list[ChoiceSet] = []
    for game, record in enumerate(records):
        out.extend(samples_from(record, game, stride=stride, omniscient=omniscient))
    return out


def split_by_game(
    samples: Sequence[ChoiceSet], *, holdout: float = 0.2, seed: int = 0
) -> tuple[list[ChoiceSet], list[ChoiceSet]]:
    """Split train/test by game.

    Splitting by position would put two nearly identical positions from the
    same game on both sides and report a test score that means nothing.
    """
    games = sorted({s.game for s in samples})
    rng = random.Random(seed)
    rng.shuffle(games)
    cut = int(len(games) * (1.0 - holdout))
    train_games = set(games[:cut])
    train = [s for s in samples if s.game in train_games]
    test = [s for s in samples if s.game not in train_games]
    return train, test
