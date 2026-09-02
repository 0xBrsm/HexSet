"""Who sits where, and what happens to a seat nobody claims.

Both of the things in here are gym concerns rather than rules, which is why
they live in the interface layer and not in `hexset.game`:

*The snake starts at the creator.* `hexset.game.start` always deals the setup
snake from seat 0 and, when setup ends, hands the first real turn to seat 0.
That is right for a duel harness, where seat 0 is always occupied. It is wrong
here: HexSet has no lobby any more, a game deals the instant somebody asks for
one, and the creator lands on a *random* seat (`api.Tables.create`). A game
has to be immediately playable by whoever just made it.

*An empty seat the snake reaches is waited out and then retired.* See
`api.Table._settle_locks` for the grace window that decides when. A locked
seat leaves the setup snake and every later turn rotation, permanently — a
claimed seat is never released, so a game's player count is fixed for good the
moment setup finishes.

Both are implemented as a **correction applied after each action**, because
`hexset.game` does not know about either. `settle` re-points the snake or the
turn rotation the moment `hexset.game` has moved it somewhere this table does
not want it. The locked set rides on the `Game` as a plain attribute so every
wire surface can read it, and `locked_of` tolerates its absence.

The known limit, and it is why `docs/engine-divergence-2026-09-02.md` files
change request R2 against dev-hexset: `hexset.game.imagine` does not carry the
attribute, so a bot searching forward from this position simulates turns for a
retired seat as though it still played. A retired seat holds nothing and
builds nothing, so those are wasted plies rather than wrong ones, but it is a
divergence, and it goes away the moment `locked` is a real `Game` field.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hexset.board.board import Board
from hexset.game import Game, Phase
from hexset.game import start as _start

SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


def start_at(
    board: Board,
    num_players: int,
    rng: random.Random | None = None,
    *,
    first: int = 0,
) -> Game:
    """`hexset.game.start`, with the setup snake rotated to begin at `first`.

    The queue `hexset.game.start` builds is `order + reversed(order)` for
    `order = range(num_players)`; rotating `order` to start at `first` and
    rebuilding keeps the snake property (whoever places first in round one
    places last in round two) whichever seat it starts from.
    """
    game = _start(board, num_players, rng)
    order = [(first + i) % num_players for i in range(num_players)]
    game.setup_queue = order + order[::-1]
    game.setup_step = 0
    game.current_player = game.setup_queue[0]
    game.locked = frozenset()  # type: ignore[attr-defined]
    return game


def locked_of(game: Game) -> frozenset[int]:
    """The retired seats. Empty for any `Game` this module did not deal —
    `imagine`'s copies included, which is change request R2."""
    return getattr(game, "locked", frozenset())


@dataclass(frozen=True)
class Before:
    """Where the game stood immediately before an action was applied — the
    three fields `settle` has to compare against to tell which rotation
    `hexset.game` just performed."""

    phase: Phase
    setup_step: int
    current_player: int


def snapshot(game: Game) -> Before:
    return Before(
        phase=game.phase, setup_step=game.setup_step, current_player=game.current_player
    )


def settle(game: Game, before: Before) -> None:
    """Correct the seat `hexset.game` just handed the move to.

    Three cases, and no fourth — they are the only points at which
    `hexset.game` chooses a next seat:

    1. The setup snake advanced (`place_initial_road`): re-point it, skipping
       retired seats.
    2. Setup just ended: `hexset.game` hands the first turn to seat 0; the
       rule is that whoever placed *first* takes it, which here is
       `setup_queue[0]` and never a retired seat.
    3. A turn ended (`end_turn`, `(p + 1) % n`): skip retired seats.
    """
    if game.phase in SETUP_PHASES:
        if game.setup_step != before.setup_step:
            advance_setup(game)
        return
    if before.phase in SETUP_PHASES:
        game.current_player = game.setup_queue[0]
        return
    if game.current_player in locked_of(game):
        game.current_player = next_unlocked(game, before.current_player)


def advance_setup(game: Game) -> None:
    """Point the snake at the next entry that isn't a retired seat, or end
    setup. Skipping advances `setup_step` past retired entries rather than
    keeping a separate "seats placed" counter — the queue keeps all
    `2 * num_players` slots, which is what keeps `hexset.game`'s own
    `setup_step >= num_players` second-round test correct however many seats
    have retired."""
    queue = game.setup_queue
    locked = locked_of(game)
    while game.setup_step < len(queue) and queue[game.setup_step] in locked:
        game.setup_step += 1
    if game.setup_step < len(queue):
        game.current_player = queue[game.setup_step]
        game.phase = Phase.SETUP_SETTLEMENT
    else:
        game.current_player = queue[0]
        game.phase = Phase.ROLL


def next_unlocked(game: Game, after: int) -> int:
    """The next seat past `after`, skipping retired ones. Always terminates:
    `setup_queue[0]` — the creator's seat — is never retired, it was occupied
    the instant the game was created."""
    n = game.state.num_players
    locked = locked_of(game)
    for step in range(1, n + 1):
        seat = (after + step) % n
        if seat not in locked:
            return seat
    raise AssertionError("every seat is retired")  # the creator's never is


def lock_seat(game: Game, seat: int) -> None:
    """Retire an empty seat the setup snake reached and waited out. A no-op if
    `seat` is already retired — the caller does not have to track that."""
    if seat in locked_of(game):
        return
    game.locked = locked_of(game) | {seat}  # type: ignore[attr-defined]
    if game.phase in SETUP_PHASES and game.current_player == seat:
        game.setup_step += 1
        advance_setup(game)
