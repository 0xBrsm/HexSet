"""Who sits where, before anyone moves.

Turn order is seat order, and nobody moves while any seat is still empty
(`api.Table.waiting_for`). A seat retires when `POST /api/close` closes it or
`POST /api/leave` gives it up, through the engine's own `hexset.game.lock_seat`,
which skips it in the setup snake and in every turn rotation from then on; a
claimed seat is never released.

The engine has no way back from `lock_seat`, because a game in progress has
none. Before the first move a table's seats are still its own to change, so a
closed seat may reopen: `unlock_seat`.
"""

from __future__ import annotations

from hexset.game import Game, Phase

SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


def advance_setup(game: Game) -> None:
    """Point the snake at the next entry that isn't retired, or end setup.
    Skipping advances `setup_step` past retired entries, so the queue keeps all
    `2 * num_players` slots and `hexset.game`'s own `setup_step >=
    num_players` second-round test stays correct."""
    queue = game.setup_queue
    locked = game.locked
    while game.setup_step < len(queue) and queue[game.setup_step] in locked:
        game.setup_step += 1
    if game.setup_step < len(queue):
        game.current_player = queue[game.setup_step]
        game.phase = Phase.SETUP_SETTLEMENT
    else:
        game.current_player = first_unlocked(game)
        game.phase = Phase.ROLL


def first_unlocked(game: Game) -> int:
    """The first entry of the setup snake that is not retired."""
    locked = game.locked
    return next(seat for seat in game.setup_queue if seat not in locked)


def unlock_seat(game: Game, seat: int) -> None:
    """Reopen a retired `seat`; a no-op if it is not retired. Valid only
    before the first move (`api.Tables.open_seat` and `seat_bot` enforce
    that). The snake is then pointed back at its first entry still playing,
    which may be `seat` itself: nothing has been placed, so the only progress
    the snake has made is past the retired entries it skipped."""
    if seat not in game.locked:
        return
    game.locked = game.locked - {seat}
    queue = game.setup_queue
    step = 0
    # Terminates on `seat` at the latest, now unlocked.
    while queue[step] in game.locked:
        step += 1
    game.setup_step = step
    game.current_player = queue[step]
    game.phase = Phase.SETUP_SETTLEMENT
