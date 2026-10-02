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
