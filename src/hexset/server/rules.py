"""The one legality authority every seat at a HexSet table shares.

This module used to stand between the engine's own enumeration and a served
table, because `hexset.actions.legal_actions`' `PROPOSE_TRADE` sample was
*omniscient*: it read every opponent's hand to skip a `want` nobody could
cover. That is a fair thing inside a duel harness, where the engine and the
players are one process, and a hand-composition leak at a table with a
person at it.

**There is nothing left to stand between.** Trading is no longer an action
(`hexset.trading`): a seat publishes a valuation vector, the engine clears
the deals, and no remaining action's legality depends on anybody else's
hand. So `legal_actions` *is* the honest mask, for every seat, and
`fair_legal_actions` is gone rather than reduced to an alias — one list, the
engine's, with no second sample to drift from it.

What is left here is the two things a *served* table wants that a duel
harness does not: an empty option list is a bug worth naming (`options_for`),
and a submitted action has to be checked against the list rather than
trusted (`is_legal`).
"""

from __future__ import annotations

from typing import Sequence

from hexset.actions import Action, legal_actions
from hexset.game import Game


class Stuck(RuntimeError):
    """Raised when a live game offers no legal action, which is always a bug."""


def options_for(game: Game) -> list[Action]:
    """`legal_actions`, for a caller that has no answer for an empty list.

    A bot on the move must be able to move. Every phase that can be reached
    offers something, so an empty list is an engine bug, and a bot that
    returned `None` here would only push the crash somewhere less obvious.
    """
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {game.current_player}")
    return options


def is_legal(game: Game, action: Action, options: Sequence[Action]) -> bool:
    """Whether `action` is one of `options` (normally `legal_actions(game)`).

    Plain membership: every action in the space is now fully enumerated and
    exactly decodable from its index, so there is no longer a case (the old
    uncapped offer language) where a legal move could sit outside the list a
    client was shown.
    """
    del game
    return action in options
