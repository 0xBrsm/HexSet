"""The one legality authority every seat at a HexSet table shares.

`hexset.actions.legal_actions` is the engine's own enumeration, and its
`PROPOSE_TRADE` sample is *omniscient*: it reads every opponent's hand to skip
a `want` nobody could cover (`hexset.actions._offer_actions`). That is a fair
thing for a duel harness, where the engine and the players are one process and
nobody is being told anything. It is not fair here, where a hand is private,
and this module is what stands between the two.

Everything that decides what a seat may play goes through
`fair_legal_actions`: the browser's own action list, `GET /api/record`'s
`options`/`action_mask`/`pair_mask`, an external bot's list, and — since this
branch — the embedded bots too (`onnxbot`, `api.spawn_bot`). PR #2 had the
embedded bots on the engine's omniscient sample and everyone else on the
honest one, which meant a checkpoint played a measurably different game
depending on how it had been seated; one mask, the honest one, for every seat
(`docs/engine-divergence-2026-09-02.md`, defect 4).

`is_legal` is the other half: what a client is *allowed* to submit, which is
deliberately wider than either sample. See its own docstring.
"""

from __future__ import annotations

from typing import Sequence

from hexset.actions import ONE_RESOURCE, Action, ActionType, legal_actions
from hexset.board.terrain import NUM_RESOURCES
from hexset.game import MAX_OFFERS_PER_TURN, Game, Phase
from hexset.trading import Offer, can_propose


def proposable_options(game: Game) -> list[Action]:
    """Every one-for-one (give, want) pair the current player could open a
    trade proposal for — from public information alone: their own hand and
    the turn's offer count.

    Deliberately not `hexset.actions.legal_actions`'s own `PROPOSE_TRADE`
    sample, which also skips any pair no opponent could currently cover.
    That's a fair thing for a bot to lean on when picking a search target —
    the engine already sees every hand, it's one shared `GameState` — but
    HexSet hands are private at a real table, and this list is what tells a
    *human* what they may propose. Reflecting that omniscient filter here,
    whether by omission or by an "isn't available" message, would hand them
    the one thing the actual board never does: proof of what's in a
    specific opponent's hand. `propose_trade` doesn't require coverage
    either — a proposal nobody can cover is still legal, it just gets no
    takers (see its own `if not willing: return`) — so this is the accurate
    rule for what a human may attempt, not a laxer one.
    """
    player = game.current_player
    # true state: only ever reads `player`'s own hand below, which is
    # public to `player` regardless of view.
    state = game.state(player, hidden=False)
    # Trading is a Main-phase act, same as `propose_trade`'s own check. Left
    # off, this offered pairs before the roll, which lit the hand up as
    # clickable and opened the trade modal on a turn where the bank half of
    # it could not be there — BANK_TRADE only exists in Main.
    if game.phase is not Phase.MAIN:
        return []
    if game.offers_made >= MAX_OFFERS_PER_TURN:
        return []
    hand = state.hands[player]
    return [
        Action(ActionType.PROPOSE_TRADE, give=ONE_RESOURCE[given], want=ONE_RESOURCE[wanted])
        for given in range(NUM_RESOURCES)
        if hand[given]
        for wanted in range(NUM_RESOURCES)
        if wanted != given
    ]


def fair_legal_actions(game: Game) -> list[Action]:
    """Every action currently legal, with `PROPOSE_TRADE`'s sample widened to
    what any client — a browser, an LLM over MCP, an external bot or an
    embedded one — may see: no seat is shown which specific opponents could
    cover an offer (see `proposable_options`), since that would leak a hand's
    exact composition beyond what the public ledger (`hexset.ledger`) already
    certifies."""
    options = [a for a in legal_actions(game) if a.type is not ActionType.PROPOSE_TRADE]
    options += proposable_options(game)
    return options


class Stuck(RuntimeError):
    """Raised when a live game offers no legal action, which is always a bug."""


def options_for(game: Game) -> list[Action]:
    """`fair_legal_actions`, for a caller that has no answer for an empty list.

    A bot on the move must be able to move. Every phase that can be reached
    offers something, so an empty list is an engine bug, and a bot that
    returned `None` here would only push the crash somewhere less obvious.

    Deliberately shadows `hexset.bots.options_for`, which wraps the engine's
    omniscient `legal_actions` instead: nothing served by this package may
    reach for that one (see the module docstring).
    """
    options = fair_legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {game.current_player}")
    return options


def is_legal(game: Game, action: Action, options: Sequence[Action]) -> bool:
    """Whether `action` is one of `options` (normally `fair_legal_actions`),
    except for `PROPOSE_TRADE`, checked against `can_propose` instead.

    Both samples only *sample* the (give, want) pairs they enumerate — but a
    proposal nobody can currently cover is still a legal move
    (`hexset.game.propose_trade` has a defensive path for exactly that,
    concluding the offer with no takers rather than raising), so gating it on
    sample membership would reject a legal offer for a reason that was never a
    real rule. It also lets a stronger client propose a bundle neither sample
    enumerates at all — a two-for-one, say. Checking `can_propose` directly
    also sidesteps `ask` entirely: it only reorders who gets asked, never who's
    eligible, so it was never part of what made an offer legal.
    """
    if action.type is ActionType.PROPOSE_TRADE:
        offer = Offer(proposer=game.current_player, give=action.give, want=action.want)
        # true state: `can_propose` only ever checks the proposer's own
        # hand, public to them regardless of view.
        return can_propose(game.state(game.current_player, hidden=False), offer)
    return action in options
