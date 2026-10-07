# SPDX-License-Identifier: GPL-3.0-only
"""A coalition: seats that play together against the rest of the table.

`Coalition(bot)` seats any bot as a member. Every seat at the table seated
with a `Coalition` is a member, and every other playing seat is a **target**,
so a lineup `[<target>, coalition:<bot>, coalition:<bot>, coalition:<bot>]`
is one entrant against a table that has agreed to stop it, whatever the
scores say. A member reads its partners off the table (`Game.gates`, the
bots seated there) when it is seated or first asked to move, so the arena's
seat rotation needs no other wiring.

It works in two layers.

**The table**, whatever the bot:

- **robber**: on the targets' best hex -- the most of their production
  blocked (pips times their settlements and cities there), then the least of
  the members' -- off the member's own hexes where a target has one
  elsewhere, stealing from a target; the bot's own choice only where no hex
  holds a target;
- **trades**: none with a target. No offer is aimed at one or asks for its
  cards, its offers and counters are passed, and its acceptances and counters
  are never picked: every trade hook the bot has (`TRADE_HOOKS`) is the bot's
  own with the targets taken out, and a hook it lacks stays absent, so the
  engine's default reads the bot's valuation with the targets priced out.

Everything else -- every other move, and how it trades with a partner -- is
the bot's.

**The bot's own objective**, where it has one: each part of the bot (both
halves of a `TradesBy` seat) that implements `PlaysAgainst` is told the
targets (`play_against`) and plays as one side against them in its own way.
A `hexset.mcts.Search` searches as one side (`Search.against`), so its builds
and roads come from the search. A bot without it gets the table layer alone.

`hexset.arena` seats one as `coalition:<entrant>`. A test opponent: nothing
here reaches a model or a value head.
"""
from __future__ import annotations

from typing import Sequence

from ..actions import Action, ActionType, legal_actions
from ..arena import RetiredSeat
from ..board.board import pips
from ..game import Game, Phase, to_move
from ..robber import occupants
from ..trading import RESPONSE_PASS, Response
from .base import Bot, play_against, seat_at

__all__ = ["Coalition", "TRADE_HOOKS", "targets_at"]

#: The trade hooks the engine looks up on a seat, each answered by the bot's
#: own with the targets taken out (`Coalition._<hook>`).
TRADE_HOOKS = frozenset({
    "candidates", "offer", "respond", "respond_any", "pick",
    "gains_many", "accepts_many", "accepts", "consent_gain",
})


def targets_at(gates: Sequence[object]) -> frozenset[int]:
    """The seats a coalition at a table seated with `gates` plays against:
    every seat neither a member nor retired."""
    return frozenset(s for s, g in enumerate(gates) if not isinstance(g, (Coalition, RetiredSeat)))


class Coalition:
    """`bot`, playing with every other `Coalition` seat at its table against
    the rest. Every name it does not answer itself is `bot`'s, read, written
    and deleted through, as `TradesBy` does, so `hexset.trading.retune`
    retunes the bot."""

    __slots__ = ("bot", "targets", "_gates", "_hex_pips")

    def __init__(self, bot: Bot) -> None:
        object.__setattr__(self, "bot", bot)
        object.__setattr__(self, "targets", None)    # read off the table when seated
        object.__setattr__(self, "_gates", None)
        object.__setattr__(self, "_hex_pips", None)

    def __reduce__(self):
        return (Coalition, (self.bot,))

    # --- seating ------------------------------------------------------------

    def seat_at(self, game: Game) -> None:
        """`hexset.bots.seat_at` on the bot, and the targets read off the
        table where it is seated."""
        seat_at(self.bot, game)
        if game.gates is not None:
            self._seat(game.gates)

    def _seat(self, gates: Sequence[object]) -> None:
        if gates is self._gates:
            return
        targets = targets_at(gates)
        object.__setattr__(self, "_gates", gates)
        if targets == self.targets:
            return
        object.__setattr__(self, "targets", targets)
        play_against(self.bot, targets)

    def _against(self) -> frozenset[int]:
        # Every seat moves in setup before any trade is put, so a trade hook
        # asked before then is a harness this seat was not built for.
        if self.targets is None:
            raise RuntimeError("a coalition seat was asked to trade before it was seated")
        return self.targets

    # --- moves ----------------------------------------------------------------

    def choose(self, game: Game) -> Action:
        if game.gates is None:
            raise ValueError("a coalition seat reads its partners off the table's seated bots")
        self._seat(game.gates)
        if game.phase is Phase.ROBBER:
            action = self._robber(game)
            if action is not None:
                return action
        return self.bot.choose(game)

    def _robber(self, game: Game) -> Action | None:
        """The robber on the targets' best hex (see the module), or `None`
        where no hex holds a target."""
        seat = to_move(game)
        cands = [a for a in legal_actions(game, seat) if a.type is ActionType.MOVE_ROBBER]
        if not cands:
            return None
        st = game.state(seat).state
        targets = self.targets
        if self._hex_pips is None or len(self._hex_pips) != len(st.board.tokens):
            object.__setattr__(self, "_hex_pips", [pips(t) for t in st.board.tokens])
        hex_pips, corners = self._hex_pips, st.board.topology.hex_vertices

        def blocked(h: int, seats) -> int:
            return hex_pips[h] * sum(st.vertex_building[v] for v in corners[h] if st.vertex_owner[v] in seats)

        hit = {a.a: blocked(a.a, targets) for a in cands}
        if max(hit.values()) == 0:
            return None
        members = {s for s in range(st.num_players) if s not in targets}
        return max(cands, key=lambda a: (hit[a.a] > 0 and seat not in occupants(st, a.a), hit[a.a],
                                         -blocked(a.a, members), a.b in targets, -a.a))

    # --- trades -----------------------------------------------------------------

    def __getattr__(self, name: str):
        hook = getattr(object.__getattribute__(self, "bot"), name)
        if name in TRADE_HOOKS and hook is not None:
            own = getattr(type(self), f"_{name}")
            return lambda *args, **kwargs: own(self, hook, *args, **kwargs)
        return hook

    def __setattr__(self, name: str, value) -> None:
        setattr(self.bot, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.bot, name)

    def _candidates(self, hook, view, counterparties, **kwargs):
        against = self._against()
        out = hook(view, [c for c in counterparties if c not in against], **kwargs)
        return [(c, b) for c, b in out if c not in against]

    def _offer(self, hook, view, candidates):
        against = self._against()
        keep = [i for i, (c, _) in enumerate(candidates) if c not in against]
        if not keep:
            return None
        index = hook(view, [candidates[i] for i in keep])
        return keep[index] if index is not None and 0 <= index < len(keep) else None

    def _respond(self, hook, view, offer):
        if offer.actor in self._against():
            return Response(view.perspective, RESPONSE_PASS, None)
        return hook(view, offer)

    _respond_any = _respond

    def _pick(self, hook, view, responses):
        against = self._against()
        keep = [i for i, r in enumerate(responses) if r.seat not in against]
        if not keep:
            return None
        index = hook(view, [responses[i] for i in keep])
        return keep[index] if index is not None and 0 <= index < len(keep) else None

    def _priced(self, hook, view, received, counterparties, refused):
        """`hook` asked about the exchanges with partners only; `refused`
        for each with a target."""
        against = self._against()
        keep = [i for i, c in enumerate(counterparties) if c not in against]
        got = iter(hook(view, [received[i] for i in keep], [counterparties[i] for i in keep]) if keep else ())
        return [refused if c in against else next(got) for c in counterparties]

    def _gains_many(self, hook, view, received, counterparties):
        return self._priced(hook, view, received, counterparties, -1.0)

    def _accepts_many(self, hook, view, received, counterparties):
        return self._priced(hook, view, received, counterparties, False)

    def _accepts(self, hook, view, received, counterparty):
        return counterparty not in self._against() and hook(view, received, counterparty)

    def _consent_gain(self, hook, view, received, counterparty, **kwargs):
        if counterparty in self._against():
            return -1.0
        return hook(view, received, counterparty, **kwargs)
