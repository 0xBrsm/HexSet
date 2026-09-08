# SPDX-License-Identifier: GPL-3.0-only
"""What a checkpoint runtime has to provide, and nothing else.

`hexset.clients.netbot` builds a bot, a leaf evaluation, a search and a trade
gate out of a checkpoint. None of that is specific to onnxruntime, and the
proof that it was not is that the training repo carried a torch copy of every
one of those classes -- a copy that drifted from this one three ways before
anybody noticed (`agents/reference/hexn-boundary-audit.md`). The fix is this
protocol: a runtime implements `Policy`, and gets the bot, the evaluator, the
searcher and the gate for free.

**The surface is stated in live positions, never in records.** A row is a
`(game, seat)` pair (with the seat's options, where a decision needs them),
because that is the only description of a position both runtimes can agree
on: how a position becomes numbers -- `hexset.onnx_record.record_from_game`
for the ONNX graph, `hexset.encoding.encode` for a torch net -- is the
runtime's own business, and the moment it appears in this protocol the
protocol has picked a side. It also means a runtime is free to encode a whole
batch its own way, which is what makes the trade gate's fan-out one forward
rather than one per candidate.

Every value the protocol hands back is in **board-seat order**: seat *i* of
the row's `game`, not seat *i* of whatever rotated frame the runtime encodes
in. Un-rotating is the runtime's job (the exported ONNX graph does it
in-graph; the torch side does it in Python), because the frame is a property
of the network, and a caller that had to know about it would be back to
knowing which runtime it holds.

`hexset.clients.onnxbot.V2Policy` satisfies `Policy` structurally -- nothing
inherits from it, and nothing should have to.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hexset.actions import Action, ActionSpace
from hexset.game import Game


class Policy(Protocol):
    """A loaded checkpoint, asked about live positions.

    Batched throughout: every method takes a list of rows and answers one
    element per row, in the same order. A single-position caller passes a
    list of one -- the cost of a network call is its dispatch, so making the
    batch the primitive is what lets the trade gate price thirty candidates
    for the price of one.
    """

    #: The flat action space this checkpoint was built against
    #: (`hexset.actions.build_space`) -- the bot's decisions and the search's
    #: priors are indexed through it, so a caller holding only the policy can
    #: still ask which slot an option occupies.
    space: ActionSpace

    def act_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[Action]:
        """One chosen action per row, each drawn from that row's own options.

        The options are passed rather than re-derived so the runtime masks
        against exactly what the caller will accept, and so a caller that has
        already enumerated them (a search leaf, a gym step) does not pay for
        it twice.
        """
        ...

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """The value head alone: one vector per row, one entry per seat, in
        board-seat order.

        A bare `(game, seat)` row carries no options because a value question
        has none to offer; a runtime that needs a legal mask to normalise
        over derives it from the position itself
        (`hexset.server.rules.options_for`).

        This is the whole of what the trade gate asks for
        (`netbot.NetworkBot._score`), which is why it takes positions: the
        gate's candidate positions are games it built, not records it
        encoded.
        """
        ...

    def score_rows(
        self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]
    ) -> list[tuple[Sequence[float], tuple[float, ...]]]:
        """`(prior, value)` per row, as `hexset.mcts` wants a leaf scored:
        the prior a non-negative weight per option of that row, aligned with
        the row's own options and summing to one; the value the same
        board-seat vector `value_rows` returns.

        Prior and value come off the same trunk in every runtime worth
        having, so asking for them separately would pay the dispatch toll
        twice for one position.
        """
        ...


class Checkpoint(Protocol):
    """A checkpoint file made playable: the `Policy` plus the facts about the
    run it came from that a caller needs to seat it.

    What `hexset.clients.onnxbot.Loaded` (and the training repo's own
    `Loaded`) already are, named here so `netbot`'s constructors can take one
    without importing either. A runtime is free to carry more (the ONNX
    loader also carries the search configuration and the iteration number);
    these four are what seating a bot reads.
    """

    #: The runtime.
    policy: Policy
    #: The action space `policy` was built against -- the same object.
    space: ActionSpace
    #: How many seats this checkpoint was trained for. A table with a
    #: different count is refused at the first `choose`, loudly.
    players: int
    #: The trade switch the run recorded training under: `0` for a no-trade
    #: referent, `None` for unbounded (the engine has no budget).
    max_trades: int | None
