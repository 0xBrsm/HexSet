# SPDX-License-Identifier: GPL-3.0-only
"""Playout copies held in the native state, for `hexset.rollout`.

A `NativeCopy` is one playout copy of a search: a `NativeState` built from the
Python copy as it stands, played on natively a move at a time. It stands in
for that copy wherever a playout reads it -- whose move it is, the legal
moves, applying one, the phase, turn count and winner -- and wherever a
contract-9 policy states a seat's decision there: the observation
(`observation`), the decision's tensors (`decision`) and the seat frame
(`frame`). Each answers exactly what the Python engine answers for the same
position: the same moves, the same tensors and the same observation bytes,
drawing chance from a copy of the same streams.

A copy is native only where `copyable` says the position and the policy's
contract are supported; otherwise a playout keeps its Python copy.
"""
from __future__ import annotations

import numpy as np

from .. import contract9
from ..actions import Action, ActionSpace, ActionType
from ..encoding import SeatFrame
from ..game import Game, Phase
from ..state import HiddenRead
from .state import NativeState, available

__all__ = ["NativeCopy", "copyable", "held"]

#: The contracts the native engine encodes (`hexset.native.supported`).
_CONTRACTS = ("9", "9.1", "9d", "9d.1")

_PHASES = tuple(Phase)
_TYPES = tuple(ActionType)
_ACTIONS: dict[tuple[int, int, int], Action] = {}


def _action(row: tuple[int, int, int]) -> Action:
    """The `Action` a native `(type, a, b)` row names, one object per row."""
    action = _ACTIONS.get(row)
    if action is None:
        action = _ACTIONS[row] = Action(_TYPES[row[0]], int(row[1]), int(row[2]))
    return action


def copyable(game: Game, contract: str | None, space: ActionSpace) -> bool:
    """Whether a playout copy of `game` may be held natively for a policy
    that reads `contract` over `space`: the native extension is built, the
    contract is one it encodes, and the policy counts every dealt seat (a
    frame of live seats is the Python copy's). Whether the position itself
    is a supported no-trade one is `held`'s to find out."""
    return (contract in _CONTRACTS and available()
            and space.num_players == game._state.num_players
            and (contract not in ("9d", "9d.1") or game._state.num_players == 2))


def held(game: Game) -> "NativeCopy | None":
    """`game` held natively, or `None` where the native engine refuses the
    position (trading or deal state, a hidden pile, another chance source):
    that copy stays the Python engine's."""
    try:
        return NativeCopy(game)
    except (ValueError, TypeError, HiddenRead):
        return None


class NativeCopy:
    """One playout copy held natively (module docstring). Build it from a
    Python copy (`NativeCopy(game)`), or `fork` one: the fork owns its state
    and its chance streams, the same as the original's at the fork."""

    __slots__ = ("state", "topology", "dealt", "locked", "_control", "_legal")

    def __init__(self, game: Game | None = None) -> None:
        if game is None:
            return
        self.state = NativeState(game)
        self.topology = game._state.board.topology
        self.dealt = game._state.num_players
        self.locked = game.locked
        self._read()

    def fork(self) -> "NativeCopy":
        child = NativeCopy()
        child.state = self.state.fork()
        child.topology, child.dealt, child.locked = self.topology, self.dealt, self.locked
        child._control, child._legal = self._control, self._legal
        return child

    def _read(self) -> None:
        self._control = self.state.core.control()
        self._legal = None

    # --- the position, as `hexset.game` reads a Game ----------------------------

    @property
    def phase(self) -> Phase:
        return _PHASES[self._control[0]]

    @property
    def current_player(self) -> int:
        return self._control[1]

    @property
    def free_roads(self) -> int:
        return self._control[6]

    @property
    def turns(self) -> int:
        return self._control[8]

    @property
    def won_by(self) -> int | None:
        won = self._control[9]
        return None if won == -1 else won

    def over(self) -> bool:
        return self._control[0] == Phase.GAME_OVER

    def to_move(self) -> int:
        """`hexset.game.to_move`: the current player, except while seats owe
        discards, where it is the lowest-indexed seat owing."""
        if self._control[0] == Phase.DISCARD:
            for seat in range(self.dealt):
                if self.state.core.legal_actions(seat):
                    return seat
        return self._control[1]

    def legal(self, seat: int | None = None) -> tuple[Action, ...]:
        """`hexset.actions.legal_actions`, in its order: the single actor's
        moves, or those of `seat`."""
        if seat is None:
            if self._legal is None:
                self._legal = tuple(_action(row) for row in self.state.core.legal_actions())
            return self._legal
        return tuple(_action(row) for row in self.state.core.legal_actions(seat))

    def inside(self) -> bool:
        """Whether a decision begun a move at a time is under way here: a
        setup road, or a Road Building card's free roads still owed
        (`hexset.decisions.inside`)."""
        if self._control[0] == Phase.SETUP_ROAD:
            return True
        if self._control[6] <= 0:
            return False
        # Owed roads are all the native engine offers while they are owed;
        # otherwise it offers no road at all, the credit having nowhere to go.
        legal = self.legal()
        return bool(legal) and all(action.type is ActionType.BUILD_ROAD for action in legal)

    def apply(self, action: Action, seat: int | None = None) -> None:
        """`hexset.actions.apply`; a discard is made by `seat`, or by the
        lowest-indexed seat owing."""
        if action.type is ActionType.DISCARD and seat is None:
            seat = self.to_move()
        self.state.apply(action, seat if action.type is ActionType.DISCARD else None)
        self._read()

    # --- what a contract-9 policy reads -----------------------------------------

    def observation(self, seat: int, contract: str) -> np.ndarray:
        """`contract9.flatten(contract9.encode9(game, seat, contract))`."""
        return np.frombuffer(self.state.core.encode9(seat, contract), dtype=np.float32)

    def frame(self, seat: int, players: int) -> SeatFrame:
        """`hexset.encoding.seat_frame` at this table, whose every dealt seat
        the policy counts (`copyable`)."""
        if players != self.dealt:
            raise ValueError(f"a native copy is held for a {self.dealt}-seat policy, not {players}")
        return SeatFrame(tuple((seat + i) % self.dealt for i in range(self.dealt)), self.dealt)

    def tensors(self, seat: int, space: ActionSpace):
        """`hexset.decisions.tensors` with no double turn: `seat`'s decision
        here as a contract-9 net states it."""
        from ..decisions import Decision
        got = self.state.core.decision(seat)
        flat = np.frombuffer(got["flat"], dtype=np.bool_)
        setup = np.frombuffer(got["setup"], dtype=np.bool_)
        if len(flat) != space.size or len(setup) != len(contract9.incidences(self.topology)):
            raise ValueError("native decision shape differs from the action space")
        roads = np.frombuffer(got["roads"], dtype=np.int16).reshape(-1, 2)
        discard = np.frombuffer(got["discard"], dtype=np.int8).reshape(-1, 10)
        return Decision(flat, setup, False, roads, discard)
