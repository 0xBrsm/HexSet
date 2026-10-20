# SPDX-License-Identifier: GPL-3.0-only
"""Decisions: everything a seat chooses before anything new comes to light.

Most actions are a decision each. Four are not, because a seat makes several
moves with nothing revealed and no other seat acting in between, and choosing
them one at a time commits it to the first before it has chosen the rest:

| `CompoundKind`  | the decision                                          |
|-----------------|-------------------------------------------------------|
| `SETUP`         | a setup settlement with its road                      |
| `SETUP_DOUBLE`  | the seat the setup snake turns on: both settlements   |
|                 | with a road each, the second the one that pays        |
| `ROAD_BUILDING` | the card with the roads it places (two, or one where  |
|                 | only one fits, or none where none does)               |
| `DISCARD`       | a seat's whole discard to a seven                     |

A `Compound` is its atomic `hexset.actions.Action`s in the order they apply,
so it applies through `actions.apply` one part at a time inside one call:
records, replays and a seat that acts a piece at a time are unchanged, and
since nothing in a decision reveals anything and nobody else acts within it,
the parts applied together leave the table exactly as the decision would.

`options(game, seat)` lists a seat's decisions in these terms; `tensors`
states them for a network (`Decision`): a flat mask over the ordinary actions
and the compound candidates beside it. Robber victims are seat-relative there,
the perspective first, as contract 9 orders the seats (`hexset.contract9`).
With `double_turn=False` both state the snake seat's double turn as two setup
turns, one after the other, each a `SETUP` of its own.

**A run of options.** An option is what one network forward chooses among.
A seat's options run on, one after another, until one shows something or
hands the table on (`closes`): the dice, a dev card drawn, a robber's steal,
the cards a Monopoly takes, a trade event the seat may offer in, its discard
done, the turn ended, or another seat to act. Everything before that is seen
by nobody else and tells the seat nothing new, so a learner can take the run
as one unit of credit (one decision) whose options are its parts: a turn's
builds after the roll, a knight with its robber, the double turn's two setup
turns.

**A whole decision, played a move at a time.** `option_of` names the
option a `(kind, index)` over a `Decision` points at (`locate` the other way);
`arrays` states a batch of decisions as the named arrays a network reads. A
table asks a seat for one atomic move at a time, so `Plans` keeps the rest of
a decision a seat chose whole: at the position before the decision's first
move the seat chooses all of it and plays its first part, and each later ask
inside the decision gets the next part. `atomic` states a position inside a
decision begun a move at a time as a decision of its legal moves alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import NamedTuple, Sequence

import numpy as np

from .actions import Action, ActionSpace, ActionType, apply, legal_actions
from .board.terrain import NUM_RESOURCES
from .contract9 import incidences
from .game import Game, Phase, may_act, pending_free_roads, to_move
from .trading import params_of
from .state import MAX_ROADS, NO_OWNER, can_place_settlement, placeable_roads, road_count
from .roads import MIN_LONGEST_ROAD, longest_road
from .victory import LONGEST_ROAD_VP, _award as award, victory_points

__all__ = [
    "CompoundKind",
    "Compound",
    "Decision",
    "DecisionKind",
    "is_double_turn",
    "opens",
    "REVEALING",
    "closes",
    "options",
    "apply_option",
    "key",
    "tensors",
    "locate",
    "flat_index",
    "double_pairs",
    "setup_parts",
    "road_building_parts",
    "discard_parts",
    "option_of",
    "ARRAYS",
    "arrays",
    "atomic",
    "Plans",
]


class CompoundKind(IntEnum):
    """The four decisions made of several moves (the table above)."""

    SETUP = 0
    SETUP_DOUBLE = 1
    ROAD_BUILDING = 2
    DISCARD = 3


class Compound(NamedTuple):
    """One decision of several moves: `parts` in the order they apply."""

    kind: CompoundKind
    parts: tuple[Action, ...]


Option = Action | Compound


def is_double_turn(game: Game) -> bool:
    """Whether the seat about to place a setup settlement places again straight
    after its road: the turn of the setup snake, where nothing is revealed and
    nobody else places between its two settlements."""
    if game.phase is not Phase.SETUP_SETTLEMENT:
        return False
    queue, step = game.setup_queue, game.setup_step + 1
    while step < len(queue) and queue[step] in game.locked:
        step += 1
    return step < len(queue) and queue[step] == game.current_player


def opens(game: Game, action: Action) -> CompoundKind | None:
    """The compound decision `action` is the first move of, at this position,
    or `None` for a decision of its own."""
    kind = action.type
    if kind is ActionType.SETUP_SETTLEMENT:
        return CompoundKind.SETUP_DOUBLE if is_double_turn(game) else CompoundKind.SETUP
    if kind is ActionType.PLAY_ROAD_BUILDING:
        return CompoundKind.ROAD_BUILDING
    if kind is ActionType.DISCARD:
        return CompoundKind.DISCARD
    return None


# --- candidates -------------------------------------------------------------------


def _setup_pairs(game: Game) -> list[tuple[int, int]]:
    """Every legal (settlement, road) of the setup turn at hand."""
    state = game._state
    player = game.current_player
    topology = state.board.topology
    out = []
    for v in range(topology.num_vertices):
        if can_place_settlement(state, player, v, connected=False):
            out.extend((v, e) for e in topology.vertex_edges[v] if state.edge_owner[e] == NO_OWNER)
    return out


def setup_parts(v: int, e: int) -> tuple[Action, Action]:
    """A setup turn's moves: the settlement on `v`, then the road on `e`."""
    return Action(ActionType.SETUP_SETTLEMENT, v), Action(ActionType.SETUP_ROAD, e)


def _discards(hand: Sequence[int], quota: int) -> list[tuple[int, ...]]:
    """Every count per resource within `hand` totalling `quota`."""
    out = []
    for counts in product(*(range(min(h, quota) + 1) for h in hand)):
        if sum(counts) == quota:
            out.append(counts)
    return out


def discard_parts(counts: Sequence[int]) -> tuple[Action, ...]:
    """A whole discard's moves: `counts[r]` of each resource, in resource order."""
    return tuple(Action(ActionType.DISCARD, r) for r in range(NUM_RESOURCES) for _ in range(counts[r]))


def _road_building(game: Game) -> list[tuple[int, ...]]:
    """The road sets a Road Building card played now can place, each in an
    order it can be placed: pairs, and singles where a road leaves no second
    (no piece left, nowhere to go, or a win). The card is offered only with a
    road to place."""
    state = game._state
    player = game.current_player
    pieces = MAX_ROADS - road_count(state, player)
    first = placeable_roads(state, player)
    if pieces < 2:
        return [(e,) for e in first]
    # Only a road that takes Longest Road can win, and only from two short.
    near = victory_points(state, player) + LONGEST_ROAD_VP >= state.rules.winning_points
    out: list[tuple[int, ...]] = []
    seen: set[frozenset[int]] = set()
    edge_owner = state.edge_owner
    for e1 in first:
        if near and _wins_with(game, e1):
            out.append((e1,))
            continue
        edge_owner[e1] = player
        try:
            # `placeable_roads` is every edge `road_placeable` takes, ascending, in one pass.
            second = placeable_roads(state, player)
        finally:
            edge_owner[e1] = NO_OWNER
        if not second:
            out.append((e1,))
            continue
        for e2 in second:
            pair = frozenset((e1, e2))
            if pair not in seen:
                seen.add(pair)
                out.append((e1, e2))
    return out


def _wins_with(game: Game, edge: int) -> bool:
    """Whether the card's first free road on `edge` ends the game (rare: a seat
    two points short whose road takes Longest Road). Read off the position: a
    road changes no points but the card's, and moves it only to its builder."""
    state = game._state
    player = game.current_player
    if state.longest_road_holder == player:
        return False
    lengths = list(state.road_lengths)
    lengths[player] = longest_road(state, player, extra_edge=edge)
    if award(lengths, state.longest_road_holder, MIN_LONGEST_ROAD) != player:
        return False
    return victory_points(state, player) + LONGEST_ROAD_VP >= state.rules.winning_points


def road_building_parts(roads: Sequence[int]) -> tuple[Action, ...]:
    """The card's moves: playing it, then each road in an order it can be placed."""
    return (Action(ActionType.PLAY_ROAD_BUILDING),) + tuple(Action(ActionType.BUILD_ROAD, e) for e in roads)


def options(game: Game, seat: int | None = None, *, double_turn: bool = True) -> list[Option]:
    """`seat`'s decisions now, compound where a decision is several moves.
    `seat` defaults to `to_move`. A position inside a compound decision (a
    setup road owed, free roads owed, a discard begun) lists its remaining
    atomic moves: a seat acting a piece at a time put it there.
    `double_turn=False` lists the snake seat's double turn as its first setup
    turn alone (`SETUP`); the second is the next position's."""
    seat = to_move(game) if seat is None else seat
    if game.phase is Phase.GAME_OVER or not may_act(game, seat):
        return []
    if game.phase is Phase.SETUP_SETTLEMENT:
        pairs = _setup_pairs(game)
        if not (double_turn and is_double_turn(game)):
            return [Compound(CompoundKind.SETUP, setup_parts(v, e)) for v, e in pairs]
        neighbours = game._state.board.topology.vertex_neighbors
        return [
            Compound(CompoundKind.SETUP_DOUBLE, setup_parts(v1, e1) + setup_parts(v2, e2))
            for v1, e1 in pairs for v2, e2 in pairs
            if v2 != v1 and v2 not in neighbours[v1]
        ]
    if game.phase is Phase.DISCARD:
        # What the seat still owes: all of its quota, or the rest of a
        # discard it began a card at a time.
        given = _discards(game._state.hands[seat], game.discard_quota[seat])
        return [Compound(CompoundKind.DISCARD, discard_parts(d)) for d in given]
    atomic = legal_actions(game, seat)
    if pending_free_roads(game) or game.phase is Phase.SETUP_ROAD:
        return list(atomic)
    out: list[Option] = [a for a in atomic if a.type is not ActionType.PLAY_ROAD_BUILDING]
    if len(out) < len(atomic):
        out.extend(Compound(CompoundKind.ROAD_BUILDING, road_building_parts(r)) for r in _road_building(game))
    return out


def apply_option(game: Game, option: Option, seat: int | None = None) -> None:
    """Apply a decision: an action, or a compound's parts in order inside this
    one call. `seat` names a discarding seat, as `actions.apply` takes it."""
    if isinstance(option, Compound):
        for part in option.parts:
            apply(game, part, seat)
            if game.phase is Phase.GAME_OVER:
                return
    else:
        apply(game, option, seat)


# The moves that show something whoever makes them: the dice, the dev card
# drawn, the robber's steal (a robber moved to rob nobody included), the
# cards a Monopoly takes, and the turn handed on. A discard shows itself once
# the seat's whole discard is made (`closes`).
REVEALING = frozenset({ActionType.ROLL, ActionType.BUY_DEV_CARD, ActionType.MOVE_ROBBER,
                       ActionType.PLAY_MONOPOLY, ActionType.END_TURN})


def closes(game: Game, option: Option, seat: int, event_before: int | None = None) -> bool:
    """Whether `seat`'s run of options (the module docstring) ends with
    `option`, `game` as it stands after it: the game is over; a part is
    `REVEALING`; the seat's discard is done; a trade event the seat may offer
    in ran inside it (`event_before` is `game.trade_event_turn` as it stood
    before the option; `None` for a caller that seats no gates); or the seat
    is no longer the one to act. Anything else -- a build, a bank trade, a
    knight before its robber, Year of Plenty, Road Building, a setup turn the
    seat follows with its own -- shows nobody anything, and the run goes on."""
    if game.phase is Phase.GAME_OVER:
        return True
    parts = option.parts if isinstance(option, Compound) else (option,)
    if any(part.type in REVEALING for part in parts):
        return True
    if any(part.type is ActionType.DISCARD for part in parts):
        return game.phase is not Phase.DISCARD or game.discard_quota[seat] <= 0
    if event_before is not None and game.trade_event_turn != event_before and game.gates is not None:
        gate = game.gates[seat] if 0 <= seat < len(game.gates) else None
        if gate is not None and params_of(gate).trade_offer_budget != 0:
            return True
    return not may_act(game, seat) or to_move(game) != seat


def key(option: Option) -> tuple:
    """What identifies a decision whatever order its parts came in: a discard
    by its counts, Road Building by its set of roads."""
    if not isinstance(option, Compound):
        return ("action", option)
    if option.kind is CompoundKind.DISCARD:
        counts = [0] * NUM_RESOURCES
        for part in option.parts:
            counts[part.a] += 1
        return (option.kind, tuple(counts))
    if option.kind is CompoundKind.ROAD_BUILDING:
        return (option.kind, frozenset(p.a for p in option.parts[1:]))
    return (option.kind, tuple(p.a for p in option.parts))


# --- the network's view of a decision ----------------------------------------------


class DecisionKind(IntEnum):
    """Which block of `Decision` the chosen option sits in."""

    FLAT = 0
    SETUP = 1
    DOUBLE = 2
    ROAD_BUILDING = 3
    DISCARD = 4


@dataclass(frozen=True)
class Decision:
    """A seat's decision as a network scores it.

    `flat` masks the ordinary actions over the `ActionSpace` (robber victims
    seat-relative). `setup` masks the (settlement, road) incidences
    (`contract9.incidences`) of a setup turn; under `double` the decision is
    an ordered pair of them, the second the paying settlement, legal where both
    are and their settlements are neither the same vertex nor neighbours.
    `road_building` lists the card's road sets, `(first, second)` with `-1` for
    a road it does not place; `discard` each whole discard as the hand kept and
    the cards given, five counts each."""

    flat: np.ndarray            # (space.size,) bool
    setup: np.ndarray           # (incidences,) bool
    double: bool
    road_building: np.ndarray   # (K, 2) int16
    discard: np.ndarray         # (K, 2 * NUM_RESOURCES) int8: kept, then given


def flat_index(space: ActionSpace, action: Action, seat: int, players: int) -> int:
    """`action`'s slot in the flat space, its robber victim seat-relative."""
    if action.type is ActionType.MOVE_ROBBER and action.b < players:
        action = Action(action.type, action.a, (action.b - seat) % players)
    return space.index(action)


def tensors(game: Game, seat: int, space: ActionSpace, *, double_turn: bool = True) -> Decision:
    """`seat`'s decision now as `Decision` states it, worked out from the
    position (a double turn's thousands of pairs are never listed).
    `double_turn=False` states the snake seat's double turn as its first
    setup turn alone, `double` false (`options`)."""
    state = game._state
    players = state.num_players
    topology = state.board.topology
    pairs = incidences(topology)
    flat = np.zeros(space.size, dtype=np.bool_)
    setup = np.zeros(len(pairs), dtype=np.bool_)
    double = False
    rb = np.zeros((0, 2), dtype=np.int16)
    discard = np.zeros((0, 2 * NUM_RESOURCES), dtype=np.int8)
    if game.phase is Phase.SETUP_SETTLEMENT:
        legal = set(_setup_pairs(game))
        setup[:] = [p in legal for p in pairs]
        double = double_turn and is_double_turn(game)
    elif game.phase is Phase.DISCARD:
        hand = state.hands[seat]
        given = _discards(hand, game.discard_quota[seat])
        discard = np.asarray([[h - d for h, d in zip(hand, g)] + list(g) for g in given],
                             dtype=np.int8).reshape(-1, 2 * NUM_RESOURCES)
    else:
        if pending_free_roads(game) or game.phase is Phase.SETUP_ROAD:
            raise ValueError("a decision begun a piece at a time has no contract-9 tensors")
        playable = False
        for action in legal_actions(game, seat):
            if action.type is ActionType.PLAY_ROAD_BUILDING:
                playable = True
                continue
            flat[flat_index(space, action, seat, players)] = True
        if playable:
            sets = _road_building(game)
            rb = np.asarray([tuple(r) + (-1,) * (2 - len(r)) for r in sets], dtype=np.int16).reshape(-1, 2)
    return Decision(flat, setup, double, rb, discard)


def locate(decision: Decision, option: Option, game: Game, seat: int, space: ActionSpace) -> tuple[DecisionKind, int]:
    """Where `option` sits in `decision`: its block and its index there. For a
    double turn the index is `first * len(incidences) + second`."""
    players = game._state.num_players
    if not isinstance(option, Compound):
        index = flat_index(space, option, seat, players)
        if not decision.flat[index]:
            raise ValueError(f"{option} is not among the decision's actions")
        return DecisionKind.FLAT, index
    pairs = incidences(game._state.board.topology)
    if option.kind is CompoundKind.SETUP:
        index = pairs.index((option.parts[0].a, option.parts[1].a))
        if decision.double or not decision.setup[index]:
            raise ValueError(f"{option} is not a legal setup turn here")
        return DecisionKind.SETUP, index
    if option.kind is CompoundKind.SETUP_DOUBLE:
        first = pairs.index((option.parts[0].a, option.parts[1].a))
        second = pairs.index((option.parts[2].a, option.parts[3].a))
        if not (decision.double and decision.setup[first] and decision.setup[second]):
            raise ValueError(f"{option} is not a legal double turn here")
        return DecisionKind.DOUBLE, first * len(pairs) + second
    if option.kind is CompoundKind.ROAD_BUILDING:
        roads = frozenset(p.a for p in option.parts[1:])
        for i, row in enumerate(decision.road_building):
            if frozenset(int(e) for e in row if e >= 0) == roads:
                return DecisionKind.ROAD_BUILDING, i
        raise ValueError(f"{option} is not among the card's road sets")
    counts = key(option)[1]
    for i, row in enumerate(decision.discard):
        if tuple(int(n) for n in row[NUM_RESOURCES:]) == counts:
            return DecisionKind.DISCARD, i
    raise ValueError(f"{option} is not among the discards")


def double_pairs(topology) -> np.ndarray:
    """`(incidences, incidences)` bool: which ordered pairs of setup turns one
    seat may place back to back, as far as the board decides it (the two
    settlements neither the same vertex nor neighbours)."""
    pairs = incidences(topology)
    out = np.ones((len(pairs), len(pairs)), dtype=np.bool_)
    for i, (v1, _) in enumerate(pairs):
        for j, (v2, _) in enumerate(pairs):
            if v2 == v1 or v2 in topology.vertex_neighbors[v1]:
                out[i, j] = False
    return out


# --- choosing a whole decision ---------------------------------------------------


def option_of(decision: Decision, kind: int, index: int, pairs, seat: int, space: ActionSpace,
              players: int) -> Option:
    """The option `(kind, index)` names in `decision` (`locate` the other way):
    an action, its robber victim back in board seats, or a `Compound`. `pairs`
    is the board's `contract9.incidences`."""
    if kind == DecisionKind.FLAT:
        action = space.decode(index)
        if action.type is ActionType.MOVE_ROBBER and action.b < players:
            action = Action(action.type, action.a, (action.b + seat) % players)
        return action
    if kind == DecisionKind.SETUP:
        return Compound(CompoundKind.SETUP, setup_parts(*pairs[index]))
    if kind == DecisionKind.DOUBLE:
        first, second = divmod(index, len(pairs))
        return Compound(CompoundKind.SETUP_DOUBLE, setup_parts(*pairs[first]) + setup_parts(*pairs[second]))
    if kind == DecisionKind.ROAD_BUILDING:
        roads = tuple(int(e) for e in decision.road_building[index] if e >= 0)
        return Compound(CompoundKind.ROAD_BUILDING, road_building_parts(roads))
    return Compound(CompoundKind.DISCARD, discard_parts(decision.discard[index][NUM_RESOURCES:]))


#: The arrays `arrays` states a batch of decisions as, by name: each
#: decision's `flat` and `setup` masks stacked, its Road Building and discard
#: candidates end to end beside the row each belongs to, and the rows whose
#: decision is a double turn.
ARRAYS = ("flat", "setup", "road_building", "road_building_row", "discard", "discard_row", "double_rows")


def arrays(decisions: Sequence["Decision | None"], space: ActionSpace, topology) -> dict[str, np.ndarray]:
    """`decisions`, one per row, as the named arrays `ARRAYS` lists: `flat`
    `(B, space.size)` and `setup` `(B, incidences)` bool, every row's masks;
    `road_building` `(Kr, 2)` and `discard` `(Kd, 10)`, every row's candidates
    end to end as `Decision` lists them, with `road_building_row` and
    `discard_row`, the row each belongs to; and `double_rows`, the rows whose
    decision is `double`. All int64 but the masks. A row whose decision is
    `None` is one with nothing to decide: its masks are empty and it has no
    candidates. `topology` is the board's (its incidences size `setup`)."""
    rows = len(decisions)
    flat = np.zeros((rows, space.size), dtype=np.bool_)
    setup = np.zeros((rows, len(incidences(topology))), dtype=np.bool_)
    rb, rb_row, disc, disc_row, double = [], [], [], [], []
    for row, decision in enumerate(decisions):
        if decision is None:
            continue
        flat[row] = decision.flat
        setup[row] = decision.setup
        if len(decision.road_building):
            rb.append(decision.road_building.astype(np.int64))
            rb_row.append(np.full(len(decision.road_building), row, dtype=np.int64))
        if len(decision.discard):
            disc.append(decision.discard.astype(np.int64))
            disc_row.append(np.full(len(decision.discard), row, dtype=np.int64))
        if decision.double:
            double.append(row)

    def joined(parts, width):
        return np.concatenate(parts) if parts else np.zeros((0, width) if width else 0, dtype=np.int64)

    return {
        "flat": flat, "setup": setup,
        "road_building": joined(rb, 2), "road_building_row": joined(rb_row, 0),
        "discard": joined(disc, 2 * NUM_RESOURCES), "discard_row": joined(disc_row, 0),
        "double_rows": np.asarray(double, dtype=np.int64),
    }


def atomic(game: Game, seat: int, space: ActionSpace) -> Decision:
    """`seat`'s legal atomic moves at `game` as a decision of flat actions
    alone (robber victims seat-relative, as `tensors` states them): how a
    position inside a decision begun a move at a time -- a setup road or a
    free road owed, which `tensors` refuses -- is asked."""
    players = game._state.num_players
    flat = np.zeros(space.size, dtype=np.bool_)
    for action in legal_actions(game, seat):
        flat[flat_index(space, action, seat, players)] = True
    return Decision(flat, np.zeros(len(incidences(game._state.board.topology)), dtype=np.bool_), False,
                    np.zeros((0, 2), dtype=np.int16), np.zeros((0, 2 * NUM_RESOURCES), dtype=np.int8))


class Plans:
    """Whole decisions, played a move at a time.

    `act` answers a seat's ask for one atomic move. Where the move opens a
    decision, it asks `decide(game, seat)` for the whole of it at the position
    before its first move, returns that first move and keeps the rest as the
    seat's plan; each later ask inside the decision gets the plan's next move.
    Nobody else acts inside a decision and nothing is revealed, so a plan
    cannot go stale; it is checked against the options all the same, and a
    position inside a decision with no plan of its own (a copy a gate
    imagined, a game resumed part-way) is finished by `finish(game, seat,
    options)` from the position as it stands: a setup road or a free road
    owed (`atomic` states it as a decision of its legal moves). A discard
    begun a card at a time is a decision of its own, the rest of the quota
    (`tensors`).

    Plans are kept per game, by identity, so one `Plans` serves every game a
    process plays; a game that ended inside a decision leaves its plan behind,
    and once `LIMIT` games hold one they are all dropped."""

    LIMIT = 256

    def __init__(self) -> None:
        self._plans: dict[int, tuple[Game, int, list[Action]]] = {}

    def act(self, game: Game, seat: int, options: Sequence[Action], decide, finish) -> Action:
        """`seat`'s move at `game` from `options`, its legal atomic moves."""
        plan = self._plans.pop(id(game), None)
        if plan is not None and plan[0] is game and plan[1] == seat and plan[2] and plan[2][0] in options:
            rest = plan[2][1:]
            if rest:
                self._plans[id(game)] = (game, seat, rest)
            return plan[2][0]
        if len(options) == 1:
            return options[0]
        if game.phase is Phase.SETUP_ROAD or pending_free_roads(game):
            return finish(game, seat, tuple(options))
        choice = decide(game, seat)
        if len(self._plans) > self.LIMIT:
            self._plans.clear()
        if isinstance(choice, Compound):
            first, rest = choice.parts[0], list(choice.parts[1:])
            if rest:
                self._plans[id(game)] = (game, seat, rest)
            choice = first
        if choice not in options:
            raise RuntimeError(f"the decision chosen opens with {choice}, not among {tuple(options)}")
        return choice
