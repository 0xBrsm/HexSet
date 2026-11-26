# SPDX-License-Identifier: GPL-3.0-only
"""What a playout ended with, for one seat: a row of numbers per copy.

A search that plays a decision's copies out can keep, besides each copy's
score, how the copy ended (`hexset.rollout.RolloutBatch` with `endings`).
`ending` reads one finished or stopped copy, a Python `Game` or a native
copy, from `seat`'s side: the first `len(FIELDS)` entries are the named
fields below, and the rest is one entry per vertex: `OPEN` for no
building, `MINE` or `THEIRS` for a settlement of `seat`'s or of another
seat's, `MINE_CITY` or `THEIRS_CITY` for a city (`holder` folds the two
levels together). What accumulates while the copy is played (`discarded`,
`other_discarded` and the `TRACE` fields) is passed in by the player
(`follow` keeps the trace).

The fields, for `seat` and for the other seats taken together:

- `won`, `lost`: 1 where `seat` won, or another seat did;
- `turns`: turns played since the playout's start;
- `points`, `other_points`: `seat`'s full points and the most any other
  seat has;
- `longest_road`, `other_longest_road`, `largest_army`,
  `other_largest_army`: 1 where `seat`, or another seat, holds the award;
- `discarded`, `other_discarded`: cards lost to sevens during the playout;
- `knights`: Knights `seat` has played;
- `cities`, `settlements`: `seat`'s buildings on the board;
- `production_wood` to `production_ore`: the pips `seat`'s buildings
  collect of each resource (a city twice), the robber's hex left out;
  `other_production`: the same, every resource, for the other seat that
  collects the most;
- `hand_wood` to `hand_ore`: the resource cards `seat` holds;
- `development_cards`: the development cards `seat` holds, bought this
  turn or before;
- `port_generic`, `port_wood` to `port_ore`: 1 where `seat` has a building
  on a 3:1 port, or on that resource's 2:1 port;
- `spots_now`: open spots (no building on or next to them) at the end of
  one of `seat`'s roads; `spots_near`: open spots one road further, along
  an edge nobody holds; `best_spot`: the most pips at any of those spots;
- `robber`: the hex the robber stands on; `blocked`, `other_blocked`: the
  pips it keeps from `seat`'s buildings there (a city twice), and from the
  other seats' together, the income the production fields leave out;
- `other_cities`, `other_settlements`: the other seats' buildings,
  together;
- the trace of the playout (`TRACE`), from the first move after the
  playout's own: `first_build`, the kind of `seat`'s first build (`BUILDS`:
  0 none, then a road, a settlement, a city or a development card bought),
  `first_build_at` its edge or vertex (-1 for a card or none) and
  `first_build_turn` the turns played before it (-1 for none), and the same
  of its second and third builds (`second_build`, `second_build_at`,
  `second_build_turn`, `third_build`, ...), so the two roads of Road
  Building and the build a traded card goes into can be followed;
  `other_first_settle`, the vertex of the first settlement another seat
  builds (-1 for none); `knights_played`, `cards_bought`: the Knights
  `seat` plays and the development cards it buys; `robber_first`, the
  first hex `seat` moves the robber to (-1 for none).

Every entry is a whole number. A row saved before a field existed reads
back (`upgrade`) with that field at `UNRECORDED`, which no copy ever
records.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hexset.actions import Action, ActionType
from hexset.board.board import pips
from hexset.board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Resource, Terrain
from hexset.game import Game, is_over
from hexset.state import NO_OWNER
from hexset.victory import victory_points

__all__ = ["BUILDS", "FIELDS", "MINE", "MINE_CITY", "NO_TRACE", "OPEN", "RESOURCE_FIELDS", "THEIRS", "THEIRS_CITY",
           "TRACE", "TRACED_BUILDS", "UNRECORDED", "NOTES", "decode_move", "decode_note", "drawn_card", "ending", "field",
           "follow", "holder", "move_code", "note_code", "play_noted", "upgrade", "vertices"]

_NAMES = tuple(r.name.lower() for r in Resource)
#: The per-resource field groups, each in `Resource` order.
RESOURCE_FIELDS = {
    "production": tuple(f"production_{n}" for n in _NAMES),
    "hand": tuple(f"hand_{n}" for n in _NAMES),
    "port": tuple(f"port_{n}" for n in _NAMES),
}
FIELDS = (
    "won", "lost", "turns", "points", "other_points", "longest_road", "other_longest_road",
    "largest_army", "other_largest_army", "discarded", "other_discarded", "knights",
    "cities", "settlements",
    *RESOURCE_FIELDS["production"], "other_production",
    *RESOURCE_FIELDS["hand"], "development_cards",
    "port_generic", *RESOURCE_FIELDS["port"],
    "spots_now", "spots_near", "best_spot",
    "robber", "blocked", "other_blocked", "other_cities", "other_settlements",
    "first_build", "first_build_at", "first_build_turn", "other_first_settle",
    "knights_played", "cards_bought", "robber_first",
    "second_build", "second_build_at", "second_build_turn", "third_build", "third_build_at", "third_build_turn",
)
#: The fields counted while a copy plays (`follow`), in this order, and
#: their values before any move.
TRACE = ("first_build", "first_build_at", "first_build_turn", "other_first_settle",
         "knights_played", "cards_bought", "robber_first",
         "second_build", "second_build_at", "second_build_turn", "third_build", "third_build_at", "third_build_turn")
NO_TRACE = (0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0, -1, -1)
#: The seat's builds the trace follows, in order: each as the fields of its
#: kind (`BUILDS`), its edge or vertex, and the turns played before it.
TRACED_BUILDS = (("first_build", "first_build_at", "first_build_turn"),
                 ("second_build", "second_build_at", "second_build_turn"),
                 ("third_build", "third_build_at", "third_build_turn"))
#: The value `upgrade` gives a field a row was saved without.
UNRECORDED = -2
#: `first_build`'s values: none, then each kind of build.
BUILDS = ("none", "road", "settlement", "city", "development card")
OPEN, MINE, THEIRS, MINE_CITY, THEIRS_CITY = 0, 1, 2, 3, 4
_INDEX = {name: i for i, name in enumerate(FIELDS)}
_BUILT = {ActionType.BUILD_ROAD: 1, ActionType.BUILD_SETTLEMENT: 2, ActionType.BUILD_CITY: 3,
          ActionType.BUY_DEV_CARD: 4}
_FIRST_BUILD, _FIRST_AT, _FIRST_TURN, _OTHER_SETTLE, _KNIGHTS, _BOUGHT, _ROBBER = range(7)
_BUILD_SLOTS = ((_FIRST_BUILD, _FIRST_AT, _FIRST_TURN), (7, 8, 9), (10, 11, 12))


def field(name: str) -> int:
    """The column of a named field in an ending row."""
    return _INDEX[name]


def vertices(rows: np.ndarray) -> np.ndarray:
    """The vertex part of ending rows (the last axis past `FIELDS`)."""
    return rows[..., len(FIELDS):]


def upgrade(rows: np.ndarray, fields) -> np.ndarray:
    """Ending rows saved with the named fields `fields` (an older `FIELDS`),
    laid out as `FIELDS` is now: each field where it is now, a field the rows
    lack at `UNRECORDED`, and the vertex marks after them. Rows already in
    this layout come back as they are."""
    rows = np.asarray(rows)
    fields = list(fields)
    if fields == list(FIELDS):
        return rows
    missing = set(fields) - set(FIELDS)
    if missing:
        raise ValueError(f"ending fields this code does not know: {sorted(missing)}")
    out = np.full((*rows.shape[:-1], len(FIELDS) + rows.shape[-1] - len(fields)), UNRECORDED, dtype=rows.dtype)
    for i, name in enumerate(fields):
        out[..., _INDEX[name]] = rows[..., i]
    out[..., len(FIELDS):] = rows[..., len(fields):]
    return out


def holder(marks: np.ndarray) -> np.ndarray:
    """Vertex marks with a city read as a settlement: `OPEN`, `MINE` or
    `THEIRS` by who holds a building there."""
    marks = np.asarray(marks)
    return np.where(marks == MINE_CITY, MINE, np.where(marks == THEIRS_CITY, THEIRS, marks))


def move_code(actor: int, action: Action, drawn: int | None = None) -> int:
    """One move of a playout as a single number: the seat that made it, its
    action and, for a development card bought, the card it drew (`drawn`, a
    `hexset.cards.DevCard` value). It fits 29 bits: `b + 1` in bits 0 to 8,
    `a + 1` in bits 9 to 20, the action's type in bits 21 to 25 and the seat
    in bits 26 to 28; a card bought carries `drawn + 1` where its `b` is."""
    b = action.b + 1 if drawn is None else drawn + 1
    return (int(actor) << 26) | (int(action.type) << 21) | ((action.a + 1) << 9) | b


def decode_move(code: int) -> tuple[int, Action, int | None]:
    """`move_code`'s seat, action and card drawn (`None` but for a card
    bought)."""
    code = int(code)
    kind = ActionType((code >> 21) & 0x1F)
    a, b = ((code >> 9) & 0xFFF) - 1, (code & 0x1FF) - 1
    if kind is ActionType.BUY_DEV_CARD:
        return code >> 26, Action(kind), b
    return code >> 26, Action(kind, a, b), None


#: The notes a record carries after a move, each a negative number
#: (`note_code`): what the move did that its code cannot say, read off the
#: position around it as the playout plays it (`play_noted`).
#:
#: | note | data |
#: |---|---|
#: | `roll` | the dice total |
#: | `gain` | a seat, a resource and the cards of it a roll paid: `seat << 8 \| resource << 5 \| n` |
#: | `steal` | the resource the robber's mover took |
#: | `take` | the cards a Monopoly took |
#: | `rate` | the cards a bank or port trade gave |
#: | `award` | `award << 8 \| (before + 1) << 4 \| (after + 1)`: longest road (0) or largest army (1) changed hands |
#: | `turn` | the turns played since the playout began, where the move ended a turn |
#: | `end` | the winner, plus one |
NOTES = ("roll", "gain", "steal", "take", "rate", "award", "turn", "end")
_NOTE = {name: k + 1 for k, name in enumerate(NOTES)}
# The moves read around: what they paid, took or changed hands.
_LOOKED = frozenset({ActionType.ROLL, ActionType.MOVE_ROBBER, ActionType.PLAY_MONOPOLY, ActionType.BANK_TRADE,
                     ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.PLAY_KNIGHT})


def note_code(name: str, data: int) -> int:
    """A note (`NOTES`) as one negative number, apart from every move code."""
    return -(1 + ((_NOTE[name] << 24) | int(data)))


def decode_note(code: int) -> tuple[str, int]:
    """`note_code`'s note and data."""
    value = -int(code) - 1
    return NOTES[(value >> 24) - 1], value & 0xFFFFFF


def _seen(game) -> tuple[list[list[int]], int, int, int]:
    """Every seat's hand, the longest road and largest army holders, and
    the last roll (-1 for none)."""
    from hexset.native.playout import NativeCopy

    if type(game) is NativeCopy:
        a = game.state.core.referee_snapshot()["arrays"]
        hands = a["hands"]
        _, road, army = a["state_control"]
        return [list(hands[i:i + 5]) for i in range(0, len(hands), 5)], road, army, a["game_control"][4]
    state = game._state
    return ([list(h) for h in state.hands], state.longest_road_holder, state.largest_army_holder,
            -1 if game.last_roll is None else game.last_roll)


def play_noted(game, actor: int, action: Action, start: int, play) -> list[int]:
    """`action` by `actor` played on `game` (`play()` applies it), as its
    record holds it: its `move_code`, then its notes (`NOTES`); `start` the
    game's turn count when the playout began."""
    kind = action.type
    turns = game.turns
    before = _seen(game) if kind in _LOOKED else None
    held = drawn_card(game, actor) if kind is ActionType.BUY_DEV_CARD else None
    play()
    grew = None
    if held is not None:
        grew = next((k for k, (x, y) in enumerate(zip(held, drawn_card(game, actor))) if y > x), None)
    codes = [move_code(actor, action, grew)]
    if before is not None:
        after = _seen(game)
        hands, was = before[0], after[0]
        if kind is ActionType.ROLL:
            codes.append(note_code("roll", max(0, after[3])))
            for s, (x, y) in enumerate(zip(hands, was)):
                for r, (p, q) in enumerate(zip(x, y)):
                    if q > p:
                        codes.append(note_code("gain", (s << 8) | (r << 5) | min(q - p, 31)))
        elif kind is ActionType.MOVE_ROBBER:
            got = [r for r in range(len(was[actor])) if was[actor][r] > hands[actor][r]]
            if got:
                codes.append(note_code("steal", got[0]))
        elif kind is ActionType.PLAY_MONOPOLY:
            codes.append(note_code("take", max(0, was[actor][action.a] - hands[actor][action.a])))
        elif kind is ActionType.BANK_TRADE:
            codes.append(note_code("rate", max(0, hands[actor][action.a] - was[actor][action.a])))
        for award, (old, new) in enumerate(((before[1], after[1]), (before[2], after[2]))):
            if old != new:
                codes.append(note_code("award", (award << 8) | ((old + 1) << 4) | (new + 1)))
    if game.turns != turns:
        codes.append(note_code("turn", game.turns - start))
    if is_over(game):
        codes.append(note_code("end", game.won_by + 1))
    return codes


def drawn_card(game, seat: int) -> tuple[int, ...]:
    """`seat`'s development cards bought this turn, a count by kind
    (`hexset.cards.DevCard` order): read before and after a purchase, the
    kind that grew is the card drawn."""
    from hexset.native.playout import NativeCopy

    if type(game) is NativeCopy:
        new = game.state.core.referee_snapshot()["arrays"]["new_dev_cards"]
        return tuple(int(n) for n in new[5 * seat: 5 * seat + 5])
    return tuple(int(n) for n in game._state.new_dev_cards[seat])


def follow(trace, action: Action, own: bool, turns: int) -> None:
    """Count `action` into a copy's `trace` (a mutable sequence in `TRACE`
    order, starting from `NO_TRACE`): `own` where the playout's seat made
    it, `turns` the turns played since the playout began."""
    kind = action.type
    if own:
        built = _BUILT.get(kind)
        if built is not None:
            for kind_at, at_at, turn_at in _BUILD_SLOTS:
                if trace[kind_at] == 0:
                    trace[kind_at] = built
                    trace[at_at] = -1 if built == 4 else action.a
                    trace[turn_at] = turns
                    break
            if built == 4:
                trace[_BOUGHT] += 1
        elif kind is ActionType.PLAY_KNIGHT:
            trace[_KNIGHTS] += 1
        elif kind is ActionType.MOVE_ROBBER and trace[_ROBBER] == -1:
            trace[_ROBBER] = action.a
    elif kind is ActionType.BUILD_SETTLEMENT and trace[_OTHER_SETTLE] == -1:
        trace[_OTHER_SETTLE] = action.a


def _read(game) -> SimpleNamespace:
    """A copy's position as plain lists: the board, the pieces, the hands,
    the awards, every seat's full points, the winner and the turn count."""
    from hexset.native.playout import NativeCopy

    if type(game) is NativeCopy:
        a = game.state.core.referee_snapshot()["arrays"]
        robber, road, army = a["state_control"]
        seats = len(a["knights_played"])
        hands = [list(a["hands"][5 * s: 5 * s + 5]) for s in range(seats)]
        held = [sum(a["dev_cards"][5 * s: 5 * s + 5]) + sum(a["new_dev_cards"][5 * s: 5 * s + 5])
                for s in range(seats)]
        p = a["ports"]
        ports = [((p[i + 1], p[i + 2]), None if p[i + 3] == -1 else p[i + 3]) for i in range(0, len(p), 5)]
        return SimpleNamespace(
            topology=game.topology, terrain=list(a["terrain"]), tokens=list(a["tokens"]), ports=ports,
            owners=list(a["vertex_owner"]), buildings=list(a["vertex_building"]), edges=list(a["edge_owner"]),
            robber=robber, road=road, army=army, knights=list(a["knights_played"]), hands=hands, held=held,
            points=list(game.state.referee_points().true_points), won_by=game.won_by, turns=game.turns)
    state = game._state
    board = state.board
    return SimpleNamespace(
        topology=board.topology, terrain=[int(t) for t in board.terrain], tokens=list(board.tokens),
        ports=[(tuple(port.vertices), None if port.resource is None else int(port.resource)) for port in board.ports],
        owners=list(state.vertex_owner), buildings=[int(b) for b in state.vertex_building],
        edges=list(state.edge_owner), robber=state.robber, road=state.longest_road_holder,
        army=state.largest_army_holder, knights=list(state.knights_played),
        hands=[list(h) for h in state.hands],
        held=[sum(state.dev_cards[s]) + sum(state.new_dev_cards[s]) for s in range(state.num_players)],
        points=[victory_points(state, p) for p in range(state.num_players)], won_by=game.won_by, turns=game.turns)


def _production(r: SimpleNamespace, seat: int) -> list[int]:
    out = [0] * NUM_RESOURCES
    for v, owner in enumerate(r.owners):
        if owner != seat:
            continue
        for h in r.topology.vertex_hexes[v]:
            resource = TERRAIN_RESOURCE.get(Terrain(r.terrain[h]))
            if resource is not None and r.tokens[h] and h != r.robber:
                out[resource] += r.buildings[v] * pips(r.tokens[h])
    return out


def _spots(r: SimpleNamespace, seat: int) -> tuple[int, int, int]:
    """Open spots at the end of the seat's roads, open spots one road
    further, and the most pips at any of them."""
    topology = r.topology
    taken = [o != NO_OWNER for o in r.owners]

    def open_(v):
        return not taken[v] and not any(taken[u] for u in topology.vertex_neighbors[v])

    ends = {v for e, owner in enumerate(r.edges) if owner == seat for v in topology.edges[e]}
    reach = {v for v in ends if not taken[v] or r.owners[v] == seat}
    now = {v for v in ends if open_(v)}
    near = set()
    for v in reach:
        for e in topology.vertex_edges[v]:
            if r.edges[e] != NO_OWNER:
                continue
            for u in topology.edges[e]:
                if u != v and u not in now and open_(u):
                    near.add(u)

    def worth(v):
        return sum(pips(r.tokens[h]) for h in topology.vertex_hexes[v]
                   if r.tokens[h] and TERRAIN_RESOURCE.get(Terrain(r.terrain[h])) is not None)

    return len(now), len(near), max((worth(v) for v in now | near), default=0)


def _blocked(r: SimpleNamespace, seat: int) -> tuple[int, int]:
    """The pips the robber's hex keeps from `seat`'s buildings, and from the
    other seats' together."""
    if not 0 <= r.robber < len(r.tokens) or not r.tokens[r.robber]:
        return 0, 0
    if TERRAIN_RESOURCE.get(Terrain(r.terrain[r.robber])) is None:
        return 0, 0
    worth = pips(r.tokens[r.robber])
    mine = theirs = 0
    for v in r.topology.hex_vertices[r.robber]:
        owner = r.owners[v]
        if owner == seat:
            mine += r.buildings[v] * worth
        elif owner != NO_OWNER:
            theirs += r.buildings[v] * worth
    return mine, theirs


def ending(game: Game, seat: int, start: int, discarded: tuple[int, int] = (0, 0),
           trace=NO_TRACE) -> np.ndarray:
    """The ending row of one copy for `seat` (module docstring), `start`
    the turn count the playout began at, `discarded` the cards `seat` and
    the other seats lost to sevens on the way, and `trace` what `follow`
    counted of the playout (`TRACE`)."""
    r = _read(game)
    others = [p for p in range(len(r.points)) if p != seat]
    row = np.zeros(len(FIELDS) + len(r.owners), dtype=np.float32)
    put = lambda name, value: row.__setitem__(_INDEX[name], value)
    put("won", float(r.won_by == seat))
    put("lost", float(r.won_by is not None and r.won_by != seat))
    put("turns", r.turns - start)
    put("points", r.points[seat])
    put("other_points", max((r.points[p] for p in others), default=0))
    put("longest_road", float(r.road == seat))
    put("other_longest_road", float(r.road not in (seat, NO_OWNER)))
    put("largest_army", float(r.army == seat))
    put("other_largest_army", float(r.army not in (seat, NO_OWNER)))
    put("discarded", discarded[0])
    put("other_discarded", discarded[1])
    put("knights", r.knights[seat])
    mine = [b for o, b in zip(r.owners, r.buildings) if o == seat]
    put("cities", sum(1 for b in mine if b == 2))
    put("settlements", sum(1 for b in mine if b == 1))
    for name, value in zip(RESOURCE_FIELDS["production"], _production(r, seat)):
        put(name, value)
    put("other_production", max((sum(_production(r, p)) for p in others), default=0))
    for name, value in zip(RESOURCE_FIELDS["hand"], r.hands[seat]):
        put(name, value)
    put("development_cards", r.held[seat])
    for vertices_, resource in r.ports:
        if any(r.owners[v] == seat for v in vertices_):
            put("port_generic" if resource is None else RESOURCE_FIELDS["port"][resource], 1.0)
    now, near, best = _spots(r, seat)
    put("spots_now", now)
    put("spots_near", near)
    put("best_spot", best)
    put("robber", r.robber)
    blocked, other_blocked = _blocked(r, seat)
    put("blocked", blocked)
    put("other_blocked", other_blocked)
    theirs = [b for o, b in zip(r.owners, r.buildings) if o not in (seat, NO_OWNER)]
    put("other_cities", sum(1 for b in theirs if b == 2))
    put("other_settlements", sum(1 for b in theirs if b == 1))
    for name, value in zip(TRACE, trace):
        put(name, value)
    row[len(FIELDS):] = [OPEN if o == NO_OWNER else
                         (MINE if o == seat else THEIRS) + (2 if b == 2 else 0)
                         for o, b in zip(r.owners, r.buildings)]
    return row
