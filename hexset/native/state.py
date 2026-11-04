# SPDX-License-Identifier: GPL-3.0-only
"""Owned native state, exact Chance streams and separate referee access."""
from __future__ import annotations

import copy
import dataclasses
import random
from operator import index

try:
    from .._native_core import ArenaState
except ImportError:
    ArenaState = None

from hexset.board.board import Board, make_board
from hexset.board.coords import Hex
from hexset.board.ports import Port
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.cards import DevCard
from hexset.chance import DiceDeck, Recording, Scripted, Live
from hexset.game import Game, Phase
from hexset.ledger import PublicLedger, SeatLedger
from hexset.rules import Rules
from hexset.state import Building, GameState, HiddenRead, is_hidden


__all__ = ["NativeState", "NativeUnavailable", "available", "supported",
           "VisibleCardFacts", "RefereePoints"]


GAME_COLUMNS = (
    "phase", "current_player", "setup_step", "last_settlement", "last_roll",
    "dev_card_played", "free_roads", "resume_phase", "turns", "won_by", "trades_made",
    "counter_steps", "trade_event_turn", "trade_mode", "trade_rule", "open_offers",
    "first", "turn_cap", "gates",
)
EMPTY_GAME_FIELDS = ("robber_promised", "deal_payments", "carried_deals", "deal_log",
                     "trades", "shown", "open_asks", "pending")
LIST_FIELDS = ("setup_queue", "discard_quota")
SET_FIELDS = ("robber_allowed", "locked", "before_roll")
STATE_ARRAYS = ("vertex_owner", "vertex_building", "edge_owner", "hands", "bank", "deck",
                "dev_cards", "new_dev_cards", "knights_played", "dev_cards_played", "road_lengths")
ADJACENCIES = ("hex_vertices", "hex_edges", "hex_neighbors", "vertex_hexes", "vertex_edges",
               "vertex_neighbors", "edge_hexes")
TRADE_MODES = ("round", "auto", "external")
TRADE_RULES = ("egalitarian", "nash", "actor")


def _copy_streams(*objects):
    """`copy.deepcopy(objects)`, with every plain `random.Random` among them
    (to three attribute levels deep, shared ones kept shared) copied by its
    state rather than element by element: the same streams, without
    deepcopy's walk over each generator's state tuple."""
    memo = {}

    def seed(obj, depth):
        if type(obj) is random.Random:
            if id(obj) not in memo:
                clone = random.Random()
                clone.setstate(obj.getstate())
                memo[id(obj)] = clone
        elif depth < 3 and hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                seed(value, depth + 1)

    for obj in objects:
        seed(obj, 0)
    return copy.deepcopy(objects, memo)


_CANONICAL: dict[int, tuple[object, bool]] = {}


def _canonical(topology) -> bool:
    """Whether `topology` is indexed as `build_topology` builds it, worked out
    once per topology (by identity)."""
    held = _CANONICAL.get(id(topology))
    if held is not None and held[0] is topology:
        return held[1]
    ok = build_topology(topology.hexes) == topology
    if len(_CANONICAL) >= 64:
        _CANONICAL.pop(next(iter(_CANONICAL)))
    _CANONICAL[id(topology)] = (topology, ok)
    return ok


def _flat(rows):
    return tuple(index(value) for row in rows for value in row)


def _ragged(rows):
    offsets, values = [0], []
    for row in rows:
        values.extend(index(value) for value in row)
        offsets.append(len(values))
    return tuple(offsets), tuple(values)


def _split(values, offsets):
    return [list(values[a:b]) for a, b in zip(offsets, offsets[1:])]


def _row(row):
    if {f.name for f in dataclasses.fields(row)} != {"known", "unknown", "want", "waste", "most"}:
        raise ValueError("SeatLedger schema changed; update native snapshot contract")
    return (*map(index, row.known), index(row.unknown), *map(index, row.want),
            *map(index, row.waste), *map(index, row.most))


def _ledger(ledger):
    if {f.name for f in dataclasses.fields(ledger)} != {"seats", "private"}:
        raise ValueError("PublicLedger schema changed; update native snapshot contract")
    private = None
    if ledger.private is not None:
        identities = [id(own) for own in ledger.private if own is not None]
        if len(identities) != len(set(identities)) or id(ledger) in identities:
            raise ValueError("aliased private ledgers are not a native referee state")
        if any(own is not None and own.private is not None for own in ledger.private):
            raise ValueError("nested private ledger is unsupported")
        private = tuple(None if own is None else tuple(_row(row) for row in own.seats)
                        for own in ledger.private)
    return dict(common=tuple(_row(row) for row in ledger.seats), private=private)


def _restore_rows(rows):
    return [SeatLedger(known=list(row[:5]), unknown=row[5], want=list(row[6:11]),
                       waste=list(row[11:16]), most=list(row[16:21])) for row in rows]


def _restore_ledger(ledger):
    return PublicLedger(_restore_rows(ledger["common"]),
        None if ledger["private"] is None else
        [None if rows is None else PublicLedger(_restore_rows(rows)) for rows in ledger["private"]])


def capture_referee(game: Game):
    """Copy all supported game data into integer-only construction buffers.

    Counts are never silently rounded: integer_index refuses float inputs.
    Unsupported trading/deal state, hidden referee piles or schema drift fails
    before creating the native object. No public View is weakened to fit it.
    """
    expected_game = (set(GAME_COLUMNS) | set(EMPTY_GAME_FIELDS) | set(LIST_FIELDS)
                     | set(SET_FIELDS) | {"_state", "rng", "chance", "ledger", "open_deal", "embargoes"})
    if {f.name for f in dataclasses.fields(game)} != expected_game:
        raise ValueError("Game schema changed; update native snapshot contract")
    expected_state = set(STATE_ARRAYS) | {"board", "num_players", "robber", "dev_ages",
        "longest_road_holder", "largest_army_holder", "rules", "dice"}
    state = game._state  # true state: referee import only, never used as policy input.
    if {f.name for f in dataclasses.fields(state)} != expected_state:
        raise ValueError("GameState schema changed; update native snapshot contract")
    if state.num_players not in (2, 4):
        raise ValueError("native state supports two/four seats")
    if game.open_deal is not None or any(getattr(game, field) for field in EMPTY_GAME_FIELDS):
        raise ValueError("native state refuses active trading/deal state")
    if game.gates is not None and (len(game.gates) != state.num_players or any(g is not None for g in game.gates)):
        raise ValueError("native state refuses trading gates")
    for field in ("hands", "dev_cards", "new_dev_cards"):
        if any(is_hidden(pile) for pile in getattr(state, field)):
            raise HiddenRead("native referee import requires concrete piles; visible facts use separate accessors")
    if is_hidden(state.deck):
        raise HiddenRead("native referee import requires a concrete deck")

    board, arrays = state.board, {"players": (index(state.num_players),)}
    if {f.name for f in dataclasses.fields(board)} != {"topology", "terrain", "tokens", "hexes_by_roll", "ports"}:
        raise ValueError("Board schema changed")
    topology = board.topology
    if not _canonical(topology):
        raise ValueError("native import requires canonical HexSet topology indices")
    arrays.update(hexes=_flat(topology.hexes), terrain=tuple(map(index, board.terrain)),
                  tokens=tuple(map(index, board.tokens)), edges=_flat(topology.edges),
                  ports=_flat((p.edge, *p.vertices, -1 if p.resource is None else p.resource, p.ratio)
                              for p in board.ports))
    for name in ADJACENCIES:
        arrays[name + "_offsets"], arrays[name + "_values"] = _ragged(getattr(topology, name))
    arrays["roll_hexes_offsets"], arrays["roll_hexes_values"] = _ragged(board.hexes_by_roll)
    for name in STATE_ARRAYS:
        values = getattr(state, name)
        arrays[name] = (_flat(values) if name in ("hands", "dev_cards", "new_dev_cards")
                        else tuple(map(index, values)))
    arrays["state_control"] = (index(state.robber), index(state.longest_road_holder), index(state.largest_army_holder))
    # Enum-vs-int representation varies between a live deal and a JSON replay.
    # The native rules use integer kinds; debug restore preserves both exactly.
    arrays["building_enum"] = tuple(int(isinstance(v, Building)) for v in state.vertex_building)
    arrays["deck_enum"] = tuple(int(isinstance(v, DevCard)) for v in state.deck)
    arrays["age_known"] = tuple(int(ages is not None) for ages in state.dev_ages)
    arrays["age_offsets"], arrays["age_values"] = _ragged(() if ages is None else ages for ages in state.dev_ages)
    if {f.name for f in dataclasses.fields(state.rules)} != {"winning_points", "discard_limit", "friendly_robber", "balanced_dice"}:
        raise ValueError("Rules schema changed")
    arrays["rules"] = (index(state.rules.winning_points), index(state.rules.discard_limit),
                       int(state.rules.friendly_robber), int(state.rules.balanced_dice))
    dice = state.dice
    if dice is not None and {f.name for f in dataclasses.fields(dice)} != {"left", "cards", "recent", "sevens", "run"}:
        raise ValueError("DiceDeck schema changed")
    arrays["dice_control"] = ((0, 0, -1, 0) if dice is None else
                              (1, index(dice.cards), -1 if dice.run[0] is None else index(dice.run[0]), index(dice.run[1])))
    arrays["dice_left"] = () if dice is None else tuple(map(index, dice.left))
    arrays["dice_recent"] = () if dice is None else tuple(map(index, dice.recent))
    arrays["dice_sevens"] = () if dice is None else _flat(dice.sevens)
    control = []
    for name in GAME_COLUMNS:
        value = getattr(game, name)
        if name == "trade_mode": value = TRADE_MODES.index(value)
        elif name == "trade_rule": value = TRADE_RULES.index(value)
        elif name == "gates": value = int(value is not None)
        elif value is None: value = -1
        control.append(index(value))
    arrays["game_control"] = tuple(control)
    for name in LIST_FIELDS:
        arrays[name] = tuple(map(index, getattr(game, name)))
    for name in SET_FIELDS:
        value = getattr(game, name)
        arrays[name] = (-1,) if value is None else tuple(sorted(map(index, value)))
    arrays["embargoes"] = _flat(sorted(game.embargoes))
    return dict(version=1, epoch=0, arrays=arrays, ledger=_ledger(game.ledger))


def _restore_board(arrays) -> Board:
    hexes = arrays["hexes"]
    topology = build_topology(Hex(*hexes[i:i+3]) for i in range(0, len(hexes), 3))
    ports = arrays["ports"]
    board = make_board(topology, tuple(Terrain(t) for t in arrays["terrain"]), tuple(arrays["tokens"]),
        tuple(Port(edge=ports[i], vertices=tuple(ports[i+1:i+3]),
                   resource=None if ports[i+3] == -1 else Resource(ports[i+3]), ratio=ports[i+4])
              for i in range(0, len(ports), 5)))
    for name in ADJACENCIES:
        if _ragged(getattr(topology, name)) != (arrays[name+"_offsets"], arrays[name+"_values"]):
            raise ValueError("native snapshot topology differs from canonical HexSet indices")
    if _flat(topology.edges) != arrays["edges"] or _ragged(board.hexes_by_roll) != (
            arrays["roll_hexes_offsets"], arrays["roll_hexes_values"]):
        raise ValueError("native snapshot board indexes differ")
    return board


def restore_referee(snapshot, rng, chance) -> Game:
    """Rebuild independent referee-only debug state using caller-owned streams."""
    arrays = snapshot["arrays"]
    players = arrays["players"][0]
    fields = dict(board=_restore_board(arrays), num_players=players)
    for name in STATE_ARRAYS:
        values = arrays[name]
        fields[name] = ([list(values[i:i+5]) for i in range(0, len(values), 5)]
                        if name in ("hands", "dev_cards", "new_dev_cards") else list(values))
    fields.update(zip(("robber", "longest_road_holder", "largest_army_holder"), arrays["state_control"]))
    fields["vertex_building"] = [Building(v) if flag else v for v, flag in zip(fields["vertex_building"], arrays["building_enum"])]
    fields["deck"] = [DevCard(v) if flag else v for v, flag in zip(fields["deck"], arrays["deck_enum"])]
    ages = _split(arrays["age_values"], arrays["age_offsets"])
    fields["dev_ages"] = [ages[s] if arrays["age_known"][s] else None for s in range(players)]
    win, discard, friendly, balanced = arrays["rules"]
    fields["rules"] = Rules(win, discard, bool(friendly), bool(balanced))
    present, cards, owner, length = arrays["dice_control"]
    fields["dice"] = None if not present else DiceDeck(
        tuple(arrays["dice_left"]), cards, tuple(arrays["dice_recent"]),
        tuple(zip(arrays["dice_sevens"][::2], arrays["dice_sevens"][1::2])),
        (None if owner == -1 else owner, length))
    state = GameState(**fields)
    control = dict(zip(GAME_COLUMNS, arrays["game_control"]))
    for name in ("phase", "resume_phase"): control[name] = Phase(control[name])
    for name in ("dev_card_played", "open_offers"): control[name] = bool(control[name])
    for name in ("last_roll", "won_by"):
        if control[name] == -1: control[name] = None
    control["trade_mode"] = TRADE_MODES[control["trade_mode"]]
    control["trade_rule"] = TRADE_RULES[control["trade_rule"]]
    control["gates"] = (None,) * players if control["gates"] else None
    for name in LIST_FIELDS: control[name] = list(arrays[name])
    for name in SET_FIELDS:
        values = arrays[name]
        control[name] = None if values == (-1,) else frozenset(DevCard(v) if name == "before_roll" else v for v in values)
    control["embargoes"] = frozenset(zip(arrays["embargoes"][::2], arrays["embargoes"][1::2]))
    return Game(_state=state, rng=rng, chance=chance, ledger=_restore_ledger(snapshot["ledger"]), **control)


@dataclasses.dataclass(frozen=True)
class VisibleCardFacts:
    """Seat-visible card totals, own cards and resource ledger bounds."""
    perspective: int
    own_hand: tuple[int, ...]
    own_dev: tuple[int, ...]
    own_new_dev: tuple[int, ...]
    hand_sizes: tuple[int, ...]
    dev_sizes: tuple[int, ...]
    new_dev_sizes: tuple[int, ...]
    deck_size: int
    ledger: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class RefereePoints:
    """Referee-only victory points, separate from visible observations."""
    true_points: tuple[int, ...]


class NativeState:
    """Persistent native arrays/ledger plus separately owned original Chance.

    The object owns its integer state, ledger and copied Chance streams.
    Referee snapshots and true points must stay outside deciding requests.
    """
    def __init__(self, game):
        if ArenaState is None:
            raise NativeUnavailable("the optional native extension is unavailable")
        snapshot = capture_referee(game)
        source = game.chance
        while isinstance(source, Recording):
            source = source.inner
        if not isinstance(source, (Scripted, Live)):
            raise ValueError("native state accepts only offline Chance implementations")
        self.core = ArenaState(snapshot)
        self._failed = None
        # Copy BOTH together: preserve any internal RNG sharing but never alias
        # the oracle's root RNG, dice/steal generators or mutable event lists.
        self._rng, self._chance = _copy_streams(game.rng, game.chance)

    def fork(self):
        self._healthy()
        child = object.__new__(type(self))
        child.core = self.core.clone()
        child._failed = None
        child._rng, child._chance = _copy_streams(self._rng, self._chance)
        return child

    def restore_referee(self):
        rng, chance = copy.deepcopy((self._rng, self._chance))
        return restore_referee(self.core.referee_snapshot(), rng, chance)

    def visible_card_facts(self, perspective):
        return VisibleCardFacts(index(perspective), **self.core.visible_cards(index(perspective)),
                                ledger=self.core.visible_ledger(index(perspective)))

    def referee_points(self):
        return RefereePoints(self.core.privileged_true_points())

    def apply(self, action, seat=None):
        """Apply one atomic no-trade action without rebuilding a Python Game.

        Compound option expansion, record emission and batching are separate
        boundary responsibilities. This method never starts a game.
        """
        self._healthy()
        kind, a, b = map(index, action)
        try:
            self.core.apply_atomic(kind, a, b, -1 if seat is None else index(seat), self.chance_event)
        except BaseException as error:
            self._failed = error
            raise

    def chance_event(self, kind, argument=None):
        """Referee-only bridge: called once per actual random event, not feature.

        The native transition must update its public DiceDeck and game state
        after the returned event; this method alone does not apply a move.
        Recording retains the original exact event protocol and RNG streams.
        """
        if kind == "roll":
            return self._chance.roll_by(None if argument is None else index(argument))
        if kind == "steal":
            return self._chance.steal(tuple(map(index, argument)))
        if kind == "deck":
            return self._chance.deck_order(list(map(index, argument)))
        raise ValueError(f"unsupported chance callback: {kind!r}")

    def _healthy(self):
        if self._failed is not None:
            raise RuntimeError("native state failed; do not retry an applied move or Chance event") from self._failed


class NativeUnavailable(ImportError):
    """The optional extension could not be imported."""


def available():
    """Whether the optional owned-state extension is installed."""
    return ArenaState is not None


def supported(game, contract="9"):
    """Check the supported no-trade referee shape before choosing a backend."""
    if not available() or contract not in ("9", "9.1", "9d", "9d.1"):
        return False
    if contract in ("9d", "9d.1") and game.num_players != 2:
        return False
    try:
        capture_referee(game)
        source = game.chance
        while isinstance(source, Recording):
            source = source.inner
        return isinstance(source, (Scripted, Live))
    except (ValueError, TypeError, HiddenRead):
        return False
