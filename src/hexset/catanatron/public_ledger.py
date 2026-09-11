"""Opt-in persistent public resource lower bounds for the Catanatron bridge.

This module is deliberately inert unless imported and used by a custom player.
It stores only typed lower bounds; unknown outcomes never read private hands.
"""
from __future__ import annotations
from dataclasses import dataclass
from hexset.ledger import PublicLedger, SeatLedger
from hexset.board.terrain import NUM_RESOURCES

@dataclass
class PublicResourceLedger:
    """Per-game public lower bounds, conservatively updated by named events."""
    ledger: PublicLedger
    game_token: object | None = None

    @classmethod
    def new(cls, players: int, token: object | None = None):
        return cls(PublicLedger([SeatLedger(unknown=0) for _ in range(players)]), token)

    def reset(self, players: int, token: object | None = None) -> None:
        self.ledger = PublicLedger([SeatLedger(unknown=0) for _ in range(players)])
        self.game_token = token

    def initialise(self, hands: list[list[int]], token: object) -> None:
        self.reset(len(hands), token)
        for seat, hand in enumerate(hands):
            self.ledger.seats[seat].unknown = sum(hand)

    def receive(self, seat: int, resource: int, count: int = 1) -> None:
        self.ledger.receive(seat, resource, count)

    def spend(self, seat: int, resource: int, count: int = 1) -> None:
        self.ledger.spend(seat, resource, count)

    def forget(self, seat: int, count: int = 1) -> None:
        """Forget certainty after a hidden steal/discard-like event."""
        row = self.ledger.seats[seat]
        for r in range(NUM_RESOURCES):
            row.known[r] = max(0, row.known[r] - count)

    def snapshot(self) -> PublicLedger:
        return self.ledger.copy()

PUBLIC_ACTIONS = frozenset({'BUILD_ROAD','BUILD_SETTLEMENT','BUILD_CITY','BUY_DEVELOPMENT_CARD','MARITIME_TRADE','ROLL','END_TURN'})
_RESOURCE_INDEX = {'WOOD': 0, 'BRICK': 1, 'SHEEP': 2, 'WHEAT': 3, 'ORE': 4}
def observe_public_roll(book, *, state, dice):
    """Credit provable production for the current public roll only.

    ``state`` is the post-roll state, so its board is exactly the board used
    for this roll.  We deliberately do not replay old rolls against today's
    board: later public builds would otherwise manufacture certainty.  A
    resource whose post-roll bank is empty is skipped because the public log
    cannot establish whether the requested payout was short before the roll.
    """
    if not isinstance(dice, (tuple, list)) or len(dice) != 2:
        return
    try:
        number = int(dice[0]) + int(dice[1])
    except (TypeError, ValueError):
        return
    if number == 7:
        return
    board = getattr(state, "board", None)
    bank = getattr(state, "resource_freqdeck", None)
    tiles = getattr(getattr(board, "map", None), "land_tiles", {})
    buildings = getattr(board, "buildings", {})
    if not tiles or bank is None:
        return
    demand = {}
    for coord, tile in tiles.items():
        if getattr(tile, "number", None) != number:
            continue
        resource = _RESOURCE_INDEX.get(str(getattr(tile, "resource", "")).upper())
        if resource is None or getattr(board, "robber_coordinate", None) == coord:
            continue
        for node_id in getattr(tile, "nodes", {}).values():
            building = buildings.get(node_id)
            if building is None:
                continue
            color, kind = building
            amount = 2 if str(kind).upper().endswith("CITY") else 1
            colors = getattr(state, "colors", ())
            try:
                seat = colors.index(color)
            except (ValueError, AttributeError):
                continue
            demand[(seat, resource)] = demand.get((seat, resource), 0) + amount
    # Check aggregate demand across every seat.  Checking each seat against
    # the same bank count would overclaim when two players jointly request
    # more cards than remain.
    total_demand = {}
    for (_seat, resource), amount in demand.items():
        total_demand[resource] = total_demand.get(resource, 0) + amount
    sufficient = {}
    for resource, amount in total_demand.items():
        try:
            sufficient[resource] = int(bank[resource]) >= amount
        except (KeyError, TypeError, IndexError):
            sufficient[resource] = False
    for (seat, resource), amount in demand.items():
        if sufficient.get(resource, False):
            book.receive(seat, resource, amount)


def observe_public_action(book, *, seat, action_type, value=None, hand_sizes=None):
    if action_type not in PUBLIC_ACTIONS:
        # Hidden outcomes invalidate every typed claim for the affected seat;
        # this reset is unconditional even when the caller has no totals.
        row = book.ledger.seats[seat]
        row.known = [0] * NUM_RESOURCES
        if hand_sizes is not None:
            for i, size in enumerate(hand_sizes):
                book.ledger.seats[i].known = [0] * NUM_RESOURCES
                book.ledger.seats[i].unknown = size
        return
    if action_type in ('ROLL', 'END_TURN'):
        return
    if action_type == 'MARITIME_TRADE':
        names={'WOOD':0,'BRICK':1,'SHEEP':2,'WHEAT':3,'ORE':4}
        for name in value[:4]:
            if name in names: book.spend(seat,names[name],1)
        if value[4] in names: book.receive(seat,names[value[4]],1)
        return
    costs={'BUILD_ROAD':((0,1),(1,1)),'BUILD_SETTLEMENT':((0,1),(1,1),(2,1),(3,1)),'BUILD_CITY':((3,2),(4,3)),'BUY_DEVELOPMENT_CARD':((2,1),(3,1),(4,1))}
    for resource,count in costs[action_type]: book.spend(seat,resource,count)

class PublicLedgerPlayerMixin:
    _public_production = False

    """Opt-in mixin for ``DevCatanPlayer`` with a persistent ledger hook."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._public_book = None
        self._public_token = None
        self._public_history_cursor = 0

    def reset_state(self):
        super().reset_state()
        self._public_book = None
        self._public_token = None
        self._public_history_cursor = 0

    def _translation_ledger(self, game):
        from catanatron.state_functions import player_num_resource_cards
        token = getattr(game, "id", id(game))
        n = len(game.state.colors)
        records = getattr(game.state, "action_records", [])
        if self._public_token != token or len(records) < self._public_history_cursor:
            self._public_book = PublicResourceLedger.new(n, token)
            hands = [[0, 0, 0, 0, 0] for _ in game.state.colors]
            self._public_book.initialise(hands, token)
            # Initial cards are public only by total, never by type.
            for i, color in enumerate(game.state.colors):
                self._public_book.ledger.seats[i].unknown = player_num_resource_cards(game.state, color)
            self._public_token = token
            self._public_history_cursor = len(records)
        for record in records[self._public_history_cursor:]:
            action = getattr(record, "action", None)
            color = getattr(action, "color", None)
            if color in game.state.colors:
                action_type = getattr(getattr(action, "action_type", None), "name", "")
                observe_public_action(self._public_book, seat=game.state.colors.index(color),
                                      action_type=action_type,
                                      value=getattr(action, "value", None),
                                      hand_sizes=[player_num_resource_cards(game.state, c)
                                                  for c in game.state.colors])
                # Only the newest record can safely use the current board.
                if self._public_production and action_type == "ROLL" and record is records[-1]:
                    observe_public_roll(self._public_book, state=game.state,
                                        dice=getattr(record, "result", None))
        self._public_history_cursor = len(records)
        for i, color in enumerate(game.state.colors):
            size = player_num_resource_cards(game.state, color)
            row = self._public_book.ledger.seats[i]
            excess = max(0, sum(row.known) - size)
            for r in sorted(range(NUM_RESOURCES), key=lambda x: row.known[x], reverse=True):
                take = min(excess, row.known[r]); row.known[r] -= take; excess -= take
            row.unknown = size - sum(row.known)
        return self._public_book.snapshot()

try:
    from hexset.catanatron.player import DevCatanPlayer
    class PublicLedgerPlayer(PublicLedgerPlayerMixin, DevCatanPlayer):
        """Opt-in DCL bridge; ``DC`` remains the default bridge."""
        pass

    class PublicProductionLedgerPlayer(PublicLedgerPlayerMixin, DevCatanPlayer):
        """DCP: conservative current-roll public production extension."""
        _public_production = True
except ImportError:  # optional Catanatron dependency
    PublicLedgerPlayer = None

if PublicLedgerPlayer is not None:
    try:
        from catanatron.cli.cli_players import register_cli_player
        register_cli_player("DCL", PublicLedgerPlayer)
        register_cli_player("DCP", PublicProductionLedgerPlayer)
    except ImportError:
        pass
