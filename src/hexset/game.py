# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Sequence

from .board.board import MAX_ROLL, MIN_ROLL, Board, pips
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Resource
from .cards import ROAD_BUILDING_ROADS, DevCard
from .chance import Chance, Live
from .devcards import buy as buy_dev_card
from .devcards import (
    mature,
    play_knight,
    play_monopoly,
    play_year_of_plenty,
    spend_card,
)
from .economy import Purchase, bank_trade, distribute, pay
from .ledger import PublicLedger
from .robber import discard, discard_count, move_robber, steal
from .trading import Bundle, Trade, checked_valuation, execute_trade, judged, trade_event
from .state import (
    NO_OWNER,
    GameState,
    can_place_road,
    copy_state,
    new_game,
    place_road,
    place_settlement,
    upgrade_to_city,
)
from .victory import WINNING_POINTS, update_largest_army, update_longest_road, victory_points

if TYPE_CHECKING:
    from .view import View

DICE = 6
MAX_TURNS = 1000

ROLL_ODDS: tuple[tuple[int, float], ...] = tuple(
    (roll, pips(roll) / DICE**2) for roll in range(MIN_ROLL, MAX_ROLL + 1)
)


class Phase(IntEnum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL = 2
    DISCARD = 3
    ROBBER = 4
    MAIN = 5
    GAME_OVER = 6


@dataclass
class Game:
    _state: GameState
    rng: random.Random
    # The one chance source (`hexset.chance`): every random draw the engine
    # makes -- the shuffled deck, a roll, a steal, a discard -- goes through
    # this, never through `rng` directly any more. `rng` itself stays: it is
    # what `Live` (the default `chance`) draws from, and it is still handed
    # to `imagine` to seed a search's own copy (`Chance` has no notion of
    # "the same stream, advanced" -- a search needs an actual generator to
    # branch from). Defaults to `Live(rng)`, so a caller that never heard of
    # `chance` gets exactly today's behaviour.
    chance: Chance
    ledger: PublicLedger
    phase: Phase = Phase.SETUP_SETTLEMENT
    current_player: int = 0
    setup_queue: list[int] = field(default_factory=list)
    setup_step: int = 0
    last_settlement: int = -1
    last_roll: int | None = None
    dev_card_played: bool = False
    discard_quota: list[int] = field(default_factory=list)
    free_roads: int = 0
    turns: int = 0
    won_by: int | None = None
    # Every seat's public valuation vector (`hexset.trading`): what each
    # resource is worth to that seat right now, in [-1, 1], positive for
    # "I want more". All-zero at `start()` -- a seat trades only once it has
    # published -- and written only by `publish`, at whichever driver is
    # stepping the game for that seat's own decisions. `trade_event` reads
    # this and never writes it.
    valuations: list[tuple[float, ...]] = field(default_factory=list)
    # This turn's executed trades and their count, cleared by `end_turn` the
    # way the offer counter was. `trades_made` is the recorded statistic;
    # `max_trades` is the knob, and `0` is the off switch for the no-trade
    # referents (a mode, not a budget -- `None`, unbounded, is the default:
    # the event stops when nothing clears, not when a counter runs out).
    trades: list[Trade] = field(default_factory=list)
    trades_made: int = 0
    max_trades: int | None = None
    # Candidates the current player's trade event found against a manual
    # (human/LLM) seat's `PendingGate` (`hexset.server.webplay`) -- a
    # snapshot of the *last* event, not an accumulating log like `trades`.
    # Recomputed (cleared, then repopulated as `trade_event` runs) at the
    # start of every trade event and cleared again by `end_turn`, so nothing
    # pending survives a change of hands it was computed against (PI
    # ratification, `docs/negotiation-interface.md`, decision 2). Not copied
    # by `imagine`, for the same reason `gates` is not: a hypothetical must
    # not leak a real seat's pending offers, and a search's own copy never
    # seats a `PendingGate` anyway.
    pending: list[Trade] = field(default_factory=list)
    # Whether this turn's *first* trade event is still waiting to run.
    # `enter_main` sets this and no longer runs the event itself -- the PI
    # amendment "publish points and the event trigger"
    # (`agents/reference/trading-design.md`): the event runs lazily, the
    # first time the current player's own `legal_actions(game)`,
    # `game.state(seat)` or `Game.publish` is reached, whichever comes
    # first (`run_pending_event`, called from all three). That is what lets
    # a driver that publishes before it ever asks for its own legal actions
    # trade on the vector it just published, while a seat that never
    # publishes (an idle human) still gets its event, on whatever vector is
    # already standing, the first time anything observes the game for it.
    # Every event after the first one in a turn (after every MAIN action,
    # `run_trade_event`'s other call sites) runs unconditionally and does
    # not touch this flag. `imagine` copies it so a search stepping its own
    # copy through simulated turns keeps arming and consuming it the same
    # way -- `run_pending_event` is a no-op whenever `gates is None`, which
    # every imagined copy's `gates` is, so a copy's own `legal_actions`
    # call never re-runs a live event with the copy's stand-in gates.
    event_pending: bool = False
    # Whether the current player still owes this turn's publish
    # (`publish_due`'s own flag, deliberately separate from `event_pending`
    # above). `enter_main` sets it, `Game.publish` clears it for the current
    # player. Regression this decoupling fixes: the server's own
    # `state_view` reads *any* viewer's poll through `game.state(current)`
    # to trigger the pending event (one of the three trigger points), so a
    # spectator's `/api/state` -- or an acting bot's own poll of its state
    # before it decides -- could consume `event_pending` before the current
    # player ever published, and `publish_due` used to be defined as
    # "`event_pending` still true", so it went permanently false for that
    # turn the instant anybody merely looked. Keying `publish_due` off this
    # flag instead means an observation can still fire the event early (the
    # turn's first event may run on this seat's standing vector from its own
    # last turn, same as it always could -- `run_pending_event`'s docstring
    # on why that is not fixed too) without ever making a still-unpublished
    # seat's publish look moot: the seat's fresh vector reaches every event
    # after that first one, the same turn, once it does publish. `imagine`
    # copies it for the same reason it copies `event_pending`.
    awaiting_publish: bool = False
    # Who answers a private gate, one per seat -- the driver's own bots
    # (`arena.play`), the gym's opponents, the server's seated players, or a
    # search's stand-in for the whole table. Gate-only: publishing a
    # valuation is a separate act (`publish`, called by the driver at that
    # seat's own decisions), not something `trade_event` asks this object
    # for any more. `None` means nobody trades, which is what a bare
    # `start()` game does until a driver seats somebody.
    #
    # **`imagine` deliberately does not copy this**, so a hypothetical never
    # trades. Two reasons, one of principle and one of cost. Principle: a
    # search must not reach the *real* opponents' private gates, any more
    # than it may read their hands, and the alternative -- every search
    # installing its own stand-in for the whole table -- is one forgotten
    # call site away from that leak. Cost: the gate is a position
    # evaluation, so re-clearing the event at every node would multiply the
    # evaluator by the branching factor to re-derive an exchange both seats
    # have already agreed improves them. A bot's trading judgement is
    # exercised for real, at every event, through `accepts`; the tree plans
    # the position that judgement hands it.
    gates: tuple[object, ...] | None = None
    # Seats retired from the game: skipped by the setup snake, skipped by
    # turn rotation, never `to_move`, never asked for a trade. Empty for every
    # game this module deals unless a caller retires a seat with `lock_seat`,
    # so a game that never locks anyone is unaffected byte-for-byte -- every
    # place this dataclass is read below checks `locked` only after the
    # ordinary computation, and an empty frozenset changes nothing it touches.
    #
    # This is `hexset.server.seating`'s per-seat setup lock, upstreamed: that
    # module built the same thing as a correction bolted onto a live `Game`
    # (`game.locked` as a plain, undeclared attribute) because this field did
    # not exist, and documented the one place that made wrong: `imagine` did
    # not know to copy an attribute it never declared, so a search forward
    # from a table with a retired seat simulated turns for it anyway
    # (`docs/engine-divergence-2026-09-02.md`, request R2). Declaring the
    # field here and having `imagine` copy it (below) is the fix; nothing
    # else about `hexset.server.seating`'s policy -- *when* a seat retires, or
    # that only a still-empty seat ever does -- belongs in the engine, which
    # is why `lock_seat` places no restriction on which seat or when.
    locked: frozenset[int] = field(default_factory=frozenset)
    # The seat `start()` opened the setup snake at (its own `first` argument,
    # `start=0` the default). Read-only after `start`: nothing in this module
    # writes it again, since the snake's start doesn't move once the table is
    # dealt. Exists as its own field, not derived from `setup_queue[0]`
    # (which happens to equal it, since the queue is built from `first` and
    # never mutated), because `hexset.record.Record` needs to carry it
    # explicitly to rebuild the same snake on replay -- `setup_queue` is not
    # itself part of a record, so there is nothing to derive it from once a
    # game is reconstructed from one. `imagine` copies it for the same
    # reason it copies `setup_queue`: a hypothetical is still the same table.
    first: int = 0

    def state(self, seat: int, *, hidden: bool = True) -> GameState | View:
        """The access path to this game's state (P0,
        `agents/reference/trading-design.md`, "Registration -- the
        one-event trade mechanic").

        `hidden=True` (the default) returns `seat`'s information-set `View`
        -- known/unknown hands, expected hands, hold probabilities,
        `sample` -- built exactly as `hexset.bots.heximax.belief.Belief.
        from_game` built it before this moved into the engine. `hidden=False`
        returns the true `GameState` (the raw field, private otherwise as
        `_state`): the same object every time, never a copy, so reading it
        costs nothing and mutating it through the returned reference works
        exactly as mutating `_state` always did. This is the only sanctioned
        way to read the true state from outside the engine; the three
        sanctioned callers are `hexset.bots.search2`, heximax's own
        `omniscient` mode, and the Catanatron adapter when it hosts a
        Catanatron bot.

        One of the three event-trigger points (`event_pending`'s
        docstring), for `hidden=True` only: building the current player's
        own information-set view is what "the seat observes the game to
        decide or publish" means, and fires this turn's pending first trade
        event, if one is still outstanding, before that view is built -- so
        what comes back already reflects the event, the same as it would
        for any observation taken after the event ran eagerly. A
        `hidden=False` read does *not* trigger it: that path is for reading
        the true state for a reason unrelated to this seat's own turn --
        `search2`'s search still reaches the trigger through its own
        `legal_actions` call, never only through this -- and letting it
        trigger here besides was a real bug: `hexset.bench.aivat.
        chance_outcomes` re-seats a *hypothetical* child with the real
        seated bots' gates for scoring purposes, and something as innocuous
        as a value function reading `child.state(0, hidden=False)` to hash
        the position was enough to fire a live trade event -- using the
        real seated bots' own judgement -- as a side effect of being asked
        to score a position, measurably diverging the real game from what
        it would otherwise have played (found via `test_aivat.py`'s
        exact-replay gate; the precise channel back into the bots was not
        pinned down further, since not triggering here at all removes it).
        """
        if hidden and seat == self.current_player:
            run_pending_event(self)
        if not hidden:
            return self._state
        from .view import View as _View

        return _View.from_game(self, seat)

    @property
    def num_players(self) -> int:
        """How many seats this game has. Public to everyone, always: it is a
        property of the table, not of anybody's hand, so reading it needs no
        view and is not an omniscient read."""
        return self._state.num_players

    def set_state(self, state: GameState) -> None:
        """Replace the true state outright.

        The one write a determinizer needs (`Heximax.worlds`'s PIMC sample,
        a session's undo) without exposing `_state` itself to code outside
        the engine.
        """
        self._state = state

    def publish_due(self, seat: int) -> bool:
        """Whether `seat` should publish its valuation right now.

        True exactly once per seat per turn: while `seat` is the current
        player, the phase is `MAIN`, and `seat` has not published since
        `enter_main` armed this turn (`awaiting_publish`). A driver checks
        this before calling `hexset.trading.publish_valuation` so a seat
        publishes once, at the engine-defined post-roll/robber point,
        rather than after every action -- the PI amendment that replaced
        "publish after every action", which measured 8.4x collection cost
        for an event that can only ever observe two publishes a turn
        (`agents/reference/trading-design.md`). A human seat is unaffected:
        it publishes whenever it likes through the server's endpoint,
        `publish_due` or not.

        Deliberately keyed off `awaiting_publish`, not `event_pending`: an
        observation (the server's `state_view`, polled by a spectator or by
        an acting bot's own runner before it decides) can fire this turn's
        first trade event -- one of the three trigger points,
        `run_pending_event`'s docstring -- before the current player has
        published, and used to make `publish_due` false for the rest of the
        turn as a result, so the seat's own publish, moments later, never
        happened. `awaiting_publish` only ever goes false when this seat
        itself publishes, so an event firing early on a still-unpublished
        vector no longer stops the publish that should have preceded it.
        """
        return (
            seat == self.current_player
            and self.phase is Phase.MAIN
            and self.awaiting_publish
        )

    def publish(self, seat: int, vector: Sequence[float]) -> None:
        """Set `seat`'s public valuation vector (`hexset.trading`).

        The driver's job, not the engine's own: called at `seat`'s own
        decision, once a turn while `publish_due(seat)` is true
        (`hexset.trading.publish_valuation` is the usual way in), never
        from inside `trade_event`, which only reads `game.valuations` --
        calling a seat's `valuation` fresh at event time was tried and cost
        one forward per seat per lane per turn in a batched collector, the
        whole of what this split avoids. Validates length and range
        (`hexset.trading.checked_valuation`) and records nothing else: no
        ledger entry, no trade, just the number a seat is currently
        standing behind.

        One of the three event-trigger points (`event_pending`'s
        docstring): when `seat` is the current player, this clears
        `awaiting_publish` (so `publish_due` reports this seat's turn-scoped
        publish as done) and then fires this turn's pending first trade
        event, if one is still outstanding, *after* the new vector is
        recorded -- so a driver that publishes before it ever observes the
        game trades on the vector it just published, not the one standing
        from its last turn.
        """
        self.valuations[seat] = checked_valuation(vector, seat)
        if seat == self.current_player:
            self.awaiting_publish = False
            run_pending_event(self)

    def execute_trade(self, proposer: int, counterparty: int, bundle: Bundle) -> Trade:
        """A manually composed exchange between `proposer` and `counterparty`
        (`docs/negotiation-interface.md` §1), bypassing `_candidates`/
        `_best_clearing` entirely -- `bundle` is taken as composed, signed
        positive towards `proposer`, so it may be any exchange both sides can
        cover, not only what the automatic event would have enumerated.

        See `hexset.trading.execute_trade` for the checks this runs (coverage,
        the counterparty's public surplus as a hard rule, the counterparty's
        private gate) and what it raises `ValueError` for. Submitting is the
        proposer's own consent -- their gate is never asked, and their public
        surplus is never checked either (PI ratification, decision 4).
        """
        return execute_trade(self, proposer, counterparty, bundle)


def start(
    board: Board,
    num_players: int,
    rng: random.Random | None = None,
    *,
    first: int = 0,
    chance: Chance | None = None,
) -> Game:
    """Start a game. `first` chooses which seat opens the setup snake and
    therefore takes the first real turn; `first=0` (the default) is today's
    behaviour exactly. Rotating the snake's start is `hexset.server.seating.start_at`
    upstreamed -- a gym seats its creator at a random index and still wants the
    snake's compensating property (whoever places first in round one places
    last in round two), which holds from any starting seat, not only seat 0.

    `chance` defaults to `Live(rng)` -- draw from `rng` exactly as before
    this module existed, so a caller that passes neither gets byte-identical
    games for every seed (`tests/test_record_engine.py::
    test_default_chance_matches_the_seeded_stream`). A caller that passes
    its own `chance` (`record.replay`'s `Scripted`) drives the game from
    that instead; `rng` is still stored on the `Game` (`imagine` still needs
    an actual generator to branch a search from), it is simply not consulted
    directly here any more.

    The deck is built unshuffled (`new_game` with no `rng` of its own) and
    then ordered by `chance.deck_order` -- the one call that must go through
    `chance` rather than through `new_game`, so a `Scripted` game replays the
    recorded shuffle instead of drawing a new one.
    """
    rng = rng or random.Random()
    chance = chance or Live(rng)
    order = [(first + i) % num_players for i in range(num_players)]
    # Snake order: the last player to place first also places first in round two,
    # which is what compensates them for choosing last.
    queue = order + order[::-1]
    state = new_game(board, num_players)
    state.deck = chance.deck_order(state.deck)
    return Game(
        _state=state,
        rng=rng,
        chance=chance,
        ledger=PublicLedger.new(num_players),
        setup_queue=queue,
        current_player=queue[0],
        discard_quota=[0] * num_players,
        # All-zero: a seat trades only after it has published something.
        valuations=[(0.0,) * NUM_RESOURCES for _ in range(num_players)],
        first=first,
    )


def imagine(
    game: Game, rng: random.Random, *, randomize_deck: bool = True
) -> Game:
    """A copy for hypothetical play, safe to mutate and to draw from.

    Three things keep a search honest. It gets its own `rng`, so exploring
    branches cannot disturb the real game's random stream and leave the result
    unreproducible. And the copied deck is shuffled by default, so a search that
    buys a development card cannot read the card the real deck is about to deal.
    A caller that cannot observe the deck before a later draw may defer that
    shuffle until the draw itself with `randomize_deck=False`.

    And the copy's `chance` is always a fresh `Live(rng)` -- never
    `game.chance` itself, whatever that is. A copy that inherited a
    `Scripted` chance would drain events from the real game's recorded
    stream as it searched, corrupting a replay in progress the moment
    anything imagined from it drew; a copy that inherited a `Recording`
    would log the search's own hypothetical draws into the real game's
    record. `Live(rng)` sidesteps both: it draws from the copy's own `rng`,
    already isolated from the real game for the same reason dice/steals
    must be.
    """
    state = copy_state(game._state)
    if randomize_deck:
        rng.shuffle(state.deck)
    return Game(
        _state=state,
        rng=rng,
        chance=Live(rng),
        ledger=game.ledger.copy(),
        phase=game.phase,
        current_player=game.current_player,
        setup_queue=game.setup_queue[:],
        setup_step=game.setup_step,
        last_settlement=game.last_settlement,
        last_roll=game.last_roll,
        dev_card_played=game.dev_card_played,
        discard_quota=game.discard_quota[:],
        free_roads=game.free_roads,
        turns=game.turns,
        won_by=game.won_by,
        valuations=[tuple(v) for v in game.valuations],
        trades=game.trades[:],
        trades_made=game.trades_made,
        max_trades=game.max_trades,
        event_pending=game.event_pending,
        awaiting_publish=game.awaiting_publish,
        locked=game.locked,
        first=game.first,
    )


def _require(game: Game, phase: Phase) -> None:
    if game.phase is not phase:
        raise ValueError(f"expected phase {phase.name}, got {game.phase.name}")


def _in_second_setup_round(game: Game) -> bool:
    return game.setup_step >= game._state.num_players


def _advance_setup(game: Game) -> None:
    """Point the snake at the next entry that is not a retired seat, or end
    setup if none remains.

    Advances `setup_step` past locked entries rather than keeping a separate
    "seats placed" count, which is what keeps `_in_second_setup_round`'s
    `setup_step >= num_players` test correct however many seats have retired
    -- the queue always holds all `2 * num_players` slots. Mirrors
    `hexset.server.seating.advance_setup`.
    """
    queue = game.setup_queue
    while game.setup_step < len(queue) and queue[game.setup_step] in game.locked:
        game.setup_step += 1
    if game.setup_step < len(queue):
        game.current_player = queue[game.setup_step]
        game.phase = Phase.SETUP_SETTLEMENT
    else:
        # Whoever placed first takes the first real turn -- `queue[0]`, which
        # `lock_seat` never retires while it is `current_player` without also
        # moving the snake off it first, so this is never a locked seat.
        game.current_player = queue[0]
        game.phase = Phase.ROLL


def _next_unlocked(game: Game, after: int) -> int:
    """The next seat past `after` in turn order, skipping retired ones.

    Mirrors `hexset.server.seating.next_unlocked`. Terminates unless every seat
    is locked, which `lock_seat` does not itself prevent -- a caller that
    retires every seat has emptied the table, and there is no seat left for
    the turn to land on.
    """
    n = game._state.num_players
    for step in range(1, n + 1):
        seat = (after + step) % n
        if seat not in game.locked:
            return seat
    raise AssertionError("every seat is locked")


def _snapshot_hands(game: Game) -> list[list[int]]:
    """A copy of every seat's hand, to diff against after a mutation whose
    resource identities are public (see `ledger.PublicLedger.apply_hand_diff`
    for which events these are and why a steal is never one of them)."""
    return [hand[:] for hand in game._state.hands]


def _record_steal(
    game: Game, thief: int, victim: int, stolen: Resource | None
) -> None:
    """The one hand mutation `_snapshot_hands`/`apply_hand_diff` must never
    see: a robber or knight steal moves one card whose identity is public to
    nobody but the thief and the victim. `stolen` is `robber.steal`'s own
    return value, read only for its `None`-ness (the victim held nothing to
    take, so nothing happened and there is nothing to record) -- never for
    the resource it names, which `ledger.PublicLedger.steal` must not be
    told: see its docstring for the identity-independent convention that
    keeps a steal's outcome unreadable from the encoded ledger, for every
    seat including the thief and the victim's neighbours."""
    if stolen is None:
        return
    game.ledger.steal(thief, victim)


def _grant_initial_resources(game: Game, vertex: int) -> None:
    state = game._state
    topology = state.board.topology
    for h in topology.vertex_hexes[vertex]:
        resource = TERRAIN_RESOURCE[state.board.terrain[h]]
        if resource is not None and state.bank[resource] > 0:
            state.bank[resource] -= 1
            state.hands[game.current_player][resource] += 1


def place_initial_settlement(game: Game, vertex: int) -> None:
    _require(game, Phase.SETUP_SETTLEMENT)
    before = _snapshot_hands(game)
    place_settlement(game._state, game.current_player, vertex, connected=False)
    game.last_settlement = vertex
    if _in_second_setup_round(game):
        _grant_initial_resources(game, vertex)
    game.ledger.apply_hand_diff(before, game._state.hands)
    update_longest_road(game._state)
    game.phase = Phase.SETUP_ROAD


def legal_initial_roads(game: Game) -> list[int]:
    topology = game._state.board.topology
    return [
        e
        for e in topology.vertex_edges[game.last_settlement]
        if game._state.edge_owner[e] == NO_OWNER
    ]


def place_initial_road(game: Game, edge: int) -> None:
    _require(game, Phase.SETUP_ROAD)
    if edge not in legal_initial_roads(game):
        raise ValueError("the opening road must touch the settlement just placed")
    place_road(game._state, game.current_player, edge)
    update_longest_road(game._state)

    game.setup_step += 1
    _advance_setup(game)


def roll_dice(game: Game, roll: int | None = None) -> int:
    """Roll, or resolve a given roll so a search can enumerate the outcomes."""
    _require(game, Phase.ROLL)
    if pending_free_roads(game):
        raise ValueError("place the remaining free roads before rolling")
    # Unplaceable credit expires with the card's resolution.
    game.free_roads = 0
    if roll is None:
        roll = game.chance.roll()
    game.last_roll = roll

    if roll == 7:
        # Quotas are fixed now rather than recomputed as hands shrink, so
        # discarding does not reduce what is still owed. A locked seat is
        # quoted 0 unconditionally: it is never `to_move`, so nothing would
        # ever resolve a nonzero quota for it, and `players_owing_discards`
        # would otherwise report a seat the table can no longer act for.
        game.discard_quota = [
            0 if p in game.locked else discard_count(game._state, p)
            for p in range(game._state.num_players)
        ]
        game.phase = Phase.DISCARD if any(game.discard_quota) else Phase.ROBBER
    else:
        before = _snapshot_hands(game)
        distribute(game._state, roll)
        game.ledger.apply_hand_diff(before, game._state.hands)
        enter_main(game)
    return roll


def players_owing_discards(game: Game) -> list[int]:
    return [p for p, owed in enumerate(game.discard_quota) if owed > 0]


def to_move(game: Game) -> int:
    """Whose decision the legal actions belong to.

    Usually the player whose turn it is, but one phase hands the decision to
    somebody else: discarding on a seven is decided by whoever owes cards.
    """
    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if owing:
            return owing[0]
    return game.current_player


def _finish_discards(game: Game) -> None:
    if not any(game.discard_quota):
        game.phase = Phase.ROBBER


def submit_discard(game: Game, player: int, cards: list[int]) -> None:
    _require(game, Phase.DISCARD)
    if game.discard_quota[player] != sum(cards):
        raise ValueError(f"player {player} must discard {game.discard_quota[player]}")
    before = _snapshot_hands(game)
    discard(game._state, player, cards, game.discard_quota[player])
    game.ledger.apply_hand_diff(before, game._state.hands)
    game.discard_quota[player] = 0
    _finish_discards(game)


def discard_one(game: Game, player: int, resource: Resource) -> None:
    _require(game, Phase.DISCARD)
    if game.discard_quota[player] < 1:
        raise ValueError(f"player {player} owes no discard")
    if game._state.hands[player][resource] < 1:
        raise ValueError(f"player {player} holds no {resource.name}")
    game._state.hands[player][resource] -= 1
    game._state.bank[resource] += 1
    game.ledger.spend(player, int(resource), 1)
    game.discard_quota[player] -= 1
    _finish_discards(game)


def move_robber_to(game: Game, target: int, victim: int | None = None) -> None:
    _require(game, Phase.ROBBER)
    move_robber(game._state, target)
    if victim is not None:
        stolen = steal(game._state, game.current_player, victim, game.chance)
        _record_steal(game, game.current_player, victim, stolen)
    enter_main(game)


def _check_win(game: Game) -> None:
    """Winning the Game (rulebook): "If you have 10 or more VPs at any point
    during YOUR turn, the game ends immediately and you are the winner" --
    scoped to whoever is on the move, not to whoever happens to be over the
    threshold. `victory.winner` scans every seat and is right for a
    state-only query (`tests/test_victory.py`), but a seat's own action can
    move VPs it does not own: breaking an opponent's Longest Road can hand
    the tile (and 2 VPs) to a *third*, off-turn seat who was already sitting
    on 8. That seat does not win here -- only `game.current_player`'s own
    total is checked, exactly as the rulebook scopes it. They still win the
    moment their own turn's `_check_win` next runs, with nothing further
    needing to happen, since nothing about their total changes in between."""
    if victory_points(game._state, game.current_player) >= WINNING_POINTS:
        game.won_by = game.current_player
        game.phase = Phase.GAME_OVER


def pending_free_roads(game: Game) -> list[int]:
    """Placeable roads still owed by a Road Building card."""
    if game.free_roads <= 0:
        return []
    return [
        edge for edge in range(game._state.board.topology.num_edges)
        if can_place_road(game._state, game.current_player, edge)
    ]


def build_road(game: Game, edge: int) -> None:
    """Build a road, spending a free road from road building if one is owed."""
    if game.phase is not Phase.ROLL or game.free_roads <= 0:
        _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    if game.free_roads > 0:
        game.free_roads -= 1
    else:
        pay(game._state, game.current_player, Purchase.ROAD)
    game.ledger.apply_hand_diff(before, game._state.hands)
    place_road(game._state, game.current_player, edge)
    update_longest_road(game._state)
    _check_win(game)
    run_trade_event(game)


def build_settlement(game: Game, vertex: int) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    pay(game._state, game.current_player, Purchase.SETTLEMENT)
    game.ledger.apply_hand_diff(before, game._state.hands)
    place_settlement(game._state, game.current_player, vertex)
    # A new settlement can cut an opponent's route, so this is not only the
    # builder's own longest road that may change.
    update_longest_road(game._state)
    _check_win(game)
    run_trade_event(game)


def build_city(game: Game, vertex: int) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    pay(game._state, game.current_player, Purchase.CITY)
    game.ledger.apply_hand_diff(before, game._state.hands)
    upgrade_to_city(game._state, game.current_player, vertex)
    _check_win(game)
    run_trade_event(game)


def buy_development_card(game: Game) -> DevCard:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    card = buy_dev_card(game._state, game.current_player)
    game.ledger.apply_hand_diff(before, game._state.hands)
    _check_win(game)
    run_trade_event(game)
    return card


def _spend_turn_card(game: Game) -> None:
    if game.dev_card_played:
        raise ValueError("only one development card may be played per turn")
    game.dev_card_played = True


def play_knight_card(
    game: Game, target: int, victim: int | None = None
) -> Resource | None:
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play a knight in {game.phase.name}")
    _spend_turn_card(game)
    stolen = play_knight(game._state, game.current_player, target, victim, game.chance)
    if victim is not None:
        _record_steal(game, game.current_player, victim, stolen)
    update_largest_army(game._state)
    _check_win(game)
    # `run_trade_event` no-ops itself when `game.phase` is `ROLL` (a knight
    # played before rolling): this is the one action legal in both phases,
    # and the interleaving only ever applies to `MAIN`.
    run_trade_event(game)
    return stolen


def play_road_building_card(game: Game) -> None:
    """Credit two free roads, placed afterwards with ordinary build actions.

    Resolving the card this way keeps the action space flat: one entry for the
    card, rather than one per pair of edges.

    Legal in `ROLL` as well as `MAIN` -- rulebook, Development Cards: "You may
    play a development card before rolling dice or at any time during the
    Action phase," which names no exception for this card. `run_trade_event`
    below no-ops itself outside `MAIN` (see its own docstring), so playing
    this before rolling credits the two free roads without touching the
    trade event. The free `build_road` calls resolve in `ROLL` before dice
    are drawn. Paid building remains MAIN-only.
    """
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play road building in {game.phase.name}")
    _spend_turn_card(game)
    spend_card(game._state, game.current_player, DevCard.ROAD_BUILDING)
    game.free_roads += ROAD_BUILDING_ROADS
    run_trade_event(game)


def play_year_of_plenty_card(game: Game, resources: list[Resource]) -> None:
    """Legal in `ROLL` as well as `MAIN`; see `play_road_building_card`."""
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play year of plenty in {game.phase.name}")
    _spend_turn_card(game)
    before = _snapshot_hands(game)
    play_year_of_plenty(game._state, game.current_player, resources)
    game.ledger.apply_hand_diff(before, game._state.hands)
    run_trade_event(game)


def play_monopoly_card(game: Game, resource: Resource) -> int:
    """Legal in `ROLL` as well as `MAIN`; see `play_road_building_card`."""
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play monopoly in {game.phase.name}")
    _spend_turn_card(game)
    before = _snapshot_hands(game)
    taken = play_monopoly(game._state, game.current_player, resource)
    # Monopoly forces every other seat to publicly hand over every card of
    # `resource`, so the transfer is fully public despite touching every
    # seat at once -- the same `apply_hand_diff` every other public mutation
    # uses, not the hidden-identity path a steal needs.
    game.ledger.apply_hand_diff(before, game._state.hands)
    run_trade_event(game)
    return taken


def trade_with_bank(game: Game, give: Resource, receive: Resource) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    bank_trade(game._state, game.current_player, give, receive)
    game.ledger.apply_hand_diff(before, game._state.hands)
    run_trade_event(game)


def run_trade_event(game: Game) -> None:
    """Clear this turn's trade event for the current player, if anybody is
    seated to answer a gate.

    Called directly, unconditionally, after every MAIN action the current
    player takes -- build, buy, a bank/port trade, a development card -- on
    whatever is currently published (the owner's review of the mechanic
    against the rulebook, 2026-09-03: trade and build interleave, rather
    than one event before the first build). The current player's hand just
    changed, so a different bundle may now clear; other seats' vectors are
    unchanged within the turn. The turn's *first* event does not come
    through here directly any more -- `run_pending_event` calls this once,
    lazily, the first time anything reaches it this turn (the PI amendment
    "publish points and the event trigger"); every call from a build/buy/
    trade/dev-card site below is one of the turn's *subsequent* events and
    always runs, pending flag or not. Never runs after `end_turn`
    (`end_turn` moves the phase to `ROLL` before this could be reached) and
    never during setup, `ROLL`, `ROBBER` or discard resolution -- every
    caller of this function already requires `game.phase is Phase.MAIN` to
    reach it (`play_knight_card` is the one action legal in both `ROLL` and
    `MAIN`, and checks explicitly), so the guard below is a second line of
    defence, not the only one.

    A game whose `gates` is `None` -- a bare `start()` with nobody seated --
    simply does not trade. The vectors `trade_event` reads (`game.valuations`)
    are not this function's concern: whichever driver stepped the acting
    seat to this point already published them (`Game.publish`, usually via
    `hexset.trading.publish_valuation`) as part of that seat's own decision.
    """
    if game.phase is not Phase.MAIN:
        return
    gates = game.gates
    if gates is None:
        return
    trade_event(
        game,
        lambda seat, view, received, other: judged(gates[seat], view, received, other),
    )


def run_pending_event(game: Game) -> None:
    """Fire this turn's *first* trade event now, if one is still pending
    and due, then clear the flag so it can never fire twice for the same
    turn.

    The lazy half of `event_pending`'s mechanism (see its docstring on
    `Game`): called from the current player's own `legal_actions(game)`,
    `game.state(seat)`, and `Game.publish`, whichever is reached first --
    the PI amendment "publish points and the event trigger"
    (`agents/reference/trading-design.md`) that replaced running this
    directly inside `enter_main`, because a batched collector needs to
    publish before this fires rather than the engine forcing a fresh
    forward pass at `enter_main` time for every seat.

    A no-op when nobody is seated to answer a gate (`game.gates is None`)
    -- deliberately, and not merely because `run_trade_event` would no-op
    too: a search's own `imagine`d copy carries `event_pending` (so a chain
    of simulated turns keeps arming and consuming it the same way a real
    game does) but never carries real gates, and stepping that copy through
    its own `legal_actions` must not so much as touch the flag, let alone
    run anything, using the copy's stand-in `gates=None`.

    Does **not** wait for the current player's own publish before firing,
    even though `awaiting_publish` (`Game`'s own docstring) may still be
    true -- deliberately, chosen over the alternative (make this a no-op
    while the current player is a seat that publishes as part of its own
    decision and has not yet done so this turn, distinguished by
    `hasattr(gates[current], "choose")`) because that alternative reaches
    past this function's own remit: `hexset.trading.publish_valuation`
    itself calls `game.state(seat)` to build the view it hands the seat's
    `valuation` -- one of the three trigger points -- *before* calling
    `Game.publish` with the result, for every driver that steps a seated
    game (`arena.play`, `record.record_game`, `bench.aivat`, the gym's
    auto-played opponents, the server's embedded bots alike). Waiting here
    would therefore also delay the arena's own turn-opening event from the
    seat's standing vector to the one it is about to compute, changing what
    every one of those drivers has always done and re-baselining the
    census for no reason connected to the regression below. So: the first
    event of a turn can still run on the current player's standing vector
    from its own last turn (`[0, ...]` on that seat's very first turn ever)
    -- a driver publishing before observing (`arena.play`'s own idiom) still
    reaches this before that happens, exactly as today. What changed is
    `publish_due` (see its docstring): an observation that spends this
    event early no longer makes a still-unpublished seat's own publish, a
    moment later, look like it already happened -- so the seat's fresh
    vector reaches the turn's *next* event (every interleaved one after a
    MAIN action) even on the turn where the first one ran early. This is
    the regression fix: `hexset.server.webplay.GameSession.state_view` used
    to read the current player's hidden view on *any* viewer's poll,
    consuming this event before an embedded bot -- or that bot's own
    runner, polling its own state before it decides -- ever got to
    publish, and `publish_due` used to go false as a result, so the
    publish that should have followed never happened, for that turn or any
    after it -- the deploy report that pinned this down called it "the
    served game never trades".
    """
    if not game.event_pending or game.phase is not Phase.MAIN or game.gates is None:
        return
    game.event_pending = False
    run_trade_event(game)


def enter_main(game: Game) -> None:
    """Enter the main phase. Arms this turn's first trade event
    (`event_pending = True`) rather than running it here: `run_pending_event`
    runs it lazily, the first time the current player's own `legal_actions`,
    `game.state`, or `Game.publish` is reached (the PI amendment "publish
    points and the event trigger"). Every path from `ROLL` or `ROBBER` into
    `MAIN` goes through here, so every driver arms the pending event without
    having to remember to. Also arms `awaiting_publish`, so `publish_due`
    reports the current player as owing a fresh publish for this turn
    regardless of whether that first event has already been observed (and so
    already run) by the time the seat gets to publish. See `run_trade_event`
    for the event itself and the interleaving with the actions that follow
    it.
    """
    game.phase = Phase.MAIN
    game.event_pending = True
    game.awaiting_publish = True


def end_turn(game: Game) -> None:
    _require(game, Phase.MAIN)
    mature(game._state, game.current_player)
    game.dev_card_played = False
    game.trades = []
    game.trades_made = 0
    game.pending = []
    # Free roads with nowhere legal to go are simply lost.
    game.free_roads = 0
    game.turns += 1
    if game.turns >= MAX_TURNS:
        game.phase = Phase.GAME_OVER
        return
    game.current_player = _next_unlocked(game, game.current_player)
    game.phase = Phase.ROLL
    # Winning the Game (rulebook): "the first player to reach 10 or more VPs
    # ... on their turn ... wins" -- the FAQ reading is that this is
    # announced at the start of a seat's own turn, not only after an action
    # of theirs. `_check_win` (see its own docstring) already scopes a win
    # to `game.current_player`, which this just made the new seat; a seat
    # that crossed 10 off-turn (a Longest Road/Largest Army tile transfer
    # during someone else's turn) therefore wins here, before it is ever
    # asked for an action -- nothing about its total needs to change for
    # that to become true, so there is nothing to wait for. Every driver
    # that steps a game (`arena.play`'s `while not is_over(game)`, a
    # search's own `imagine`d copy calling this same function) sees
    # `Phase.GAME_OVER` on the very next check, before requesting a move
    # from the new current player.
    _check_win(game)


def is_over(game: Game) -> bool:
    return game.phase is Phase.GAME_OVER


def lock_seat(game: Game, seat: int) -> None:
    """Retire `seat`. A no-op if it is already retired.

    From here on `seat` is skipped by the setup snake and by turn rotation,
    can never be `to_move`, and is never a counterparty in a trade event. This is the primitive `hexset.server.seating.lock_seat`
    (`ui:seating.py:148-154`) implemented as a post-apply correction because
    `hexset.game` had no such field -- see `Game.locked`'s docstring and
    `docs/engine-divergence-2026-09-02.md` request R2.

    It also matches that function's answer for what happens to a retired
    seat's pieces and hand: nothing. `lock_seat` never touches `game._state`.
    `hexset.server.api.Table._settle_locks` only ever calls
    it on a seat the setup snake reached still empty, so there is nothing
    built or held to clear; a caller that retires a seat mid-game gets the
    same treatment, and its hand and pieces are simply abandoned in place
    rather than erased or returned to the bank. Retiring a seat is a
    decision about whose turn it is, not a rules event -- it does not
    forfeit pieces the way `robber.discard` or `economy.pay` does.
    """
    if seat in game.locked:
        return
    game.locked = game.locked | {seat}

    if seat < len(game.discard_quota):
        game.discard_quota[seat] = 0
    if game.phase is Phase.DISCARD:
        _finish_discards(game)

    if game.current_player != seat:
        return
    if game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        game.setup_step += 1
        _advance_setup(game)
    elif game.phase in (Phase.ROLL, Phase.ROBBER, Phase.MAIN):
        # The only phases where `to_move` reads `current_player` directly
        # (see its docstring); DISCARD is already handled above, GAME_OVER
        # has no next turn to hand off.
        game.current_player = _next_unlocked(game, seat)
