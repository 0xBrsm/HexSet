# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Iterable, Sequence

from .board.board import MAX_ROLL, MIN_ROLL, Board, pips
from .board.terrain import TERRAIN_RESOURCE, Resource, check_resource
from .cards import ROAD_BUILDING_ROADS, DevCard
from .chance import Chance, Live, for_rules as chance_for
from .devcards import buy as buy_dev_card
from .devcards import (
    buy_drawn,
    mature,
    play_knight,
    play_monopoly,
    play_year_of_plenty,
    spend_card,
)
from .economy import Purchase, bank_trade, check_afford, distribute, hand_size, pay
from .ledger import PublicLedger
from .robber import allowed_targets, check_target, discard, discard_count, move_robber, steal, victims
from .rules import STANDARD, STANDARD_GAME, GameType, Rules
from .trading import (
    TRADE_RULES,
    params_of,
    Trade,
    trade_event,
    trade_round,
)
from .state import (
    MAX_ROADS,
    NO_OWNER,
    GameState,
    check_city,
    check_road,
    check_settlement,
    copy_state,
    is_hidden,
    new_game,
    observed_by,
    place_road,
    place_settlement,
    road_count,
    road_placeable,
    upgrade_to_city,
)
from .victory import (
    public_victory_points,
    update_largest_army,
    update_longest_road,
    victory_points,
)

if TYPE_CHECKING:
    from .view import View

__all__ = [
    "DICE",
    "MAX_TURNS",
    "UNSTRUCTURED_TURN_CAP",
    "NO_TURN_CAP",
    "ROLL_ODDS",
    "Phase",
    "Game",
    "start",
    "imagine",
    "observe",
    "place_initial_settlement",
    "legal_initial_roads",
    "place_initial_road",
    "roll_dice",
    "players_owing_discards",
    "to_move",
    "may_act",
    "submit_discard",
    "discard_one",
    "move_robber_to",
    "pending_free_roads",
    "build_road",
    "build_settlement",
    "build_city",
    "buy_development_card",
    "play_knight_card",
    "play_road_building_card",
    "play_year_of_plenty_card",
    "play_monopoly_card",
    "trade_with_bank",
    "run_trade_event",
    "event_hand_sizes",
    "publish_trade_event",
    "enter_main",
    "end_turn",
    "is_over",
    "lock_seat",
]


DICE = 6
#: Turns -- `end_turn` calls, summed over every seat -- after which a game is
#: abandoned without a winner. Reaching it is a defect, not a long game, and
#: `hexset.arena.compete` refuses to score a run that does.
#:
#: A flat total rather than a per-seat budget, because the longest game barely
#: moves with table size: more seats means proportionally fewer turns each.
#: Seats playing to win finish well inside it -- the longest such games run
#: to about 225 turns -- so 300 leaves room for a slow game while stopping a
#: stuck one before it costs more than a game or two of compute.
MAX_TURNS = 300

#: The horizon unstructured play needs. Random bots are not stuck, they are
#: hopeless: 40 games of four of them run to a median of 285 turns and a
#: maximum of 702, and two of them to a median of 327 and a maximum of 1352.
#: A cap set to accommodate that catches no bug at all, so it is a separate number and a caller asks for it by name.
UNSTRUCTURED_TURN_CAP = 3000

#: No horizon at all, for a caller that is not playing a game forward: a
#: replay is driven by a recorded action list, which already knows where its
#: game ended, so re-capping it could only truncate it somewhere else.
NO_TURN_CAP = 1 << 62

ROLL_ODDS: tuple[tuple[int, float], ...] = tuple(
    (roll, pips(roll) / DICE**2) for roll in range(MIN_ROLL, MAX_ROLL + 1)
)


class Phase(IntEnum):
    """Where a game is in its turn structure; `legal_actions` reads it."""

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL = 2
    DISCARD = 3
    ROBBER = 4
    MAIN = 5
    GAME_OVER = 6


@dataclass
class Game:
    """One game: the state, the chance source every draw goes through, the
    public ledger, and the turn's phase and bookkeeping. Played through
    `actions.apply`; `Game.state(seat)` is what a seat may read."""

    _state: GameState
    rng: random.Random
    # The one chance source: every random draw the engine makes goes through
    # this, never through `rng` directly. `rng` is what the default
    # `Live(rng, split=True)` draws its deck and its dice and steal
    # generators from; `imagine` takes a fresh `rng` of its caller's and
    # never reads this one.
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
    # Which phase to resume once this turn's robber phase resolves: `MAIN`
    # after a seven or a knight played from `MAIN`, `ROLL` for a knight
    # played before rolling. Only meaningful while `phase is ROBBER`; reset
    # by `end_turn` so no stale value survives into the next turn.
    resume_phase: Phase = Phase.MAIN
    # A host's rule for *this* robber move: the hexes the robber may go to,
    # or `None` for the rulebook's "anywhere but where it stands". Some
    # tables forbid more -- a "friendly robber" rule keeps it off hexes next
    # to a seat with few points, and tells the client only which hexes
    # remain -- and a policy offered the rulebook's set there picks
    # placements the table will not take. Set by whoever mirrors such a
    # table (`legal_actions` honours it), cleared by `move_robber_to` once
    # the move is made so it cannot outlive the prompt it answered; copied
    # by `imagine` so a search sees the same table.
    robber_allowed: frozenset[int] | None = None
    turns: int = 0
    won_by: int | None = None
    # This turn's executed trades and their count, cleared by `end_turn`.
    trades: list[Trade] = field(default_factory=list)
    trades_made: int = 0
    # This turn's offers and answers put on the public ledger
    # (`trading.show`), as `(len(trades) when shown, seat, received)`, so a
    # record can put each back between the same exchanges. Cleared by
    # `end_turn` with `trades`.
    shown: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    # How a turn's trading is run; one axis, because these are exclusive.
    # "round" (the default) repeats `trading.trade_round` while the actor
    # has offers left and the cap allows; "auto" is the exhaustive clearing
    # house `trading.trade_event`, which fitting or training against teaches
    # a game nobody plays; "external" means somebody else drives and the
    # engine runs nothing, for a served table answering over a wire.
    trade_mode: str = "round"
    # How many counters one exchange of a "round" may carry, a seat's answer
    # to the offer the first: 1 is a counter the actor takes or leaves; more
    # lets the actor counter it back and the seat answer again, in turn
    # (`trading.counter_thread`). The served table runs one.
    counter_steps: int = 1
    # The `turns` value the trade event last ran in, so a knight played in
    # MAIN cannot re-run it when its robber move re-enters MAIN.
    trade_event_turn: int = -1
    # Candidates the trade event found against a manual seat's
    # `PendingGate`: a snapshot of the *last* event, not an accumulating log
    # like `trades`, so nothing pending survives a change of hands it was
    # computed against. Not copied by `imagine`, as `gates` is not.
    pending: list[Trade] = field(default_factory=list)
    # Which candidate the automatic trade event clears among those both
    # gates price above their own floors: `"egalitarian"` (the default)
    # maximises the smaller of the two private gains, `"nash"` their
    # product, `"actor"` the current player's own. Validated at `start()`.
    trade_rule: str = "egalitarian"
    # Who answers a private gate, one per seat, asked fresh at every trade
    # event. `None` means nobody trades, as a bare `start()` game does.
    # **`imagine` deliberately does not copy this**, so a hypothetical never
    # trades: a search must not reach the real opponents' private gates any
    # more than it may read their hands, and re-clearing the event at every
    # node would multiply a position evaluation by the branching factor.
    gates: tuple[object, ...] | None = None
    # Seats retired from the game: skipped by the setup snake and by turn
    # rotation, never `to_move`, never asked for a trade. Copied by
    # `imagine`, so a search forward from a table with a retired seat does
    # not simulate turns for it. Whether and when a seat retires is the
    # caller's policy; `lock_seat` restricts neither.
    locked: frozenset[int] = field(default_factory=frozenset)
    # The seat `start()` opened the setup snake at; read-only afterwards.
    # Its own field rather than `setup_queue[0]` because a
    # `hexset.record.Record` carries it to rebuild the snake on replay, and
    # `setup_queue` is not part of a record.
    first: int = 0
    # Turns after which this game is abandoned with no winner. A property of
    # the run rather than of the module, because how long a game legitimately
    # takes is a property of who is playing it: `MAX_TURNS` is read off agents
    # that are trying to win, and a run measuring unstructured play has to say
    # so out loud rather than have its games quietly truncated. Carried by
    # `imagine` so a search sees the same horizon the real game has.
    turn_cap: int = MAX_TURNS

    def state(self, seat: int, *, hidden: bool = True) -> GameState | View:
        """The access path to this game's state.

        `hidden=True` (the default) returns `seat`'s information-set `View`.
        `hidden=False` returns the true `GameState` -- the same object every
        time, never a copy, so mutating it through the returned reference
        mutates the game. It is the public way to read the true state; the
        package's own modules also reach it as `_state`. Information-set
        policies must use the default view.
        """
        if not hidden:
            return self._state
        from .view import View as _View

        return _View.from_game(self, seat)

    @property
    def num_players(self) -> int:
        """How many seats this game has -- a property of the table, not of
        anybody's hand, so reading it is not an omniscient read."""
        return self._state.num_players

    def set_state(self, state: GameState) -> None:
        """Replace the true state outright -- the one write a determinizer
        needs, without exposing `_state` to code outside the engine."""
        self._state = state



def start(
    board: Board,
    num_players: int,
    rng: random.Random | None = None,
    *,
    first: int = 0,
    chance: Chance | None = None,
    trade_rule: str = "egalitarian",
    game_type: GameType = STANDARD_GAME,
    locked: Iterable[int] = (),
    turn_cap: int = MAX_TURNS,
) -> Game:
    """Start a game of `game_type` at a table of `num_players` seats.

    `game_type` is the contract: a ruleset together with the seat counts it is
    played at. It is checked against the seats that will actually *play* --
    `num_players` less `locked` -- so a four-seat deal with two seats retired
    is a two-seat game and `DUEL_VARIANT_GAME` accepts it, while the same deal
    with nobody retired is a four-seat game and it does not. This is the only
    door: `new_game` takes a bare `Rules` and checks nothing, because it
    builds a state rather than a playable game.

    `locked` retires those seats before the first action, which is what a
    table with closed seats is -- the server dealing four and seating two, or
    the arena seating fewer players than the table holds. Retiring here rather
    than afterwards is what lets the contract be checked at all, and it points
    the setup snake past them from the outset. `lock_seat` remains for a seat
    that leaves *during* a game, which does not re-open the contract.

    `first` chooses which seat opens the setup snake and therefore takes the
    first real turn; the snake's compensating property -- whoever places
    first in round one places last in round two -- holds from any start. A
    retired `first` is skipped like any other.

    `chance` defaults to the source the rules ask for -- `Balanced(rng)` under
    `balanced_dice`, `Live(rng)` otherwise, split either way
    (`chance.for_rules`): the same seed rolls the same dice turn for turn
    whatever the seats play. A caller passing its own
    (`Scripted`, for replay) drives the game from that instead, whatever the
    rules say: a recording already holds the rolls the deck produced. The deck
    is ordered by `chance.deck_order`, so a scripted game replays the recorded
    shuffle.
    """
    if trade_rule not in TRADE_RULES:
        raise ValueError(f"unknown trade rule: {trade_rule!r}")
    retired = frozenset(locked)
    stranger = sorted(s for s in retired if not 0 <= s < num_players)
    if stranger:
        raise ValueError(
            f"cannot retire {stranger} at a {num_players}-seat table"
        )
    game_type.check(num_players - len(retired), dealt=num_players)
    rules = game_type.rules
    rng = rng or random.Random()
    chance = chance or chance_for(rules, rng)
    order = [(first + i) % num_players for i in range(num_players)]
    # Snake order: the last player to place first also places first in round two,
    # which is what compensates them for choosing last.
    queue = order + order[::-1]
    state = new_game(board, num_players, rules=rules)
    state.deck = chance.deck_order(state.deck)
    game = Game(
        _state=state,
        rng=rng,
        chance=chance,
        ledger=PublicLedger.new(num_players),
        setup_queue=queue,
        current_player=queue[0],
        discard_quota=[0] * num_players,
        trade_rule=trade_rule,
        locked=retired,
        first=first,
        turn_cap=turn_cap,
    )
    if retired:
        # The same walk `lock_seat` would do, but before anything has been
        # placed: point the snake at the first seat that is actually playing.
        _advance_setup(game)
    return game


def imagine(
    game: Game, rng: random.Random, *, randomize_deck: bool = True
) -> Game:
    """A copy for hypothetical play, safe to mutate and to draw from.

    Three things keep a search honest. The copy gets its own `rng`. The
    copied deck is shuffled by default, so a search that buys a development
    card cannot read the card the real deck is about to deal;
    `randomize_deck=False` defers that to the draw. And the copy's `chance`
    is always a fresh `Live(rng)`, never `game.chance`, which would
    otherwise drain or pollute the real game's recorded stream (or, split,
    hand the search the real game's coming rolls). It is unsplit: a search
    takes its dice and steals off its own `rng` in turn, as it always has.

    On an observed state (`state.Hidden`) hidden piles copy as counts and
    `randomize_deck` must be `False` -- shuffling a deck known only by its
    length is an identity read and raises `state.HiddenRead`.
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
        resume_phase=game.resume_phase,
        robber_allowed=game.robber_allowed,
        turns=game.turns,
        won_by=game.won_by,
        trades=game.trades[:],
        trades_made=game.trades_made,
        shown=game.shown[:],
        trade_mode=game.trade_mode,
        counter_steps=game.counter_steps,
        trade_event_turn=game.trade_event_turn,
        trade_rule=game.trade_rule,
        locked=game.locked,
        first=game.first,
        turn_cap=game.turn_cap,
    )


def observe(game: Game, seat: int, chance: Chance | None = None) -> Game:
    """`game` as `seat` sees it, to be played on from that side of the table.

    A copy whose other seats' hands and development cards, and the deck, are
    hidden piles of the same size (`state.observed_by`), with everything
    public carried over -- the board, the ledger, whose turn it is and every
    flag of it. It is advanced by the ordinary `actions.apply` as the table's
    events arrive, every seat's actions alike: a hidden hand takes each
    public change by name into its count (`state.HiddenHand.move`, which the
    ledger reads), and what no seat here can draw for itself -- the dice, a
    steal, a purchase, a Monopoly's takings -- is taken from `chance`, a
    `chance.Hosted` source the caller feeds (a fresh one by default).

    It is a seat's game, not a referee's. `legal_actions` answers for `seat`
    and for nobody else, whose hands it cannot read; what another seat may
    do is the host's to rule, and applying it is taking the host's word.
    Nothing here asks a gate: `trade_mode` is `"external"`, and a trade the
    table makes goes in through `trading.execute_agreed` with nobody asked.
    A hidden seat's victory point cards cannot be counted, so a win they
    decide is the host's to announce (`_check_win`).

    Played in lockstep with the true game it observes, it stays equal to
    `observed_by` of that game, ledger included (`tests/test_observed.py`).
    """
    from .chance import Hosted

    seen = imagine(game, random.Random(0), randomize_deck=False)
    seen.set_state(observed_by(seen._state, seat))
    seen.chance = Hosted() if chance is None else chance
    seen.trade_mode = "external"
    return seen


def _require(game: Game, phase: Phase) -> None:
    if game.phase is not phase:
        raise ValueError(f"expected phase {phase.name}, got {game.phase.name}")


def _require_main(game: Game) -> None:
    """`MAIN`, with no placeable road owed: a Road Building card resolves
    when played, so while it has a road left to place nothing else may be
    done (`legal_actions` offers only those roads)."""
    _require(game, Phase.MAIN)
    if pending_free_roads(game):
        raise ValueError("place the remaining free roads first")


def _require_seat(game: Game, seat: int) -> None:
    if not 0 <= seat < game._state.num_players:
        raise ValueError(f"there is no seat {seat}")


def _in_second_setup_round(game: Game) -> bool:
    return game.setup_step >= game._state.num_players


def _advance_setup(game: Game) -> None:
    """Point the snake at the next entry that is not a retired seat, or end
    setup if none remains. Advances `setup_step` past locked entries rather
    than keeping a "seats placed" count, which keeps
    `_in_second_setup_round`'s test correct however many seats retired: the
    queue always holds all `2 * num_players` slots."""
    queue = game.setup_queue
    while game.setup_step < len(queue) and queue[game.setup_step] in game.locked:
        game.setup_step += 1
    if game.setup_step < len(queue):
        game.current_player = queue[game.setup_step]
        game.phase = Phase.SETUP_SETTLEMENT
    else:
        # Whoever placed first takes the first real turn -- unless that seat
        # retired before it ever placed, which `lock_seat` cannot correct for
        # because the snake was never pointed at it. Retiring a seat ahead of
        # setup is what the arena does when it seats fewer players than the
        # table holds.
        first = queue[0]
        game.current_player = (
            first if first not in game.locked else _next_unlocked(game, first)
        )
        game.phase = Phase.ROLL


def _next_unlocked(game: Game, after: int) -> int:
    """The next seat past `after` in turn order, skipping retired ones.
    Raises `AssertionError` if every seat is locked, which `lock_seat` does
    not itself prevent."""
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
    see: a steal moves one card whose identity is public to nobody but the
    thief and the victim. `stolen` is read only for its `None`-ness (the
    victim held nothing), never for the resource it names, which
    `PublicLedger.steal` must not be told."""
    if stolen is None:
        return
    game.ledger.steal(thief, victim)


def _grant_initial_resources(game: Game, vertex: int) -> None:
    state = game._state
    topology = state.board.topology
    hand = state.hands[game.current_player]
    for h in topology.vertex_hexes[vertex]:
        resource = TERRAIN_RESOURCE[state.board.terrain[h]]
        if resource is not None and state.bank[resource] > 0:
            state.bank[resource] -= 1
            if is_hidden(hand):
                hand.move(resource, 1)
            else:
                hand[resource] += 1


def place_initial_settlement(game: Game, vertex: int) -> None:
    """Place the current seat's setup settlement on `vertex`, no road needed.
    In the second setup round the seat also takes one card per adjacent
    producing hex, as far as the bank holds them. Raises `ValueError` outside
    `Phase.SETUP_SETTLEMENT` or when `vertex` cannot be settled."""
    _require(game, Phase.SETUP_SETTLEMENT)
    before = _snapshot_hands(game)
    place_settlement(game._state, game.current_player, vertex, connected=False)
    game.last_settlement = vertex
    if _in_second_setup_round(game):
        _grant_initial_resources(game, vertex)
    game.ledger.apply_hand_diff(before, game._state.hands)
    update_longest_road(
        game._state, settlement_vertex=vertex, settlement_owner=game.current_player
    )
    game.phase = Phase.SETUP_ROAD


def legal_initial_roads(game: Game) -> list[int]:
    """The free edges touching the settlement just placed: where
    `place_initial_road` may go."""
    topology = game._state.board.topology
    return [
        e
        for e in topology.vertex_edges[game.last_settlement]
        if game._state.edge_owner[e] == NO_OWNER
    ]


def place_initial_road(game: Game, edge: int) -> None:
    """Place the current seat's setup road on `edge` and move setup on, to the
    next seat's settlement or the first turn's `Phase.ROLL`. Raises
    `ValueError` outside `Phase.SETUP_ROAD` or unless `edge` is one of
    `legal_initial_roads`."""
    _require(game, Phase.SETUP_ROAD)
    if edge not in legal_initial_roads(game):
        raise ValueError("the opening road must touch the settlement just placed")
    place_road(game._state, game.current_player, edge)
    update_longest_road(game._state, road_owner=game.current_player)

    game.setup_step += 1
    _advance_setup(game)


def roll_dice(game: Game, roll: int | None = None) -> int:
    """Roll, or resolve a given roll so a search can enumerate the outcomes."""
    _require(game, Phase.ROLL)
    if pending_free_roads(game):
        raise ValueError("place the remaining free roads before rolling")
    if roll is None:
        roll_by = getattr(game.chance, "roll_by", None)
        roll = roll_by(game.current_player) if roll_by is not None else game.chance.roll()
    if not MIN_ROLL <= roll <= MAX_ROLL:
        raise ValueError(f"two dice cannot roll {roll}")
    # Unplaceable credit expires with the card's resolution.
    game.free_roads = 0
    game.last_roll = roll

    if roll == 7:
        # Quotas are fixed here rather than recomputed as hands shrink, so
        # discarding does not reduce what is still owed. A locked seat is
        # quoted 0: it is never `to_move`, so nothing would resolve a
        # nonzero quota for it.
        game.discard_quota = [
            0 if p in game.locked else discard_count(game._state, p)
            for p in range(game._state.num_players)
        ]
        # A seven always resumes into MAIN once discard/robber resolve.
        game.resume_phase = Phase.MAIN
        game.phase = Phase.DISCARD if any(game.discard_quota) else Phase.ROBBER
    else:
        before = _snapshot_hands(game)
        distribute(game._state, roll)
        game.ledger.apply_hand_diff(before, game._state.hands)
        enter_main(game)
    return roll


def players_owing_discards(game: Game) -> list[int]:
    """The seats still owing cards to the current discard round, lowest
    first."""
    return [p for p, owed in enumerate(game.discard_quota) if owed > 0]


def to_move(game: Game) -> int:
    """Whose decision the legal actions belong to, when exactly one seat is
    wanted: the current player, except in `Phase.DISCARD`, where it is the
    lowest-indexed seat still owing cards.

    **Discarding is not a turn.** Every seat over the limit discards at the
    same instant, each bounded only by its own hand and quota. Serializing
    that round to one seat is correct for a caller wanting one decision at a
    time, but wrong for a live table where seat 3 must discard without
    waiting on seat 0; a server asks `may_act` instead.
    """
    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if owing:
            return owing[0]
    return game.current_player


def may_act(game: Game, seat: int) -> bool:
    """Whether `seat` may act right now -- true for every seat still owing a
    discard, so a discard round resolves in whatever order the seats answer
    in, and equivalent to `seat == to_move(game)` everywhere else.

    Any interleaving of a round's discards reaches the same position: a
    discard reads and writes only that seat's own hand, its own quota and
    the bank, and draws no chance event.
    """
    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if owing:
            return seat in owing
    return seat == to_move(game)


def _finish_discards(game: Game) -> None:
    """Close the discard round once nothing is owed: on to the robber, or --
    when the seat that rolled the seven retired during the round -- straight
    to the next seat's turn, there being nobody left to move the robber."""
    if any(game.discard_quota):
        return
    if game.current_player in game.locked:
        _close_turn(game)
    else:
        game.phase = Phase.ROBBER


def submit_discard(game: Game, player: int, cards: list[int]) -> None:
    """Discard `player`'s whole quota at once, `cards` counted per resource.
    Raises `ValueError` outside `Phase.DISCARD`, for no such seat, when the
    seat owes nothing, or when `cards` does not total the quota or the hand
    cannot cover it."""
    _require(game, Phase.DISCARD)
    _require_seat(game, player)
    if game.discard_quota[player] < 1:
        raise ValueError(f"player {player} owes no discard")
    if game.discard_quota[player] != sum(cards):
        raise ValueError(f"player {player} must discard {game.discard_quota[player]}")
    before = _snapshot_hands(game)
    discard(game._state, player, cards, game.discard_quota[player])
    game.ledger.apply_hand_diff(before, game._state.hands)
    game.discard_quota[player] = 0
    _finish_discards(game)


def discard_one(game: Game, player: int, resource: Resource) -> None:
    """Discard one `resource` towards `player`'s quota. Raises `ValueError`
    outside `Phase.DISCARD`, for no such seat or resource, when the seat owes
    nothing, or when it holds no such card (no card at all, for a hidden
    hand)."""
    _require(game, Phase.DISCARD)
    _require_seat(game, player)
    check_resource(resource)
    if game.discard_quota[player] < 1:
        raise ValueError(f"player {player} owes no discard")
    hand = game._state.hands[player]
    if is_hidden(hand):
        # A discard is public; of a hidden hand only the count can be checked.
        if len(hand) < 1:
            raise ValueError(f"player {player} holds no cards")
        hand.move(resource, -1)
    else:
        if hand[resource] < 1:
            raise ValueError(f"player {player} holds no {resource.name}")
        hand[resource] -= 1
    game._state.bank[resource] += 1
    game.ledger.spend(player, int(resource), 1)
    game.discard_quota[player] -= 1
    _finish_discards(game)


def move_robber_to(game: Game, target: int, victim: int | None = None) -> None:
    """Resolve the robber phase, entered after a seven or after a knight --
    the same phase and the same move either way (rulebook: a knight "acts
    like the dice roll of a 7").

    `victim` must be one of `robber.victims` on `target` when there is any,
    and `None` only when there is none: a seat with cards on the hex has to
    be robbed.

    Returns to `game.resume_phase` and runs the trade event unconditionally;
    `run_trade_event` no-ops itself outside `MAIN`.
    """
    _require(game, Phase.ROBBER)
    state = game._state
    thief = game.current_player
    check_target(state, target)
    allowed = allowed_targets(state, thief, game.robber_allowed)
    if allowed is not None and target not in allowed:
        raise ValueError(f"the table does not allow the robber on hex {target}")
    robbable = victims(state, target, thief)
    if victim is None and robbable:
        raise ValueError(f"the robber on hex {target} must rob one of seats {list(robbable)}")
    if victim is not None and victim not in robbable:
        raise ValueError(f"seat {victim} cannot be robbed on hex {target}")
    # The steal draws before the robber moves, so a chance source that
    # refuses (a host's stream out of step) leaves the board as it was.
    stolen = None if victim is None else steal(state, thief, victim, game.chance)
    move_robber(state, target)
    game.robber_allowed = None      # the rule was for this move alone
    if victim is not None:
        _record_steal(game, thief, victim, stolen)
    game.phase = game.resume_phase
    run_trade_event(game)


def _check_win(game: Game) -> None:
    """End the game if `game.current_player` is at the win threshold.

    Scoped to the seat on the move, as the rulebook scopes it. A seat's own
    action can move VPs it does not own -- breaking an opponent's Longest
    Road can hand the tile to a third, off-turn seat already on 8 -- and
    that seat does not win here; it wins when its own turn's `_check_win`
    runs.

    On an observed state a seat whose cards are hidden may hold victory
    point cards nobody here can count, so only its public points are read:
    they decide a win they reach, and otherwise it is the host who says the
    game is over (`end`)."""
    state = game._state
    player = game.current_player
    points = (
        public_victory_points(state, player)
        if is_hidden(state.dev_cards[player]) or is_hidden(state.new_dev_cards[player])
        else victory_points(state, player)
    )
    if points >= state.rules.winning_points:
        game.won_by = player
        game.phase = Phase.GAME_OVER


def pending_free_roads(game: Game) -> list[int]:
    """Placeable roads still owed by a Road Building card.

    On the hot path, so the piece limit is read once rather than once per
    edge: `can_place_road` opens with a `road_count`, itself a scan of every
    edge, which would make this quadratic in the board. `road_placeable` is
    `can_place_road` without that check, and exists for this.
    """
    if game.free_roads <= 0:
        return []
    state = game._state
    player = game.current_player
    if road_count(state, player) >= MAX_ROADS:
        return []
    return [
        edge for edge in range(state.board.topology.num_edges)
        if road_placeable(state, player, edge)
    ]


def build_road(game: Game, edge: int) -> None:
    """Build a road, spending a free road from road building if one is owed.

    Like every entry point here, it checks the whole action before it
    changes anything: an illegal call raises `ValueError` and leaves the
    game exactly as it was."""
    free = game.free_roads > 0
    if game.phase is not Phase.ROLL or not free:
        _require(game, Phase.MAIN)
    check_road(game._state, game.current_player, edge)
    before = _snapshot_hands(game)
    if free:
        game.free_roads -= 1
    else:
        pay(game._state, game.current_player, Purchase.ROAD)
    game.ledger.apply_hand_diff(before, game._state.hands)
    place_road(game._state, game.current_player, edge)
    update_longest_road(game._state, road_owner=game.current_player)
    _check_win(game)


def build_settlement(game: Game, vertex: int) -> None:
    """Build and pay for the current seat's settlement on `vertex`, ending the
    game if it wins. Raises `ValueError`, changing nothing, outside
    `Phase.MAIN`, while a Road Building road is owed, or when the seat cannot
    settle there or afford it."""
    _require_main(game)
    check_settlement(game._state, game.current_player, vertex)
    before = _snapshot_hands(game)
    pay(game._state, game.current_player, Purchase.SETTLEMENT)
    game.ledger.apply_hand_diff(before, game._state.hands)
    place_settlement(game._state, game.current_player, vertex)
    # A new settlement can cut an opponent's route, so this is not only the
    # builder's own longest road that may change.
    update_longest_road(
        game._state, settlement_vertex=vertex, settlement_owner=game.current_player
    )
    _check_win(game)


def build_city(game: Game, vertex: int) -> None:
    """Build and pay for the current seat's city on its settlement at `vertex`,
    ending the game if it wins. Raises `ValueError`, changing nothing, outside
    `Phase.MAIN`, while a Road Building road is owed, or when the seat cannot
    upgrade there or afford it."""
    _require_main(game)
    check_city(game._state, game.current_player, vertex)
    before = _snapshot_hands(game)
    pay(game._state, game.current_player, Purchase.CITY)
    game.ledger.apply_hand_diff(before, game._state.hands)
    upgrade_to_city(game._state, game.current_player, vertex)
    _check_win(game)


def buy_development_card(game: Game) -> DevCard | int:
    """Buy the top card. On an observed state the deck is a length and the
    host drew the card: `game.chance.draw()` says which, or `chance.UNSEEN`
    for a seat whose cards are hidden -- which is then what this returns."""
    _require_main(game)
    state = game._state
    if not state.deck:
        raise ValueError("the development deck is empty")
    # Checked before the host is asked which card it drew, so a refused
    # purchase consumes no chance event.
    check_afford(state, game.current_player, Purchase.DEV_CARD)
    before = _snapshot_hands(game)
    if is_hidden(game._state.deck):
        card = buy_drawn(game._state, game.current_player, game.chance.draw())
    else:
        card = buy_dev_card(game._state, game.current_player)
    game.ledger.apply_hand_diff(before, game._state.hands)
    _check_win(game)
    return card


def _check_turn_card(game: Game, name: str) -> None:
    """Whether a development card may be played now at all: before rolling
    or in `MAIN`, and only one a turn. `dev_card_played` is set by the play
    itself, once everything about it has been checked."""
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play {name} in {game.phase.name}")
    if game.phase is Phase.MAIN:
        _require_main(game)
    if game.dev_card_played:
        raise ValueError("only one development card may be played per turn")


def play_knight_card(game: Game) -> None:
    """Play a knight: legal before rolling or in the Action phase.

    Resolves in two steps. This spends the card, credits Largest Army and
    checks the win -- a knight reaching the winning total ends the game
    immediately, with **no robber move at all**. Otherwise the robber move
    happens through the same robber phase a seven enters, with
    `game.resume_phase` remembering where to return; no steal is possible
    until that phase's own `MOVE_ROBBER` names a victim.
    """
    _check_turn_card(game, "a knight")
    play_knight(game._state, game.current_player)
    game.dev_card_played = True
    update_largest_army(game._state)
    _check_win(game)
    if game.phase is Phase.GAME_OVER:
        return
    game.resume_phase = game.phase
    game.phase = Phase.ROBBER


def play_road_building_card(game: Game) -> None:
    """Credit two free roads, placed afterwards with ordinary build actions
    -- which keeps the action space flat: one entry for the card rather than
    one per pair of edges.

    Legal in `ROLL` as well as `MAIN`, and the free `build_road` calls
    likewise resolve in `ROLL` before dice are drawn. Paid building remains
    MAIN-only.
    """
    _check_turn_card(game, "road building")
    if road_count(game._state, game.current_player) >= MAX_ROADS:
        raise ValueError("road building needs a road piece left")
    spend_card(game._state, game.current_player, DevCard.ROAD_BUILDING)
    game.dev_card_played = True
    game._state.dev_cards_played[DevCard.ROAD_BUILDING] += 1
    game.free_roads += ROAD_BUILDING_ROADS


def play_year_of_plenty_card(game: Game, resources: list[Resource]) -> None:
    """Legal in `ROLL` as well as `MAIN`; see `play_road_building_card`."""
    _check_turn_card(game, "year of plenty")
    before = _snapshot_hands(game)
    play_year_of_plenty(game._state, game.current_player, resources)
    game.dev_card_played = True
    game.ledger.apply_hand_diff(before, game._state.hands)


def play_monopoly_card(game: Game, resource: Resource) -> int:
    """Legal in `ROLL` as well as `MAIN`; see `play_road_building_card`."""
    _check_turn_card(game, "monopoly")
    before = _snapshot_hands(game)
    taken = play_monopoly(game._state, game.current_player, resource, game.chance)
    game.dev_card_played = True
    # Monopoly's transfer is fully public despite touching every seat at
    # once, so it takes `apply_hand_diff`, not a steal's hidden path.
    game.ledger.apply_hand_diff(before, game._state.hands)
    return taken


def trade_with_bank(game: Game, give: Resource, receive: Resource) -> None:
    """The current seat's `economy.bank_trade` of `give` for one `receive`.
    Raises `ValueError` outside `Phase.MAIN`, while a Road Building road is
    owed, or as `bank_trade` does."""
    _require_main(game)
    before = _snapshot_hands(game)
    bank_trade(game._state, game.current_player, give, receive)
    game.ledger.apply_hand_diff(before, game._state.hands)


def _run_trade_rounds(game: Game, gates) -> list[Trade]:
    """This turn's trade rounds, under the acting gate's own
    `trade_offer_budget`, which caps offers rather than trades -- a round
    clears at most one exchange. The table has no say: how hard a seat
    bargains is the seat's own business.

    `-1` -- what a gate declaring no limit reads as -- keeps offering until
    the actor runs out of offers it has not made, rather than stopping at the
    first refusal. `already_offered` is what makes that terminate:
    `default_offer` is a pure function of the position, so a rerun would
    repeat the same bundle for ever.
    """
    actor = game.current_player
    gate = gates[actor] if gates is not None and 0 <= actor < len(gates) else None
    budget = params_of(gate).trade_offer_budget
    completed: list[Trade] = []
    offered: set = set()
    made: list = []
    asked = 0
    while budget < 0 or asked < budget:
        asked += 1
        before = len(made)
        cleared = trade_round(game, gates, already_offered=offered, offers_made=made,
                              counter_steps=game.counter_steps)
        if len(made) == before:
            break
        completed.extend(cleared)
    return completed


def run_trade_event(game: Game) -> None:
    """Clear this turn's one trade event for the current player, if anybody
    is seated to answer a gate.

    Runs at most once a turn, called from `enter_main` and
    `move_robber_to` -- the two transitions into `MAIN` -- and guarded by
    `trade_event_turn`, so a knight whose robber move re-enters `MAIN` does
    not reopen it. A game whose `gates` is `None` does not trade.

    When the event opens is the actor's gate's own business. A gate with a
    `trade_now(game) -> bool` method is asked at every main-phase decision
    point until the event is used -- on entering `MAIN`, and again after
    each of its main-phase actions (`actions.apply`) -- and the event runs
    at the first point it says yes. Asked with the `Game`, exactly as
    `choose` is, because it is the same decision: the choice of this ply
    includes whether to go to the table first. A "no" leaves the window
    open; a turn it never says yes in does not trade. A gate without
    `trade_now` opens on entering `MAIN`.
    """
    if game.phase is not Phase.MAIN:
        return
    if game.trade_mode not in ("round", "auto", "external"):
        raise ValueError(f"unknown trade mode: {game.trade_mode!r}")
    if game.trade_event_turn == game.turns:
        return
    gates = game.gates
    if gates is not None and game.trade_mode != "external":
        seated = gates[game.current_player] if 0 <= game.current_player < len(gates) else None
        ask = getattr(seated, "trade_now", None)
        if ask is not None and not ask(game):
            return
    game.trade_event_turn = game.turns
    if gates is None or game.trade_mode == "external":
        return
    # A gate that makes no offers is not asked at all, under either
    # mechanism: refusing to open is the seat's own declaration, and the
    # table has nothing to add to it.
    actor_gate = gates[game.current_player] if 0 <= game.current_player < len(gates) else None
    if params_of(actor_gate).trade_offer_budget == 0:
        return
    if game.trade_mode == "auto":
        # The clearing house ranks with the raw `gains_many` and never asks a
        # gate to price its own consent, so a policy whose limits live in
        # `consent_gain` -- an outgoing-card cap, a per-card risk -- is
        # silently not in force here. Refuse rather than run it out of spec.
        for seat, gate in enumerate(gates):
            if getattr(gate, "consent_gain", None) is not None:
                raise ValueError(
                    f"seat {seat}'s gate prices its own consent, which the "
                    f'automatic clearing house does not ask for: run it under '
                    f'trade_mode="round"'
                )
    hand_sizes = event_hand_sizes(game)
    if game.trade_mode == "auto":
        completed = trade_event(game)
    else:
        completed = _run_trade_rounds(game, gates)
    publish_trade_event(game, gates, hand_sizes, completed)


def event_hand_sizes(game: Game) -> tuple[int, ...]:
    """Every seat's hand size as a trade event opens -- public, and what
    `publish_trade_event` reports alongside the exchanges it cleared."""
    return tuple(hand_size(game._state, s) for s in range(game._state.num_players))


def publish_trade_event(
    game: Game, gates, hand_sizes: tuple[int, ...], completed: Sequence[Trade]
) -> None:
    """Tell every gate with `observe_trade` what this turn's event did: the
    turn, the actor, the hand sizes it opened with and who completed an
    exchange. Public facts only. Called once per event by the engine's own
    driver and once per closed round by the served table, so a bot's public
    activity model sees the same stream whichever table it sits at; an
    observer ignores a repeated turn.
    """
    observers = [observe for gate in gates
                 if gate is not None and (observe := getattr(gate, "observe_trade", None)) is not None]
    if not observers:
        return
    participants = tuple((trade.a, trade.b) for trade in completed)
    for observe in observers:
        observe(turn=game.turns, actor=game.current_player,
                hand_sizes=hand_sizes, trade_participants=participants)


def enter_main(game: Game) -> None:
    """Enter the main phase and clear this turn's first trade event.

    Every path from `ROLL` or `ROBBER` into `MAIN` goes through here, so no
    driver has to remember to run the event itself.
    """
    game.phase = Phase.MAIN
    run_trade_event(game)


def end_turn(game: Game) -> None:
    """End the current seat's turn and start the next playing seat's at
    `Phase.ROLL`, or end the game at the turn cap. Raises `ValueError` outside
    `Phase.MAIN` or while a Road Building road is owed."""
    _require_main(game)
    _close_turn(game)


def _close_turn(game: Game) -> None:
    """End `current_player`'s turn, whatever phase it is in, and start the
    next playing seat's at `ROLL`: every per-turn field is reset, so nothing
    of this turn survives into the next. `end_turn` is this from `MAIN`;
    `lock_seat` reaches it from anywhere in a turn."""
    mature(game._state, game.current_player)
    game.dev_card_played = False
    game.trades = []
    game.trades_made = 0
    game.shown = []
    game.pending = []
    # Free roads with nowhere legal to go are simply lost.
    game.free_roads = 0
    game.resume_phase = Phase.MAIN
    game.robber_allowed = None
    game.turns += 1
    if game.turns >= game.turn_cap:
        game.phase = Phase.GAME_OVER
        return
    game.current_player = _next_unlocked(game, game.current_player)
    game.phase = Phase.ROLL
    # A win is announced at the start of a seat's own turn, not only after
    # an action of theirs, so a seat that crossed the threshold off-turn (a
    # Longest Road or Largest Army transfer during someone else's turn) wins
    # here, before it is ever asked for an action.
    _check_win(game)


def is_over(game: Game) -> bool:
    """Whether `game` is in `Phase.GAME_OVER`."""
    return game.phase is Phase.GAME_OVER


def lock_seat(game: Game, seat: int) -> None:
    """Retire `seat`, a no-op if it is already retired. From here on it is
    skipped by the setup snake and by turn rotation, can never be `to_move`,
    and is never a counterparty in a trade event.

    Retiring the seat whose turn it is ends that turn, as `end_turn` would,
    and the next playing seat starts its own at `ROLL` -- it does not
    inherit the rest of the retired seat's. During a discard round the
    hand-off waits until the other seats have discarded; the robber is then
    not moved, there being nobody to move it.

    Nothing happens to its hand or pieces: its board and cards are
    abandoned in place rather than returned to the bank.
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
        # DISCARD is handled above (`_finish_discards` hands the turn on once
        # the round closes); GAME_OVER has no turn to hand off.
        _close_turn(game)
