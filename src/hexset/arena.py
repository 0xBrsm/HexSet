# SPDX-License-Identifier: GPL-3.0-only
"""Play bots against each other and report win rates worth quoting.

Two things here exist because the published Catan work tends to skip them.
Seat position matters, so a lineup is rotated through every seat and each
entrant plays each seat the same number of times. And win rates carry a Wilson
interval, because at the few hundred games prior work reports the answer is
only good to several points either way.

An entrant is a description rather than a constructed bot: a frozen dataclass
of what to build, not a closure that builds it. That is what lets a tournament
fan out over processes, since a closure cannot be pickled, and it means a
lineup can be written into a run manifest and read back verbatim.

Every seat's terminal victory points are kept alongside the winner. A duel
spends a whole game to learn one bit; the losing seats say how close they came,
and subtracting two entrants' points *within* a game cancels most of the board
and dice variance rather than averaging it away. That is what lets a few
hundred games speak to differences worth a fraction of a point.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from multiprocessing import get_context
from typing import TYPE_CHECKING, Callable, Sequence

from .actions import apply
from .board.board import Board, random_base_board
from .economy import hand_size
from .game import Game, is_over, start, to_move
from .placement import PlacementBot
from .state import city_count, road_count, settlement_count
from .victory import victory_points

if TYPE_CHECKING:
    # Runtime imports are local to avoid cycles through preset registration
    # and record.MAX_ACTIONS.
    from .bots import Bot
    from .record import Record

Z_95 = 1.959964

# Runtimes register their network/MCTS factories explicitly.
_ENTRANT_KIND_FACTORIES: dict[str, Callable[[Entrant, Board, random.Random], Bot]] = {}
_NETWORK_KINDS = frozenset({"network", "mcts"})
_RUNTIME_HINT = "requires a runtime loader registered with hexset.clients.netbot.register_entrants"


def register_entrant_kind(kind: str, factory) -> None:
    """Register a bot-building factory for an `Entrant.kind` hexset does not
    implement itself. `factory(entrant, board, rng) -> Bot`."""
    _ENTRANT_KIND_FACTORIES[kind] = factory


def register_preset(name: str, entrant: "Entrant") -> None:
    """Register a named lineup shortcut (`PRESETS[name]`, resolved by
    `entrant_from_name`/`lineup_from_names`) for an entrant hexset does not
    ship itself -- how `hexset.bots.heximax` makes "heximax"
    resolvable by name once imported."""
    PRESETS[name] = entrant


MAX_ACTIONS = 20000


# --- the game law -----------------------------------------------------------
#
# A run's `index`-th game is a pure function of `(seed, index)`: the board from
# one key, every draw the engine makes from another. Two things depend on that
# holding exactly -- a tournament fanned out over processes (`_play_one`, which
# must play the same game whichever worker draws it) and a lockstep collector
# (`hexset.gym.lanes`, which must play the same game whichever lane draws it and
# however many lanes are in flight). It was written twice, once here and once in
# the training package, kept in step by a comment; the four functions below are
# the one copy both call, so "the same game" is a call and not a claim.


def board_key(seed: int, index: int) -> str:
    """The rng key the `index`-th game's board is drawn from."""
    return f"{seed}:{index}:board"


def game_key(seed: int, index: int) -> str:
    """The rng key the `index`-th game's own draws come from.

    Handed out as a string rather than a generator because it is also what a
    `hexset.record.Record` stores as its seed -- a record names the stream it
    replays.
    """
    return f"{seed}:{index}:game"


def deal_board(seed: int, index: int) -> Board:
    """The `index`-th game's board."""
    return random_base_board(random.Random(board_key(seed, index)))


def deal_game(
    seed: int,
    index: int,
    players: int,
    *,
    board: Board | None = None,
    chance: Callable[[random.Random], object] | None = None,
) -> Game:
    """The `index`-th game of run `seed`, started and ready for its first action.

    `board` overrides the dealt one, which is how a caller pins every game to
    one geometry (`compete`'s antithetic pairing deals the board for the pair
    and passes it in here; so does a lane environment asked for a fixed board).

    `chance` wraps the game's own generator before the game is started -- the
    seam `_play_and_record` uses to record every draw. It takes the generator
    rather than being one so that the wrapper is built on exactly the stream
    the game would have used, which is what keeps a recorded game identical to
    the unrecorded one.

    Nothing is seated here: gates, `max_trades` and the loop are the caller's,
    because the seating is what differs between a tournament and a collector
    and the game is what must not.
    """
    if board is None:
        board = deal_board(seed, index)
    rng = random.Random(game_key(seed, index))
    return start(board, players, rng, chance=None if chance is None else chance(rng))


@dataclass(frozen=True)
class Entrant:
    """What to build, not a built bot. Picklable, so it can cross a process."""

    name: str
    kind: str = "heximax"
    # Fitted evaluation weights, or — for `kind="network"` — the path to a
    # training checkpoint. Both are "what this entrant plays with", and both
    # have to survive a pickle to a worker, which a loaded network would not.
    weights: object | None = None
    depth: int = 2
    width: int | None = None
    # None uses the selected bot's search objective.
    stance: str | None = None
    # Whether the opening settlements come from the fitted placement prior
    # rather than from whatever this entrant would otherwise do. Orthogonal to
    # `kind`, so any entrant can be duelled against itself with only the eight
    # setup picks differing.
    placement: bool = False
    # The trade off switch: `0` means this entrant's gate refuses every
    # exchange, so it never trades. Independent of the table's per-turn
    # budget (`Game.max_trades`), and self-imposed rather than engine-wide
    # so a duel can see what trading is worth: only a bot that declines what
    # its opponent still has can price it.
    max_trades: int | None = None
    # `kind="mcts"` only: how many descents the tree gets per decision, and how
    # many it may launch before expanding new leaves. They are separate because
    # a wider wave changes collision rate as well as network batch size.
    simulations: int = 128
    wave: int = 16
    # Heximax follows public trade activity unless pinned for an experiment.
    # This does not toggle willingness to trade (max_trades does that).
    pin_weights: float | None = None
    # Only for explicit custom weight vectors, e.g. fitting or frozen controls.
    expansion_value: float | None = None
    # `kind="heximax"` only: determinized worlds searched per decision (PIMC).
    # P1½ (`heximax.md` §8) found no k > 1 beat k = 1 beyond the instrument's
    # resolution over 400 games; `k = 1` ships and the field stays for anyone
    # who wants to re-open the question, not for a preset to vary.
    k: int = 1
    # `kind="heximax"` only, `win` stance: the temperature the vector is read
    # at, `None` meaning the bot's own constant. A fitted `weights` and its
    # temperature are identified jointly, so a candidate carries both.
    temperature: float | None = None

    def renamed(self, name: str) -> Entrant:
        return replace(self, name=name)


PRESETS: dict[str, Entrant] = {
    "random": Entrant("random", kind="random"),
    "random-placement": Entrant("random-placement", kind="random", placement=True),
}


def spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    bot = _spawn(entrant, board, rng)
    return PlacementBot(bot) if entrant.placement else bot


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    from .bots import RandomBot

    if entrant.kind == "random":
        return RandomBot(rng)
    if entrant.kind == "catanatron" and entrant.kind not in _ENTRANT_KIND_FACTORIES:
        _load_presets(("catanatron",))
    if entrant.kind in _ENTRANT_KIND_FACTORIES:
        return _ENTRANT_KIND_FACTORIES[entrant.kind](entrant, board, rng)
    if entrant.kind in _NETWORK_KINDS:
        raise ValueError(f"entrant kind {entrant.kind!r} {_RUNTIME_HINT}")
    raise ValueError(f"unknown bot kind: {entrant.kind}")


def wilson(wins: int, games: int, z: float = Z_95) -> tuple[float, float]:
    """Score interval for a proportion.

    Preferred to the normal approximation because it stays inside [0, 1] and
    behaves at the extremes, which matters when a baseline shuts an opponent
    out entirely.

    The returned interval is also forced to contain the estimate itself. At
    `p = 1` the upper bound is analytically exactly 1, but in floating point it
    lands an ulp short -- 0.9999999999999999 -- so a clean sweep produced an
    interval that excluded its own point estimate. Only the rounding is being
    corrected here; the arithmetic is unchanged.
    """
    if games == 0:
        return (0.0, 1.0)
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = (
        z * sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    )
    return (min(p, max(0.0, centre - spread)), max(p, min(1.0, centre + spread)))


def seat_of(entrant: int, game: int, seats: int) -> int:
    """Where an entrant sits in a given game.

    A cyclic shift, so over any `seats` consecutive games each entrant occupies
    each seat exactly once and seat bias cancels out of the standings.
    """
    return (entrant + game) % seats


@dataclass(frozen=True)
class Standing:
    name: str
    wins: int
    games: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def interval(self, z: float = Z_95) -> tuple[float, float]:
        return wilson(self.wins, self.games, z)


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    samples: int


def mean_interval(samples: Sequence[float], z: float = Z_95) -> Estimate:
    """Normal interval for a paired mean.

    Callers supply hundreds of bounded integer differences, where the central
    limit approximation is ample. With fewer than two samples there is no
    variance estimate, so the deliberately useless infinite interval reports no
    evidence either way rather than a confident zero.
    """
    if not samples:
        return Estimate(mean=0.0, lower=float("-inf"), upper=float("inf"), samples=0)
    mean = statistics.mean(samples)
    if len(samples) < 2:
        return Estimate(
            mean=mean, lower=float("-inf"), upper=float("inf"), samples=len(samples)
        )
    error = z * statistics.stdev(samples) / len(samples) ** 0.5
    return Estimate(
        mean=mean, lower=mean - error, upper=mean + error, samples=len(samples)
    )


@dataclass(frozen=True)
class ClearedTrade:
    """One exchange the engine cleared, with what a `Record`'s own
    `(step, a, b, received)` tuple deliberately does not carry.

    A record holds what is needed to replay a game; a census needs to
    describe a trade without replaying it -- when it happened, and both
    sides' hands and private gains at the time. `a`/`b` are seats and
    `received` is signed towards `a`, both as `hexset.trading.Trade` has
    them. `hand_a`/`hand_b` are each side's total card count at the start of
    the step this trade cleared inside: the last moment the engine's state
    holds a hand rather than a hand mid-transaction.
    """

    step: int
    turn: int
    phase: str
    a: int
    b: int
    received: tuple[int, ...]
    hand_a: int
    hand_b: int
    gain_a: float
    gain_b: float


@dataclass(frozen=True)
class Tournament:
    standings: tuple[Standing, ...]
    games: int
    unfinished: int
    mean_turns: float
    seconds: float
    # Wins by board seat rather than by entrant. Rotation is what makes the
    # standings fair; this is what shows it worked. With one bot in every seat
    # these should come out even, and a skew would be a bug in setup order or
    # the snake draft rather than anything about the bots.
    seat_wins: tuple[int, ...] = ()
    # Per game, in entrant order: who won, and every seat's terminal points.
    winners: tuple[int | None, ...] = ()
    points: tuple[tuple[int, ...], ...] = ()
    # Per game, in the same order as `winners` and `points`: `game.turns`, the
    # engine's own counter. `mean_turns` already folds this down to one number;
    # this is the raw sequence a caller needs to ask a finer question of it —
    # e.g. whether length moves with a parameter, which a mean cannot answer.
    turns: tuple[int, ...] = ()
    # Per game, in entrant order alongside `points`: what each entrant had
    # standing on the board at the end. Roads are what a weight sweep reads to
    # see *how* a vector won rather than only that it did, so the census is
    # kept here rather than in a bench script's own copy of the play loop --
    # counting three fields off a terminal state costs nothing next to the
    # game that produced it.
    roads: tuple[tuple[int, ...], ...] = ()
    settlements: tuple[tuple[int, ...], ...] = ()
    cities: tuple[tuple[int, ...], ...] = ()
    # Per game: which seat each entrant took, in entrant order. The rotation
    # is what makes the standings fair, and a caller reading anything the
    # engine reports by *seat* -- a cleared trade, say -- needs it to say
    # which entrant that seat was.
    seating: tuple[tuple[int, ...], ...] = ()
    # Per game, every trade the engine cleared, in order. Filled only when
    # `compete(records=True)` asked, alongside `records`: a `Record` is what
    # replays a game, this is what describes it.
    cleared: tuple[tuple[ClearedTrade, ...], ...] = ()
    # One `Record` per game, in the same order as `winners`/`points`/`turns`
    # -- only when `compete(records=True)` asked for them (empty otherwise,
    # never partially filled). `hexset.bench.duel`'s `--records` is the
    # caller: every game a duel counts toward its verdict is also what a
    # `--records` file holds, because both come from the same job.
    records: tuple["Record", ...] = ()

    def seat_balance(self, z: float = Z_95) -> list[tuple[int, int, tuple[float, float]]]:
        decided = sum(self.seat_wins)
        return [
            (seat, wins, wilson(wins, decided, z))
            for seat, wins in enumerate(self.seat_wins)
        ]

    def decided(self) -> list[tuple[int, tuple[int, ...]]]:
        """Winner and every seat's points, for the games that reached a winner."""
        return [
            (winner, row)
            for winner, row in zip(self.winners, self.points)
            if winner is not None
        ]


def play(
    bots: Sequence[Bot],
    board: Board,
    rng: random.Random,
    *,
    action_cap: int = MAX_ACTIONS,
) -> Game:
    """One game, each bot seated at its own index.

    Seating a bot also seats its private gate: `game.gates` is the lineup
    itself, so the engine's one trade event a turn (`hexset.trading`) asks
    each seat's own `gains_many` rather than this loop having to remember to
    run anything -- a gate is a pure function of the position, asked fresh
    at every event, so there is nothing for this loop to publish. A bot
    that defines none of `gains_many`/`accepts_many`/`accepts` never
    trades, which is how `RandomBot` and any external bot that predates the
    mechanic behave.
    """
    return play_game(start(board, len(bots), rng), bots, action_cap=action_cap)


def play_game(
    game: Game,
    bots: Sequence[Bot],
    *,
    action_cap: int = MAX_ACTIONS,
    trade_mode: str = "round",
    max_trades: int = 1,
) -> Game:
    """`play`'s loop over a game somebody else dealt (`deal_game`).

    Dealing a game and playing it are separate acts: a tournament deals from
    `(seed, index)` and plays it here, while a lockstep environment deals from
    the same law and steps the loop itself, one action per lane per tick.

    `trade_mode`/`max_trades` are the table's bargaining rules, and default to
    one propose-and-respond round per turn. Pass `trade_mode="auto"` and
    `max_trades=-1` for the exhaustive automatic house, which is what
    every study recorded before this defaulted the other way -- see
    `Game.trade_mode` for why that is a comparability option now rather than
    the thing to fit against.
    """
    game.gates = tuple(bots)
    game.trade_mode = trade_mode
    game.max_trades = max_trades
    actions = 0
    while not is_over(game) and actions < action_cap:
        seat = to_move(game)
        bot = bots[seat]
        apply(game, bot.choose(game))
        actions += 1
    return game


@dataclass(frozen=True)
class Outcome:
    """One played game, as `_play_one` hands it back to `compete`.

    `points`/`roads`/`settlements`/`cities` are in entrant rather than seat
    order, so they can be compared across games that rotated the lineup
    differently; `seating` is the rotation that produced them. The cleared
    trades stay in seat order, because that is the order the engine reports
    them in and `seating` is what turns one into the other.
    """

    winner: int | None
    seat: int | None
    turns: int
    seating: tuple[int, ...]
    points: tuple[int, ...]
    roads: tuple[int, ...]
    settlements: tuple[int, ...]
    cities: tuple[int, ...]
    cleared: tuple[ClearedTrade, ...]
    record: "Record | None"


def _play_one(
    job: tuple[tuple[Entrant, ...], int, int, int, bool, bool, str, int],
) -> Outcome:
    """Play game `index` and return its `Outcome`.

    `record` is a `hexset.record.Record` of the game just played when the
    job's `records` flag is set, `None` otherwise -- never partially built,
    so a caller that never asked for one never pays the extra bookkeeping
    either. The cleared-trade census rides along under the same flag, for
    the same reason.

    Module level and taking only picklable arguments, so a pool can call it.
    Every random stream is derived from the seed and the game index, so a game
    plays identically whichever worker draws it and however many there are.
    """
    entrants, index, seed, action_cap, antithetic, records, trade_mode, max_trades = job
    seats = len(entrants)
    # Antithetic pairing: the board comes from the pair, the rotation from the
    # position within it, so the two halves of a pair are the same board played
    # under complementary seat assignments and the seat term cancels per board
    # rather than only in the mean. `seats // 2` is the complementary shift --
    # with an [a, a, b, b] lineup it exchanges the two sides' seat pairs exactly.
    #
    # Deliberately two-fold and not the full `seats`-way rotation. A full
    # rotation also cancels the seat term but costs `seats`x the distinct
    # boards, and the board-to-board variance it gives up is not reduced by
    # replaying one board. At the measured ratio -- the seat residual is ~55% of
    # a duel's variance -- the four-way trade is a net loss and the two-way one
    # is a net gain.
    if antithetic:
        pair, half = divmod(index, 2)
        board_index = pair
        rotation = pair + half * (seats // 2)
    else:
        board_index = rotation = index
    board = deal_board(seed, board_index)
    seats_taken = [seat_of(e, rotation, seats) for e in range(seats)]

    # Every stream keys off `board_index`, not `index`. Under antithetic the two
    # halves of a pair must differ in the seat assignment and in *nothing else*
    # -- same board, same dice, same per-entrant stream -- or the seat term does
    # not cancel and the pair is simply two different games. Keying the dice to
    # `index` here made an identical-entrant self-duel read 20/28 instead of the
    # 24/24 the design guarantees.
    lineup: list[Bot] = [None] * seats  # type: ignore[list-item]
    for e, entrant in enumerate(entrants):
        lineup[seats_taken[e]] = spawn(
            entrant, board, random.Random(f"{seed}:{board_index}:{e}")
        )

    record = None
    cleared: tuple[ClearedTrade, ...] = ()
    if records:
        game, record, cleared = _play_and_record(
            lineup, board, seed, board_index, action_cap,
            trade_mode=trade_mode, max_trades=max_trades,
        )
    else:
        game = play_game(
            deal_game(seed, board_index, seats, board=board),
            lineup,
            action_cap=action_cap,
            trade_mode=trade_mode,
            max_trades=max_trades,
        )
    # true state: the verdict's own victory points include hidden
    # victory-point dev cards, so the final score and the build census are
    # both read off the truth rather than off one seat's view of it.
    states = [game.state(seats_taken[e], hidden=False) for e in range(seats)]
    won = None if game.won_by is None else seats_taken.index(game.won_by)
    return Outcome(
        winner=won,
        seat=None if won is None else game.won_by,
        turns=game.turns,
        seating=tuple(seats_taken),
        points=tuple(victory_points(s, seats_taken[e]) for e, s in enumerate(states)),
        roads=tuple(road_count(s, seats_taken[e]) for e, s in enumerate(states)),
        settlements=tuple(
            settlement_count(s, seats_taken[e]) for e, s in enumerate(states)
        ),
        cities=tuple(city_count(s, seats_taken[e]) for e, s in enumerate(states)),
        cleared=cleared,
        record=record,
    )


def _play_and_record(
    lineup: list[Bot],
    board: Board,
    seed: int,
    index: int,
    action_cap: int,
    *,
    trade_mode: str = "round",
    max_trades: int = 1,
) -> tuple[Game, "Record", tuple[ClearedTrade, ...]]:
    """`play`'s own loop, with a `hexset.record.Tape` running alongside it.

    `--records` must record exactly the game `play` would have played, not
    an approximation of it, so this is not `play` calling out to a separate
    recorder but the same loop instrumented in place -- and the instrument
    is the same `Tape` `record_game` and `hexset.gym.lanes` file their steps
    on, so a tournament's records and a lane's are records of the same kind.

    The same pass builds the `ClearedTrade` census, off the one before/after
    `game.trades` diff the record already takes. A census used to mean a
    second copy of this loop in a bench script, which is how it came to
    report each side's hand *after* the turn's trades under the name
    `hand_before`: here the hand sizes are read once, off the true state, at
    the top of the step the trade cleared inside.
    """
    from .record import Tape, recording

    game = deal_game(seed, index, len(lineup), board=board, chance=recording)
    game.gates = tuple(lineup)
    game.trade_mode = trade_mode
    game.max_trades = max_trades
    tape = Tape()
    cleared: list[ClearedTrade] = []
    while not is_over(game) and len(tape.actions) < action_cap:
        seat = to_move(game)
        bot = lineup[seat]
        before = len(game.trades)
        # true state: a census of who was flush cannot be read off one seat's
        # view of the table.
        true_state = game.state(0, hidden=False)
        hands = [hand_size(true_state, s) for s in range(game.num_players)]
        turn, phase = game.turns, game.phase.name
        step = len(tape.actions)
        action = bot.choose(game)
        apply(game, action)
        for trade in game.trades[before:]:
            cleared.append(
                ClearedTrade(
                    step=step,
                    turn=turn,
                    phase=phase,
                    a=trade.a,
                    b=trade.b,
                    received=tuple(trade.received),
                    hand_a=hands[trade.a],
                    hand_b=hands[trade.b],
                    gain_a=trade.gain_a,
                    gain_b=trade.gain_b,
                )
            )
            # Several trades can clear inside one step; each later one sees
            # the hands the earlier ones left behind.
            moved = sum(trade.received)
            hands[trade.a] += moved
            hands[trade.b] -= moved
        tape.step(action, game.trades[before:])

    return game, tape.sealed(game, seed=game_key(seed, index)), tuple(cleared)


def compete(
    entrants: Sequence[Entrant],
    games: int,
    *,
    seed: int = 0,
    action_cap: int = MAX_ACTIONS,
    workers: int = 1,
    antithetic: bool = True,
    records: bool = False,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    start_method: str | None = None,
    trade_mode: str = "round",
    max_trades: int = 1,
) -> Tournament:
    """Run `games` games, rotating the lineup so every entrant sits every seat.

    `trade_mode`/`max_trades` are the table's bargaining rules for every game
    (`play_game`): the default is one propose-and-respond round a turn;
    `trade_mode="auto"` with `max_trades=-1` is the unbounded automatic
    clearing house every study before HexSet 0.50 was recorded under, kept
    so a policy trained against it can be read in its native environment.

    `games` must be a multiple of the lineup size, otherwise the rotation is
    incomplete and the seat bias it exists to cancel leaks into the result.
    Antithetic runs with odd seat counts require twice that many games to
    complete both halves of every board pair and balance all seats.

    `workers` only changes the wall clock. Results are identical at any worker
    count, which is the property that makes a parallel run quotable.
    Custom runtimes can register their entrant factories with
    ``worker_initializer(*worker_initargs)``. It runs once in each worker
    (or in the calling process for workers=1). With spawn/forkserver the
    initializer must be importable at module scope. ``start_method`` selects
    a multiprocessing context without changing the process-wide default.

    `records=True` has every job build a `hexset.record.Record` of its own
    game alongside the verdict (`_play_and_record`), returned as
    `Tournament.records` in the same order as `winners`/`points`/`turns` --
    the games a `--records` file holds are exactly the games the verdict
    counted, because both come from the one job. Off by default: the extra
    bookkeeping is skipped entirely, not merely discarded, when nobody asks.
    """
    seats = len(entrants)
    if seats < 2:
        raise ValueError("a tournament needs at least two entrants")
    if games <= 0:
        raise ValueError("games must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if action_cap <= 0:
        raise ValueError("action_cap must be positive")
    if games % seats:
        raise ValueError(f"{games} games does not divide evenly over {seats} seats")
    if antithetic and seats % 2 and games % (2 * seats):
        raise ValueError("antithetic runs with odd seat counts require games divisible by twice the seat count")

    lineup = tuple(entrants)
    jobs = [
        (lineup, i, seed, action_cap, antithetic, records, trade_mode, max_trades)
        for i in range(games)
    ]
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context(start_method),
            initializer=worker_initializer, initargs=worker_initargs,
        ) as pool:
            outcomes = list(pool.map(_play_one, jobs, chunksize=1))
    else:
        if worker_initializer is not None:
            worker_initializer(*worker_initargs)
        outcomes = [_play_one(job) for job in jobs]
    elapsed = time.perf_counter() - started

    wins = [0] * seats
    seat_wins = [0] * seats
    for outcome in outcomes:
        if outcome.winner is not None:
            wins[outcome.winner] += 1
            seat_wins[outcome.seat] += 1

    return Tournament(
        standings=tuple(
            Standing(name=entrant.name, wins=wins[e], games=games)
            for e, entrant in enumerate(lineup)
        ),
        games=games,
        unfinished=sum(1 for o in outcomes if o.winner is None),
        mean_turns=statistics.mean(o.turns for o in outcomes) if outcomes else 0.0,
        seconds=elapsed,
        seat_wins=tuple(seat_wins),
        winners=tuple(o.winner for o in outcomes),
        points=tuple(o.points for o in outcomes),
        turns=tuple(o.turns for o in outcomes),
        roads=tuple(o.roads for o in outcomes),
        settlements=tuple(o.settlements for o in outcomes),
        cities=tuple(o.cities for o in outcomes),
        seating=tuple(o.seating for o in outcomes),
        cleared=tuple(o.cleared for o in outcomes) if records else (),
        records=tuple(o.record for o in outcomes) if records else (),
    )


# A checkpoint cannot be a preset, since its path is only known at the command
# line. `network:/path/to/latest.pt` names one wherever a preset name is taken,
# and the entrant it builds still pickles to a worker verbatim.
NETWORK = "network:"

MCTS = "mcts:"


def _load_presets(names: Sequence[str]) -> None:
    # Resolve built-ins even when the caller imports only hexset.arena.
    from . import bots  # noqa: F401

    if "catanatron" in names and "catanatron" not in PRESETS:
        try:
            from .catanatron import bot  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "catanatron" or exc.name.startswith("catanatron.")):
                raise ValueError("catanatron requires the 'catanatron' extra") from exc
            raise


def entrant_from_name(name: str) -> Entrant:
    """Resolve a preset, a Heximax testing spec, or a checkpoint spec."""
    if name.startswith("heximax:"):
        _load_presets(("heximax",))
        settings = {}
        for option in name.split(":")[1:]:
            key, separator, value = option.partition("=")
            if not separator or key in settings:
                raise ValueError(f"invalid or repeated Heximax option: {option!r}")
            if key == "pin-weights" and value in ("0", "1"):
                settings[key] = int(value)
            elif key == "trading" and value in ("on", "off"):
                settings[key] = value
            else:
                raise ValueError(f"unknown Heximax option: {option!r}; use pin-weights=0|1 or trading=on|off")
        return replace(
            PRESETS["heximax"], name=name,
            pin_weights=settings.get("pin-weights"),
            max_trades=0 if settings.get("trading") == "off" else None,
        )
    if name.startswith(NETWORK):
        # `network:<path>@<trades>` switches trading off with `@0`, mirroring
        # `mcts:<path>@<simulations>`. Nothing else is a meaningful value:
        # the engine has no trade budget to tune, only an off switch.
        path, separator, trades = name[len(NETWORK) :].partition("@")
        return Entrant(
            name=f"network-trades{trades}" if separator else "network",
            kind="network",
            weights=path,
            max_trades=int(trades) if separator else None,
        )
    if name.startswith(MCTS):
        path, _, search = name[len(MCTS) :].partition("@")
        budget, separator, wave = search.partition("w")
        simulations = int(budget) if budget else 128
        width = int(wave) if separator else 16
        label = f"mcts{simulations}w{width}" if separator else f"mcts{budget}"
        return Entrant(
            name=label if budget or separator else "mcts",
            kind="mcts",
            weights=path,
            simulations=simulations,
            wave=width,
        )
    _load_presets((name,))
    return PRESETS[name]


CHECKPOINT_KINDS = (NETWORK, MCTS)


def lineup_from_names(names: Sequence[str]) -> list[Entrant]:
    """Resolve names to entrants, numbering repeats so standings stay readable."""
    _load_presets(names)
    unknown = sorted(
        name
        for name in set(names)
        if name not in PRESETS and not name.startswith((*CHECKPOINT_KINDS, "heximax:"))
    )
    if unknown:
        raise ValueError(f"unknown bots: {', '.join(unknown)}")
    entrants = [entrant_from_name(name) for name in names]
    taken = [entrant.name for entrant in entrants]
    repeated = {name for name in taken if taken.count(name) > 1}
    seen: dict[str, int] = {}
    lineup = []
    for entrant in entrants:
        if entrant.name in repeated:
            base = entrant.name
            entrant = entrant.renamed(f"{base}#{seen.get(base, 0)}")
            seen[base] = seen.get(base, 0) + 1
        lineup.append(entrant)
    return lineup


def base_name(name: str) -> str:
    """The name before the repeat number, so two seats of one bot pool."""
    return name.split("#", 1)[0]


def pooled(standings: Sequence[Standing], games: int) -> list[Standing]:
    """Standings grouped by base name.

    A duel is two entrants a side, so the number worth quoting is the side's
    share of the games rather than either seat's quarter of them. Grouping here
    rather than in the reader's head is what stops "60.8%" and "30.4%" being the
    same measurement written two ways.
    """
    order: list[str] = []
    wins: dict[str, int] = {}
    for standing in standings:
        name = base_name(standing.name)
        if name not in wins:
            order.append(name)
        wins[name] = wins.get(name, 0) + standing.wins
    return [Standing(name=name, wins=wins[name], games=games) for name in order]
