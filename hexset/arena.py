# SPDX-License-Identifier: GPL-3.0-only
"""Play bots against each other and report win rates worth quoting.

Seat position matters, so a lineup is rotated through every seat and each
entrant plays each seat the same number of times; win rates carry a Wilson
interval, because at a few hundred games the answer is only good to several
points either way.

An entrant is a description rather than a constructed bot -- a frozen
dataclass of what to build, not a closure -- so a tournament can fan out
over processes and a lineup can be read back verbatim from a manifest.

Every seat's terminal victory points are kept alongside the winner:
subtracting two entrants' points *within* a game cancels most of the board
and dice variance rather than averaging it away.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from math import sqrt
from multiprocessing import get_context
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

from . import gamelog
from .actions import apply
from .board.board import Board, random_base_board
from .economy import hand_size
from .game import MAX_TURNS, Game, is_over, start, to_move
from .rules import STANDARD_GAME, GameType
from .state import city_count, road_count, settlement_count
from .trading import TradeParams
from .victory import victory_points

if TYPE_CHECKING:
    # Runtime imports are local to avoid cycles through preset registration
    # and record.MAX_ACTIONS.
    from .bots import Bot
    from .record import Record, Tape

Z_95 = 1.959964


class Exhausted(RuntimeError):
    """A game reached `MAX_TURNS` with nobody at the win threshold.

    Raised by `compete` and `hexset.bench.versus.compete_batched` on the
    first one, because there is no reading of a stuck game that belongs in a
    measurement: scoring it as a loss for side A is a silent, directional lie
    about strength, and it costs about ten ordinary games of compute besides.
    Carries what it takes to replay the game on its own --
    `deal_game(seed, index, seats)` -- and its `seating`: the seat each
    entrant took under `compete`, the policy id in each seat under
    `compete_batched`.

    Not raised by `play_game`, so a served table with a wedged bot is the
    server's problem to survive rather than an exception through a live game.
    """

    def __init__(self, *, seed: int, index: int, seating: tuple[int, ...], turns: int):
        self.seed, self.index, self.seating, self.turns = seed, index, seating, turns
        super().__init__(
            f"game {index} of seed {seed} reached {turns} turns with no winner "
            f"(seating {list(seating)}). A game that cannot finish is a defect, "
            f"not a result; it is not scored. Replay it with "
            f"deal_game({seed}, {index}, {len(seating)})."
        )

# Bots hexset does not ship register here: a factory per `Entrant.kind`, a
# preset per name, a parser per spec prefix. A runtime module makes those
# calls when imported; `load_runtime` imports it in each process that spawns
# entrants.
_ENTRANT_KIND_FACTORIES: dict[str, Callable[[Entrant, Board, random.Random], Bot]] = {}
_SPEC_PARSERS: dict[str, Callable[[str], "Entrant"]] = {}
_NETWORK_KINDS = frozenset({"network", "mcts"})
_RUNTIME_HINT = "requires a runtime loader registered with hexset.clients.netbot.register_entrants"

#: The kinds and presets hexset resolves itself, and the spec prefixes it
#: parses itself. No runtime registers over them: a lineup naming `random`
#: means the same bot in every process. `catanatron`'s own module registers
#: the `catanatron` names when its extra is loaded.
SHIPPED_NAMES = frozenset({"random", "retired", "catanatron"})
SHIPPED_PREFIXES = ("network:", "mcts:", "catanatron:")

_ABSENT = object()


def _register(table: dict, key: str, value, what: str) -> None:
    """Add `value` under `key`, or refuse a different one already there.
    Registering the same value again is a no-op, so a runtime imported twice
    is harmless; anything else needs an explicit `unregister_*` first."""
    held = table.get(key, _ABSENT)
    if held is not _ABSENT and held is not value and held != value:
        raise ValueError(
            f"{what} {key!r} is already registered ({held!r}); unregister it "
            f"first to replace it"
        )
    table[key] = value


def _refuse_shipped(name: str, what: str) -> None:
    if name in SHIPPED_NAMES:
        raise ValueError(f"{what} {name!r} is one hexset ships; give yours another name")


def register_entrant_kind(kind: str, factory) -> None:
    """Register a bot-building factory for an `Entrant.kind` hexset does not
    implement itself. `factory(entrant, board, rng) -> Bot`.

    A kind hexset ships (`SHIPPED_NAMES`) is refused, and so is a second,
    different factory for a kind already registered; the same factory again
    is a no-op. The checkpoint kinds `network` and `mcts` are the exception:
    they build from whatever checkpoint loader the process installed
    (`hexset.clients.netbot.register_entrants`), and installing another
    replaces it."""
    _refuse_shipped(kind, "entrant kind")
    if kind in _NETWORK_KINDS:
        _ENTRANT_KIND_FACTORIES[kind] = factory
        return
    _register(_ENTRANT_KIND_FACTORIES, kind, factory, "entrant kind")


def register_preset(name: str, entrant: "Entrant") -> None:
    """Register a named lineup shortcut (`PRESETS[name]`, resolved by
    `entrant_from_name`/`lineup_from_names`) for an entrant hexset does not
    ship itself. A shipped name is refused, as is a name already registered
    to an unequal entrant; an equal one again is a no-op."""
    _refuse_shipped(name, "preset")
    _register(PRESETS, name, entrant, "preset")


def register_spec(prefix: str, parse: Callable[[str], "Entrant"]) -> None:
    """Register a parser for lineup names starting with `prefix` (say
    ``"mybot:"``): `entrant_from_name` hands it the whole name and seats the
    `Entrant` it returns.

    A name that is a preset resolves as the preset, and the longest
    registered prefix a name starts with is the one that parses it. A prefix
    overlapping one hexset parses itself (`SHIPPED_PREFIXES`) is refused,
    since it would either never be reached or capture shipped names."""
    if not prefix:
        raise ValueError("a spec prefix cannot be empty")
    for shipped in SHIPPED_PREFIXES:
        if prefix.startswith(shipped) or shipped.startswith(prefix):
            raise ValueError(
                f"spec prefix {prefix!r} overlaps {shipped!r}, which hexset parses itself"
            )
    _register(_SPEC_PARSERS, prefix, parse, "spec prefix")


def unregister_entrant_kind(kind: str) -> None:
    """Undo `register_entrant_kind`. A kind never registered is a no-op; a
    shipped one is refused."""
    _refuse_shipped(kind, "entrant kind")
    _ENTRANT_KIND_FACTORIES.pop(kind, None)


def unregister_preset(name: str) -> None:
    """Undo `register_preset`. A name never registered is a no-op; a shipped
    one is refused."""
    _refuse_shipped(name, "preset")
    PRESETS.pop(name, None)


def unregister_spec(prefix: str) -> None:
    """Undo `register_spec`. A prefix never registered is a no-op; a shipped
    one is refused."""
    if prefix in SHIPPED_PREFIXES:
        raise ValueError(f"spec prefix {prefix!r} is one hexset parses itself")
    _SPEC_PARSERS.pop(prefix, None)


def load_runtime(*modules: str) -> None:
    """Import each module for its side effect: its `register_*` calls.
    Strings, not callables, because a spawned worker gets this as its
    `compete(worker_initializer=...)` and a closure would not survive the
    pickle."""
    for module in modules:
        importlib.import_module(module)


#: The `--runtime` help every command that names bots shares.
RUNTIME_HELP = (
    "import this module first, in every worker too, so the bots it registers "
    "can be named (repeatable)"
)


MAX_ACTIONS = 20000


# --- the game law -----------------------------------------------------------
#
# A run's `index`-th game is a pure function of `(seed, index)`: the board from
# one key, every draw the engine makes from another. A tournament fanned out
# over processes and a lockstep collector both depend on that exactly, and both
# call the four functions below rather than deriving the keys themselves.


def board_key(seed: int, index: int) -> str:
    """The rng key the `index`-th game's board is drawn from."""
    return f"{seed}:{index}:board"


def game_key(seed: int, index: int) -> str:
    """The rng key the `index`-th game's own draws come from. A string
    rather than a generator because a `hexset.record.Record` stores it as
    its seed: a record names the stream it replays."""
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
    game_type: GameType = STANDARD_GAME,
    locked: Iterable[int] = (),
    turn_cap: int = MAX_TURNS,
    first: int = 0,
) -> Game:
    """The `index`-th game of run `seed`, started and ready for its first
    action. Nothing is seated: gates, their budgets and the loop are the
    caller's.

    `board` overrides the dealt one, pinning every game to one geometry.
    `chance` takes the game's generator rather than being one, so the
    wrapper is built on exactly the stream the game would have used and a
    recorded game is identical to the unrecorded one.

    `game_type` and `locked` are the table's contract, checked by `start`
    against the seats that will actually play. A run seating fewer players
    than the table holds names them here rather than retiring them after the
    deal, which is what lets the contract be checked at all.

    `first` is the seat that opens the setup snake (`hexset.game.start`).
    """
    if board is None:
        board = deal_board(seed, index)
    rng = random.Random(game_key(seed, index))
    return start(
        board, players, rng,
        chance=None if chance is None else chance(rng),
        game_type=game_type,
        locked=locked,
        turn_cap=turn_cap,
        first=first,
    )


@dataclass(frozen=True)
class Entrant:
    """What to build, not a built bot. Picklable, so it can cross a process."""

    name: str
    # Which factory builds it: one hexset ships (`random`, `retired`,
    # `catanatron`) or one a runtime registered (`register_entrant_kind`).
    kind: str
    # The kind's own data, as its factory reads it: for `network` and
    # `mcts`, the path to a checkpoint. Data rather than a loaded object,
    # because it has to survive a pickle to a worker.
    weights: object | None = None
    # Two generic search settings, for a kind whose factory reads them:
    # `catanatron` reads `depth` as its alpha-beta depth; a kind that reads
    # neither ignores both.
    depth: int = 2
    width: int | None = None
    # Read by `kind="mcts"`: descents per decision, and how many may be
    # launched before expanding new leaves. Separate because a wider wave
    # changes collision rate as well as network batch size.
    simulations: int = 128
    wave: int = 16
    # How this entrant's gate bargains, for any kind that has one: card caps
    # and their direction, clearing floor, responder price, offer budget
    # and fragment plan. `None` takes the bot's own default -- whatever a
    # checkpoint declares for a network. Overriding it is how one arm plays
    # another arm's bargaining with its own valuation.
    trade: "TradeParams | None" = None
    # Determinized worlds searched per decision, for a kind that samples
    # them from the mover's own `View` (`mcts` reads it).
    k: int = 1
    # Settings only the factory for `kind` reads, as `(key, value)` pairs
    # (or a mapping) so the entrant stays frozen and picklable; `option`
    # reads one. Stored sorted by key, so two entrants naming the same
    # settings in a different order are equal, hash equal and journal as
    # one run.
    options: tuple[tuple[str, object], ...] = ()
    # A preset or spec whose bot answers every trade for this entrant
    # (`hexset.bots.TradesBy`); its moves stay this entrant's. `None` trades
    # with the entrant's own gate. Named `<entrant>~<trader>` in a lineup.
    trader: str | None = None

    def __post_init__(self) -> None:
        pairs = self.options.items() if isinstance(self.options, Mapping) else self.options
        options = tuple(sorted((tuple(pair) for pair in pairs), key=lambda pair: pair[0]))
        for pair in options:
            if len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError(f"an entrant option is a (name, value) pair, not {pair!r}")
        keys = [key for key, _ in options]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(f"entrant options name {', '.join(repeated)} more than once")
        object.__setattr__(self, "options", options)
        if self.trader is not None and self.trade is not None:
            raise ValueError(
                "an entrant that trades through a `trader` has no gate of its own "
                "to set limits on; name the limits on the trader instead"
            )

    def option(self, key: str, default: object = None) -> object:
        """The value `options` gives `key`, or `default`."""
        return dict(self.options).get(key, default)

    def renamed(self, name: str) -> Entrant:
        return replace(self, name=name)

    def trade_params(self, default: "TradeParams") -> "TradeParams":
        """The config this entrant names for its gate, falling back to the
        bot's own `default`."""
        return default if self.trade is None else self.trade


#: An entrant that never plays. Its seat is retired before the first action,
#: so the table is smaller than the lineup and the game type is checked
#: against the seats that actually play. This is a lineup entry rather than a
#: post-deal `lock_seat` call because the contract has to be known at the
#: deal: a four-seat lineup with two of these is a two-seat game.
RETIRED = "retired"


class RetiredSeat:
    """A seat that is not playing. Never asked for a move, and defines none of
    the gate hooks, so it is never a counterparty either."""

    def choose(self, game):  # pragma: no cover - a retired seat never moves
        raise AssertionError("a retired seat was asked to move")


PRESETS: dict[str, Entrant] = {
    "random": Entrant("random", kind="random"),
    RETIRED: Entrant(RETIRED, kind=RETIRED),
}
_SHIPPED_PRESETS = frozenset(PRESETS)


def registered_presets() -> list[str]:
    """Every preset beyond `random` and `retired`, in the order it was
    registered (`register_preset`): the runtimes' bots, and `catanatron` once
    its module is loaded (the served picker lists it where the extra is
    installed)."""
    return [name for name in PRESETS if name not in _SHIPPED_PRESETS]


def spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    """A fresh `Bot` playing `entrant` on `board` off `rng`, its moves over the
    entrant's trader when it names one (`traded`). Raises `ValueError` for a
    kind no loaded runtime registers."""
    return traded(_spawn(entrant, board, rng), entrant.trader, board, rng)


def traded(bot: Bot, trader: str | None, board: Board, rng: random.Random) -> Bot:
    """`bot`'s moves over `trader`'s trading (`hexset.bots.TradesBy`), the
    trader named as a lineup names it; `bot` itself where `trader` is `None`.
    The trader is seeded off `rng`'s state without drawing from it, so it
    leaves the mover's own draws exactly as they were, and built bare
    (`_spawn`, no trader of its own), since a trader only ever trades."""
    if trader is None:
        return bot
    from .bots import TradesBy

    seed = int.from_bytes(hashlib.sha256(repr(rng.getstate()).encode()).digest()[:8], "big")
    return TradesBy(bot, _spawn(entrant_from_name(trader), board, random.Random(seed)))


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    from .bots import RandomBot

    if entrant.kind == RETIRED:
        return RetiredSeat()
    if entrant.kind == "random":
        return RandomBot(rng)
    if entrant.kind == "catanatron" and entrant.kind not in _ENTRANT_KIND_FACTORIES:
        _load_presets(("catanatron",))
    if entrant.kind in _ENTRANT_KIND_FACTORIES:
        return _ENTRANT_KIND_FACTORIES[entrant.kind](entrant, board, rng)
    if entrant.kind in _NETWORK_KINDS:
        raise ValueError(f"entrant kind {entrant.kind!r} {_RUNTIME_HINT}")
    raise ValueError(f"unknown bot kind: {entrant.kind!r}; {KIND_HINT}")


def wilson(wins: int, games: int, z: float = Z_95) -> tuple[float, float]:
    """Score interval for a proportion. Preferred to the normal
    approximation because it stays inside [0, 1] and behaves at the
    extremes. Clamped to contain the point estimate: at `p = 1` the upper
    bound is analytically 1 but lands an ulp short in floating point."""
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
    """Where an entrant sits in a given game: a cyclic shift, so over any
    `seats` consecutive games each entrant occupies each seat exactly once
    and seat bias cancels out of the standings."""
    return (entrant + game) % seats


@dataclass(frozen=True)
class Standing:
    """One entrant's wins over the games it played, with its win rate and
    Wilson interval."""

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
    """A mean over `samples` with its `lower`/`upper` interval
    (`mean_interval`)."""

    mean: float
    lower: float
    upper: float
    samples: int


def mean_interval(samples: Sequence[float], z: float = Z_95) -> Estimate:
    """Normal interval for a paired mean, over the hundreds of bounded
    integer differences callers supply. With fewer than two samples there is
    no variance estimate, so the interval is infinite -- no evidence either
    way, rather than a confident zero."""
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
    """One exchange the engine cleared, described rather than replayable.

    `a`/`b` are seats and `received` is signed towards `a`, as
    `hexset.trading.Trade` has them. `hand_a`/`hand_b` are each side's
    resource-card count just before this trade: at the start of the step it
    cleared inside, moved by any earlier trade in the same step. Nothing
    else that step does is counted, so a trade inside a roll's step sees the
    hands from before the roll's production.
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
    """What `compete` played: each entrant's `standings`, game and turn
    counts, and the per-game winners, points and turns in entrant order."""

    standings: tuple[Standing, ...]
    games: int
    unfinished: int
    mean_turns: float
    seconds: float
    # Wins by board seat rather than by entrant. With one bot in every seat
    # these should come out even; a skew is a bug in setup order or the
    # snake draft, not in the bots.
    seat_wins: tuple[int, ...] = ()
    # Per game, in entrant order: who won, and every seat's terminal points.
    winners: tuple[int | None, ...] = ()
    points: tuple[tuple[int, ...], ...] = ()
    # Per game, in the same order as `winners` and `points`: `game.turns`,
    # the raw sequence `mean_turns` folds down.
    turns: tuple[int, ...] = ()
    # Per game, in entrant order alongside `points`: what each entrant had
    # standing on the board at the end -- *how* an entrant won rather than
    # only that it did.
    roads: tuple[tuple[int, ...], ...] = ()
    settlements: tuple[tuple[int, ...], ...] = ()
    cities: tuple[tuple[int, ...], ...] = ()
    # Per game: which seat each entrant took, in entrant order -- what turns
    # anything the engine reports by *seat* back into an entrant.
    seating: tuple[tuple[int, ...], ...] = ()
    # Per game, every trade the engine cleared, in order. Filled only when
    # `compete(records=True)` asked, alongside `records`.
    cleared: tuple[tuple[ClearedTrade, ...], ...] = ()
    # One `Record` per game, in the same order as `winners`/`points`/`turns`,
    # only when `compete(records=True)` asked (empty otherwise, never
    # partially filled).
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
    game_type: GameType = STANDARD_GAME,
    locked: Iterable[int] = (),
    turn_cap: int = MAX_TURNS,
) -> Game:
    """One game, each bot seated at its own index.

    Seating a bot also seats its private gate: `game.gates` is the lineup
    itself. A bot defining none of `gains_many`/`accepts_many`/`accepts`
    never trades.
    """
    return play_game(
        start(board, len(bots), rng, game_type=game_type, locked=locked,
              turn_cap=turn_cap),
        bots,
        action_cap=action_cap,
    )


def seat_bots(game: Game, bots: Sequence[Bot], trade_mode: str = "round") -> None:
    """Seat `bots` at `game` as its gates and choose the mechanism.

    That is all a table decides. How hard a seat bargains, and whether it
    trades at all, is that seat's own config -- an arm that does not trade is
    built as a bot that does not trade, not as a table that forbids it.
    """
    game.gates = tuple(bots)
    game.trade_mode = trade_mode


def run_seated(
    game: Game,
    bots: Sequence[Bot],
    action_cap: int = MAX_ACTIONS,
    *,
    tape: "Tape | None" = None,
    cleared: list[ClearedTrade] | None = None,
) -> None:
    """The engine's one loop: step `game` with `bots` until it is over or
    `action_cap` is reached. `game` must already be seated (`seat_bots`).

    `tape` and `cleared` are hooks, not branches: a caller that passes
    neither pays for neither, and the steps taken are identical whatever
    combination is attached.
    """
    actions = 0
    while not is_over(game) and actions < action_cap:
        seat = to_move(game)
        bot = bots[seat]
        watching = tape is not None or cleared is not None
        before = len(game.trades) if watching else 0
        shown_before = len(game.shown) if tape is not None else 0
        if cleared is not None:
            # true state: a census of who was flush cannot be read off one
            # seat's view of the table.
            true_state = game.state(0, hidden=False)
            hands = [hand_size(true_state, s) for s in range(game.num_players)]
            turn, phase = game.turns, game.phase.name
            step = actions
        action = bot.choose(game)
        apply(game, action)
        if tape is not None:
            # An END_TURN clears `game.shown`, and shows nothing itself.
            tape.step(action, game.trades[before:], game.shown[shown_before:], before=before)
        if cleared is not None:
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
                # Several trades can clear inside one step; each later one
                # sees the hands the earlier ones left behind.
                moved = sum(trade.received)
                hands[trade.a] += moved
                hands[trade.b] -= moved
        actions += 1


def play_game(
    game: Game,
    bots: Sequence[Bot],
    *,
    action_cap: int = MAX_ACTIONS,
    trade_mode: str = "round",
) -> Game:
    """`play`'s loop over a game somebody else dealt (`deal_game`).

    `trade_mode` is the mechanism, and the only bargaining choice a table
    makes; everything else is each seated bot's own config. `"auto"`
    is the exhaustive automatic house, a comparability option rather than
    the thing to fit against (see `Game.trade_mode`).
    """
    seat_bots(game, bots, trade_mode)
    run_seated(game, bots, action_cap)
    return game


@dataclass(frozen=True)
class Outcome:
    """One played game, as `_play_one` hands it back to `compete`.
    `points`/`roads`/`settlements`/`cities` are in entrant rather than seat
    order, so they compare across differently rotated games; `seating` is
    the rotation that produced them, and `cleared` stays in seat order."""

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
    # Which game of the run this is, and the board it was dealt: under
    # antithetic pairing the two halves of a pair share `board_index`.
    index: int | None = None
    board_index: int | None = None


def half_turn(seats: int) -> tuple[int, ...]:
    """The default antithetic complement (`compete(complement=...)`): entrant
    `e` takes the seat entrant `e + seats // 2` had. It exchanges the two
    sides of an `[a, a, b, b]` or `[a, b]` lineup exactly, and is no side
    swap for any other: an interleaved `[a, b, a, b]` maps each side onto
    itself."""
    return tuple((e + seats // 2) % seats for e in range(seats))


def deal_seats(
    entrants: Sequence[Entrant], index: int, complement: Sequence[int] | None = None,
) -> tuple[int, tuple[int, ...], int]:
    """Where game `index` of a `compete` run is dealt: `(board_index,
    seating, first)`, `seating[e]` being entrant `e`'s seat and `first` the
    seat that opens.

    Without a `complement` the board and the rotation are the game's own
    index. With one (antithetic pairing) they come from the pair, and the
    second game of a pair seats entrant `e` where entrant `complement[e]`
    sat in the first: the same board under the complementary seating, so the
    seat term cancels per board rather than only in the mean. Two-fold
    rather than the full `seats`-way rotation, which costs `seats` times the
    distinct boards to cancel the same term.

    The opener is seat 0 whenever every entrant plays. With `RETIRED`
    entrants seat 0 may be closed, and the lowest playing seat opening
    instead would give side A of `[a, b, retired, retired]` three first
    turns in four; so the opener rotates among the playing entrants, each
    opening once in every `len(playing)` consecutive rotations. Both games
    of a pair open from the same seat, so the complement changes who holds
    it.
    """
    seats = len(entrants)
    if complement is None:
        board_index = rotation = index
        order: Sequence[int] = range(seats)
    else:
        board_index = rotation = index // 2
        order = complement if index % 2 else range(seats)
    seating = tuple(seat_of(order[e], rotation, seats) for e in range(seats))
    playing = [e for e, entrant in enumerate(entrants) if entrant.kind != RETIRED]
    # Entrant `-rotation` sits at seat 0, so with nobody retired this is 0.
    first = seat_of(playing[-rotation % len(playing)], rotation, seats)
    return board_index, seating, first


def _play_one(
    job: tuple[tuple[Entrant, ...], int, int, int, "bool | tuple[int, ...]", bool, str,
               GameType, int],
) -> Outcome:
    """Play game `index` and return its `Outcome`. `record` and the
    cleared-trade census are built only when the job's `records` flag is
    set, never partially.

    Module level and taking only picklable arguments, so a pool can call it.
    Every random stream derives from the seed and the game index, so a game
    plays identically whichever worker draws it. The job's `antithetic` slot
    is the complement itself, `True` for `half_turn`, or `False`.
    """
    (entrants, index, seed, action_cap, antithetic, records, trade_mode,
     game_type, turn_cap) = job
    seats = len(entrants)
    complement = half_turn(seats) if antithetic is True else (antithetic or None)
    board_index, seating, first = deal_seats(entrants, index, complement)
    seats_taken = list(seating)
    board = deal_board(seed, board_index)
    # Read off the lineup, not off the spawned bots: the deal needs to know
    # which seats are playing before it starts, because that is what the game
    # type is checked against.
    retired = frozenset(
        seats_taken[e] for e, entrant in enumerate(entrants) if entrant.kind == RETIRED
    )

    # Every stream keys off `board_index`, not `index`: under antithetic the
    # two halves of a pair must differ in the seat assignment and in
    # *nothing else* -- same board, same dice, same per-entrant stream -- or
    # the seat term does not cancel and the pair is two different games.
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
            trade_mode=trade_mode, game_type=game_type, locked=retired,
            turn_cap=turn_cap, first=first,
        )
    else:
        game = play_game(
            deal_game(
                seed, board_index, seats, board=board,
                game_type=game_type, locked=retired, turn_cap=turn_cap, first=first,
            ),
            lineup,
            action_cap=action_cap,
            trade_mode=trade_mode,
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
        index=index,
        board_index=board_index,
    )


def _journal_entry(outcome: Outcome, *, exhausted: bool = False) -> dict:
    """`outcome` as one journal line: `kind` "game", or "exhausted" for a game
    that reached the turn cap with no winner -- written, then raised."""
    from .record import to_json

    return {
        "kind": "exhausted" if exhausted else "game",
        "index": outcome.index, "board_index": outcome.board_index,
        "winner": outcome.winner, "seat": outcome.seat, "turns": outcome.turns,
        "seating": list(outcome.seating), "points": list(outcome.points),
        "roads": list(outcome.roads), "settlements": list(outcome.settlements),
        "cities": list(outcome.cities),
        "cleared": [asdict(c) for c in outcome.cleared],
        "record": None if outcome.record is None else json.loads(to_json(outcome.record)),
    }


def _outcome_from_entry(entry: dict) -> Outcome:
    from .record import from_json

    return Outcome(
        winner=entry["winner"], seat=entry["seat"], turns=entry["turns"],
        seating=tuple(entry["seating"]), points=tuple(entry["points"]),
        roads=tuple(entry["roads"]), settlements=tuple(entry["settlements"]),
        cities=tuple(entry["cities"]),
        cleared=tuple(ClearedTrade(**{**c, "received": tuple(c["received"])})
                      for c in entry["cleared"]),
        record=None if entry["record"] is None else from_json(json.dumps(entry["record"])),
        index=entry["index"], board_index=entry["board_index"],
    )


def read_journal(path: str | os.PathLike) -> tuple[dict, dict[int, Outcome]]:
    """A `compete` journal's header and every finished game in it, by game
    index -- readable while the run is still going, or after it died. Games
    that exhausted the turn cap are not among them."""
    entries = gamelog.read_lines(path)
    if not entries or entries[0].get("kind") != "header":
        raise ValueError(f"{path} is not a compete journal: no header line")
    games = {e["index"]: _outcome_from_entry(e) for e in entries[1:] if e.get("kind") == "game"}
    return entries[0], games


def journal_tournament(path: str | os.PathLike) -> "Tournament":
    """The `Tournament` of the games a journal holds so far, in game order:
    what a run stopped part-way reports, with `games` the games it finished."""
    header, games = read_journal(path)
    outcomes = [games[i] for i in sorted(games)]
    return _tournament(header["entrants"], outcomes, header["records"], seconds=0.0)


def _tournament(names: Sequence[str], outcomes: Sequence[Outcome], records: bool,
                *, seconds: float) -> "Tournament":
    seats = len(names)
    wins = [0] * seats
    seat_wins = [0] * seats
    for outcome in outcomes:
        if outcome.winner is not None:
            wins[outcome.winner] += 1
            seat_wins[outcome.seat] += 1
    games = len(outcomes)
    return Tournament(
        standings=tuple(
            Standing(name=name, wins=wins[e], games=games) for e, name in enumerate(names)
        ),
        games=games,
        unfinished=sum(1 for o in outcomes if o.winner is None),
        mean_turns=statistics.mean(o.turns for o in outcomes) if outcomes else 0.0,
        seconds=seconds,
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


def _play_and_record(
    lineup: list[Bot],
    board: Board,
    seed: int,
    index: int,
    action_cap: int,
    *,
    trade_mode: str = "round",
    game_type: GameType = STANDARD_GAME,
    locked: Iterable[int] = (),
    turn_cap: int = MAX_TURNS,
    first: int = 0,
) -> tuple[Game, "Record", tuple[ClearedTrade, ...]]:
    """`play_game`'s own loop (`run_seated`) with a `hexset.record.Tape` and the
    `ClearedTrade` census attached as hooks, so a record is exactly the game
    `play` would have played."""
    from .record import Tape, recording

    game = deal_game(
        seed, index, len(lineup), board=board,
        chance=lambda rng: recording(rng, game_type.rules),
        game_type=game_type, locked=locked, turn_cap=turn_cap, first=first,
    )
    seat_bots(game, lineup, trade_mode)
    tape = Tape()
    cleared: list[ClearedTrade] = []
    run_seated(game, lineup, action_cap, tape=tape, cleared=cleared)
    return game, tape.sealed(game, seed=game_key(seed, index)), tuple(cleared)


def compete(
    entrants: Sequence[Entrant],
    games: int,
    *,
    seed: int = 0,
    action_cap: int = MAX_ACTIONS,
    workers: int = 1,
    antithetic: bool = True,
    complement: Sequence[int] | None = None,
    records: bool = False,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    start_method: str | None = None,
    trade_mode: str = "round",
    game_type: GameType = STANDARD_GAME,
    turn_cap: int = MAX_TURNS,
    progress: Callable[[int, int, Outcome], None] | None = None,
    journal: str | os.PathLike | None = None,
    resume: bool = False,
) -> Tournament:
    """Run `games` games, rotating the lineup so every entrant sits every
    seat.

    `games` must be a multiple of the lineup size, or the rotation is
    incomplete and the seat bias it exists to cancel leaks into the result;
    antithetic runs with odd seat counts need twice that many, to complete
    both halves of every board pair.

    `antithetic` plays every board twice, the second time under the
    `complement` seating (`deal_seats`): entrant `e` takes the seat entrant
    `complement[e]` had. The default is `half_turn`, which swaps the sides
    of `[a, a, b, b]` and `[a, b]`; any other two-sided lineup names its own
    side swap (`hexset.bench.duel` builds one from its geometry), or the
    pair is the same board under a seating that is not the complement. A
    complement must keep `RETIRED` entrants on retired seats, or the two
    games of a pair are different tables.

    `workers` only changes the wall clock: results are identical at any
    worker count, which is what makes a parallel run quotable.
    ``worker_initializer(*worker_initargs)`` runs once in each worker (or in
    the calling process for workers=1), for a custom runtime to register its
    entrant factories; under spawn/forkserver it must be importable at
    module scope. ``start_method`` selects a multiprocessing context without
    changing the process-wide default.

    A game that reaches `turn_cap` with no winner raises `Exhausted`
    immediately, rather than being scored as a loss for whoever sat side A.
    The run stops on the first one.

    `turn_cap` defaults to `hexset.game.MAX_TURNS`, which is read off agents
    that are trying to win. A run measuring unstructured play -- random bots
    take about four times the turns real play does -- raises it explicitly,
    which is the point: a cap that accommodates the worst imaginable player
    catches no bugs at all.

    `game_type` is the contract every game is dealt under, checked against
    the entrants that actually play -- a `RETIRED` entrant holds a seat at the
    table without occupying it, so a four-entrant lineup with two of them is a
    two-seat game and `DUEL_VARIANT_GAME` accepts it. Who opens rotates among
    the playing entrants (`deal_seats`).

    `records=True` has every job build a `hexset.record.Record` alongside
    the verdict, returned as `Tournament.records` in the same order as
    `winners`/`points`/`turns`; off by default, and then skipped entirely.

    `progress(done, games, outcome)` is called in the calling process as each
    game's outcome arrives, in game order -- so `done` counts finished games
    from the front of the schedule and can trail the workers by a few games.
    A long run reports through it instead of going silent until the end. It
    is for reporting, not keeping: a game resumed from a journal is delivered
    to it again.

    `journal` is where a run keeps its games (`hexset.gamelog`): a header
    line with the run's settings, then one line per game -- its outcome and,
    under `records=True`, its record -- written and fsynced the moment the
    game finishes, in whatever order the workers finish them. A run that
    dies, is killed or raises keeps every game it finished, games still
    running included where it can wait for them; `read_journal` and
    `journal_tournament` read it part-way. A journal that already holds a
    run is refused unless `resume=True`, which checks its header against
    this run's settings and plays only the games it lacks: every game is a
    function of `seed` and its index, so the result is the run's own.
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

    playing = sum(1 for entrant in entrants if entrant.kind != RETIRED)
    # Checked once here as well as per game, so a lineup that can never be
    # dealt fails before any worker starts rather than `games` times over.
    game_type.check(playing, dealt=seats)
    lineup = tuple(entrants)
    pairing = _complement(lineup, complement) if antithetic else False
    if not antithetic and complement is not None:
        raise ValueError("a complement is the antithetic pair's second seating; "
                         "it needs antithetic=True")
    jobs = {
        i: (lineup, i, seed, action_cap, pairing, records, trade_mode, game_type,
            turn_cap)
        for i in range(games)
    }
    header = _journal_header(
        lineup, games, seed=seed, action_cap=action_cap, antithetic=antithetic,
        records=records, trade_mode=trade_mode, game_type=game_type, turn_cap=turn_cap,
        # Only a complement other than the default is a setting of its own.
        complement=None if not pairing or pairing == half_turn(seats) else list(pairing),
    )
    if resume and journal is None:
        raise ValueError("resume=True needs the journal to resume from")
    finished = {} if journal is None else _open_journal(journal, header, resume)
    started = time.perf_counter()
    outcomes: list[Outcome] = []
    ready: dict[int, Outcome] = dict(finished)
    kept: set[int] = set(finished)
    delivered = [0]

    def exhausted(outcome: Outcome) -> bool:
        return outcome.winner is None and outcome.turns >= turn_cap

    def keep(index: int, outcome: Outcome) -> None:
        # Written the moment a game finishes, whatever its place in the order.
        if journal is not None and index not in kept:
            gamelog.append(journal, _journal_entry(outcome, exhausted=exhausted(outcome)))
            kept.add(index)

    def release() -> None:
        # `progress` and the verdict read games in game order.
        while delivered[0] in ready:
            index = delivered[0]
            outcome = ready.pop(index)
            if exhausted(outcome):
                raise Exhausted(
                    seed=seed, index=index, seating=outcome.seating, turns=outcome.turns,
                )
            outcomes.append(outcome)
            delivered[0] += 1
            if progress is not None:
                progress(len(outcomes), games, outcome)

    def arrive(index: int, outcome: Outcome) -> None:
        keep(index, outcome)
        ready[index] = outcome
        release()

    release()
    pending = [i for i in range(games) if i not in finished]
    if workers > 1 and pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context(start_method),
            initializer=worker_initializer, initargs=worker_initargs,
        ) as pool:
            futures = {pool.submit(_play_one, jobs[i]): i for i in pending}
            try:
                for future in as_completed(futures):
                    arrive(futures[future], future.result())
            except Exception:
                # Stop dealing new games, and keep the ones already running.
                for future in futures:
                    future.cancel()
                for future, index in futures.items():
                    if future.cancelled() or index in kept:
                        continue
                    try:
                        keep(index, future.result())
                    except Exception:
                        continue
                raise
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    elif pending:
        if worker_initializer is not None:
            worker_initializer(*worker_initargs)
        for index in pending:
            arrive(index, _play_one(jobs[index]))
    elapsed = time.perf_counter() - started
    return _tournament([entrant.name for entrant in lineup], outcomes, records, seconds=elapsed)


def _complement(lineup: Sequence[Entrant], complement: Sequence[int] | None) -> tuple[int, ...]:
    """`complement`, or `half_turn`, checked: a permutation of the entrants
    that keeps the retired ones on retired seats."""
    seats = len(lineup)
    pairing = half_turn(seats) if complement is None else tuple(complement)
    if sorted(pairing) != list(range(seats)):
        raise ValueError(f"complement {list(pairing)} is not a permutation of the {seats} entrants")
    retired = {e for e, entrant in enumerate(lineup) if entrant.kind == RETIRED}
    if {pairing[e] for e in retired} != retired:
        raise ValueError(
            f"complement {list(pairing)} moves a retired entrant onto a playing seat, "
            "so the two games of a pair would be different tables; name a "
            "complement that swaps the playing sides (`compete(complement=...)`), "
            "or pass antithetic=False"
        )
    return pairing


def _journal_header(lineup: Sequence[Entrant], games: int, **settings) -> dict:
    """What a journal's games are a function of: the lineup, the run's
    settings and the engine. A resumed run must match it exactly."""
    from . import __version__

    game_type = settings.pop("game_type")
    return {
        "kind": "header", "journal": gamelog.JOURNAL_VERSION, "hexset": __version__,
        "entrants": [entrant.name for entrant in lineup],
        "lineup": hashlib.sha256(repr(tuple(lineup)).encode()).hexdigest(),
        "games": games, "game_type": repr(game_type), **settings,
    }


def _open_journal(path: str | os.PathLike, header: dict, resume: bool) -> dict[int, Outcome]:
    """Start a journal at `path`, or pick up the one there: the games it
    already holds, by index."""
    held = os.path.exists(path) and os.path.getsize(path) > 0
    if not held:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        gamelog.append(path, header)
        return {}
    if not resume:
        raise FileExistsError(
            f"{path} already holds a run: resume it with resume=True, or give this "
            "run a journal of its own"
        )
    gamelog.cut_torn_tail(path)
    found, games = read_journal(path)
    differs = sorted(k for k in set(found) | set(header) if found.get(k) != header.get(k))
    if differs:
        raise ValueError(
            f"{path} is a different run ({', '.join(differs)} differ): a resumed "
            "run's games have to be the run's own"
        )
    return games


# A checkpoint cannot be a preset, its path being known only at the command
# line. `network:/path/to/latest.pt` names one wherever a preset name is taken.
NETWORK = "network:"

MCTS = "mcts:"

#: `catanatron:<key>=<value>[:...]`, parsed by `hexset.catanatron.bot`.
CATANATRON = "catanatron:"


def _load_presets(names: Sequence[str]) -> None:
    wanted = any(name == "catanatron" or name.startswith(CATANATRON) for name in names)
    if wanted and "catanatron" not in PRESETS:
        try:
            from .catanatron import bot  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "catanatron" or exc.name.startswith("catanatron.")):
                raise ValueError("catanatron requires the 'catanatron' extra") from exc
            raise


def _registered_spec(name: str) -> str | None:
    """The longest registered prefix `name` starts with, if any."""
    matches = [prefix for prefix in _SPEC_PARSERS if name.startswith(prefix)]
    return max(matches, key=len) if matches else None


def _network(name: str) -> Entrant:
    # `network:<path>@0` is the no-trade arm. It is the one bargaining
    # setting a spec names, and it names it by replacing the checkpoint's
    # own config rather than layering over it: an arm that does not trade
    # has nothing else to say about how it trades.
    path, separator, trades = name[len(NETWORK) :].partition("@")
    if separator and trades != "0":
        raise ValueError(
            f"{name!r}: `@` is the no-trade switch (`@0`), not an offer "
            f"budget -- a checkpoint declares that in its own metadata"
        )
    return Entrant(
        name="network-notrade" if separator else "network",
        kind="network",
        weights=path,
        trade=TradeParams(max_offers=0) if separator else None,
    )


_MCTS_FORM = "mcts:<path>[@[<simulations>][w<wave>][:k=<worlds>]]"


def _mcts(name: str) -> Entrant:
    # `mcts:<path>@<simulations>w<wave>:k=<worlds>`; every part after the
    # path is optional. `k` is `Entrant.k`: determinized worlds searched
    # per decision.
    path, at, search = name[len(MCTS) :].partition("@")
    if not at and any("=" in part for part in path.split(":")[1:]):
        raise ValueError(
            f"{name!r}: options follow the `@`, as in {_MCTS_FORM} "
            "(`mcts:<path>@:k=4` for the default budget)"
        )
    search, _, options = search.partition(":")
    budget, separator, wave = search.partition("w")
    if not (budget == "" or budget.isdigit()) or (separator and not wave.isdigit()):
        raise ValueError(f"{name!r}: the search budget reads {_MCTS_FORM}")
    simulations = int(budget) if budget else 128
    width = int(wave) if separator else 16
    worlds = 1
    seen = set()
    for option in (part for part in options.split(":") if part):
        key, assigned, value = option.partition("=")
        if key in seen:
            raise ValueError(f"repeated MCTS option: {option!r}")
        seen.add(key)
        if key == "k" and assigned and value.isdigit() and int(value) >= 1:
            worlds = int(value)
        else:
            raise ValueError(f"unknown MCTS option: {option!r}; use k=<n>")
    label = f"mcts{simulations}w{width}" if separator else f"mcts{budget}"
    if worlds > 1:
        label = f"{label}k{worlds}" if (budget or separator) else f"mctsk{worlds}"
    return Entrant(
        name=label if budget or separator or worlds > 1 else "mcts",
        kind="mcts",
        weights=path,
        simulations=simulations,
        wave=width,
        k=worlds,
    )


def entrant_from_name(name: str) -> Entrant:
    """Resolve a name, any of them as `<entrant>~<trader>` to trade through
    another's gate: a checkpoint spec (`network:`, `mcts:`), a preset, or a
    spec a runtime registered, the longest matching prefix parsing it."""
    if TRADER in name:
        mover, _, trader = name.partition(TRADER)
        if TRADER in trader:
            raise ValueError(f"{name!r}: one `{TRADER}` names one trader")
        entrant = entrant_from_name(mover)
        entrant_from_name(trader)  # refuse an unknown trader here, not at the deal
        return replace(entrant, name=f"{entrant.name}{TRADER}{trader}", trader=trader)
    if name.startswith(NETWORK):
        return _network(name)
    if name.startswith(MCTS):
        return _mcts(name)
    _load_presets((name,))
    if name in PRESETS:
        return PRESETS[name]
    prefix = _registered_spec(name)
    if prefix is not None:
        return _SPEC_PARSERS[prefix](name)
    raise ValueError(f"unknown bot: {name!r}; {UNKNOWN_HINT}")


#: What an unknown name usually means: the runtime that registers it was
#: never loaded.
UNKNOWN_HINT = "a bot hexset does not ship is registered by importing its runtime (`load_runtime`, `--runtime`)"

#: What an unknown `Entrant.kind` usually means: the process building it
#: never loaded the runtime that registers it.
KIND_HINT = (
    "a kind hexset does not ship is registered by its runtime in every process "
    "that spawns entrants -- `compete(worker_initializer=load_runtime, "
    "worker_initargs=(module,))`, or `--runtime` -- because a worker started by "
    "spawn or forkserver (the default start method on some platforms and "
    "Python versions) does not inherit this process's registrations"
)

CHECKPOINT_KINDS = (NETWORK, MCTS)

#: Separates an entrant from the preset or spec that answers its trades.
TRADER = "~"


def lineup_from_names(names: Sequence[str]) -> list[Entrant]:
    """Resolve names to entrants, numbering repeats so standings stay readable."""
    parts = {part for name in names for part in name.split(TRADER)}
    _load_presets(tuple(parts))
    unknown = sorted(
        name
        for name in parts
        if name not in PRESETS and not name.startswith((*CHECKPOINT_KINDS, *_SPEC_PARSERS))
    )
    if unknown:
        raise ValueError(f"unknown bots: {', '.join(unknown)}; {UNKNOWN_HINT}")
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
    share of the games rather than either seat's quarter of them.
    """
    order: list[str] = []
    wins: dict[str, int] = {}
    for standing in standings:
        name = base_name(standing.name)
        if name not in wins:
            order.append(name)
        wins[name] = wins.get(name, 0) + standing.wins
    return [Standing(name=name, wins=wins[name], games=games) for name in order]


__all__ = [
    # what to seat, and the names that resolve to it
    "Entrant", "RETIRED", "RetiredSeat", "PRESETS", "SHIPPED_NAMES", "SHIPPED_PREFIXES",
    "NETWORK", "MCTS", "CATANATRON", "CHECKPOINT_KINDS", "TRADER",
    "entrant_from_name", "lineup_from_names", "base_name",
    # runtimes
    "register_entrant_kind", "register_preset", "register_spec",
    "unregister_entrant_kind", "unregister_preset", "unregister_spec",
    "registered_presets", "load_runtime", "RUNTIME_HELP", "UNKNOWN_HINT", "KIND_HINT",
    "spawn", "traded",
    # the game law
    "MAX_ACTIONS", "board_key", "game_key", "deal_board", "deal_game", "deal_seats",
    "half_turn", "seat_of",
    # playing
    "play", "play_game", "seat_bots", "run_seated", "compete", "Exhausted",
    # results
    "Tournament", "Standing", "Outcome", "ClearedTrade", "Estimate", "wilson",
    "mean_interval", "pooled", "read_journal", "journal_tournament",
]
