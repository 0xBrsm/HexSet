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
from math import sqrt
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable, Sequence

from .actions import apply
from .board.board import Board, random_base_board
from .board.topology import Topology
from .game import Game, is_over, start, to_move
from .placement import PlacementBot
from .victory import victory_points

if TYPE_CHECKING:
    # `Bot` is annotation-only here (`from __future__ import annotations`
    # makes every annotation a string) -- never imported for real. The three
    # names `_spawn` actually calls at runtime (`RandomBot`, `SearchBot`,
    # `greedy`) are imported locally inside `_spawn` instead of at module
    # level, because `hexset.bots` imports `hexset.bots.heximax`, which
    # imports this module back (for `Entrant`/`register_entrant_kind`/
    # `register_preset`) -- a module-level `from .bots import ...` here
    # would deadlock that cycle on whichever of the two is cold-started
    # first. See `hexset.bots.heximax`'s own docstring for the full cycle.
    from .bots import Bot

Z_95 = 1.959964

# Network-backed entrant kinds and evaluators are not implemented here: they
# need torch and a trained checkpoint, which live in the `hexnet` package, and
# hexset must never import hexnet. Instead `hexnet.netbot` registers factories
# here at import time -- so any HexNet entry point (train, league, collect,
# its duel wrapper) that imports `hexnet.netbot` makes "network"/"mcts"
# entrants and the "network" evaluator spawnable, and a hexset-only process
# that never imports it gets a clear error naming the package that provides
# them rather than a silent `ImportError` on torch.
_ENTRANT_KIND_FACTORIES: dict[str, Callable[[Entrant, Board, random.Random], Bot]] = {}
_EVALUATOR_PROVIDERS: dict[str, Callable[[object, Board], object]] = {}
_CHECKPOINT_LOADER: Callable[[str, Topology, str], object] | None = None
_LEAF_EVALUATOR_FACTORY: Callable[..., object] | None = None

# Kinds/evaluators hexset knows the *name* of but does not implement itself --
# used only to tell "not registered yet" apart from "not a real kind at all".
_NETWORK_KINDS = frozenset({"network", "mcts"})
_NETWORK_EVALUATORS = frozenset({"network"})
_HEXIMAX_KINDS = frozenset({"heximax"})

_HEXNET_HINT = (
    "is provided by the hexnet package; import hexnet.netbot (or an entry "
    "point that does, such as hexnet.train/hexnet.league/hexnet.collect) "
    "before spawning it"
)
_HEXIMAX_HINT = "is provided by hexset.bots.heximax; import hexset.bots before spawning it"


def register_entrant_kind(kind: str, factory) -> None:
    """Register a bot-building factory for an `Entrant.kind` hexset does not
    implement itself. `factory(entrant, board, rng) -> Bot`."""
    _ENTRANT_KIND_FACTORIES[kind] = factory


def register_evaluator_provider(name: str, factory) -> None:
    """Register a leaf-evaluation factory for an `Entrant.evaluator` value
    hexset does not implement itself. `factory(weights, board) -> Evaluator`,
    matching `EVALUATORS[name](board, weights)`'s role for the built-in ones --
    the object just needs an `evaluate_game(game, seat)` method and may carry
    its own `max_trades`."""
    _EVALUATOR_PROVIDERS[name] = factory


def register_preset(name: str, entrant: "Entrant") -> None:
    """Register a named lineup shortcut (`PRESETS[name]`, resolved by
    `entrant_from_name`/`lineup_from_names`) for an entrant hexset does not
    ship itself -- how `hexset.bots.heximax` makes "heximax",
    "heximax-omni" and "heximax-notrade" resolvable by name once imported."""
    PRESETS[name] = entrant


def register_checkpoint_loader(loader) -> None:
    """Register `hexnet.netbot.load`-shaped loader: `(path, topology, device)
    -> Loaded`, an object with `.policy`, `.space` and `.max_trades`. Lets
    `hexset.bench.aivat`/`hexset.bench.human_agreement` load a checkpoint without
    importing hexnet themselves."""
    global _CHECKPOINT_LOADER
    _CHECKPOINT_LOADER = loader


def register_leaf_evaluator_factory(factory) -> None:
    """Register a `hexnet.netbot.LeafEvaluator`-shaped factory:
    `(policy, space, pad_to=None) -> object` with an `evaluate(leaves)`
    method matching `hexset.mcts.Evaluator`."""
    global _LEAF_EVALUATOR_FACTORY
    _LEAF_EVALUATOR_FACTORY = factory


def load_checkpoint(path: str, topology: Topology, device: str = "cpu"):
    """A trained checkpoint, loaded through whatever registered
    `register_checkpoint_loader`. Raises with a clear message if nothing has."""
    if _CHECKPOINT_LOADER is None:
        raise RuntimeError(f"loading a checkpoint {_HEXNET_HINT}")
    return _CHECKPOINT_LOADER(path, topology, device)


def leaf_evaluator(policy, space, pad_to: int | None = None):
    """A `hexset.mcts.Evaluator` over a loaded network policy, through
    whatever registered `register_leaf_evaluator_factory`."""
    if _LEAF_EVALUATOR_FACTORY is None:
        raise RuntimeError(f"a network leaf evaluator {_HEXNET_HINT}")
    return _LEAF_EVALUATOR_FACTORY(policy, space, pad_to)

# The engine caps turns, but nothing caps actions within a turn, so a policy
# that liked trading in circles would never reach the turn cap. A game that
# trips this is a bug worth seeing, not a result worth counting.
#
# Raised once the old offer protocol landed: negotiating cost an action per
# offer and one per response, so a random four-player game went from about
# 1400 actions to 3400 and 5000 had stopped being a guard. Trading is one
# engine event now and costs no actions at all, so the headroom is larger
# than it needs to be -- kept, because a guard is not a target.
MAX_ACTIONS = 20000


@dataclass(frozen=True)
class Entrant:
    """What to build, not a built bot. Picklable, so it can cross a process."""

    name: str
    kind: str = "greedy"
    # Fitted evaluation weights, or — for `kind="network"` — the path to a
    # training checkpoint. Both are "what this entrant plays with", and both
    # have to survive a pickle to a worker, which a loaded network would not.
    weights: object | None = None
    depth: int = 1
    width: int | None = None
    # Which evaluation to score with. `weights` has to be the matching type,
    # since the two evaluations do not share a term set — that is the point of
    # keeping both.
    evaluator: str = "default"
    # How the per-seat vector is read: see `hexset.bots.STANCES`. Defaults to
    # the stance that wins; `greedy-own` reproduces the plain max^n baseline.
    stance: str = "relative"
    # Whether the opening settlements come from the fitted placement prior
    # rather than from whatever this entrant would otherwise do. Orthogonal to
    # `kind`, so any entrant can be duelled against itself with only the eight
    # setup picks differing.
    placement: bool = False
    # The trade off switch: `0` means this entrant publishes no valuation and
    # refuses every exchange, so it never trades. Not a budget -- the engine
    # has no cap (`hexset.trading`) -- and self-imposed rather than engine-wide
    # so a duel can see what trading is worth: only a bot that declines what
    # its opponent still has can price it.
    max_trades: int | None = None
    # `kind="mcts"` only: how many descents the tree gets per decision, and how
    # many it may launch before expanding new leaves. They are separate because
    # a wider wave changes collision rate as well as network batch size.
    simulations: int = 128
    wave: int = 16
    # `kind="heximax"` only: which of `heximax.MODES` to build --
    # `honest` (the referent), `omniscient` (the information price), or
    # `notrade` (the no-trade weights, declining everything). Defaulted so
    # every other entrant is unchanged.
    mode: str = "honest"
    # `kind="heximax"` only: determinized worlds searched per decision (PIMC).
    # P1½ (`heximax.md` §8) found no k > 1 beat k = 1 beyond the instrument's
    # resolution over 400 games; `k = 1` ships and the field stays for anyone
    # who wants to re-open the question, not for a preset to vary.
    k: int = 1

    def renamed(self, name: str) -> Entrant:
        return replace(self, name=name)


# `Evaluator`/`TieredEvaluator` are not imported at module level, and
# `EVALUATORS` is not a module-level literal -- see `_evaluators` below for
# why: `hexset.bots` now imports `heximax`, which imports this module back
# for `Entrant`/`register_entrant_kind`/`register_evaluator_provider`/
# `register_preset`.
_EVALUATORS: dict[str, type] | None = None


def _evaluators() -> dict[str, type]:
    """`{"default": Evaluator, "tiered": TieredEvaluator}`, built on first use
    and cached.

    Deferred rather than a module-level import + literal:
    `hexset.bots.evaluate` is a submodule of the `hexset.bots` *package*, so
    importing it requires `hexset/bots/__init__.py` to finish running first,
    and that module imports `hexset.bots.heximax`, which imports this module
    back for the four names above. A module-level import here would deadlock that cycle
    on whichever of `hexset.arena`/`hexset.tuning` is cold-started first, so
    the whole dependency is pushed to first use, well after every module
    involved has finished importing. See `hexset.bots.heximax`'s own docstring
    for the full cycle and why `hexset.mcts` carries the same pattern for
    `STANCES`.
    """
    global _EVALUATORS
    if _EVALUATORS is None:
        from .bots.evaluate import Evaluator
        from .evaluate_tiered import Evaluator as TieredEvaluator

        _EVALUATORS = {"default": Evaluator, "tiered": TieredEvaluator}
    return _EVALUATORS

PRESETS: dict[str, Entrant] = {
    "random": Entrant("random", kind="random"),
    "greedy": Entrant("greedy", kind="greedy"),
    "search2": Entrant("search2", kind="search", depth=2, width=6),
    "greedy-own": Entrant("greedy-own", kind="greedy", stance="own"),
    "search2-own": Entrant(
        "search2-own", kind="search", depth=2, width=6, stance="own"
    ),
    "search3": Entrant("search3", kind="search", depth=3, width=4),
    # Scored with the same stance as the default, so a comparison between them
    # differs only in the feature set. The stance barely moves it either way:
    # 38.5% relative against 36.7% own, intervals overlapping.
    "greedy-tiered": Entrant("greedy-tiered", kind="greedy", evaluator="tiered"),
    "search2-tiered": Entrant(
        "search2-tiered", kind="search", depth=2, width=6, evaluator="tiered"
    ),
    "greedy-relative": Entrant("greedy-relative", kind="greedy", stance="relative"),
    "greedy-paranoid": Entrant("greedy-paranoid", kind="greedy", stance="paranoid"),
    # The no-trade referents: same bot, trade switch off, so a duel between
    # them and their trading twins prices the mechanic and a duel between two
    # of them is a game with no trading in it at all.
    "greedy-notrade": Entrant("greedy-notrade", kind="greedy", max_trades=0),
    "search2-notrade": Entrant(
        "search2-notrade", kind="search", depth=2, width=6, max_trades=0
    ),
    "search2-relative": Entrant(
        "search2-relative", kind="search", depth=2, width=6, stance="relative"
    ),
    # The placement prior bolted onto entrants that otherwise pick their opening
    # at random. Paired with their plain versions above, the duel isolates the
    # eight setup settlements from everything else.
    "random-placement": Entrant("random-placement", kind="random", placement=True),
    "greedy-placement": Entrant("greedy-placement", kind="greedy", placement=True),
    # "heximax"/"heximax-omni"/"heximax-notrade" are not built in -- they are
    # registered by `hexset.bots.heximax` at import time via `register_preset`
    # (see that package's "registration" section), the same way `hexnet`
    # registers "network"/"mcts".
}


def spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    bot = _spawn(entrant, board, rng)
    return PlacementBot(bot) if entrant.placement else bot


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Bot:
    from .bots import RandomBot, SearchBot, greedy

    if entrant.kind == "random":
        return RandomBot(rng)
    if entrant.kind in _ENTRANT_KIND_FACTORIES:
        return _ENTRANT_KIND_FACTORIES[entrant.kind](entrant, board, rng)
    if entrant.kind in _NETWORK_KINDS:
        raise ValueError(f"entrant kind {entrant.kind!r} {_HEXNET_HINT}")
    if entrant.kind in _HEXIMAX_KINDS:
        raise ValueError(f"entrant kind {entrant.kind!r} {_HEXIMAX_HINT}")

    max_trades = entrant.max_trades
    if entrant.evaluator in _EVALUATOR_PROVIDERS:
        evaluator = _EVALUATOR_PROVIDERS[entrant.evaluator](entrant.weights, board)
        if max_trades is None:
            max_trades = getattr(evaluator, "max_trades", None)
    elif entrant.evaluator in _NETWORK_EVALUATORS:
        raise ValueError(f"evaluator {entrant.evaluator!r} {_HEXNET_HINT}")
    elif entrant.evaluator not in _evaluators():
        raise ValueError(f"unknown evaluator: {entrant.evaluator}")
    else:
        evaluator = _evaluators()[entrant.evaluator](board, entrant.weights)

    if entrant.kind == "greedy":
        return greedy(
            evaluator,
            rng,
            stance=entrant.stance,
            max_trades=max_trades,
        )
    if entrant.kind == "search":
        return SearchBot(
            evaluator,
            depth=entrant.depth,
            width=entrant.width,
            rng=rng,
            stance=entrant.stance,
            max_trades=max_trades,
        )
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

    Seating a bot also seats what it brings to a trade: `game.traders` is
    the lineup itself, so the engine's one trade event a turn
    (`hexset.trading`) asks each seat's own `valuation`/`accepts` rather
    than this loop having to remember to run anything. A bot that defines
    neither never trades, which is how `RandomBot` and any external bot that
    predates the mechanic behave.
    """
    game = start(board, len(bots), rng)
    game.traders = tuple(bots)
    game.max_trades = None
    actions = 0
    while not is_over(game) and actions < action_cap:
        apply(game, bots[to_move(game)].choose(game))
        actions += 1
    return game


def _play_one(
    job: tuple[tuple[Entrant, ...], int, int, int],
) -> tuple[int | None, int | None, int, tuple[int, ...]]:
    """Play game `index`. Returns (winning entrant, winning seat, turns, points).

    Points are in entrant rather than seat order, so they can be compared
    across games that rotated the lineup differently.

    Module level and taking only picklable arguments, so a pool can call it.
    Every random stream is derived from the seed and the game index, so a game
    plays identically whichever worker draws it and however many there are.
    """
    entrants, index, seed, action_cap, antithetic = job
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
    board = random_base_board(random.Random(f"{seed}:{board_index}:board"))
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

    game = play(
        lineup, board, random.Random(f"{seed}:{board_index}:game"), action_cap=action_cap
    )
    # true state: the verdict's own victory points include hidden
    # victory-point dev cards, so the final score is read off the truth.
    points = tuple(
        victory_points(game.state(seats_taken[e], hidden=False), seats_taken[e])
        for e in range(seats)
    )
    if game.won_by is None:
        return None, None, game.turns, points
    return seats_taken.index(game.won_by), game.won_by, game.turns, points


def compete(
    entrants: Sequence[Entrant],
    games: int,
    *,
    seed: int = 0,
    action_cap: int = MAX_ACTIONS,
    workers: int = 1,
    antithetic: bool = True,
) -> Tournament:
    """Run `games` games, rotating the lineup so every entrant sits every seat.

    `games` must be a multiple of the lineup size, otherwise the rotation is
    incomplete and the seat bias it exists to cancel leaks into the result.

    `workers` only changes the wall clock. Results are identical at any worker
    count, which is the property that makes a parallel run quotable.
    """
    seats = len(entrants)
    if seats < 2:
        raise ValueError("a tournament needs at least two entrants")
    if games % seats:
        raise ValueError(f"{games} games does not divide evenly over {seats} seats")

    lineup = tuple(entrants)
    jobs = [(lineup, i, seed, action_cap, antithetic) for i in range(games)]
    started = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            outcomes = pool.map(_play_one, jobs, chunksize=1)
    else:
        outcomes = [_play_one(job) for job in jobs]
    elapsed = time.perf_counter() - started

    wins = [0] * seats
    seat_wins = [0] * seats
    for winner, seat, _, _ in outcomes:
        if winner is not None:
            wins[winner] += 1
            seat_wins[seat] += 1

    return Tournament(
        standings=tuple(
            Standing(name=entrant.name, wins=wins[e], games=games)
            for e, entrant in enumerate(lineup)
        ),
        games=games,
        unfinished=sum(1 for winner, _, _, _ in outcomes if winner is None),
        mean_turns=statistics.mean(t for _, _, t, _ in outcomes) if outcomes else 0.0,
        seconds=elapsed,
        seat_wins=tuple(seat_wins),
        winners=tuple(winner for winner, _, _, _ in outcomes),
        points=tuple(row for _, _, _, row in outcomes),
        turns=tuple(t for _, _, t, _ in outcomes),
    )


# A checkpoint cannot be a preset, since its path is only known at the command
# line. `network:/path/to/latest.pt` names one wherever a preset name is taken,
# and the entrant it builds still pickles to a worker verbatim.
NETWORK = "network:"

# The same checkpoint, read as a leaf evaluation instead of as a policy: the
# handcrafted search with learned leaves. `search2` is the entrant it has to be
# compared against, since the two then differ only in what scores a leaf.
NETSEARCH = "netsearch:"
NETGREEDY = "netgreedy:"

# The same checkpoint again, under the batched PUCT search rather than the
# handcrafted one. `mcts:<path>@<simulations>w<wave>` names both quantities;
# the `w<wave>` suffix is optional and defaults to 16 for compatibility with
# the runs already on record.
MCTS = "mcts:"


def entrant_from_name(name: str) -> Entrant:
    """One preset name, or one `<kind>:<checkpoint>` spec, as an entrant."""
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
    if name.startswith(NETSEARCH):
        return Entrant(
            name="netsearch",
            kind="search",
            depth=2,
            width=6,
            evaluator="network",
            weights=name[len(NETSEARCH) :],
        )
    if name.startswith(NETGREEDY):
        return Entrant(
            name="netgreedy",
            kind="greedy",
            evaluator="network",
            weights=name[len(NETGREEDY) :],
        )
    return PRESETS[name]


CHECKPOINT_KINDS = (NETWORK, NETSEARCH, NETGREEDY, MCTS)


def lineup_from_names(names: Sequence[str]) -> list[Entrant]:
    """Resolve names to entrants, numbering repeats so standings stay readable."""
    unknown = sorted(
        name
        for name in set(names)
        if name not in PRESETS and not name.startswith(CHECKPOINT_KINDS)
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
