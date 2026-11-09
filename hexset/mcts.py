# SPDX-License-Identifier: GPL-3.0-only
"""Monte Carlo tree search with PUCT selection and batched policy/value inference.

Leaves are collected in waves; virtual loss discourages duplicate descents.
Nodes store per-seat value vectors and select with the mover's stance, or as
one side against named seats (`Search.against`). Dice,
steals and dev-card draws are sampled per simulation, not frozen at expansion.

Where the evaluator's policy reads native positions (`native_contract` and
`space`, as `hexset.clients.netbot.LeafEvaluator` states them) and the native
engine holds the world (`hexset.native.playout.copyable`), every node of that
world's tree holds a `hexset.native.playout.NativeCopy` instead of a Python
`Game`. The search draws each roll and steal from its own stream exactly as
the Python tree does and hands the outcome to the native copy, so the two
trees make the same draws, read the same positions and choose the same moves.
Before a development card's draw the native deck is shuffled from the same
stream, as the Python tree shuffles its copied deck. Setting `HEXSET_NATIVE_PLAYOUTS=0` keeps every node a Python `Game`.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Collection, Protocol, Sequence

import numpy as np

from .actions import (
    Action,
    ActionType,
    apply,
    legal_actions,
    victim_of,
)
from .chance import Live
from .game import ROLL_ODDS, Game, imagine, is_over, roll_dice, roll_odds, to_move
from .native.playout import NativeCopy, copyable, held
from .victory import public_victory_points
from .view import HoldReading

__all__ = [
    "Leaf",
    "Evaluation",
    "Evaluator",
    "HIDDEN_DRAW",
    "draws_hidden",
    "sampled_children",
    "Node",
    "STANCE_ROWS",
    "lost_value",
    "live_seats",
    "side_value",
    "calibrate_own",
    "Decision",
    "Search",
    "visit_policy",
]


@dataclass(frozen=True)
class Leaf:
    """A position the search wants a prior and a value for."""

    game: Game
    seat: int
    options: tuple[Action, ...]


@dataclass(frozen=True)
class Evaluation:
    """A leaf's prior and value, optionally with a value vector per observer.

    ``value`` is the vector for the leaf mover, preserving the ordinary
    evaluator contract. ``observer_values[p]`` is the full board-seat value
    vector as seen from seat ``p``'s information frame. Search uses the frame
    of each choosing node when it backs a leaf up its path. Legacy evaluators
    may continue returning ``(prior, value)``; that shared vector is used at
    every ancestor as before.
    """

    prior: Sequence[float]
    value: Sequence[float]
    observer_values: Sequence[Sequence[float]] | None = None

    # Keep the historical two-item evaluator result usable by callers that
    # unpack, index, or take its length. Observer rows are additive metadata.
    def __iter__(self):
        yield self.prior
        yield self.value

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        return (self.prior, self.value)[index]


class Evaluator(Protocol):
    """Scores a whole wave of leaves at once: per leaf, a prior over that leaf's
    `options` and a value per seat. The prior must be normalised over the legal
    options. A value is each seat's chance of winning, the scale on which a
    finished game is its one-hot winner; `lost_value` reads a loss off it."""

    def evaluate(
        self, leaves: Sequence[Leaf]
    ) -> Sequence[tuple[Sequence[float], Sequence[float]] | Evaluation]: ...

    def terminal(self, game: Game) -> Sequence[float]:
        """The per-seat value of a finished game, board-seat order, on this
        evaluator's own scale -- a backup mixes two scales otherwise. No
        rotation: `Node.mover` and `STANCE_ROWS` index it by board seat."""
        ...


class _Chance:
    """The outcomes of one chance edge, keyed by outcome -- a roll for `ROLL`,
    `_drawn`'s card index for the edges that steal or buy. A draw is not a
    choice, so it carries no visit statistics of its own: the parent's edge
    counts aggregate over the outcomes drawn, the sampled expectimax average."""

    __slots__ = ("outcomes",)

    def __init__(self) -> None:
        self.outcomes: dict[int, Node] = {}


HIDDEN_DRAW = frozenset({ActionType.MOVE_ROBBER, ActionType.BUY_DEV_CARD})


def draws_hidden(game: Game, action: Action) -> bool:
    """Whether taking this action resolves hidden information. A robber move
    that names nobody steals nothing, so its child stays on the deterministic
    path. The tree's own test for a chance edge, for a caller expanding
    children outside it (`sampled_children`)."""
    if action.type not in HIDDEN_DRAW:
        return False
    if action.type is ActionType.BUY_DEV_CARD:
        return True
    if type(game) is NativeCopy:
        return action.b < game.dealt
    return victim_of(game, action.b) is not None


def sampled_children(
    game: Game, action: Action, *, draws: int, rng, extra
) -> list[Game]:
    """This edge's outcome drawn `draws` times, as the positions it produced.

    The first draw comes from `rng` and only draws two and up from `extra`, so
    adding a chance row cannot shift the chance-free rows after it. Unlike the
    tree's `_sample`, identical outcomes keep their own positions: a caller
    rolling them out for plies sees the deck order beneath the top card."""
    if draws < 1:
        raise ValueError("a child needs at least one draw")
    first = imagine(game, rng)
    apply(first, action)
    if draws == 1 or not draws_hidden(game, action):
        return [first]
    children = [first]
    for _ in range(draws - 1):
        child = imagine(game, extra)
        apply(child, action)
        children.append(child)
    return children


def _drawn(before: Game, after: Game, action: Action) -> int:
    """Which card this edge's draw produced, as the key of its chance slot.

    Read off the state because `apply` returns nothing: exactly one count goes
    up. `-1` is the no-draw outcome, from an emptied hand."""
    player = before.current_player
    bought = action.type is ActionType.BUY_DEV_CARD
    was, now = _own_cards(before, player, bought), _own_cards(after, player, bought)
    for index, (old, new) in enumerate(zip(was, now)):
        if new > old:
            return index
    return -1


def _own_cards(game, player: int, bought: bool) -> tuple[int, ...]:
    """`player`'s own resource hand, or with `bought` its development cards
    bought this turn, from a Python `Game` or a native copy alike."""
    if type(game) is NativeCopy:
        facts = game.state.visible_card_facts(player)
        return tuple(facts.own_new_dev if bought else facts.own_hand)
    # true state: identifying which card was drawn needs the actual hand/
    # dev-card delta, not an information-set estimate of it.
    state = game.state(player, hidden=False)
    return tuple(state.new_dev_cards[player] if bought else state.hands[player])


class _Rolled:
    """What a native tree position draws a roll from: the one roll the
    search drew for this edge (`Search._roll`). It draws nothing else."""

    __slots__ = ("total",)

    def __init__(self, total: int) -> None:
        self.total = total

    def roll_by(self, seat: int | None) -> int:
        return self.total

    def steal(self, hand):
        raise RuntimeError("a roll edge drew a steal")

    def deck_order(self, deck):
        raise RuntimeError("a roll edge drew a deck order")


class _NoChance:
    """What a native tree position plays a chance-free edge with: any draw is
    an error, since the Python tree draws nothing there either."""

    __slots__ = ()

    def roll_by(self, seat):
        raise RuntimeError("a chance-free edge drew a roll")

    def steal(self, hand):
        raise RuntimeError("a chance-free edge drew a steal")

    def deck_order(self, deck):
        raise RuntimeError("a chance-free edge drew a deck order")


_NO_CHANCE = _NoChance()


@dataclass
class Node:
    """One position in a `Search` tree: its legal `options`, the evaluator's
    `prior` and `value`, and per edge the visits, value totals, the mover's
    ranked totals, in-flight descents and children."""

    game: Game
    mover: int
    options: tuple[Action, ...]
    value: tuple[float, ...] | None = None
    prior: np.ndarray | None = None
    visits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    totals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    ranked: np.ndarray = field(default_factory=lambda: np.zeros(0))
    virtual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    children: list[object] = field(default_factory=list)
    # The determinizations this root combines, as `(share, root)` pairs --
    # empty on every node but a root. A root searched in one world is that
    # world's own node and lists itself; a root combining several is a
    # synthetic node whose `children` are empty and whose subtrees live here.
    worlds: tuple[tuple[float, "Node"], ...] = ()
    # Full value vectors by observer, set by frame-aware evaluators. ``value``
    # remains the vector from this node mover's frame for existing consumers.
    # Appended to preserve the positional constructor of the public Node.
    observer_values: tuple[tuple[float, ...], ...] | None = None
    # The balanced-dice deck at a native position (`game` a `NativeCopy`), as
    # the Python tree reads it off `Game._state.dice`: what `Search._roll`
    # draws by. `None` under independent dice and on every Python node.
    dice: object = None

    @property
    def expanded(self) -> bool:
        return self.value is not None

    @property
    def terminal(self) -> bool:
        return not self.options


@dataclass
class _Run:
    root: Node
    done: int = 0
    # This world's own simulation budget; see `Search.run_many`.
    budget: int = 0
    share: float = 1.0


def _own_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    return vectors[:, seat]


def _relative_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    seats = vectors.shape[1]
    return (vectors[:, seat] * seats - vectors.sum(axis=1)) / (seats - 1)


def _paranoid_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    return vectors[:, seat] - np.delete(vectors, seat, axis=1).max(axis=1)


# Whole-matrix forms of `hexset.bots.STANCES`, pinned to those canonical scalar
# forms by test. `_relative_rows` reassociates `v[s] - sum(others)/(n-1)` into
# `(v[s]*n - sum(all))/(n-1)`: equal in exact arithmetic, within a rounding step
# in floating point, which is why that test uses a tolerance.
STANCE_ROWS = {
    "own": _own_rows,
    "relative": _relative_rows,
    "paranoid": _paranoid_rows,
}


def side_value(vector: Sequence[float], against: Sequence[int]) -> float:
    """What a seat playing as one side against `against` reads a value
    vector of win chances as: the chance none of them wins."""
    return 1.0 - float(sum(vector[s] for s in against))


def calibrate_own(
    vector: Sequence[float], seat: int, calibration: tuple[float, float]
) -> tuple[float, ...]:
    """`vector` with `seat`'s win chance moved to `sigmoid(a + b * logit(p))`
    for `calibration = (a, b)`, and the other seats' chances scaled together so
    the vector still sums to what it did. A recalibration of the value head's
    reading of one seat -- the searching seat's -- against outcomes it was not
    trained on (a table whose players are weaker or stronger than its own
    self-play copies), not a change of what the search plays for. `(0, 1)` is
    the identity."""
    a, b = calibration
    values = [float(v) for v in vector]
    total = sum(values)
    p = min(max(values[seat], 1e-6), 1.0 - 1e-6)
    lifted = 1.0 / (1.0 + math.exp(-(a + b * math.log(p / (1.0 - p)))))
    lifted *= total
    rest = total - values[seat]
    if rest <= 0.0:
        share = (total - lifted) / max(1, len(values) - 1)
        return tuple(lifted if s == seat else share for s in range(len(values)))
    scale = (total - lifted) / rest
    return tuple(lifted if s == seat else v * scale for s, v in enumerate(values))


@lru_cache(maxsize=None)
def lost_value(stance: str, seats: int) -> float:
    """What `stance` reads a game another seat won as: the bottom of the
    scale its means sit on. An in-flight descent counts against its edge as
    a visit worth this, so it lowers that edge's mean whatever the mean's
    sign without pushing it below a sure loss. 0 for `own`, `-1/(seats-1)`
    for `relative`, -1 for `paranoid`."""
    won_by_other = np.zeros((1, seats))
    won_by_other[0, 1] = 1.0
    return float(STANCE_ROWS[stance](won_by_other, 0)[0])


def live_seats(game: Game) -> int:
    """The seats still playing `game`: its seats less those retired at the
    deal (`Game.locked`). A duel is dealt as four seats with two retired, so
    its scores read over two: `relative` is then `2p - 1`, 0 an even game,
    and `lost_value` -1. Over the board's four seats, 0 would read a 25%
    win chance, below even, and every move the tree has not tried would
    start there."""
    if type(game) is NativeCopy:
        return game.dealt - len(game.locked)
    return game.num_players - len(game.locked)



@dataclass(frozen=True)
class Decision:
    """One decision as `Search.decide` made it: the root's `options`, the
    evaluator's `prior` over them and the root's `value` (board-seat order),
    combined over the decision's worlds as the root is (`Search._combine`);
    each option's root `visits` and `means`, the mover's stance reading of
    the values backed up through it (`nan` for an option never visited); and
    the option `chosen`, the most visited. `searched` is false for a root the
    search credited without descending: one legal move, or a prior at or
    above `decisive` in every world. `worlds` counts the distinct worlds the
    root was searched in.

    What it was made from, for the log: the root's `turn` and `phase`, the
    deciding `seat`, every seat's public victory `points` as that seat sees
    them, the `simulations` budget of each world's tree, and `cpu_seconds`,
    the process time the decision took."""

    options: tuple[Action, ...]
    prior: np.ndarray
    value: tuple[float, ...]
    visits: np.ndarray
    means: tuple[float, ...]
    chosen: int
    searched: bool
    worlds: int
    turn: int = 0
    phase: str = ""
    seat: int = 0
    points: tuple[int, ...] = ()
    simulations: int = 0
    cpu_seconds: float = 0.0

    def to_json(self) -> dict:
        """The decision as one JSON object. A decision it did not search keeps
        its top prior and the option count; a searched one lists every visited
        option, most visited first."""
        out = {
            "turn": self.turn, "phase": self.phase, "seat": self.seat,
            "points": list(self.points), "options": len(self.options),
            "top_prior": round(float(self.prior.max()), 4) if self.prior.size else None,
            "chosen": self.chosen, "searched": self.searched, "worlds": self.worlds,
            "simulations": self.simulations, "cpu_seconds": round(self.cpu_seconds, 4),
        }
        if not self.searched:
            return out
        r = lambda x: round(float(x), 4)  # noqa: E731
        visited = sorted((i for i in range(len(self.options)) if self.visits[i] > 0),
                         key=lambda i: (-self.visits[i], i))
        out.update(
            candidates=visited,
            actions=[[self.options[i].type.name, self.options[i].a, self.options[i].b] for i in visited],
            prior=[r(self.prior[i]) for i in visited],
            visits=[r(self.visits[i]) for i in visited],
            means=[r(self.means[i]) for i in visited],
            value=[r(v) for v in self.value],
        )
        return out

class Search:
    """One tree per determinized world per decision. Not reused across moves --
    see `run`.

    The roots are drawn from the mover's own `View`, never from the true state:
    `k` samples are folded to the distinct worlds among them, each world is
    searched to the full `simulations` budget, and the root visit counts are
    combined weighted by each world's share of the draws
    (`hexset.bots.determinized`, the PIMC arrangement). That is what keeps an
    opponent's hand out of the answer: within one world the tree still
    resolves steals, draws and terminal points exactly, but the hands it
    resolves them against are sampled, not read.

    `hidden=False` roots the tree on the true state instead -- the omniscient
    search, an analysis tool only. Nothing a bot seats may pass it. `hold` is
    a caller's reading of other seats' held development cards for those
    samples (`View.sample`); without one every unseen card is dealt alike.

    `exploration` is PUCT's c_puct; the values it trades against are the
    stance's reading of win chances: [0, 1] for `own`, [-1/(seats-1), 1] for
    `relative` over the seats still playing (`live_seats`: a duel dealt as four
    seats reads [-1, 1], 0 even), [-1, 1] for `paranoid`. `stance` is restricted to `STANCE_ROWS`'s keys, what this
    tree's backup implements; anything else is refused at construction.

    `own_calibration = (a, b)` recalibrates, at every evaluated position, the
    win chance of the seat the search decides for (each decision's root
    mover) as `calibrate_own` does; finished games keep their exact values.
    `None`, the default, reads the evaluator's values as they are.

    `decisive` plays a confident root without searching it: where the
    evaluator's prior on one of the root's moves is at least `decisive`, that
    move is credited with the whole budget at the root's own value, as a root
    with one legal move is, and no simulation is spent. `None`, the default,
    searches every root with more than one move.

    `decision_log` keeps every decision `decide` (and so `choose`) made, in
    order, as a `Decision`; `run` and `run_many` log nothing.

    `against` names seats the search plays against as one side (`side_value`):
    at a node whose mover is not one of them, the mover ranks a position by
    the chance none of them wins, on [0, 1] like `own`; a seat in `against`
    still moves by `stance`. Empty, the default, every seat moves by
    `stance`. `play_against` sets it on a search already built."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        simulations: int = 128,
        wave: int = 16,
        exploration: float = 1.25,
        stance: str = "relative",
        root_noise: float = 0.0,
        noise_fraction: float = 0.25,
        k: int = 1,
        hidden: bool = True,
        hold: HoldReading | None = None,
        rng: random.Random | None = None,
        against: Collection[int] = (),
        own_calibration: tuple[float, float] | None = None,
        decisive: float | None = None,
    ) -> None:
        if stance not in STANCE_ROWS:
            raise ValueError(
                f"stance {stance!r} is not one this tree backs up; "
                f"use one of {sorted(STANCE_ROWS)}"
            )
        if simulations < 1 or wave < 1:
            raise ValueError("a search needs at least one simulation and one wave")
        if k < 1:
            raise ValueError("a search needs at least one determinized world")
        if decisive is not None and not 0.0 < decisive <= 1.0:
            raise ValueError(f"decisive is a prior in (0, 1], not {decisive!r}")
        self.evaluator = evaluator
        self.simulations = simulations
        self.wave = wave
        self.exploration = exploration
        self.stance = stance
        self.rank_rows = STANCE_ROWS[stance]
        self.root_noise = root_noise
        self.noise_fraction = noise_fraction
        self.k = k
        self.hidden = hidden
        self.hold = hold
        self.rng = rng or random.Random()
        self.own_calibration = (
            None if own_calibration is None
            else (float(own_calibration[0]), float(own_calibration[1]))
        )
        self.decisive = None if decisive is None else float(decisive)
        self.decision_log: list[Decision] = []
        self.play_against(against)

    def play_against(self, seats: Collection[int]) -> None:
        """Search as one side against `seats` (`against`); empty, every seat
        for itself."""
        self.against = tuple(sorted({int(s) for s in seats}))

    def _sided(self, mover: int) -> bool:
        """Whether `mover` ranks as the side against `against`."""
        return bool(self.against) and mover not in self.against

    def worlds(self, game: Game) -> list[tuple[float, Game]]:
        """The determinizations this decision is searched in, each with its
        share of the `k` draws.

        `holdings_signature` keys them: the deck's order is
        a chance stream a `BUY_DEV_CARD` edge reshuffles anyway, so two draws
        differing only there are one world. `hidden=False` skips sampling and
        answers the true state, the omniscient search."""
        # Imported here for the same reason `STANCES` is.
        from .bots.determinized import distinct_worlds, holdings_signature

        if not self.hidden or is_over(game):
            return [(1.0, imagine(game, self.rng, randomize_deck=False))]
        seat = to_move(game)
        out = []
        for share, state in distinct_worlds(
            game.state(seat), self.rng, self.k, holdings_signature, self.hold
        ):
            world = imagine(game, self.rng, randomize_deck=False, seat=seat)
            world.set_state(state)
            out.append((share, world))
        return out

    def _options(self, game) -> tuple[Action, ...]:
        if is_over(game):
            return ()
        if type(game) is NativeCopy:
            return game.legal()
        return tuple(legal_actions(game))

    def _held(self, world: Game):
        """`world` as this tree's root position: held natively where the
        evaluator's policy reads native positions and the engine takes the
        world (module docstring), else the `Game` itself."""
        contract = getattr(self.evaluator, "native_contract", None)
        space = getattr(self.evaluator, "space", None)
        if contract is None or space is None or not copyable(world, contract, space):
            return world
        native = held(world)
        return world if native is None else native

    def _node(self, game, dice: object = None) -> Node:
        options = self._options(game)
        native = type(game) is NativeCopy
        if native:
            mover = 0 if is_over(game) else game.to_move()
            seats = game.dealt
        else:
            mover = 0 if is_over(game) else to_move(game)
            # true state: `num_players` is a fixed board property.
            seats = game.state(0, hidden=False).num_players
        node = Node(
            game=game,
            mover=mover,
            options=options,
            children=[None] * len(options),
            visits=np.zeros(len(options)),
            virtual=np.zeros(len(options)),
            totals=np.zeros((len(options), seats)),
            ranked=np.zeros(len(options)),
            dice=dice if native else None,
        )
        if node.terminal:
            node.value = tuple(float(v) for v in self.evaluator.terminal(game))
            node.observer_values = tuple(node.value for _ in node.value)
            node.prior = np.zeros(0)
        return node

    def _advance(self, node: Node, index: int, roll: int | None) -> Game:
        """The position one action on, before it is wrapped in a `Node`: a
        chance edge reads the outcome off it before it knows whether it already
        holds a node, and wrapping one costs a `legal_actions`."""
        action = node.options[index]
        if type(node.game) is NativeCopy:
            return self._advance_native(node.game, action, roll)
        # The encoder sees only deck size, never order, so the hidden-deck
        # randomisation is deferred until an edge actually draws a card.
        child = imagine(
            node.game,
            self.rng,
            randomize_deck=action.type is ActionType.BUY_DEV_CARD,
        )
        if action.type is ActionType.ROLL:
            roll_dice(child, roll)
        else:
            apply(child, action)
        return child

    def _advance_native(self, game: NativeCopy, action: Action, roll: int | None):
        """`_advance` from a native position, drawing what the Python tree
        draws from `self.rng`, in the same order: nothing on a chance-free
        edge, the steal off `Live(self.rng)` (what `imagine` gives a copy) on
        a robber edge that robs someone, the given `roll` on a roll edge, and
        on a development card's purchase the deck's shuffle, from `self.rng`
        as `imagine` shuffles a Python copy's before the draw."""
        if action.type is ActionType.BUY_DEV_CARD:
            child = game.fork(_NO_CHANCE)
            child.shuffle_deck(self.rng)
            child.apply(action)
            return child
        if action.type is ActionType.ROLL:
            chance = _Rolled(roll)
        elif draws_hidden(game, action):
            chance = Live(self.rng)
        else:
            chance = _NO_CHANCE
        child = game.fork(chance)
        child.apply(action)
        return child

    def _dice_after(self, node: Node, index: int, roll: int | None) -> object:
        """The balanced-dice deck one edge on from a native node: rolled on a
        roll edge, as `roll_dice` rolls `Game._state.dice`; unchanged else."""
        if node.dice is None or node.options[index].type is not ActionType.ROLL:
            return node.dice
        return node.dice.rolled(node.game.current_player, roll)

    def _step(self, node: Node, index: int, roll: int | None) -> Node:
        return self._node(self._advance(node, index, roll), self._dice_after(node, index, roll))

    def _draws_hidden(self, game: Game, action: Action) -> bool:
        """See `draws_hidden`."""
        return draws_hidden(game, action)

    def _sample(self, node: Node, index: int, slot: _Chance) -> Node:
        """Draw this edge's hidden card once, reusing the child that outcome has.

        Discarding the repeat's fresh copy is exact, not approximate: the copies
        differ only in the deck order beneath the top card, which nothing
        downstream observes. The draw comes from `self.rng` in descent order, so
        the search stays a pure function of its seed."""
        child = self._advance(node, index, None)
        outcome = _drawn(node.game, child, node.options[index])
        held = slot.outcomes.get(outcome)
        if held is not None:
            return held
        fresh = self._node(child, node.dice)
        slot.outcomes[outcome] = fresh
        return fresh

    def _select(self, node: Node) -> int:
        assert node.prior is not None
        counts = node.visits + node.virtual
        total = float(counts.sum())
        # An unvisited edge scores zero rather than the parent's value, so the
        # prior decides what gets tried first. The stance reads the mean vector,
        # not the mean of what it read per visit; ranking the mean is the max^n
        # backup, and the two differ for the non-linear `paranoid`. A descent
        # still in flight counts as a lost visit (`lost_value`).
        sided = self._sided(node.mover)
        if not sided and self.stance == "paranoid":
            totals = self.rank_rows(node.totals, node.mover) + lost_value(
                self.stance, live_seats(node.game)
            ) * node.virtual
            means = np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
            bonus = self.exploration * node.prior * math.sqrt(max(total, 1e-8)) / (1 + counts)
            return int(np.argmax(means + bonus))
        # The same scores as the array form above, edge by edge in plain
        # floats -- the same operations in the same order, so the same bits --
        # since a node's few edges cost less as a loop than as array calls.
        # A side's loss is one of `against` winning: 0, as for `own`.
        lost = 0.0 if sided else lost_value(self.stance, live_seats(node.game))
        root = math.sqrt(max(total, 1e-8))
        exploration = self.exploration
        best, best_score = 0, -math.inf
        for i, (count, ranked, virtual, prior) in enumerate(zip(
            counts.tolist(), node.ranked.tolist(), node.virtual.tolist(), node.prior.tolist()
        )):
            if count > 0:
                mean = (ranked if sided else ranked + lost * virtual) / (count if count > 1 else 1)
            else:
                mean = 0.0
            score = mean + exploration * prior * root / (1 + count)
            if score > best_score:
                best, best_score = i, score
        return best

    def _perturb(self, roots: Sequence[Node]) -> None:
        """Dirichlet noise on root priors, self-play only.

        Without it visits go in prior order and the target is a sharpened copy of
        the prior, collapsing the policy toward its own argmax. Zero by default:
        perturb the roots you learn from, never those you evaluate on."""
        if self.root_noise <= 0.0 or self.noise_fraction <= 0.0:
            return
        for node in roots:
            assert node.prior is not None
            draw = np.array(
                [self.rng.gammavariate(self.root_noise, 1.0) for _ in node.prior]
            )
            total = float(draw.sum())
            if total <= 0.0:
                continue
            node.prior = (1.0 - self.noise_fraction) * node.prior + (
                self.noise_fraction * draw / total
            )

    def _descend(self, root: Node) -> tuple[list[tuple[Node, int]], Node]:
        path: list[tuple[Node, int]] = []
        node = root
        while node.expanded and not node.terminal:
            index = self._select(node)
            node.virtual[index] += 1
            path.append((node, index))
            slot = node.children[index]
            action = node.options[index]
            if action.type is ActionType.ROLL:
                if slot is None:
                    slot = _Chance()
                    node.children[index] = slot
                assert isinstance(slot, _Chance)
                roll = self._roll(node)
                child = slot.outcomes.get(roll)
                if child is None:
                    child = self._step(node, index, roll)
                    slot.outcomes[roll] = child
                node = child
            elif self._draws_hidden(node.game, action):
                if slot is None:
                    slot = _Chance()
                    node.children[index] = slot
                assert isinstance(slot, _Chance)
                node = self._sample(node, index, slot)
            else:
                if slot is None:
                    slot = self._step(node, index, None)
                    node.children[index] = slot
                assert isinstance(slot, Node)
                node = slot
        return path, node

    def _roll(self, node: Node) -> int:
        """A roll for the seat on turn, at the position's odds (`roll_odds`;
        at a native node, its tracked dice deck's)."""
        game = node.game
        if type(game) is NativeCopy:
            odds = ROLL_ODDS if node.dice is None else node.dice.odds(game.current_player)
        else:
            odds = roll_odds(game)
        draw = self.rng.random()
        cumulative = 0.0
        for roll, weight in odds:
            cumulative += weight
            if draw < cumulative:
                return roll
        return odds[-1][0]

    def _backup(
        self,
        path: Sequence[tuple[Node, int]],
        value: Sequence[float] | Sequence[Sequence[float]],
    ) -> None:
        vectors = np.asarray(value, dtype=np.float64)
        for node, index in path:
            # An observer-aware evaluation keeps a coherent full vector for
            # each seat. The parent node's mover is the chooser for this edge;
            # do not use the child leaf's frame for its Q value.
            vector = vectors[node.mover] if vectors.ndim == 2 else vectors
            node.virtual[index] -= 1
            self._credit(node, index, vector, 1, float(vector.sum()))

    def _credit(
        self, node: Node, index: int, vector: np.ndarray, visits: int, total: float | None = None
    ) -> None:
        """`visits` visits of edge `index`, each worth `vector`; `total`, the
        vector's sum where the caller has it."""
        node.visits[index] += visits
        node.totals[index] += vector if visits == 1 else visits * vector
        if self._sided(node.mover):
            node.ranked[index] += visits * side_value(vector, self.against)
        elif self.stance == "own":
            node.ranked[index] += visits * vector[node.mover]
        elif self.stance == "relative":
            # Over the seats still playing (`live_seats`): a retired seat
            # neither counts nor adds its column to the total.
            seats = live_seats(node.game)
            if total is None:
                total = float(vector.sum())
            for retired in node.game.locked:
                total -= float(vector[retired])
            node.ranked[index] += visits * (vector[node.mover] * seats - total) / (seats - 1)

    def _expand(self, nodes: Sequence[Node], seats: Sequence[int] | None = None) -> None:
        """Give a whole wave of leaves its prior and value in one call.
        `seats`, one per node, is the seat each node's decision is searched
        for: the seat `own_calibration` recalibrates."""
        wanted = [Leaf(node.game, node.mover, node.options) for node in nodes]
        scored = self.evaluator.evaluate(wanted)
        if len(scored) != len(wanted):
            raise ValueError(f"evaluator answered {len(scored)} of {len(wanted)} leaves")
        for node, result in zip(nodes, scored):
            if isinstance(result, Evaluation):
                prior, value = result.prior, result.value
                observer_values = result.observer_values
            else:
                prior, value = result
                observer_values = None
            node.prior = np.asarray(prior, dtype=np.float64)
            if node.prior.shape != (len(node.options),):
                raise ValueError(
                    f"prior over {node.prior.shape} for {len(node.options)} options"
                )
            vector = np.asarray(value, dtype=np.float64)
            seats_count = node.totals.shape[1]
            if vector.shape != (seats_count,) or not np.isfinite(vector).all():
                raise ValueError(f"value vector must have {seats_count} finite seats")
            if observer_values is None:
                matrix = None
                node.value = tuple(float(v) for v in vector)
            else:
                matrix = np.asarray(observer_values, dtype=np.float64)
                if matrix.shape != (seats_count, seats_count) or not np.isfinite(matrix).all():
                    raise ValueError(
                        f"observer values must be a finite {seats_count}x{seats_count} matrix"
                    )
                if not np.allclose(matrix[node.mover], vector, rtol=0.0, atol=1e-7):
                    raise ValueError("value must match the leaf mover's observer row")
                node.observer_values = tuple(
                    tuple(float(v) for v in row) for row in matrix
                )
                node.value = node.observer_values[node.mover]
        if self.own_calibration is not None and seats is not None:
            for node, seat in zip(nodes, seats):
                if node.observer_values is None:
                    node.value = calibrate_own(node.value, seat, self.own_calibration)
                else:
                    node.observer_values = tuple(
                        calibrate_own(row, seat, self.own_calibration)
                        for row in node.observer_values
                    )
                    node.value = node.observer_values[node.mover]

    def _combine(self, group: Sequence[_Run]) -> Node:
        """One root from this decision's worlds: root statistics summed
        weighted by each world's share of the draws.

        One world is its own answer and is returned as it was searched, so a
        `k=1` decision is an ordinary tree. Several are summed into a synthetic
        root, which carries the combined visits, priors and values a caller
        reads and lists the per-world roots in `worlds`; it has no children of
        its own, because two worlds' subtrees are not the same positions."""
        if len(group) == 1:
            root = group[0].root
            root.worlds = ((group[0].share, root),)
            return root
        order: dict[Action, int] = {}
        for run in group:
            for action in run.root.options:
                order.setdefault(action, len(order))
        options = tuple(order)
        seats = group[0].root.totals.shape[1]
        combined = Node(
            game=group[0].root.game,
            mover=group[0].root.mover,
            options=options,
            children=[None] * len(options),
            visits=np.zeros(len(options)),
            virtual=np.zeros(len(options)),
            totals=np.zeros((len(options), seats)),
            ranked=np.zeros(len(options)),
            prior=np.zeros(len(options)),
            worlds=tuple((run.share, run.root) for run in group),
        )
        values = np.zeros((seats, seats))
        for run in group:
            root, share = run.root, run.share
            where = [order[action] for action in root.options]
            combined.visits[where] += share * root.visits
            combined.ranked[where] += share * root.ranked
            combined.totals[where] += share * root.totals
            if root.prior is not None and root.prior.size:
                combined.prior[where] += share * root.prior
            matrix = root.observer_values
            if matrix is None:
                matrix = tuple(root.value for _ in range(seats))
            values += share * np.asarray(matrix)
        combined.observer_values = tuple(tuple(float(v) for v in row) for row in values)
        combined.value = combined.observer_values[combined.mover]
        mass = float(combined.prior.sum())
        if mass > 0:
            combined.prior = combined.prior / mass
        return combined

    def run_many(
        self, games: Sequence[Game]
    ) -> list[tuple[Node, tuple[Action, ...], np.ndarray]]:
        """Search independent decisions together, batching their leaves.

        Each decision is searched in the distinct worlds `worlds` draws for it
        and answered by `_combine`. `simulations` is the budget of one world's
        tree, not of the decision: it counts descents that cross at least one
        edge, and a world contributes at most `wave` of them before an
        expansion, so every per-world tree is the tree a `k=1` search builds
        and combined visit counts still sum to `simulations` (the shares sum
        to one). A `k>1` decision therefore costs a tree per distinct world.

        A root with one legal move is evaluated but not searched: its only
        edge is credited with `simulations` visits at the root's own value,
        which is what the evaluator says that move leads to. Its value and
        visits read like any other root's. Under `decisive`, so is a root
        whose prior on one move reaches it: that move takes the visits.

        The tree is built fresh each decision: reuse is unsound, since the roll
        that happened is one of eleven the subtree averaged over."""
        # Root evaluation cannot observe deck order, and a later BUY edge
        # shuffles immediately before drawing, so the real order leaks nothing.
        groups: list[list[_Run]] = []
        runs: list[_Run] = []
        for game in games:
            group = [
                # true state: the balanced-dice deck is public.
                _Run(self._node(self._held(world), world._state.dice), budget=self.simulations, share=share)
                for share, world in self.worlds(game)
            ]
            groups.append(group)
            runs.extend(group)
        searchable = [run for run in runs if not run.root.terminal]
        for run in runs:
            if run.root.terminal:
                run.done = run.budget
        if searchable:
            self._expand([run.root for run in searchable], [run.root.mover for run in searchable])
            forced = [run for run in searchable if len(run.root.options) == 1]
            for run in forced:
                vector = np.asarray(run.root.value, dtype=np.float64)
                self._credit(run.root, 0, vector, run.budget)
                run.done = run.budget
            open_roots = [run for run in searchable if len(run.root.options) > 1]
            if self.decisive is not None:
                for run in open_roots:
                    # Read before any root noise: the evaluator's own prior.
                    best = int(np.argmax(run.root.prior))
                    if float(run.root.prior[best]) >= self.decisive:
                        vector = np.asarray(run.root.value, dtype=np.float64)
                        self._credit(run.root, best, vector, run.budget)
                        run.done = run.budget
                open_roots = [run for run in open_roots if run.done < run.budget]
            self._perturb([run.root for run in open_roots])

        while any(run.done < run.budget for run in runs):
            # Virtual loss makes collisions rare, not impossible: a wave wider
            # than the branching factor must reuse edges, so two descents on one
            # unexpanded leaf share an evaluation and back up separately.
            waiting: dict[
                int, tuple[_Run, Node, list[list[tuple[Node, int]]]]
            ] = {}
            wanted: list[Node] = []
            wanted_seats: list[int] = []
            for run in runs:
                for _ in range(min(self.wave, run.budget - run.done)):
                    path, leaf = self._descend(run.root)
                    if leaf.expanded:
                        self._backup(path, leaf.observer_values or leaf.value or ())
                        run.done += 1
                        continue
                    entry = waiting.get(id(leaf))
                    if entry is None:
                        entry = (run, leaf, [])
                        waiting[id(leaf)] = entry
                        wanted.append(leaf)
                        wanted_seats.append(run.root.mover)
                    entry[2].append(path)

            if wanted:
                self._expand(wanted, wanted_seats)
                for run, leaf, paths in waiting.values():
                    for path in paths:
                        self._backup(path, leaf.observer_values or leaf.value or ())
                        run.done += 1

        answers = [self._combine(group) for group in groups]
        return [(root, root.options, root.visits) for root in answers]

    def run(self, game: Game) -> tuple[Node, tuple[Action, ...], np.ndarray]:
        """Search one position; the scalar form of `run_many`."""
        return self.run_many([game])[0]

    def decide(self, game: Game) -> Decision:
        """Search one position and log it (`decision_log`): the `Decision`,
        whose `chosen` is the most-visited root action."""
        began = time.process_time()
        root, options, visits = self.run(game)
        if not options:
            raise ValueError("no legal action to choose from")
        seat = root.mover
        seen = game.state(seat).state
        means = tuple(
            float(root.ranked[i] / visits[i]) if visits[i] > 0 else math.nan
            for i in range(len(options))
        )
        searched = any(
            len(world.options) > 1 and not (
                self.decisive is not None and world.prior is not None
                and float(world.prior.max()) >= self.decisive
            )
            for _, world in root.worlds
        )
        decision = Decision(
            options, np.asarray(root.prior, dtype=np.float64), tuple(root.value),
            np.asarray(visits, dtype=np.float64), means, int(np.argmax(visits)), searched,
            len(root.worlds), turn=game.turns, phase=game.phase.name, seat=seat,
            points=tuple(public_victory_points(seen, s) for s in range(seen.num_players)),
            simulations=self.simulations, cpu_seconds=time.process_time() - began,
        )
        self.decision_log.append(decision)
        return decision

    def choose(self, game: Game) -> Action:
        """The most-visited root action: a rarely-visited edge can hold a high
        mean off one lucky rollout, and PUCT's guarantees are about visits."""
        decision = self.decide(game)
        return decision.options[decision.chosen]


def visit_policy(visits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """The root's visit counts as a distribution -- expert iteration's target.
    `temperature` 1 is proportional to visit share, 0 is argmax with ties split
    evenly."""
    if visits.size == 0:
        return visits
    if temperature <= 0:
        best = visits == visits.max()
        return best / best.sum()
    weighted = visits ** (1.0 / temperature)
    total = weighted.sum()
    if total <= 0:
        return np.full(visits.shape, 1.0 / visits.size)
    return weighted / total
