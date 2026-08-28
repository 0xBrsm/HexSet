"""PUCT over a learned policy and value, with the leaves evaluated in batches.

The shape of this module is decided by one measurement: a forward costs a
~1.5 ms fixed dispatch toll plus ~25 µs per position, so a search that
evaluated one leaf per call would spend
essentially all of its time in dispatch. A hundred leaves evaluated singly cost
1.25 seconds; batched they cost about fifteen milliseconds. **Leaves are
therefore gathered into waves and handed to the evaluator together**, which is
what virtual loss is for — without it every simulation in a wave picks the same
path and the wave is worth one simulation.

This is deliberately not `hexset_ui.search2`'s `SearchBot` with a network
evaluator. That combination is the thing this replaces: it evaluates one leaf
at a time, and it lost to the handcrafted `search2` 13.3% to 86.7%. Two
separate problems are tangled there — batching and the value head's accuracy
off-policy — and only the first is addressed here.

Four places this departs from the Go setting, each for a reason already
measured in this project rather than imported from a paper:

**Per-seat vectors and a stance, not a scalar and a sign flip.** Four seats with
non-opposed fortunes need max^n, so a node backs up the whole vector and each
node maximises its own mover's reading of it. How a seat reads the vector is
`STANCE_ROWS`: `relative` — own less the mean of the others —
beat plain max^n 53.6% over 2000 games on this engine. The literature's
alternative here is CatAnalysis's κ=0.8 damping at another seat's node; it is
not used, because the stance was measured on this codebase and κ was not.

**Chance nodes are sampled, not expanded.** `SearchBot` expands a roll into all
eleven outcomes weighted by probability, which is exact and multiplies the leaf
count by eleven at every roll. Under a fixed simulation budget that is the wrong
trade: the budget should be spent where the search finds it useful, and the
frequencies of repeated simulations approximate the same distribution. A roll
edge keeps one child per outcome actually drawn.

**The tree stores its positions.** Replaying from the root would cost ~19 µs of
engine per ply against ~25 µs for a batched network evaluation, so a path of any
depth would cost more to walk than to evaluate.

**Terminal nodes are not evaluated.** A finished game has a known value, and it
is `relative_points` below — the same quantity the value head is trained to
predict, so the two are on one scale and a backup can mix them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .actions import Action, ActionType, apply, legal_actions, within_offer_budget
from .game import ROLL_ODDS, Game, imagine, is_over, roll_dice, to_move
from .victory import WINNING_POINTS, victory_points


@dataclass(frozen=True)
class Leaf:
    """A position the search wants a prior and a value for."""

    game: Game
    seat: int
    options: tuple[Action, ...]


class Evaluator(Protocol):
    """Scores a whole wave of leaves at once.

    Returns, per leaf, a prior over that leaf's `options` and a value per seat.
    The prior is expected to be normalised over the legal options; a search
    cannot use probability mass parked on actions it will never select.
    """

    def evaluate(
        self, leaves: Sequence[Leaf]
    ) -> Sequence[tuple[Sequence[float], Sequence[float]]]: ...


class _Chance:
    """The outcomes of one roll edge, kept apart from the decision tree.

    A roll is not a choice, so it gets no visit statistics of its own: the
    parent's edge counts aggregate over whatever outcomes were drawn, which is
    the sampled expectimax average.
    """

    __slots__ = ("outcomes",)

    def __init__(self) -> None:
        self.outcomes: dict[int, Node] = {}


@dataclass
class Node:
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


def _own_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    return vectors[:, seat]


def _relative_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    seats = vectors.shape[1]
    return (vectors[:, seat] * seats - vectors.sum(axis=1)) / (seats - 1)


def _paranoid_rows(vectors: np.ndarray, seat: int) -> np.ndarray:
    return vectors[:, seat] - np.delete(vectors, seat, axis=1).max(axis=1)


# These read the whole `totals` matrix at once. `_select` needs a reading for
# every child of every node it passes through — 300k scalar calls in a 400-move
# game at 64 simulations, and 12% of the whole search when done one at a time.
#
# `_relative_rows` reassociates: `v[s] - sum(others)/(n-1)` becomes
# `(v[s]*n - sum(all))/(n-1)`, which is the same number in exact arithmetic and
# within a rounding step in floating point. That is deliberate: the search has
# no published result measured against it, so there is no baseline for a
# last-bit difference to disturb.
STANCE_ROWS = {
    "own": _own_rows,
    "relative": _relative_rows,
    "paranoid": _paranoid_rows,
}


def relative_points(points: tuple[int, ...]) -> tuple[float, ...]:
    """Each seat's terminal points less the mean of the others, over 10.

    Exactly zero-sum: the per-seat values sum to zero for any input, since
    subtracting the mean of the others is an affine transform whose total
    cancels. That says in the value what the game already says — HexSet has one
    winner, so a position is only worth what it is worth compared to the table,
    and a move that lifts every seat equally earns nothing.

    Scaled by the 10 points that win a game, so a seat lands in about [-1, +1]
    and the value head does not have to learn the units.
    """
    seats = len(points)
    if seats < 2:
        raise ValueError("a relative reward needs at least two seats")
    total = sum(points)
    return tuple(
        (own - (total - own) / (seats - 1)) / WINNING_POINTS for own in points
    )


def _relative(game: Game) -> tuple[float, ...]:
    seats = game.state.num_players
    return relative_points(tuple(victory_points(game.state, s) for s in range(seats)))


class Search:
    """One tree per decision. Not reused across moves — see `run`.

    `exploration` is PUCT's c_puct. The values it trades against are relative
    terminal points over the ten that win a game, so they sit in about [-1, +1]
    and a constant near 1 is the right order; a constant tuned for a [0, 1] win
    probability would be twice as exploratory as intended.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        simulations: int = 128,
        wave: int = 16,
        exploration: float = 1.25,
        stance: str = "relative",
        max_offers: int | None = None,
        root_noise: float = 0.0,
        noise_fraction: float = 0.25,
        rng: random.Random | None = None,
    ) -> None:
        if stance not in STANCE_ROWS:
            raise ValueError(f"unknown stance: {stance}")
        if simulations < 1 or wave < 1:
            raise ValueError("a search needs at least one simulation and one wave")
        self.evaluator = evaluator
        self.simulations = simulations
        self.wave = wave
        self.exploration = exploration
        self.stance = stance
        self.rank_rows = STANCE_ROWS[stance]
        self.max_offers = max_offers
        self.root_noise = root_noise
        self.noise_fraction = noise_fraction
        self.rng = rng or random.Random()

    def _options(self, game: Game) -> tuple[Action, ...]:
        if is_over(game):
            return ()
        return tuple(within_offer_budget(game, legal_actions(game), self.max_offers))

    def _node(self, game: Game) -> Node:
        options = self._options(game)
        node = Node(
            game=game,
            mover=0 if is_over(game) else to_move(game),
            options=options,
            children=[None] * len(options),
            visits=np.zeros(len(options)),
            virtual=np.zeros(len(options)),
            totals=np.zeros((len(options), game.state.num_players)),
            ranked=np.zeros(len(options)),
        )
        if node.terminal:
            node.value = _relative(game)
            node.prior = np.zeros(0)
        return node

    def _step(self, node: Node, index: int, roll: int | None) -> Node:
        action = node.options[index]
        # The encoder sees only deck size, never order. Defer the expensive
        # hidden-deck randomisation until an edge actually draws a card; this
        # keeps every draw uniform without reshuffling on every other edge.
        child = imagine(
            node.game,
            self.rng,
            randomize_deck=action.type is ActionType.BUY_DEV_CARD,
        )
        if action.type is ActionType.ROLL:
            roll_dice(child, roll)
        else:
            apply(child, action)
        return self._node(child)

    def _select(self, node: Node) -> int:
        assert node.prior is not None
        counts = node.visits + node.virtual
        total = float(counts.sum())
        # An unvisited edge scores zero rather than the parent's value. Both are
        # defensible; zero is AlphaZero's and it makes the prior, not the
        # parent's optimism, decide what gets tried first.
        #
        # The stance reads the mean vector, rather than the mean of what the
        # stance read at each visit. Those agree for `own` and `relative`, which
        # are linear, and differ for `paranoid`, whose max is not. Ranking the
        # mean is the one that stays a max^n backup: the edge's estimate of the
        # position is the averaged vector, and the mover reads that.
        totals = (
            self.rank_rows(node.totals, node.mover)
            if self.stance == "paranoid"
            else node.ranked
        )
        means = np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
        bonus = self.exploration * node.prior * math.sqrt(max(total, 1e-8)) / (1 + counts)
        return int(np.argmax(means + bonus))

    def _perturb(self, roots: Sequence[Node]) -> None:
        """Dirichlet noise on root priors, self-play only.

        Without it the search only ever looks where the policy already points:
        an unvisited edge scores zero, so visits go in prior order and the visit
        target is close to a sharpened copy of the prior. Training on that is
        self-distillation with a sharpening operator, and the policy collapses
        toward its own argmax instead of improving. Left at zero by default —
        AlphaZero perturbs the roots it learns from and never the ones it is
        evaluated on.
        """
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
            if node.options[index].type is ActionType.ROLL:
                if slot is None:
                    slot = _Chance()
                    node.children[index] = slot
                assert isinstance(slot, _Chance)
                roll = self._roll()
                child = slot.outcomes.get(roll)
                if child is None:
                    child = self._step(node, index, roll)
                    slot.outcomes[roll] = child
                node = child
            else:
                if slot is None:
                    slot = self._step(node, index, None)
                    node.children[index] = slot
                assert isinstance(slot, Node)
                node = slot
        return path, node

    def _roll(self) -> int:
        draw = self.rng.random()
        cumulative = 0.0
        for roll, weight in ROLL_ODDS:
            cumulative += weight
            if draw < cumulative:
                return roll
        return ROLL_ODDS[-1][0]

    def _backup(self, path: Sequence[tuple[Node, int]], value: Sequence[float]) -> None:
        vector = np.asarray(value, dtype=np.float64)
        total = float(vector.sum()) if self.stance == "relative" else 0.0
        for node, index in path:
            node.visits[index] += 1
            node.virtual[index] -= 1
            node.totals[index] += vector
            if self.stance == "own":
                node.ranked[index] += vector[node.mover]
            elif self.stance == "relative":
                seats = vector.size
                node.ranked[index] += (
                    vector[node.mover] * seats - total
                ) / (seats - 1)

    def _expand(self, nodes: Sequence[Node]) -> None:
        """Give a whole wave of leaves its prior and value in one call."""
        wanted = [Leaf(node.game, node.mover, node.options) for node in nodes]
        scored = self.evaluator.evaluate(wanted)
        if len(scored) != len(wanted):
            raise ValueError(f"evaluator answered {len(scored)} of {len(wanted)} leaves")
        for node, (prior, value) in zip(nodes, scored):
            node.prior = np.asarray(prior, dtype=np.float64)
            if node.prior.shape != (len(node.options),):
                raise ValueError(
                    f"prior over {node.prior.shape} for {len(node.options)} options"
                )
            node.value = tuple(float(v) for v in value)

    def run_many(
        self, games: Sequence[Game]
    ) -> list[tuple[Node, tuple[Action, ...], np.ndarray]]:
        """Search independent roots together, batching their leaves.

        `simulations` counts descents that cross at least one edge, so the visit
        counts always sum to it and a policy target built from them means the
        same thing at any wave size. A root still contributes at most `wave`
        descents before an expansion, preserving the search that was measured;
        roots are combined only at the evaluator boundary, where one network
        call can serve every game in a collector tick.

        The tree is built fresh each decision rather than carried over. Reuse is
        the usual optimisation and it is unsound here without more care than it
        is worth yet: the sampled roll that actually happened is one of eleven
        the subtree averaged over, so the statistics under it are not the
        statistics of the position reached.
        """
        # Root evaluation cannot observe deck order. A later BUY edge shuffles
        # immediately before drawing, so the copied real order leaks nothing.
        runs = [
            _Run(self._node(imagine(game, self.rng, randomize_deck=False)))
            for game in games
        ]
        searchable = []
        for run in runs:
            if run.root.terminal:
                run.done = self.simulations
                continue
            if len(run.root.options) == 1:
                run.root.visits = np.ones(1)
                run.done = self.simulations
            else:
                searchable.append(run.root)
        if searchable:
            self._expand(searchable)
            self._perturb(searchable)

        while any(run.done < self.simulations for run in runs):
            # Virtual loss makes collisions rare, not impossible: a wave wider
            # than the branching factor must reuse edges. Two descents that land
            # on the same unexpanded leaf share one evaluation and back up
            # separately, because a leaf queued earlier in this wave has no value
            # yet and would otherwise be sent a second time.
            waiting: dict[
                int, tuple[_Run, Node, list[list[tuple[Node, int]]]]
            ] = {}
            wanted: list[Node] = []
            for run in runs:
                for _ in range(min(self.wave, self.simulations - run.done)):
                    path, leaf = self._descend(run.root)
                    if leaf.expanded:
                        self._backup(path, leaf.value or ())
                        run.done += 1
                        continue
                    entry = waiting.get(id(leaf))
                    if entry is None:
                        entry = (run, leaf, [])
                        waiting[id(leaf)] = entry
                        wanted.append(leaf)
                    entry[2].append(path)

            if wanted:
                self._expand(wanted)
                for run, leaf, paths in waiting.values():
                    for path in paths:
                        self._backup(path, leaf.value or ())
                        run.done += 1

        return [(run.root, run.root.options, run.root.visits) for run in runs]

    def run(self, game: Game) -> tuple[Node, tuple[Action, ...], np.ndarray]:
        """Search one position; the scalar form of `run_many`."""
        return self.run_many([game])[0]

    def choose(self, game: Game) -> Action:
        """The most-visited root action.

        Visit count rather than mean value: a rarely-visited edge can hold a
        high mean off one lucky rollout, and the visit distribution is the
        quantity PUCT's guarantees are about.
        """
        _, options, visits = self.run(game)
        if not options:
            raise ValueError("no legal action to choose from")
        return options[int(np.argmax(visits))]


