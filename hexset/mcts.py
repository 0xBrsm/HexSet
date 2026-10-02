# SPDX-License-Identifier: GPL-3.0-only
"""Monte Carlo tree search with PUCT selection and batched policy/value inference.

Leaves are collected in waves; virtual loss discourages duplicate descents.
Nodes store per-seat value vectors and select with the mover's stance. Dice,
steals and dev-card draws are sampled per simulation, not frozen at expansion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .actions import (
    Action,
    ActionType,
    apply,
    legal_actions,
    victim_of,
)
from .game import ROLL_ODDS, Game, imagine, is_over, roll_dice, to_move
from .view import HoldReading

__all__ = [
    "Leaf",
    "Evaluator",
    "HIDDEN_DRAW",
    "draws_hidden",
    "sampled_children",
    "Node",
    "STANCE_ROWS",
    "Search",
    "visit_policy",
]


@dataclass(frozen=True)
class Leaf:
    """A position the search wants a prior and a value for."""

    game: Game
    seat: int
    options: tuple[Action, ...]


class Evaluator(Protocol):
    """Scores a whole wave of leaves at once: per leaf, a prior over that leaf's
    `options` and a value per seat. The prior must be normalised over the legal
    options. A value is each seat's chance of winning, the scale on which a
    finished game is its one-hot winner."""

    def evaluate(
        self, leaves: Sequence[Leaf]
    ) -> Sequence[tuple[Sequence[float], Sequence[float]]]: ...

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
    # true state: identifying which card was drawn needs the actual hand/
    # dev-card delta, not an information-set estimate of it.
    if action.type is ActionType.BUY_DEV_CARD:
        was = before.state(player, hidden=False).new_dev_cards[player]
        now = after.state(player, hidden=False).new_dev_cards[player]
    else:
        was = before.state(player, hidden=False).hands[player]
        now = after.state(player, hidden=False).hands[player]
    for index, (old, new) in enumerate(zip(was, now)):
        if new > old:
            return index
    return -1


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
    `relative`, [-1, 1] for `paranoid`. `stance` is restricted to `STANCE_ROWS`'s keys, what this
    tree's backup implements; anything else is refused at construction."""

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
            world = imagine(game, self.rng, randomize_deck=False)
            world.set_state(state)
            out.append((share, world))
        return out

    def _options(self, game: Game) -> tuple[Action, ...]:
        if is_over(game):
            return ()
        return tuple(legal_actions(game))

    def _node(self, game: Game) -> Node:
        options = self._options(game)
        node = Node(
            game=game,
            mover=0 if is_over(game) else to_move(game),
            options=options,
            children=[None] * len(options),
            visits=np.zeros(len(options)),
            virtual=np.zeros(len(options)),
            # true state: `num_players` is a fixed board property.
            totals=np.zeros((len(options), game.state(0, hidden=False).num_players)),
            ranked=np.zeros(len(options)),
        )
        if node.terminal:
            node.value = tuple(float(v) for v in self.evaluator.terminal(game))
            node.prior = np.zeros(0)
        return node

    def _advance(self, node: Node, index: int, roll: int | None) -> Game:
        """The position one action on, before it is wrapped in a `Node`: a
        chance edge reads the outcome off it before it knows whether it already
        holds a node, and wrapping one costs a `legal_actions`."""
        action = node.options[index]
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

    def _step(self, node: Node, index: int, roll: int | None) -> Node:
        return self._node(self._advance(node, index, roll))

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
        fresh = self._node(child)
        slot.outcomes[outcome] = fresh
        return fresh

    def _select(self, node: Node) -> int:
        assert node.prior is not None
        counts = node.visits + node.virtual
        total = float(counts.sum())
        # An unvisited edge scores zero rather than the parent's value, so the
        # prior decides what gets tried first. The stance reads the mean vector,
        # not the mean of what it read per visit; ranking the mean is the max^n
        # backup, and the two differ for the non-linear `paranoid`.
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
                roll = self._roll()
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
        value = np.zeros(seats)
        for run in group:
            root, share = run.root, run.share
            where = [order[action] for action in root.options]
            combined.visits[where] += share * root.visits
            combined.ranked[where] += share * root.ranked
            combined.totals[where] += share * root.totals
            if root.prior is not None and root.prior.size:
                combined.prior[where] += share * root.prior
            value += share * np.asarray(root.value or np.zeros(seats))
        combined.value = tuple(float(v) for v in value)
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

        The tree is built fresh each decision: reuse is unsound, since the roll
        that happened is one of eleven the subtree averaged over."""
        # Root evaluation cannot observe deck order, and a later BUY edge
        # shuffles immediately before drawing, so the real order leaks nothing.
        groups: list[list[_Run]] = []
        runs: list[_Run] = []
        for game in games:
            group = [
                _Run(self._node(world), budget=self.simulations, share=share)
                for share, world in self.worlds(game)
            ]
            groups.append(group)
            runs.extend(group)
        searchable = []
        for run in runs:
            if run.root.terminal:
                run.done = run.budget
                continue
            if len(run.root.options) == 1:
                run.root.visits = np.ones(1)
                run.done = run.budget
            else:
                searchable.append(run.root)
        if searchable:
            self._expand(searchable)
            self._perturb(searchable)

        while any(run.done < run.budget for run in runs):
            # Virtual loss makes collisions rare, not impossible: a wave wider
            # than the branching factor must reuse edges, so two descents on one
            # unexpanded leaf share an evaluation and back up separately.
            waiting: dict[
                int, tuple[_Run, Node, list[list[tuple[Node, int]]]]
            ] = {}
            wanted: list[Node] = []
            for run in runs:
                for _ in range(min(self.wave, run.budget - run.done)):
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

        answers = [self._combine(group) for group in groups]
        return [(root, root.options, root.visits) for root in answers]

    def run(self, game: Game) -> tuple[Node, tuple[Action, ...], np.ndarray]:
        """Search one position; the scalar form of `run_many`."""
        return self.run_many([game])[0]

    def choose(self, game: Game) -> Action:
        """The most-visited root action: a rarely-visited edge can hold a high
        mean off one lucky rollout, and PUCT's guarantees are about visits."""
        _, options, visits = self.run(game)
        if not options:
            raise ValueError("no legal action to choose from")
        return options[int(np.argmax(visits))]


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
