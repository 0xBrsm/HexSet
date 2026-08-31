# SPDX-License-Identifier: GPL-3.0-only
"""Collect self-play experience from many games at once.

The network costs a ~1.5 ms fixed dispatch toll per forward plus ~25 µs per
position, so a collector that steps one game and calls the net per move would
spend essentially all of its time in dispatch. That single measurement decides
the shape of this module: hold `lanes` games in flight, step them in lockstep,
and gather one batch of decisions per tick so a single forward serves every
lane. Exactly one seat acts at a time in a given game, so a lane contributes
exactly one position per tick and a tick is one batch of size `lanes`.

The policy is behind `BatchPolicy` — a batched `hexset.bots.Bot` — which is what
keeps this file numpy-only. PyTorch cannot be installed on the development
phone, so the plumbing is tested and timed here against a trivial policy and
only the torch-backed policy is deferred to the training box.

**Reward is deliberately not here.** A collector emits per-seat trajectories and
the terminal `Outcome`; turning that into returns is the caller's to decide,
because the choice between terminal win/loss and terminal victory points is
still open. Nothing in this module assumes either.

Trajectories come out demultiplexed by seat. A Catan game interleaves four
seats' decisions into one action stream, and a seat's next state is not the one
that immediately follows its action — it is the next position that seat was
asked about. Keeping one list per seat is what makes a transition mean anything
to PPO.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, Sequence

import numpy as np

from .actions import (
    Action,
    ActionSpace,
    apply,
    legal_actions,
    space_for,
    within_offer_budget,
)
from .arena import MAX_ACTIONS
from .board.board import Board, random_base_board
from .bots import Bot
from .encoding import Observation, encode, encode_batch
from .game import Game, is_over, start, to_move
from .play import Stuck
from .victory import victory_points


@dataclass(frozen=True)
class Request:
    """One lane's decision, put to the policy alongside every other lane's.

    `options` rides along with `mask` because the two are not interchangeable:
    an offer is ten numbers, so `PROPOSE_TRADE` occupies one slot in the flat
    space and its bundles cannot be recovered from an index. A policy that
    samples the flat categorical still has to name the offer it meant.

    `seat` is `to_move`, which is not always `current_player` — discarding on a
    seven and answering an offer belong to somebody else — and it is both the
    perspective `observation` was encoded from and the seat the transition is
    filed under.

    `game` is the live lane state, and it rides along for a policy that searches:
    a tree needs positions to step, and an observation is a lossy encoding of
    one. A policy that only reads the encoding can ignore it. Handed out rather
    than copied because `hexset.mcts` copies at its own root; a policy that
    mutates this corrupts the lane.
    """

    lane: int
    seat: int
    observation: Observation
    mask: np.ndarray
    options: tuple[Action, ...]
    game: Game | None = None


@dataclass(frozen=True)
class Choice:
    """What the policy did, and what PPO will want to have recorded with it.

    `value` is the per-seat vector the value head emits, in the mover's frame
    like the encoder. Empty means the policy has no estimate, which is the case
    for every scripted policy.

    `aux` is carried through onto the `Transition` untouched and never read
    here. PPO has to recompute the log-prob of a stored decision under the
    updated network, and for `PROPOSE_TRADE` that needs the set of offers that
    were legal at the time — which is in `Request.options` and is recoverable
    from nothing the transition otherwise keeps. Rather than teach this module
    what an offer is, it hands the policy a pocket. Default empty, so no
    scripted policy notices.
    """

    action: Action
    log_prob: float = 0.0
    value: tuple[float, ...] = ()
    aux: object = None


class BatchPolicy(Protocol):
    """The batched analogue of `hexset.bots.Bot`.

    One call per tick, so a torch implementation collates, moves and runs the
    network once for the whole batch. Must return one `Choice` per `Request`,
    in order.
    """

    def act(self, requests: Sequence[Request]) -> Sequence[Choice]: ...


def action_mask(space: ActionSpace, options: Sequence[Action]) -> np.ndarray:
    """Mark already-enumerated actions without enumerating them again."""
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True
    return mask


@dataclass
class RandomPolicy:
    """Uniform over the legal actions.

    Torch-free, so it measures the cost of the plumbing alone and gives the
    tests a policy whose behaviour they can predict. Its `log_prob` is the real
    one for a uniform choice rather than a placeholder.
    """

    rng: random.Random = field(default_factory=random.Random)

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        return [
            Choice(
                action=self.rng.choice(request.options),
                log_prob=-math.log(len(request.options)),
            )
            for request in requests
        ]


class BotPolicy:
    """A `hexset.bots.Bot` behind `BatchPolicy`, one bot per board.

    A scripted bot answers one position at a time and pays no dispatch toll, so
    batching buys it nothing — but the handcrafted evaluators cache per-vertex
    pips for the board they were built on, and every lane plays its own board.
    So the wrapper spawns a bot per board it meets, keyed by the board object
    while a reference is held (an `id` is only a stable key while its object is
    alive), and evicts oldest-first so a long run does not keep one bot per
    finished game. An evicted board still in play is respawned on its next
    request, which costs a spawn and nothing else.

    `Choice.log_prob` and `value` stay at their scripted defaults, which is
    right because a casting collector never records these seats — see
    `Collector`.
    """

    def __init__(self, spawn: Callable[[Board], Bot], capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("a bot per board needs room for at least one")
        self.spawn = spawn
        self.capacity = capacity
        self._bots: OrderedDict[int, tuple[Board, Bot]] = OrderedDict()

    def _bot(self, board: Board) -> Bot:
        key = id(board)
        held = self._bots.get(key)
        if held is not None:
            self._bots.move_to_end(key)
            return held[1]
        bot = self.spawn(board)
        self._bots[key] = (board, bot)
        while len(self._bots) > self.capacity:
            self._bots.popitem(last=False)
        return bot

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        for request in requests:
            if request.game is None:
                raise ValueError("a scripted bot needs the live game on the request")
        return [
            Choice(action=self._bot(request.game.state.board).choose(request.game))
            for request in requests
        ]


@dataclass(frozen=True)
class Transition:
    """One decision, filed under the seat that made it.

    `step` is its position in the lane's whole action stream, so the interleaved
    order is recoverable even though the storage is per seat.

    `aux` is whatever the policy attached to its `Choice`, carried verbatim.
    """

    seat: int
    step: int
    observation: Observation
    mask: np.ndarray
    action: Action
    index: int
    log_prob: float
    value: tuple[float, ...]
    aux: object = None


@dataclass(frozen=True)
class Outcome:
    """How a game ended, with the scalarisation left to the caller.

    Both candidate rewards are here: `winner` for terminal win/loss and `points`
    for terminal victory points. `truncated` says the action cap stopped it,
    which is a different thing from a game that ran out of turns and a very
    different thing from a game that was won.
    """

    winner: int | None
    points: tuple[int, ...]
    turns: int
    actions: int
    truncated: bool


@dataclass(frozen=True)
class Episode:
    """One finished game: per-seat trajectories plus how it ended.

    `index` and `seed` are enough to rebuild the game exactly, so an episode can
    be replayed against the engine rather than trusted.

    `cast` is who played each seat — 0 the learner, `k` the collector's
    `opponents[k - 1]` — and empty means everyone was the learner. Opponent
    seats have empty trajectories, so `stream()` on a cast episode has gaps and
    cannot replay the game; the outcome still describes the whole table.
    """

    index: int
    seed: int
    players: int
    trajectories: tuple[tuple[Transition, ...], ...]
    outcome: Outcome
    cast: tuple[int, ...] = ()

    def stream(self) -> list[Transition]:
        """Every transition back in the order it was taken."""
        return sorted(
            (t for seat in self.trajectories for t in seat), key=lambda t: t.step
        )

    def __len__(self) -> int:
        return sum(len(seat) for seat in self.trajectories)


def owned(episodes: Sequence[Episode], learner: int) -> list[Episode]:
    """Each episode with only `learner`'s seats kept — the table league's
    assemble-side counterpart to the collector's recording gate.

    An empty `cast` means the pre-league convention: every seat was learner 0.
    Seats belonging to other learners are emptied, not dropped, so `assemble`
    skips them exactly as it has always skipped opponent seats, and per-seat
    indexing (rewards, rotation) is untouched.
    """
    out = []
    for episode in episodes:
        cast = episode.cast or (0,) * episode.players
        trajectories = tuple(
            seat_transitions if cast[seat] == learner else ()
            for seat, seat_transitions in enumerate(episode.trajectories)
        )
        out.append(
            Episode(
                index=episode.index,
                seed=episode.seed,
                players=episode.players,
                trajectories=trajectories,
                outcome=episode.outcome,
                cast=episode.cast,
            )
        )
    return out


def new_game(seed: int, index: int, players: int, board: Board | None = None) -> Game:
    """The `index`-th game of a run, reproducible from the seed alone.

    Same derivation as `hexset.arena._play_one`, so a game plays identically
    however many lanes are in flight and whichever lane happens to draw it.
    """
    if board is None:
        board = random_base_board(random.Random(f"{seed}:{index}:board"))
    return start(board, players, random.Random(f"{seed}:{index}:game"))


@dataclass
class _Lane:
    index: int
    game: Game
    by_seat: list[list[Transition]]
    cast: tuple[int, ...] = ()
    actions: int = 0


class Collector:
    """`lanes` games in flight, stepped in lockstep, one batch per tick.

    A finished lane is refilled on the spot, so the batch stays full and a long
    game never stalls the others. Every lane plays its own board by default;
    they share a topology, which is what lets one batch cross into the model
    against a single index set.

    `opponents` and `caster` put other policies on some seats. The caster maps
    a game index to one policy id per seat — 0 the learner, `k` for
    `opponents[k - 1]` — and must be a pure function of the index, so a resumed
    run casts the same games the same way. Each tick still makes one `act` call
    per policy, so the learner's batch stays batched. **Opponent decisions are
    never recorded**: their seats' trajectories stay empty, which is exactly
    what `hexset.ppo.assemble` skips, so opponent play shapes the games the
    learner sees without ever entering an update.
    """

    def __init__(
        self,
        policy: BatchPolicy,
        *,
        lanes: int = 8,
        players: int = 4,
        seed: int = 0,
        action_cap: int = MAX_ACTIONS,
        board: Board | None = None,
        max_offers: int | None = None,
        first_game: int = 0,
        deal: int | None = None,
        fill: bool = True,
        opponents: Sequence[BatchPolicy] = (),
        caster: Callable[[int], Sequence[int]] | None = None,
        learners: Sequence[int] = (0,),
        stride: int = 1,
        pair_boards: bool = False,
    ) -> None:
        """`first_game` is where the game counter starts.

        A training run that crashes and resumes would otherwise replay the games
        it had already learned from, since a game is a pure function of the seed
        and its index. Checkpointing `games_started` and passing it back here is
        what makes a resumed run continue rather than repeat.

        `deal` bounds how many games are ever started. Left `None`, a lane is
        refilled the moment its game ends and the collector runs forever, which
        is what a training run wants. An *evaluation* wants a fixed cohort, and
        without a bound the only way to get one is to keep dealing replacements
        and throw them away after playing them in full — which is where a
        400-game duel went to spend ten minutes.

        `stride` deals every `stride`-th index instead of every one, which is
        how parallel collectors shard one run's games: worker `w` of `K` takes
        `first_game = base + w, stride = K` and the workers' index sets are
        disjoint by construction while every game stays the same pure function
        of `(seed, index)` it always was.

        `pair_boards` transplants the duel pairing to collection: games `2k`
        and `2k+1` share the board keyed `f"{seed}:{2k}:board"` while each
        keeps its own `f"{seed}:{index}:game"` rng — same geometry, independent
        dice and play. The even half's board key is the one unpaired dealing
        already uses for that index, so this extends the (seed, index) law
        rather than amending it, and a pair straddling two strided workers
        still shares its board because each derives it from the same key.
        """
        if lanes < 1:
            raise ValueError("a collector needs at least one lane")
        if deal is not None and deal < 1:
            raise ValueError("a collector cannot be asked to deal nothing")
        if stride < 1:
            raise ValueError("a collector cannot deal backwards or stand still")
        if caster is not None and not opponents:
            raise ValueError("a caster without opponents has nobody to cast")
        if pair_boards and board is not None:
            raise ValueError(
                "pair_boards deals each pair its own shared board; a fixed "
                "board= would put every game on one board and the pairing "
                "would compare nothing"
            )
        self.policy = policy
        self.opponents = tuple(opponents)
        self.caster = caster
        # Which policy ids record their seats. {0} is every run before gen3 —
        # one learner, opponents as scenery. The table league seats several
        # learners in one game: id 0 is `policy`, id k>0 is `opponents[k-1]`,
        # and a seat records iff its id is here. Dispatch needs no change —
        # `_answers` already routes requests per policy id — so the whole
        # single-learner assumption was this gate and the assemble-side
        # `owned` filter.
        self._learners = frozenset(learners)
        if any(not 0 <= pid <= len(self.opponents) for pid in self._learners):
            raise ValueError("a learner id names a policy that is not seated")
        self.players = players
        self.seed = seed
        self.action_cap = action_cap
        self.max_offers = max_offers
        self.board = board
        self.pair_boards = pair_boards
        self.ticks = 0
        self.steps = 0
        self.games = 0
        self.stride = stride
        self._next = first_game
        self._stop = None if deal is None else first_game + deal * stride
        self._lanes: list[_Lane | None] = (
            [self._fresh() for _ in range(lanes)] if fill else [None] * lanes
        )
        first = next((lane.game for lane in self._lanes if lane is not None), None)
        # `cohort` starts empty, so the space cannot come from a dealt lane. It is
        # a pure function of the rules, not of the game, so any position gives it.
        self.space: ActionSpace = space_for(
            first if first is not None else new_game(seed, first_game, players, board)
        )

    def _fresh(self) -> _Lane | None:
        if self._stop is not None and self._next >= self._stop:
            return None
        index = self._next
        self._next += self.stride
        board = self.board
        if self.pair_boards:
            # `2 * (index // 2)` names the pair's even half, whose key is
            # exactly the one unpaired dealing derives for that index — so the
            # even games' boards replay bit-identically and only the odd games
            # move onto their mate's geometry. The game rng below stays keyed
            # to `index` itself, which is where the two halves diverge.
            board = random_base_board(
                random.Random(f"{self.seed}:{2 * (index // 2)}:board")
            )
        cast = (0,) * self.players
        if self.caster is not None:
            cast = tuple(self.caster(index))
            if len(cast) != self.players or any(
                not 0 <= pid <= len(self.opponents) for pid in cast
            ):
                raise ValueError(f"cast {cast} does not fit game {index}")
        return _Lane(
            index=index,
            game=new_game(self.seed, index, self.players, board),
            by_seat=[[] for _ in range(self.players)],
            cast=cast,
        )

    def _ask(
        self, lane: _Lane, slot: int, observation: Observation | None = None
    ) -> Request:
        game = lane.game
        seat = to_move(game)
        options = legal_actions(game)
        if not options:
            raise Stuck(f"no legal action in {game.phase.name} for player {seat}")
        # Filtered before the mask is built, so the policy never sees a slot the
        # budget forbids and cannot be trained to want one.
        options = within_offer_budget(game, options, self.max_offers)
        return Request(
            lane=slot,
            seat=seat,
            observation=(
                observation
                if observation is not None
                else encode(game, seat)
            ),
            # Built from the actions already enumerated rather than by calling
            # `legal_mask`, which would enumerate them a second time.
            mask=action_mask(self.space, options),
            options=tuple(options),
            game=game,
        )

    def _requests(self, live: Sequence[tuple[int, _Lane]]) -> list[Request]:
        """Build one tick's observations through the vectorized encoder."""
        seats = [to_move(lane.game) for _, lane in live]
        observations = encode_batch([lane.game for _, lane in live], seats)
        return [
            self._ask(lane, slot, observation)
            for (slot, lane), observation in zip(
                live, observations, strict=True
            )
        ]

    def _answers(
        self, live: Sequence[tuple[int, _Lane]], requests: Sequence[Request]
    ) -> list[Choice]:
        """One `act` call per policy, reassembled in request order."""
        if not self.opponents:
            choices = list(self.policy.act(requests))
            if len(choices) != len(requests):
                raise ValueError(
                    f"policy answered {len(choices)} of {len(requests)} requests"
                )
            return choices
        policies = (self.policy, *self.opponents)
        shares: list[list[int]] = [[] for _ in policies]
        for i, ((_, lane), request) in enumerate(zip(live, requests)):
            shares[lane.cast[request.seat]].append(i)
        out: list[Choice | None] = [None] * len(requests)
        for policy, share in zip(policies, shares):
            if not share:
                continue
            answers = policy.act([requests[i] for i in share])
            if len(answers) != len(share):
                raise ValueError(
                    f"policy answered {len(answers)} of {len(share)} requests"
                )
            for i, choice in zip(share, answers):
                out[i] = choice
        return out  # type: ignore[return-value]

    def _harvest(self, lane: _Lane, slot: int) -> Episode:
        game = lane.game
        episode = Episode(
            index=lane.index,
            seed=self.seed,
            players=self.players,
            trajectories=tuple(tuple(seat) for seat in lane.by_seat),
            cast=lane.cast,
            outcome=Outcome(
                winner=game.won_by,
                points=tuple(
                    victory_points(game.state, seat) for seat in range(self.players)
                ),
                turns=game.turns,
                actions=lane.actions,
                truncated=game.won_by is None and not is_over(game),
            ),
        )
        self.games += 1
        self._lanes[slot] = self._fresh()
        return episode

    def tick(self) -> list[Episode]:
        """Step every live lane once. Returns the games that ended on this tick."""
        live = [(s, lane) for s, lane in enumerate(self._lanes) if lane is not None]
        if not live:
            return []
        requests = self._requests(live)
        choices = self._answers(live, requests)

        finished: list[Episode] = []
        for (slot, lane), request, choice in zip(live, requests, choices):
            if lane.cast[request.seat] in self._learners:
                lane.by_seat[request.seat].append(
                    Transition(
                        seat=request.seat,
                        step=lane.actions,
                        observation=request.observation,
                        mask=request.mask,
                        action=choice.action,
                        index=self.space.index(choice.action),
                        log_prob=choice.log_prob,
                        value=tuple(choice.value),
                        aux=choice.aux,
                    )
                )
            apply(lane.game, choice.action)
            lane.actions += 1
            if is_over(lane.game) or lane.actions >= self.action_cap:
                finished.append(self._harvest(lane, slot))

        self.ticks += 1
        self.steps += len(live)
        return finished

    @property
    def running(self) -> bool:
        """False once a bounded collector has played out everything it dealt."""
        return any(lane is not None for lane in self._lanes)

    def drain(self) -> list[Episode]:
        """Play every dealt game to completion. Requires a `deal` bound."""
        if self._stop is None:
            raise ValueError("an unbounded collector never drains; pass `deal`")
        out: list[Episode] = []
        while self.running:
            out.extend(self.tick())
        return out

    def run(self, ticks: int) -> list[Episode]:
        out: list[Episode] = []
        for _ in range(ticks):
            out.extend(self.tick())
        return out

    def collect(self, episodes: int) -> list[Episode]:
        """Tick until `episodes` games have finished.

        Terminates: the action cap ends every lane within `action_cap` ticks, so
        this cannot spin however badly the policy plays.
        """
        if self._stop is not None:
            # A bounded collector stops dealing, so asking for more than it has
            # left would spin on empty ticks rather than block on a slow game.
            left = (self._stop - self._next) // self.stride + sum(
                lane is not None for lane in self._lanes
            )
            if left < episodes:
                raise ValueError(f"{episodes} games wanted, {left} left to finish")
        out: list[Episode] = []
        while len(out) < episodes:
            out.extend(self.tick())
        return out

    def cohort(self, games: int) -> list[Episode]:
        """Deal `games` fresh games and play every one of them to completion.

        This is what a PPO iteration wants and `collect` is not. `collect`
        refills a lane the moment its game ends, so its batch is part
        replacement games and part whatever happened to be mid-game when the
        last one finished — and those unfinished lanes carry across the
        learner's weight sync, which is how a trajectory ends up stitched from
        several policy generations. A cohort starts with empty lanes, deals
        nothing beyond `games`, and leaves nothing in flight, so every position
        it returns was played under one set of weights.

        It also removes the length bias `deal` exists to prevent: taking the
        first `n` games to finish selects for short ones, and game length is
        not independent of who is winning.

        `lanes` stays free — it is the concurrency, not the cohort. Below
        `games` the lanes refill until the cohort is dealt out and only the
        last wave tails off; at `games` every game starts together and the
        longest one ticks alone at the end. That tail is what a cohort costs.
        """
        if self._stop is not None:
            raise ValueError("a bounded collector deals its one cohort at build time")
        if games < 1:
            raise ValueError("a cohort needs at least one game")
        if any(lane is not None for lane in self._lanes):
            raise RuntimeError(
                "the lanes still hold games; a cohort collector must be built "
                "with fill=False and is empty again after every cohort"
            )
        self._stop = self._next + games * self.stride
        self._lanes = [self._fresh() for _ in range(len(self._lanes))]
        try:
            return self.drain()
        finally:
            self._stop = None

    def in_flight(self) -> Iterator[Game]:
        return (lane.game for lane in self._lanes if lane is not None)

    def pending(self) -> tuple[int, ...]:
        """Actions taken so far in each live lane's unfinished game."""
        return tuple(lane.actions for lane in self._lanes if lane is not None)

    def games_started(self) -> int:
        """How many games this collector has dealt out, finished or not.

        Pass it back as `first_game` to carry on where a run left off. The games
        still in flight are lost on a resume — they hold engine state, not data
        — which costs at most `lanes` partial games once per crash.
        """
        return self._next
