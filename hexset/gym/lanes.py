# SPDX-License-Identifier: GPL-3.0-only
"""Batched game collection without a Gymnasium or neural-network dependency.

Each tick exposes one decision per live lane; the caller batches inference and
answers with one legal action per request. Requests expose the live engine
state, not a snapshot, so encode before stepping.

A seat a caller names in `decisions` is asked for whole decisions instead
(`hexset.decisions`): its request carries the decision's `tensors` and is
answered with an action or a `Compound`, whose parts are applied one at a
time, each its own step of the lane's action stream and its record. Every
other seat, a bot's included, acts one move at a time. The snake seat's
double setup is asked as two setup turns, the second on the board after the
first (`hexset.decisions.tensors`' `double_turn=False`).

Every request names the run of options it is a part of
(`hexset.decisions.closes`): `Request.decision_id` counts a game's runs from
0 over every seat, `Request.part` the request's place in its run. A run lasts
from a seat's first option to the first that shows something or hands the
table on, so a learner can take it as one unit of credit while each part is
still one forward's choice. `Decision.decision_id` files every move under it.

During a seven's discards several seats owe at once. `discard_order="seat"`
asks the lowest owing seat first (`hexset.game.to_move`); `"random"` draws the
next seat uniformly among those owing, from a stream of the game's own that
draws nothing from its dice or deck, and lets it finish its whole discard
before drawing again.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import decisions as D
from ..actions import Action, ActionSpace, Stuck, apply, legal_actions, options_for, space_for
from ..arena import MAX_ACTIONS, deal_game, game_key
from ..board.board import BOARD_MODES, Board
from ..game import MAX_TURNS, Game, Phase, is_over, players_owing_discards, to_move
from ..record import Record, Tape, recording
from ..rules import STANDARD
from ..victory import victory_points

if TYPE_CHECKING:  # annotation-only
    from ..bots import Bot
    from ..view import View


# How a discard round's seats are ordered (`LaneEnv`'s `discard_order`).
DISCARD_ORDERS = ("seat", "random")


@dataclass
class Request:
    """One lane's decision. `seat` is the seat to act, not always
    `current_player` (a discard is another seat's), and is both `view`'s
    perspective and the seat this is filed under; `step` indexes the lane's
    whole action stream, at the decision's first move. `game` is live state,
    private information included, and no information boundary: read `view`,
    never mutate `game`, and encode before `step`.

    A seat asked for whole decisions (`LaneEnv`'s `decisions`) gets its
    decision as `hexset.decisions.tensors` states it in `decision`, and empty
    `options`: the answer is any option the tensors hold, an action or a
    `hexset.decisions.Compound`. Every other request lists its atomic
    `options` and has no `decision`.

    `decision_id` is the run of options this request is a part of, counted
    from 0 per game over every seat, and `part` its place there, 0 the first
    (the module docstring)."""

    lane: int
    index: int
    seat: int
    policy: int
    step: int
    options: tuple[Action, ...]
    game: Game
    decision: D.Decision | None = None
    decision_id: int = 0
    part: int = 0
    _view: View | None = None

    @property
    def view(self) -> View:
        if self._view is None:
            self._view = self.game.state(self.seat)
        return self._view


@dataclass(frozen=True)
class Decision:
    """One action, filed under the seat that took it. `step` is its position in
    the lane's action stream, so a caller can key its own record by
    `(seat, step)`; `decision_id` the run of options it was made in
    (`Request.decision_id`)."""

    seat: int
    step: int
    policy: int
    action: Action
    decision_id: int = 0


@dataclass(frozen=True)
class Outcome:
    """How a game ended, scalarisation left to the caller. `points` is read off
    the true state, so hidden VP cards count; `truncated` means the action cap
    stopped it before the engine ended it, and `exhausted` that it ran to the
    turn cap (`LaneEnv.turn_cap`) with no winner. Either way `winner` is
    `None` and the game was not decided; one with a winner is neither.
    `trades` counts exchanges over the whole game, since
    `Game.trades`/`trades_made` are turn-scoped."""

    winner: int | None
    points: tuple[int, ...]
    turns: int
    actions: int
    truncated: bool
    trades: int = 0
    exhausted: bool = False


@dataclass(frozen=True)
class Episode:
    """One finished game: per-seat decisions plus how it ended. `seed`/`index`
    identify the default deal only, so use `records=True` for a self-contained
    replay. `trades` is every cleared exchange as `(step, a, b, received)`,
    carried explicitly because a replay seats no gates."""

    index: int
    seed: int
    players: int
    cast: tuple[int, ...]
    decisions: tuple[tuple[Decision, ...], ...]
    outcome: Outcome
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    # Replay from this, not by rebuilding from `(seed, index)`.
    record: Record | None = None

    def stream(self) -> list[Decision]:
        """Every decision back in the order it was taken."""
        return sorted(
            (d for seat in self.decisions for d in seat), key=lambda d: d.step
        )

    def __len__(self) -> int:
        return sum(len(seat) for seat in self.decisions)


def _bind_gates(game: Game) -> None:
    """`seat_at(game)` on every gate `game` asks that has one (a network
    bot, a search over one, a `TradesBy` seat's trader)."""
    for gate in game.gates or ():
        seat_at = getattr(gate, "seat_at", None)
        if seat_at is not None:
            seat_at(game)


class BoardBots:
    """One bot per board, for a policy id played by a scripted bot; keyed by
    board object and evicted oldest-first.

    One bot serves every seat and lane on the same board and policy id, so its
    factory must support that sharing: mutable state, random generators
    included, is shared, and stochastic choices can therefore depend on request
    order and lane count. Seat-specific trade gates belong in `LaneEnv.gates`.
    """

    def __init__(self, spawn: Callable[[Board], Bot], capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("a bot per board needs room for at least one")
        self.spawn = spawn
        self.capacity = capacity
        self._bots: OrderedDict[int, tuple[Board, Bot]] = OrderedDict()

    def bot(self, board: Board) -> Bot:
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


@dataclass
class _Lane:
    """One game in flight and the bookkeeping that leaves with it."""

    index: int
    game: Game
    cast: tuple[int, ...]
    by_seat: list[list[Decision]]
    actions: int = 0
    trades: int = 0
    trade_log: list[tuple[int, int, int, tuple[int, ...]]] = field(default_factory=list)
    # The record under construction, or `None` when nobody asked for one.
    tape: Tape | None = None
    # Under `discard_order="random"`: the seat discarding now, and the
    # game's own stream that draws it (made at the first discard).
    discarding: int | None = None
    discard_rng: random.Random | None = None
    # Each seat's run of options under way, `(decision_id, next part)`, and
    # how many runs the game has begun.
    running: dict[int, tuple[int, int]] = field(default_factory=dict)
    runs: int = 0


class LaneEnv:
    """`lanes` games in flight, stepped one action each per tick: `requests()`
    hands out one `Request` per live lane, `step(actions)` answers them all and
    refills the lanes that ended.

    `caster(index) -> (policy id per seat)` must be a pure function of the game
    index. An id in `bots` is played by that bot and may be answered `None`;
    any other id is the caller's. That bot is also its own gate if it
    implements the optional trading methods, where a caller-driven seat instead
    supplies `gates[id](game, seat)` -- a `NetworkBot` used only as a gate must
    be bound with `seat_at(game)`, and a missing gate disables trading there.
    A bot serves every lane on its board (`BoardBots`), so before each action
    every gate in that lane with a `seat_at` is seated at the lane's game.

    `deal_game(seed, index, players)` fixes each game's board and chance stream
    independently of lane count; `first_game`/`stride` shard the index sequence,
    worker `w` of `K` using `first_game=w, stride=K`. The discard order under
    `"random"` is drawn per game too (`f"discard-order:{game_key(seed, index)}"`),
    so it does not depend on the lanes either.
    """

    def __init__(
        self,
        players: int = 4,
        seed: int = 0,
        lanes: int = 8,
        *,
        deal: int | None = None,
        action_cap: int = MAX_ACTIONS,
        turn_cap: int = MAX_TURNS,
        board: Board | Callable[[int], Board | None] | None = None,
        board_mode: str = "random",
        caster: Callable[[int], Sequence[int]] | None = None,
        bots: Mapping[int, Callable[[Board], Bot]] | None = None,
        gates: Mapping[int, Callable[[Game, int], object]] | None = None,
        first_game: int = 0,
        stride: int = 1,
        bot_capacity: int = 256,
        records: bool = False,
        discard_order: str = "seat",
        decisions: Collection[int] = (),
    ) -> None:
        """`deal` bounds how many games are started; left `None` the
        environment refills forever, where an evaluation wants a fixed cohort
        since the first `n` games to finish select for short ones.
        `first_game` is where the counter starts. `board` pins every game to
        one geometry, or as a callable is a board law
        `board(index) -> Board | None`, `None` meaning the default deal,
        which `board_mode` names (`hexset.arena.deal_board`).
        `records` has each `Episode` carry a `hexset.record.Record`.
        `discard_order` is `DISCARD_ORDERS`' choice (the module docstring).
        `decisions` names the caller's policy ids asked for whole decisions;
        a bot's id cannot be one.
        """
        if players < 2:
            raise ValueError("a game needs at least two seats")
        if lanes < 1:
            raise ValueError("a lane environment needs at least one lane")
        if deal is not None and deal < 1:
            raise ValueError("a lane environment cannot be asked to deal nothing")
        if stride < 1:
            raise ValueError("a lane environment cannot deal backwards or stand still")
        if action_cap < 1:
            raise ValueError("an action cap below one action ends every game empty")
        if board_mode not in BOARD_MODES:
            raise ValueError(f"unknown board mode {board_mode!r}: {sorted(BOARD_MODES)}")
        if discard_order not in DISCARD_ORDERS:
            raise ValueError(f"unknown discard order {discard_order!r}: {DISCARD_ORDERS}")
        if set(decisions) & set(bots or {}):
            raise ValueError("a bot acts a move at a time; it cannot be asked for decisions")
        self.players = players
        self.seed = seed
        self.action_cap = action_cap
        self.turn_cap = turn_cap
        self.board = board
        self.board_mode = board_mode
        self.caster = caster
        self.gates = dict(gates or {})
        self.bots = {
            pid: BoardBots(spawn, capacity=bot_capacity)
            for pid, spawn in (bots or {}).items()
        }
        self.stride = stride
        self.records = records
        self.discard_order = discard_order
        self.decisions = frozenset(decisions)
        self._spaces: dict[tuple[int, ...], ActionSpace] = {}
        self.ticks = 0
        self.steps = 0
        self.games = 0
        self._next = first_game
        self._stop = None if deal is None else first_game + deal * stride
        self._lanes: list[_Lane | None] = [self._fresh() for _ in range(lanes)]
        self._outstanding: tuple[Request, ...] | None = None

    # --- dealing ----------------------------------------------------------

    def _cast(self, index: int) -> tuple[int, ...]:
        if self.caster is None:
            return (0,) * self.players
        cast = tuple(self.caster(index))
        if len(cast) != self.players:
            raise ValueError(
                f"cast {cast} seats {len(cast)} of game {index}'s {self.players}"
            )
        if any(pid < 0 for pid in cast):
            raise ValueError(f"cast {cast} names a policy that cannot exist")
        return cast

    def _seat_gates(self, game: Game, cast: Sequence[int]) -> tuple[object, ...]:
        """Who answers each seat's private gate; a bot-played seat is seated as
        its own bot, so the two cannot disagree."""
        out: list[object] = []
        for seat, pid in enumerate(cast):
            bench = self.bots.get(pid)
            if bench is not None:
                out.append(bench.bot(game.state(0, hidden=False).board))
                continue
            make = self.gates.get(pid)
            out.append(None if make is None else make(game, seat))
        return tuple(out)

    def _fresh(self) -> _Lane | None:
        if self._stop is not None and self._next >= self._stop:
            return None
        index = self._next
        self._next += self.stride
        cast = self._cast(index)
        board = self.board(index) if callable(self.board) else self.board
        game = deal_game(
            self.seed,
            index,
            self.players,
            board=board,
            board_mode=self.board_mode,
            # Lanes deal the standard game (`deal_game`'s default type).
            chance=(lambda rng: recording(rng, STANDARD)) if self.records else None,
            turn_cap=self.turn_cap,
        )
        game.gates = self._seat_gates(game, cast)
        return _Lane(
            index=index,
            game=game,
            cast=cast,
            by_seat=[[] for _ in range(self.players)],
            tape=Tape() if self.records else None,
        )

    # --- the tick ---------------------------------------------------------

    def requests(self) -> tuple[Request, ...]:
        """One decision per live lane, in lane order; idempotent within a tick.
        Raises `hexset.actions.Stuck` if a live game offers a seat no legal
        action, which is always an engine bug."""
        if self._outstanding is not None:
            return self._outstanding
        batch: list[Request] = []
        for slot, lane in enumerate(self._lanes):
            if lane is None:
                continue
            game = lane.game
            seat = self._actor(lane)
            policy = lane.cast[seat]
            run = lane.running.get(seat)
            if run is None:
                run = lane.running[seat] = (lane.runs, 0)
                lane.runs += 1
            if policy in self.decisions:
                # A seat that acts in whole decisions never leaves itself
                # inside one, so `tensors` refusing such a position is a bug.
                options: tuple[Action, ...] = ()
                decision = D.tensors(game, seat, self._space(game), double_turn=False)
            else:
                decision = None
                options = tuple(options_for(game) if seat == to_move(game) else legal_actions(game, seat))
                if not options:
                    raise Stuck(f"no legal action in {game.phase.name} for player {seat}")
            batch.append(
                Request(
                    lane=slot,
                    index=lane.index,
                    seat=seat,
                    policy=policy,
                    step=lane.actions,
                    options=options,
                    game=game,
                    decision=decision,
                    decision_id=run[0],
                    part=run[1],
                )
            )
        self._outstanding = tuple(batch)
        return self._outstanding

    def _actor(self, lane: _Lane) -> int:
        """Who acts next in `lane`: `to_move`, but in a discard round under
        `discard_order="random"` the seat drawn among those owing, kept until
        its discard is done."""
        game = lane.game
        if self.discard_order == "seat" or game.phase is not Phase.DISCARD:
            lane.discarding = None
            return to_move(game)
        if lane.discarding is None or game.discard_quota[lane.discarding] <= 0:
            if lane.discard_rng is None:
                lane.discard_rng = random.Random(f"discard-order:{game_key(self.seed, lane.index)}")
            lane.discarding = lane.discard_rng.choice(players_owing_discards(game))
        return lane.discarding

    def _space(self, game: Game) -> ActionSpace:
        topology = game._state.board.topology
        key = (topology.num_vertices, topology.num_edges, topology.num_hexes, game._state.num_players)
        space = self._spaces.get(key)
        if space is None:
            space = self._spaces[key] = space_for(game)
        return space

    def _legal(self, request: Request, option: D.Option) -> bool:
        if request.decision is None:
            return option in request.options
        try:
            D.locate(request.decision, option, request.game, request.seat, self._space(request.game))
        except (ValueError, KeyError, IndexError):
            return False
        return True

    def step(
        self, actions: Sequence[D.Option | None] | Mapping[int, D.Option]
    ) -> list[Episode]:
        """Apply one answer per outstanding request; return the games that ended.

        `actions` is a sequence in `requests()` order, entries possibly `None`,
        or a lane-to-answer mapping. An answer is an action, or for a request
        with a `decision` any of its options: a `Compound`'s parts are applied
        in order, each a step of its own, until the game ends or the action
        cap stops it. An unanswered request is played by its seat's bot, and
        is an error with no bot either. Missing, illegal or inactive-lane
        responses raise before any game advances, though bot calls may
        already have advanced their own generators.
        """
        requests = self.requests()
        if isinstance(actions, Mapping):
            unknown = actions.keys() - {r.lane for r in requests}
            if unknown:
                raise ValueError(f"actions name inactive lanes: {sorted(unknown)}")
            chosen: list[D.Option | None] = [actions.get(r.lane) for r in requests]
        else:
            chosen = list(actions)
            if len(chosen) != len(requests):
                raise ValueError(
                    f"{len(chosen)} actions answer {len(requests)} requests"
                )
        # Validate the whole response before advancing any lane: a missing
        # answer late in the batch must not leave earlier lanes a step ahead.
        for request, action in zip(requests, chosen):
            if action is None and request.policy not in self.bots:
                raise ValueError(
                    f"lane {request.lane} seat {request.seat} is policy "
                    f"{request.policy}, which has no bot to answer for it"
                )
            if action is not None and not self._legal(request, action):
                raise ValueError(f"illegal action {action!r} for lane {request.lane}")
        resolved: list[D.Option] = []
        for request, action in zip(requests, chosen):
            if action is None:
                action = self._bot_action(request)
                if action not in request.options:
                    raise ValueError(f"illegal bot action {action!r} for lane {request.lane}")
            resolved.append(action)
        self._outstanding = None

        finished: list[Episode] = []
        for request, option in zip(requests, resolved):
            lane = self._lanes[request.lane]
            assert lane is not None  # a request is only built for a live lane
            parts = option.parts if isinstance(option, D.Compound) else (option,)
            event_before = lane.game.trade_event_turn
            done = False
            for action in parts:
                self.steps += 1
                if self._play(lane, request, action):
                    finished.append(self._harvest(lane, request.lane))
                    done = True
                    break
            if done:
                continue
            if D.closes(lane.game, option, request.seat, event_before):
                del lane.running[request.seat]
            else:
                lane.running[request.seat] = (request.decision_id, request.part + 1)

        self.ticks += 1
        return finished

    def _play(self, lane: _Lane, request: Request, action: Action) -> bool:
        """Apply one move of `request`'s seat in `lane` and file it; whether
        the game is done (over, or at the action cap)."""
        game = lane.game
        lane.by_seat[request.seat].append(
            Decision(
                seat=request.seat,
                step=lane.actions,
                policy=request.policy,
                action=action,
                decision_id=request.decision_id,
            )
        )
        # Only a discard's actor is not fixed by the position.
        discarding = game.phase is Phase.DISCARD
        actor = request.seat if discarding and request.seat != to_move(game) else None
        # `trades_made` and `game.trades` are turn-scoped, so the rise and
        # the slice since `before` are this action's own exchanges.
        cleared = game.trades_made
        before = len(game.trades)
        shown_before = len(game.shown)
        # The action can open a trade event, which asks every gate here;
        # one shared with another lane was last seated at that lane.
        _bind_gates(game)
        apply(game, action, request.seat if discarding else None)
        lane.trades += max(0, game.trades_made - cleared)
        for trade in game.trades[before:]:
            lane.trade_log.append(
                (lane.actions, trade.a, trade.b, tuple(trade.received))
            )
        if lane.tape is not None:
            lane.tape.step(action, game.trades[before:],
                           game.shown[shown_before:], before=before, actor=actor)
        lane.actions += 1
        return is_over(game) or lane.actions >= self.action_cap

    def _bot_action(self, request: Request) -> Action:
        bench = self.bots.get(request.policy)
        if bench is None:
            raise ValueError(
                f"lane {request.lane} seat {request.seat} is policy "
                f"{request.policy}, which has no bot to answer for it"
            )
        # public field: a bot reads the board off the true state.
        bot = bench.bot(request.game.state(0, hidden=False).board)
        game = request.game
        if request.seat == to_move(game):
            return bot.choose(game)
        # A discard drawn out of seat order: a bot answers for `to_move`, so
        # it is asked with the seat alone owing, and the round put back.
        quota = game.discard_quota
        game.discard_quota = [n if s == request.seat else 0 for s, n in enumerate(quota)]
        try:
            return bot.choose(game)
        finally:
            game.discard_quota = quota

    def _harvest(self, lane: _Lane, slot: int) -> Episode:
        game = lane.game
        # true state: terminal victory points include hidden dev cards.
        state = game.state(0, hidden=False)
        # The record names the stream the game's draws came from, not the run
        # seed: `hexset.record.replay` checks recorded chance against it.
        record = (
            None
            if lane.tape is None
            else lane.tape.sealed(game, seed=game_key(self.seed, lane.index))
        )
        episode = Episode(
            index=lane.index,
            seed=self.seed,
            players=self.players,
            cast=lane.cast,
            decisions=tuple(tuple(seat) for seat in lane.by_seat),
            trades=tuple(lane.trade_log),
            record=record,
            outcome=Outcome(
                winner=game.won_by,
                points=tuple(
                    victory_points(state, seat) for seat in range(self.players)
                ),
                turns=game.turns,
                actions=lane.actions,
                truncated=game.won_by is None and not is_over(game),
                trades=lane.trades,
                exhausted=game.won_by is None and is_over(game),
            ),
        )
        self.games += 1
        self._lanes[slot] = self._fresh()
        return episode

    # --- driving ----------------------------------------------------------

    @property
    def running(self) -> bool:
        """False once a bounded environment has played out everything it dealt."""
        return any(lane is not None for lane in self._lanes)

    def drain(
        self,
        answer: Callable[[Sequence[Request]], Sequence[Action | None]] | None = None,
    ) -> list[Episode]:
        """Play every dealt game to completion; requires a `deal` bound.
        `answer` is the batched policy, one action per request in order, or
        `None` to let every seat's own bot play."""
        if self._stop is None:
            raise ValueError("an unbounded environment never drains; pass `deal`")
        out: list[Episode] = []
        while self.running:
            requests = self.requests()
            out.extend(
                self.step([None] * len(requests) if answer is None else answer(requests))
            )
        return out

    def in_flight(self) -> Iterator[Game]:
        return (lane.game for lane in self._lanes if lane is not None)

    def pending(self) -> tuple[int, ...]:
        """Actions taken so far in each live lane's unfinished game."""
        return tuple(lane.actions for lane in self._lanes if lane is not None)

    def cohort(self, games: int) -> None:
        """Re-arm a bounded environment for `games` more from where the counter
        stands, refilling idle lanes; an unbounded one refuses."""
        if games < 1:
            raise ValueError("a cohort needs at least one game")
        if self._stop is None:
            raise ValueError("an unbounded environment has no cohorts; build it with `deal`")
        self._stop = self._next + games * self.stride
        self._outstanding = None
        self._lanes = [lane if lane is not None else self._fresh() for lane in self._lanes]

    def games_started(self) -> int:
        """How many games have been dealt out, finished or not. Pass it back as
        `first_game` to resume; games still in flight are lost."""
        return self._next


__all__ = [
    "BoardBots",
    "DISCARD_ORDERS",
    "Decision",
    "Episode",
    "LaneEnv",
    "Outcome",
    "Request",
]
