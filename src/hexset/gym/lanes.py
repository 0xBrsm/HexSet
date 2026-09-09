# SPDX-License-Identifier: GPL-3.0-only
"""Many games in flight at once, stepped in lockstep, one batch per tick.

A learner that steps a single game and asks its network for each move spends
essentially all of its time in dispatch: the measured toll is ~1.5 ms fixed per
forward against ~25 µs per position, so one position per call wastes the
hardware by a factor of fifty. That measurement is the whole reason this module
exists. Hold `lanes` games in flight, step every one of them once per tick, and
hand the caller the whole tick as one batch, so a single forward serves every
lane. Exactly one seat is ever to move in a Catan position, so a live lane
contributes exactly one decision per tick and a tick is a batch of `lanes`.

`hexset.gym.aec` is the same engine for a different consumer: one game, one
agent per seat, PettingZoo's turn-taking convention. That environment seats no
trade gate, so an agent played through it cannot trade -- there was nobody to
ask. Here a gate is seated at *every* seat (`game.gates`), so the engine's own
trade event (`hexset.trading`) runs exactly as it does under `hexset.arena`,
and a lane environment is the first gym entry point at which a learner can
trade at all.

Nothing about the model is here, and nothing about reward. A tick's `Request`
carries the position, the seat's information set and its legal actions; the
caller answers with an `Action`. What a decision is worth, what it is encoded
as, and what the trainer keeps alongside it are the caller's, because the
choice between terminal win/loss and terminal victory points is not the
engine's to make -- both are on the `Outcome`.

Decisions come back demultiplexed by seat. One game interleaves four seats'
decisions into a single action stream, and a seat's next position is not the
one that follows its action but the next one that seat was asked about; keeping
a list per seat with the stream index on every entry is what makes a transition
mean anything to a policy-gradient update, while `Episode.stream()` puts the
interleaving back.

This module is numpy-free and imports neither `pettingzoo` nor `gymnasium` --
it is a plain engine driver, and a training loop should not have to install a
gym API it does not use to get one.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..actions import Action, apply
from ..arena import MAX_ACTIONS, deal_game, game_key
from ..board.board import Board
from ..actions import options_for
from ..game import Game, is_over, to_move
from ..record import Record, Tape, recording
from ..victory import victory_points

if TYPE_CHECKING:  # annotation-only, so no import cost and no cycle
    from ..bots import Bot
    from ..view import View


@dataclass
class Request:
    """One lane's decision, put to the caller alongside every other lane's.

    `seat` is `to_move`, which is not always `current_player` -- discarding on
    a seven belongs to somebody else -- and it is both the perspective `view`
    is taken from and the seat the decision is filed under.

    `policy` is which policy id the cast put at this seat (`caster`), so a
    driver holding several policies can route the batch without consulting the
    episode. `step` is the decision's index in the lane's whole action stream,
    the same number the resulting `Decision` carries.

    `game` is the live lane state, handed out rather than copied because a
    searching caller needs positions to step and `hexset.mcts` copies at its
    own root. A caller that mutates it corrupts the lane.

    `view` is the seat's information set (`game.state(seat)`), computed on
    first access rather than up front: a seat played by its own bot is never
    asked for one, and a tick that built every lane's view whether or not
    anybody read it would pay for the opponents as well as the learner.
    """

    lane: int
    index: int
    seat: int
    policy: int
    step: int
    options: tuple[Action, ...]
    game: Game
    _view: View | None = None

    @property
    def view(self) -> View:
        if self._view is None:
            self._view = self.game.state(self.seat)
        return self._view


@dataclass(frozen=True)
class Decision:
    """One action, filed under the seat that took it.

    `step` is its position in the lane's whole action stream, so the
    interleaved order survives being stored per seat, and so a caller can key
    its own record of the decision -- an encoding, a log-probability, a value
    estimate -- to this one by `(seat, step)` without the engine having to know
    what any of those are.
    """

    seat: int
    step: int
    policy: int
    action: Action


@dataclass(frozen=True)
class Outcome:
    """How a game ended, with the scalarisation left to the caller.

    Both candidate rewards are here: `winner` for terminal win/loss and
    `points` for terminal victory points, per seat and read off the true state
    so hidden victory-point development cards count. `truncated` says the
    action cap stopped the game, which is a different thing from a game that
    ran out of turns and a very different thing from a game that was won.

    `trades` is how many exchanges the engine's trade events cleared over the
    whole game. Counted as the game was played rather than read off it at the
    end, because `Game.trades`/`trades_made` are turn-scoped -- `end_turn`
    clears both -- and because a trade is not an action and so appears nowhere
    in the decision stream.
    """

    winner: int | None
    points: tuple[int, ...]
    turns: int
    actions: int
    truncated: bool
    trades: int = 0


@dataclass(frozen=True)
class Episode:
    """One finished game: per-seat decisions plus how it ended.

    `seed` and `index` are the whole game (`hexset.arena.deal_game`), so an
    episode can be replayed against the engine rather than trusted.

    `cast` is which policy id held each seat, the caster's verdict for this
    index. `decisions` is one tuple per seat; `trades` is every exchange the
    table cleared, `(step, a, b, received)` -- the shape
    `hexset.record.Record` keeps, with `step` indexing into `stream()` exactly
    as `Decision.step` does. A replay that only re-applied the actions would
    play a trade-free game however the table traded, because a replay seats no
    gates (`hexset.record.advance`: nobody is there to ask), so the exchanges
    are carried explicitly.
    """

    index: int
    seed: int
    players: int
    cast: tuple[int, ...]
    decisions: tuple[tuple[Decision, ...], ...]
    outcome: Outcome
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    # The game itself, when the environment was built with `records=True`:
    # a `hexset.record.Record` of exactly the game this lane played, which
    # `hexset.record.replay`/`replay_to` reopen at any ply. `None`
    # otherwise, never partly built -- an environment that was not asked for
    # records does not even wrap its chance source.
    #
    # This is what a driver should replay from. Rebuilding the game from
    # `(seed, index)` and walking `stream()` forward is a second replay path
    # beside `hexset.record`, and two paths that must agree about a position
    # eventually do not.
    record: Record | None = None

    def stream(self) -> list[Decision]:
        """Every decision back in the order it was taken."""
        return sorted(
            (d for seat in self.decisions for d in seat), key=lambda d: d.step
        )

    def __len__(self) -> int:
        return sum(len(seat) for seat in self.decisions)


class BoardBots:
    """One bot per board, for a policy id that is played by a scripted bot.

    The handcrafted evaluators cache per-vertex pips for the board they were
    built on and every lane plays its own board, so a single bot cannot serve
    the table. Bots are keyed by the board object while a reference to it is
    held -- an `id` is only a stable key while its object is alive -- and
    evicted oldest-first, so a long run does not accumulate one bot per
    finished game. An evicted board still in play is respawned on its next
    request, which costs a spawn and nothing else.

    One bot serves every seat of a board that casts the same id twice. That is
    deliberate: a `hexset.bots` bot is a pure function of the position it is
    handed (its `choose` and its `gains_many` both take the seat's own view),
    so two seats sharing one are indistinguishable from two seats holding one
    each, and the pips cache is shared rather than built twice.
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


class LaneEnv:
    """`lanes` games in flight, stepped one action each per tick.

    The loop is two calls: `requests()` hands out one `Request` per live lane,
    `step(actions)` applies one `Action` per request and returns the games that
    ended. A finished lane is refilled on the spot, so the batch stays full and
    one long game never stalls the others.

    **Casting.** `caster(index) -> (policy id per seat)` decides who sits where
    and must be a pure function of the game index, so a resumed or reshared run
    casts the same games the same way. Every request carries its seat's id.
    An id in `bots` is played by that bot: the caller may answer such a request
    with `None` (or simply leave it out of a mapping) and the seated bot
    chooses. Any other id is the caller's to answer.

    **Gates.** Every seat is seated as a trade gate on `game.gates`, so the
    engine's own trade event (`hexset.trading`) asks each seat's private
    verdict and a game played here trades exactly as the same table would under
    `hexset.arena.play`. A bot is its own gate -- `hexset.bots.Bot` carries
    `gains_many` -- so an id in `bots` needs nothing further; an id the caller
    drives supplies one through `gates[id](game, seat)`, which is where a
    network-backed gate is seated. An id with neither is seated as `None`,
    which `hexset.trading` reads as "this seat never trades".

    **The games.** Every game is `hexset.arena.deal_game(seed, index, players)`
    -- the same law `hexset.arena.compete` deals from -- so a game is the same
    whichever lane draws it and however many lanes are in flight, and a lane
    environment and a tournament playing index `i` of seed `s` play the same
    game. `first_game` and `stride` shard that sequence: worker `w` of `K`
    takes `first_game=w, stride=K` and the workers' index sets are disjoint by
    construction while every game stays what it was.
    """

    def __init__(
        self,
        players: int = 4,
        seed: int = 0,
        lanes: int = 8,
        *,
        deal: int | None = None,
        action_cap: int = MAX_ACTIONS,
        board: Board | Callable[[int], Board | None] | None = None,
        caster: Callable[[int], Sequence[int]] | None = None,
        bots: Mapping[int, Callable[[Board], Bot]] | None = None,
        gates: Mapping[int, Callable[[Game, int], object]] | None = None,
        first_game: int = 0,
        stride: int = 1,
        max_trades: int | None = None,
        bot_capacity: int = 256,
        records: bool = False,
    ) -> None:
        """`deal` bounds how many games are ever started.

        Left `None`, a lane refills the moment its game ends and the
        environment runs forever, which is what a training run wants. An
        *evaluation* wants a fixed cohort, and taking the first `n` games to
        finish instead selects for short ones -- game length is not independent
        of who is winning.

        `first_game` is where the game counter starts, so a run that resumes
        from a checkpoint carries on rather than replaying the games it has
        already learned from (`games_started`).

        `board` pins every game to one geometry, which a diagnostic wants and a
        training run does not: sharing a board across lanes costs the boards'
        share of the variance. Given as a callable it is a *board law*,
        `board(index) -> Board | None`, the board for that game (`None` for
        the default, `hexset.arena.deal_board(seed, index)`): how a paired
        evaluation gives games `2k` and `2k+1` one board while each keeps its
        own dice (`hexset.casting.paired` is the casting half of the same
        pairing).

        `records` has every finished `Episode` carry a
        `hexset.record.Record` of its own game, built on the same
        `hexset.record.Tape` `hexset.arena` records a tournament game with,
        so a lane's record and a tournament's are records of the same kind.
        Off by default and skipped entirely rather than discarded: a game
        recorded here is dealt with `hexset.record.recording` wrapping its
        chance source, and an environment nobody asked for records from
        deals the plain `hexset.chance.Live` it always did.
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
        self.players = players
        self.seed = seed
        self.action_cap = action_cap
        self.board = board
        self.caster = caster
        self.max_trades = max_trades
        self.gates = dict(gates or {})
        self.bots = {
            pid: BoardBots(spawn, capacity=bot_capacity)
            for pid, spawn in (bots or {}).items()
        }
        self.stride = stride
        self.records = records
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
        """Who answers each seat's private gate for this game.

        A bot is its own gate, so a bot-played seat is seated as its bot and
        the two can never disagree about what that seat would trade.
        """
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
            chance=recording if self.records else None,
        )
        game.max_trades = self.max_trades
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
        """One decision per live lane, in lane order.

        Idempotent within a tick: calling it twice before `step` hands back the
        same batch rather than re-enumerating the positions. Raises
        `hexset.play.Stuck` if a live game offers a seat no legal action, which
        is always an engine bug and never a position to skip.
        """
        if self._outstanding is not None:
            return self._outstanding
        batch: list[Request] = []
        for slot, lane in enumerate(self._lanes):
            if lane is None:
                continue
            seat = to_move(lane.game)
            batch.append(
                Request(
                    lane=slot,
                    index=lane.index,
                    seat=seat,
                    policy=lane.cast[seat],
                    step=lane.actions,
                    options=tuple(options_for(lane.game)),
                    game=lane.game,
                )
            )
        self._outstanding = tuple(batch)
        return self._outstanding

    def step(
        self, actions: Sequence[Action | None] | Mapping[int, Action]
    ) -> list[Episode]:
        """Apply one action per outstanding request; return the games that ended.

        `actions` is either a sequence in `requests()` order, whose entries may
        be `None`, or a mapping from lane to action, which is the ergonomic
        form for a driver that answers only the seats it holds. Either way a
        request left unanswered is played by its seat's bot, and a request with
        no answer and no bot is an error -- a lane cannot be skipped, because
        the whole point of lockstep is that every live lane advances together.
        """
        requests = self.requests()
        if isinstance(actions, Mapping):
            chosen: list[Action | None] = [actions.get(r.lane) for r in requests]
        else:
            chosen = list(actions)
            if len(chosen) != len(requests):
                raise ValueError(
                    f"{len(chosen)} actions answer {len(requests)} requests"
                )
        self._outstanding = None

        finished: list[Episode] = []
        for request, action in zip(requests, chosen):
            lane = self._lanes[request.lane]
            assert lane is not None  # a request is only built for a live lane
            if action is None:
                action = self._bot_action(request)
            lane.by_seat[request.seat].append(
                Decision(
                    seat=request.seat,
                    step=lane.actions,
                    policy=request.policy,
                    action=action,
                )
            )
            # `trades_made` counts this turn's exchanges and `end_turn` clears
            # it, so a rise is what this one action's trade event cleared and a
            # fall is the turn ending. `game.trades` is turn-scoped the same
            # way, so the slice since `before` is exactly this action's own
            # cleared exchanges -- the same diff `hexset.record.record_game`
            # and `hexset.arena._play_and_record` take around their own
            # `apply`.
            cleared = lane.game.trades_made
            before = len(lane.game.trades)
            apply(lane.game, action)
            lane.trades += max(0, lane.game.trades_made - cleared)
            for trade in lane.game.trades[before:]:
                lane.trade_log.append(
                    (lane.actions, trade.a, trade.b, tuple(trade.received))
                )
            if lane.tape is not None:
                lane.tape.step(action, lane.game.trades[before:])
            lane.actions += 1
            if is_over(lane.game) or lane.actions >= self.action_cap:
                finished.append(self._harvest(lane, request.lane))

        self.ticks += 1
        self.steps += len(requests)
        return finished

    def _bot_action(self, request: Request) -> Action:
        bench = self.bots.get(request.policy)
        if bench is None:
            raise ValueError(
                f"lane {request.lane} seat {request.seat} is policy "
                f"{request.policy}, which has no bot to answer for it"
            )
        # public field: a bot reads the board off the true state, as every
        # `hexset.arena` caller does.
        return bench.bot(request.game.state(0, hidden=False).board).choose(
            request.game
        )

    def _harvest(self, lane: _Lane, slot: int) -> Episode:
        game = lane.game
        # true state: terminal victory points include hidden dev cards.
        state = game.state(0, hidden=False)
        # The seed the record names is the stream the game's own draws came
        # from (`hexset.arena.game_key`), not the run seed: a record names
        # the stream it replays, and `hexset.record.replay` checks the
        # recorded chance against it. It stays right when the environment
        # pinned a board, because a pinned board changes the board and not
        # the draws.
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
        """Play every dealt game to completion. Requires a `deal` bound.

        `answer` is the caller's batched policy, one action per request in
        order; left out, every seat is played by its own bot, which is what a
        bots-only evaluation wants and what a test drives.
        """
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
        """Re-arm a bounded environment for `games` more games from where the
        counter stands, refilling idle lanes. A training iteration wants a
        fresh bounded cohort per call without rebuilding the environment
        (and losing its counters); an environment built unbounded stays
        unbounded and refuses this."""
        if games < 1:
            raise ValueError("a cohort needs at least one game")
        if self._stop is None:
            raise ValueError("an unbounded environment has no cohorts; build it with `deal`")
        self._stop = self._next + games * self.stride
        self._outstanding = None
        self._lanes = [lane if lane is not None else self._fresh() for lane in self._lanes]

    def games_started(self) -> int:
        """How many games have been dealt out, finished or not.

        Pass it back as `first_game` to carry on where a run left off. The
        games still in flight are lost on a resume -- they hold engine state,
        not data -- which costs at most `lanes` partial games once per stop.
        """
        return self._next


__all__ = [
    "BoardBots",
    "Decision",
    "Episode",
    "LaneEnv",
    "Outcome",
    "Request",
]
