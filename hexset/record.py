# SPDX-License-Identifier: GPL-3.0-only
"""Record played games as data that outlives the code that produced them: the
board and the actions taken, not encoded features, plus the game's own chance
stream, so a record re-encodes freely and replays without a seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Sequence

from .actions import Action, ActionType, apply, legal_actions
from .arena import MAX_ACTIONS, run_seated as _run, seat_bots as _seat
from .board.board import Board, make_board
from .board.coords import Hex
from .board.ports import Port
from .board.terrain import Resource, Terrain
from .board.topology import build as build_topology
from .bots import Bot
from .cards import DevCard, make_deck
from .chance import Balanced, Chance, ChanceError, Live, Recording, Scripted, for_rules
from .game import MAX_TURNS, NO_TURN_CAP, Game, lock_seat, may_act, start, to_move
from .rules import STANDARD, STANDARD_GAME, GameType, Rules, rules_from
from .trading import Trade, apply_trades, show

__all__ = [
    "VERSION",
    "ReplayError",
    "Record",
    "board_fields",
    "board_of",
    "recording",
    "Tape",
    "record_game",
    "actions_of",
    "moves",
    "shown_by_step",
    "advance",
    "open_record",
    "replay",
    "replay_to",
    "to_json",
    "from_json",
    "from_journal",
    "from_events",
    "write",
    "read",
]


VERSION = 2


class ReplayError(RuntimeError):
    """Raised when a record no longer describes the game it claims to."""


@dataclass(frozen=True)
class Record:
    """A finished game as data: the board, the seat count, every action and
    every chance outcome in order, the winner and the turns. `replay` plays
    it back exactly; `to_json`/`from_json` move it on disk."""

    layout: tuple[tuple[int, int, int], ...]
    terrain: tuple[int, ...]
    tokens: tuple[int, ...]
    ports: tuple[tuple[int, int, int | None], ...]
    num_players: int
    actions: tuple[tuple[int, int, int], ...]
    # The chance stream in draw order: ("deck", card) per card of the shuffled
    # deck (bottom first -- `devcards.buy` pops the end), ("roll", n), ("steal",
    # resource) per steal that took a card.
    chance: tuple[tuple[str, int], ...]
    winner: int | None
    turns: int
    # The seed the game was dealt with, if any. `replay` only cross-checks
    # `chance` against it; `seed=None` replays just as well.
    seed: int | str | None = None
    # The setup snake's start seat, passed back to `hexset.game.start`. Wrong
    # here, every action from the second setup placement on fails as illegal.
    first: int = 0
    # Trades, sparse by step: `(step, a, b, received)` per exchange cleared
    # inside the action at `step`, `received` signed towards `a`. Not derivable
    # from the actions, so recording it is what makes a record replayable.
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    # Offers and answers put to the table, sparse by step: `(step, order,
    # seat, received)` per one shown inside the action at `step`, after
    # `order` of that step's exchanges, `received` signed towards `seat`.
    # They move no card but go on the public ledger (`trading.show`), so a
    # replay without them is a different position. Absent from a record
    # written before this field existed.
    shown: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    # The ruleset the game was played under. Absent from a record written
    # before this field existed, which is exactly the standard ruleset, so the
    # default reads those correctly rather than guessing at them.
    rules: Rules = STANDARD
    # Who took the action, sparse by step: `(step, seat)`. Only for `DISCARD`,
    # whose seats are served simultaneously (`hexset.game.may_act`). Empty means
    # "ask `to_move`", i.e. ascending seat order.
    actors: tuple[tuple[int, int], ...] = ()
    # Whether `chance` was drawn from the dice deck (`chance.Balanced`) rather
    # than independent dice. A fact about the stream, not the rules: a record
    # may name `balanced_dice` in `rules` and still hold independent rolls.
    # Absent means independent. Read only by the seed cross-check, which must
    # regenerate the same draws.
    balanced_dice: bool = False
    # Whether the dice and the steals were drawn from generators of their own
    # (`chance.Live`'s `split`), as every game dealt since 1.10.0 is, rather
    # than one stream shared with the deck. Absent means shared. Read only by
    # the seed cross-check, which must regenerate the same draws.
    split_streams: bool = False
    # The seats retired at the deal (`hexset.game.start`'s `locked`): dealt
    # in, but skipped by the setup snake and turn rotation from the first
    # action. A record holds no seat retired later. Absent means none.
    locked: tuple[int, ...] = ()

    @property
    def decided(self) -> bool:
        return self.winner is not None


def board_fields(board: Board) -> dict[str, tuple]:
    """The parts of a board a record needs. Port vertices are left out — they
    follow from the edge once the topology is rebuilt."""
    return {
        "layout": tuple(tuple(h) for h in board.topology.hexes),
        "terrain": tuple(int(t) for t in board.terrain),
        "tokens": tuple(board.tokens),
        "ports": tuple(
            (p.edge, p.ratio, None if p.resource is None else int(p.resource))
            for p in board.ports
        ),
    }


def board_of(record: Record) -> Board:
    """The `Board` `record` was played on, rebuilt from its `board_fields`."""
    topology = build_topology(Hex(*h) for h in record.layout)
    ports = tuple(
        Port(
            edge=edge,
            vertices=(topology.edges[edge][0], topology.edges[edge][1]),
            resource=None if resource is None else Resource(resource),
            ratio=ratio,
        )
        for edge, ratio, resource in record.ports
    )
    return make_board(
        topology,
        tuple(Terrain(t) for t in record.terrain),
        tuple(record.tokens),
        ports,
    )


def recording(rng: random.Random, rules: Rules) -> Recording:
    """The chance source a game must be dealt with to be recordable: a
    `Record`'s `chance` is whatever this wrapper logged.

    `rules` is required, and is the ruleset the game is dealt under: the
    wrapper logs the source those rules ask for (`chance.for_rules`), so a
    recorded game rolls the same dice as the unrecorded one."""
    return Recording(for_rules(rules, rng))


@dataclass
class Tape:
    """A record under construction. A loop calls `step` where it applies an
    action and `sealed` where its game ends."""

    actions: list[tuple[int, int, int]] = field(default_factory=list)
    trades: list[tuple[int, int, int, tuple[int, ...]]] = field(default_factory=list)
    shown: list[tuple[int, int, int, tuple[int, ...]]] = field(default_factory=list)

    def step(
        self, action: Action, cleared: Sequence[Trade],
        shown: Sequence[tuple[int, int, tuple[int, ...]]] = (), *, before: int = 0,
    ) -> None:
        """File one applied action, the exchanges that cleared inside it and
        the offers and answers shown around them.

        `cleared` is the `game.trades` slice across that action's own `apply`,
        and `before` its length going in; `shown` the `game.shown` slice across
        the same `apply`. The turn's trade event runs inside the action that
        enters the main phase (a `ROLL`, or the `MOVE_ROBBER` after a seven),
        or, for a gate with `trade_now`, inside a later main-phase action."""
        step = len(self.actions)
        for trade in cleared:
            self.trades.append((step, trade.a, trade.b, tuple(trade.received)))
        for traded, seat, received in shown:
            self.shown.append((step, int(traded) - before, int(seat), tuple(received)))
        self.actions.append((int(action.type), action.a, action.b))

    def sealed(self, game: Game, *, seed: int | str | None = None) -> Record:
        """The finished `Record` of `game`, as this tape watched it played.
        `game` must have been dealt with `recording`; any other chance source is
        refused. Its retired seats are recorded as retired at the deal, which
        is the only way a recorded loop retires one."""
        chance = game.chance
        if not isinstance(chance, Recording):
            raise TypeError(
                "a recorded game must be dealt with hexset.record.recording as "
                f"its chance source, not {type(chance).__name__}"
            )
        # true state: the board is public either way, and `state(0)` is the
        # accessor every caller in this package already reads it through.
        state = game.state(0, hidden=False)
        board = state.board
        return Record(
            num_players=game.num_players,
            rules=state.rules,
            seed=seed,
            first=game.first,
            actions=tuple(self.actions),
            chance=tuple(chance.events),
            balanced_dice=isinstance(chance.inner, Balanced),
            split_streams=getattr(chance.inner, "split", False),
            trades=tuple(self.trades),
            shown=tuple(self.shown),
            winner=game.won_by,
            turns=game.turns,
            locked=tuple(sorted(game.locked)),
            **board_fields(board),
        )


def record_game(
    bots: Sequence[Bot],
    board: Board,
    seed: int,
    *,
    action_cap: int = MAX_ACTIONS,
    trade_mode: str = "round",
    turn_cap: int = MAX_TURNS,
    game_type: GameType = STANDARD_GAME,
) -> Record:
    """Play one game and record it. Each bot is seated at its own index."""
    rng = random.Random(seed)
    game = start(board, len(bots), rng, chance=recording(rng, game_type.rules), turn_cap=turn_cap,
                 game_type=game_type)
    _seat(game, bots, trade_mode)
    tape = Tape()
    _run(game, bots, action_cap, tape=tape)
    return tape.sealed(game, seed=seed)


def actions_of(record: Record) -> Iterator[Action]:
    """The recorded actions, in order."""
    for kind, a, b in record.actions:
        yield Action(ActionType(kind), a, b)


def moves(record: Record) -> Iterator[tuple[int | None, Action, tuple[Trade, ...]]]:
    """Each step as `(actor, action, trades)`. `actor` is `None` where the
    record names none (`Record.actors`): whoever `hexset.game.to_move` says."""
    by_step: dict[int, list[Trade]] = {}
    for step, a, b, received in record.trades:
        by_step.setdefault(step, []).append(Trade(a, b, tuple(received)))
    actors = dict(record.actors)
    for step, action in enumerate(actions_of(record)):
        yield actors.get(step), action, tuple(by_step.get(step, ()))


def shown_by_step(record: Record) -> dict[int, tuple[tuple[int, int, tuple[int, ...]], ...]]:
    """`Record.shown` by step, each as `(order, seat, received)`: what
    `advance` takes as its `shown`."""
    by_step: dict[int, list] = {}
    for step, order, seat, received in record.shown:
        by_step.setdefault(step, []).append((order, seat, tuple(received)))
    return {step: tuple(rows) for step, rows in by_step.items()}


def advance(
    game: Game, action: Action, trades: Sequence[Trade], seat: int | None = None,
    shown: Sequence[tuple[int, int, tuple[int, ...]]] = (),
) -> None:
    """Apply one recorded step: the action, then the trades it cleared, with
    the offers and answers shown among them (`shown_by_step`) back on the
    public ledger where they went, so the position is the one played.

    A replayed game seats no bots, so its own trade event never fires and the
    recorded exchanges are re-executed here. `seat` is the recorded actor, read
    only for a `DISCARD`; `None` means the lowest-indexed owing seat."""
    apply(game, action, seat)
    if not shown:
        apply_trades(game, trades)
        return
    for traded, trade in enumerate(trades):
        for order, who, received in shown:
            if order == traded:
                show(game, who, received)
        apply_trades(game, (trade,))
    for order, who, received in shown:
        if order >= len(trades):
            show(game, who, received)


class _SeedChecked(Chance):
    """Replay from `record.chance` (`Scripted`), cross-checked against the
    source the game was dealt with (`Live`, or `Balanced` when the record says
    its stream came from the dice deck), seeded the way the game was. Only
    built when `record.seed` is set."""

    def __init__(self, scripted: Scripted, seeded: Live) -> None:
        self._scripted = scripted
        self._seeded = seeded

    def _check(self, kind: str, got, want) -> None:
        if got != want:
            raise ReplayError(
                f"chance diverges at event {self._scripted.index - 1} ({kind}): "
                f"recorded {got!r}, the seed would draw {want!r}"
            )

    def deck_order(self, deck: list[int]) -> list[int]:
        got = self._scripted.deck_order(deck)
        want = self._seeded.deck_order(make_deck(None))
        self._check("deck", got, want)
        return got

    def roll(self) -> int:
        got = self._scripted.roll()
        want = self._seeded.roll()
        self._check("roll", got, want)
        return got

    def roll_by(self, seat: int | None) -> int:
        got = self._scripted.roll()
        want = self._seeded.roll_by(seat)
        self._check("roll", got, want)
        return got

    def steal(self, hand):
        got = self._scripted.steal(hand)
        want = self._seeded.steal(hand)
        self._check("steal", got, want)
        return got


def open_record(record: Record) -> Game:
    """Open a record at its initial position, preserving chance, first seat
    and retired seats. Walk it with `moves`/`advance`, or use `replay` for
    full validation."""
    scripted = Scripted(record.chance)
    if record.seed is None:
        chance = scripted
    else:
        seeded = random.Random(record.seed)
        split = record.split_streams
        chance = _SeedChecked(
            scripted,
            Balanced(seeded, split=split) if record.balanced_dice else Live(seeded, split=split),
        )
    # No turn cap on a replay. The cap is a live-play policy -- how long a
    # game is allowed to run before it is called stuck -- and a record is
    # already authoritative about where its game ended. Applying a cap here
    # would re-truncate a game that was recorded under a looser one, and the
    # replay would reject the record's own remaining actions as illegal in
    # GAME_OVER, which is a false accusation against the record.
    # No contract check on a replay, for the same reason there is no turn cap:
    # the record is authoritative about the game it holds. Checking `seats`
    # here would reject a record of a game that was actually played, which is a
    # false accusation against the record rather than a finding about it. So
    # the retired seats are retired after the deal, before the first action,
    # as a served table closes them: a table one seat played still opens.
    game = start(board_of(record), record.num_players, random.Random(record.seed),
                 first=record.first, chance=chance, turn_cap=NO_TURN_CAP,
                 game_type=GameType("recorded", record.rules, (record.num_players,)))
    for seat in record.locked:
        lock_seat(game, seat)
    return game


def replay(record: Record) -> Game:
    """Re-play a record, checking it still describes the game it claims to.

    Driven by `record.chance`, not by the seed; with a seed present every
    outcome is also cross-checked against that seed's stream. Legality is
    checked against the recorded actor (`Record.actors`), not `to_move`."""
    game = open_record(record)
    shown = shown_by_step(record)
    for step, (actor, action, trades) in enumerate(moves(record)):
        seat = to_move(game) if actor is None else actor
        if not may_act(game, seat):
            raise ReplayError(
                f"step {step}: seat {seat} may not act in {game.phase.name}"
            )
        if action not in legal_actions(game, seat):
            raise ReplayError(
                f"step {step}: {action} is not legal for seat {seat} in {game.phase.name}"
            )
        try:
            advance(game, action, trades, seat, shown.get(step, ()))
        except ChanceError as error:
            raise ReplayError(f"step {step}: {error}") from error

    scripted = game.chance._scripted if isinstance(game.chance, _SeedChecked) else game.chance
    if scripted.index != len(record.chance):
        raise ReplayError(f"unconsumed chance events: {len(record.chance) - scripted.index}")
    if (game.won_by, game.turns) != (record.winner, record.turns):
        raise ReplayError(
            f"replay ended {game.won_by} after {game.turns} turns, "
            f"record says {record.winner} after {record.turns}"
        )
    return game


def replay_to(record: Record, ply: int) -> Game:
    """The live `Game` after `ply` of `record`'s recorded actions.

    `ply` counts actions applied: 0 is the opening position, `len(actions)` the
    final one, without `replay`'s checks. A `ply` outside it is refused."""
    if ply < 0:
        raise ValueError("a ply is a non-negative number of actions applied")
    if ply > len(record.actions):
        raise ValueError(
            f"record holds {len(record.actions)} actions, ply {ply} asked for more"
        )
    game = open_record(record)
    shown = shown_by_step(record)
    for taken, (actor, action, trades) in enumerate(moves(record)):
        if taken >= ply:
            break
        advance(game, action, trades, actor, shown.get(taken, ()))
    return game


def to_json(record: Record) -> str:
    """`record` as one compact line of JSON, stamped with `VERSION`."""
    data = asdict(record)
    data["version"] = VERSION
    return json.dumps(data, separators=(",", ":"))


def from_json(line: str) -> Record:
    """The `Record` a `to_json` line holds. Raises `ValueError` for a record of
    any version but `VERSION`."""
    raw = json.loads(line)
    version = raw.get("version")
    if version != VERSION:
        raise ValueError(
            f"record is version {version!r}, not {VERSION}: version 1 records "
            "(no chance stream, a required seed) are refused. Re-emit "
            "through record_game/write."
        )
    return Record(
        layout=tuple(tuple(h) for h in raw["layout"]),
        terrain=tuple(raw["terrain"]),
        tokens=tuple(raw["tokens"]),
        ports=tuple(tuple(p) for p in raw["ports"]),
        num_players=raw["num_players"],
        actions=tuple(tuple(a) for a in raw["actions"]),
        chance=tuple((kind, value) for kind, value in raw["chance"]),
        trades=tuple(
            (step, a, b, tuple(received))
            for step, a, b, received in raw.get("trades", ())
        ),
        shown=tuple(
            (step, order, seat, tuple(received))
            for step, order, seat, received in raw.get("shown", ())
        ),
        actors=tuple((step, seat) for step, seat in raw.get("actors", ())),
        balanced_dice=raw.get("balanced_dice", False),
        split_streams=raw.get("split_streams", False),
        rules=rules_from(raw.get("rules", {})),
        winner=raw["winner"],
        turns=raw["turns"],
        seed=raw.get("seed"),
        first=raw.get("first", 0),
        locked=tuple(raw.get("locked", ())),
    )


def from_journal(path) -> Record:
    """Convert a server journal (`hexset.server._journal`) into a seedless `Record`.

    A journalled `Phase.DISCARD` is the one action whose `actor` must be
    believed rather than recomputed; those alone go into `Record.actors`. A
    manual trade folds into the preceding action's `trades`, so an `undo`'s
    `back_to` is mapped through journal step numbers. Each offer, acceptance
    and counter a trade round noted goes into `Record.shown` where the table
    showed it, signed towards the seat that showed it, as the served table
    puts it on the ledger. A seat retired at the deal (the header's
    `locked`) or closed before the first action goes into `Record.locked`; one that left later is not a seat retired at the deal,
    and is not recorded. A journal with no `result` line is refused."""
    from .server import _journal as game_journal

    return from_events(game_journal.read(path), where=path)


def from_events(events: Sequence[dict], *, where="journal", partial: bool = False) -> Record:
    """`from_journal` over events already read. `partial` takes a journal with
    no `result` line as the game so far: no winner, and `turns` the END_TURNs
    played. That is what opens a journal with no seed to deal from
    (`hexset.server.api`), whose chance stream is its own recorded effects."""
    from .server import _journal as game_journal

    if not events or events[0].get("kind") != "game":
        raise ValueError(f"not a journal (no header): {where}")
    header = events[0]

    steps: list[list] = []  # [action, trades, chance_events, actor], mutable for a folded trade
    # Journal step number -> index into `steps` of the action at or before it,
    # so an undo's `back_to` can be applied to the action list.
    at_step: dict[int, int] = {}
    # (journal step, index into `steps`, order, seat, received) per shown note.
    shown: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    # Headers written before they named the seats retired at the deal name
    # none; those tables closed theirs with `locked` lines instead.
    locked: set[int] = {int(seat) for seat in header.get("locked", ())}
    result: dict | None = None
    for event in events[1:]:
        kind = event.get("kind")
        if kind in ("locked", "unlocked"):
            if int(event.get("at_step", 0)) == 0:
                if kind == "locked":
                    locked.add(int(event["seat"]))
                else:
                    locked.discard(int(event["seat"]))
            continue
        if kind == "note":
            bundle = event.get("bundle")
            if event["note"] not in ("offer", "accept", "counter") or bundle is None:
                continue
            if not steps:
                raise ValueError(f"journal notes a trade round before any action: {where}")
            # As the served table shows it (`webplay.GameSession._note`): an
            # offer is the actor's own, an answer is signed towards the actor.
            sign = 1 if event["note"] == "offer" else -1
            received = tuple(sign * int(n) for n in bundle)
            shown.append((int(event["step"]), len(steps) - 1, len(steps[-1][1]),
                          int(event["seat"]), received))
            continue
        if kind == "action":
            action = game_journal.action_of(event)
            trades = game_journal.trades_of(event)
            actor = event["actor"]
            chance_events: list[tuple[str, int]] = []
            if action.type is ActionType.ROLL:
                chance_events.append(("roll", event["roll"]))
            if action.type is ActionType.MOVE_ROBBER:
                stole = event.get("stole")
                if stole is not None and stole.get("resource") is not None:
                    chance_events.append(("steal", int(Resource[stole["resource"]])))
            at_step[int(event["step"])] = len(steps)
            steps.append([action, trades, tuple(chance_events), actor])
        elif kind == "trade":
            if not steps:
                raise ValueError(f"journal executes a trade before any action: {where}")
            a, b, received = event["trade"]
            steps[-1][1] = tuple(steps[-1][1]) + (Trade(a, b, tuple(received)),)
            at_step[int(event["step"])] = len(steps)  # an undo back to here keeps the action before
        elif kind == "undo":
            back_to = int(event["back_to"])
            cut = at_step.get(back_to, len(steps))
            del steps[cut:]
            at_step = {step: index for step, index in at_step.items() if index < cut}
            shown = [row for row in shown if row[0] < back_to and row[1] < cut]
        elif kind == "result":
            result = event

    if result is None:
        if not partial:
            raise ValueError(f"journal has no result line, cannot record: {where}")
        ended = sum(1 for action, _, _, _ in steps if action.type is ActionType.END_TURN)
        result = {"winner": None, "turns": ended}

    deck = tuple(("deck", int(DevCard[name])) for name in header["deck"])
    chance = deck + tuple(event for _, _, events, _ in steps for event in events)
    actions = tuple((int(action.type), action.a, action.b) for action, _, _, _ in steps)
    trades = tuple(
        (step, trade.a, trade.b, tuple(trade.received))
        for step, (_, step_trades, _, _) in enumerate(steps)
        for trade in step_trades
    )
    actors = tuple(
        (step, actor)
        for step, (action, _, _, actor) in enumerate(steps)
        if action.type is ActionType.DISCARD
    )

    return Record(
        layout=tuple(tuple(h) for h in header["layout"]),
        terrain=tuple(header["terrain"]),
        tokens=tuple(header["tokens"]),
        ports=tuple(tuple(p) for p in header["ports"]),
        num_players=header["num_players"],
        actions=actions,
        chance=chance,
        trades=trades,
        actors=actors,
        shown=tuple((index, order, seat, received) for _, index, order, seat, received in shown),
        winner=result["winner"],
        turns=result["turns"],
        seed=header.get("seed"),
        first=header.get("first", 0),
        # A header written before it carried `rules` is a standard game.
        rules=rules_from(header.get("rules", {})),
        # How the seed drew its dice, for the seed cross-check. Absent: shared.
        split_streams=header.get("split_streams", False),
        locked=tuple(sorted(locked)),
    )


def write(path: str, records: Iterable[Record]) -> int:
    """Append records as JSON lines, each flushed as it goes, so a generator
    fed game by game leaves every finished record on disk. Returns how many
    were written."""
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(to_json(record) + "\n")
            handle.flush()
            written += 1
    return written


def read(path: str) -> Iterator[Record]:
    """Every record in `path`. A torn last line -- the one a dying writer was
    part-way through -- is skipped; a bad line anywhere before it raises."""
    with open(path, encoding="utf-8") as handle:
        previous = None
        for line in handle:
            if previous is not None and previous.strip():
                yield from_json(previous)
            previous = line
        if previous is not None and previous.strip():
            if previous.endswith("\n"):
                yield from_json(previous)
            else:
                try:
                    yield from_json(previous)
                except (json.JSONDecodeError, ValueError):
                    return
