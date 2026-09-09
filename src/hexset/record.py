# SPDX-License-Identifier: GPL-3.0-only
"""Record played games as data that outlives the code that produced them.

A record stores the board in full and the actions taken, not encoded features.
Feature layouts change every time the encoder is touched, and a dataset frozen
in one of them is worth nothing afterwards; a replayable action sequence can be
re-encoded however the model wants, as many times as it wants.

The board is written out rather than stored as the seed that generated it, for
the same reason: a seed only reproduces a board for as long as the board
generator is untouched. Hex coordinates go in too, so a Seafarers layout records
exactly like the base board.

Version 2 (registered `agents/reference/game-records.md`, 2026-09-04): a
record carries its own chance -- the shuffled deck, every roll, every steal,
every random discard -- as an explicit event stream (`chance`), rather than
depending on `seed` to reproduce them from the engine's random draws. That
made a record unreadable across an engine change to how chance is resolved,
and unbuildable at all for a game this engine never played (a colonist.io
game, a Catanatron game, this project's own server journal). `seed` is now
optional: present, `replay` uses it as an extra check that the recorded
chance stream is what that seed would actually have produced
(`ReplayError` on divergence); absent, `replay` drives the game purely from
`chance`. Version 1 lines (no `chance`, `seed` required) are refused by
`from_json` -- the only version-1 file this project has, the trade-lab bank,
is re-emitted as version 2 by re-running `record_game`.

`actors` joins `trades` and `first` as a field added inside version 2 rather
than by bumping it: it defaults to empty, an older file without it reads and
replays exactly as before, and bumping would refuse every record already
written. It names who took a step where the position cannot say -- a seven's
discards, which the engine serves simultaneously -- so a real table's discard
order round-trips instead of being flattened into ascending seat order. A
reader older than the field will drop it and replay such a record in the
engine's own order, which reaches the same position but is not the same
history.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Sequence

from .actions import Action, ActionType, apply, legal_actions
from .arena import MAX_ACTIONS
from .board.board import Board, make_board
from .board.coords import Hex
from .board.ports import Port
from .board.terrain import Resource, Terrain
from .board.topology import build as build_topology
from .bots import Bot
from .cards import DevCard, make_deck
from .chance import Chance, ChanceError, Live, Recording, Scripted
from .game import Game, is_over, may_act, start, to_move
from .trading import Trade, apply_trades

VERSION = 2


class ReplayError(RuntimeError):
    """Raised when a record no longer describes the game it claims to."""


@dataclass(frozen=True)
class Record:
    layout: tuple[tuple[int, int, int], ...]
    terrain: tuple[int, ...]
    tokens: tuple[int, ...]
    ports: tuple[tuple[int, int, int | None], ...]
    num_players: int
    actions: tuple[tuple[int, int, int], ...]
    # The chance stream, in the order the engine drew it: `("deck", card)`
    # once per card (the shuffled development deck, bottom of the deck
    # first -- `devcards.buy` pops the end), `("roll", n)` per dice roll,
    # `("steal", resource)` per robber/knight steal that took a card,
    # `("discard", resource)` per card a seat that does not choose its own
    # discards gave up. This is what `replay` drives the game from
    # (`chance.Scripted`) -- the porting surface: a converter (`from_journal`
    # below; colonist.io's own converter lives in dev-HexN's private
    # `colonists/`) builds one of these from a game this engine never
    # played, with no seed at all.
    chance: tuple[tuple[str, int], ...]
    winner: int | None
    turns: int
    # Optional: the seed the game was actually dealt with, when there was
    # one. `replay` uses it only as an extra check -- that `chance` is what
    # this seed's stream would have produced -- never as a dependency: a
    # record with `seed=None` replays exactly as well. `str` as well as
    # `int`, because `random.Random` accepts either and `hexset.arena`'s own
    # per-game seed is a composite string (`f"{seed}:{board_index}:game"),
    # not a bare int -- `arena.compete`'s own `--records` path stores that
    # string here rather than giving up the seed check for every game it
    # records.
    seed: int | str | None = None
    # The setup snake's start seat (`hexset.game.start`'s own `first`
    # argument, and its `Game.first` field). Defaults to 0, matching every
    # caller that never chooses it (`record_game`, `arena`'s and
    # `trade_census`'s own recording paths never pass `first` either), so an
    # existing record is unaffected. `replay` passes it back to `start` --
    # without it, a record whose game opened the snake anywhere but seat 0
    # (a server-journalled game with a rotated deal, in particular) would
    # replay a different setup order than the one actually played, and every
    # action from the second setup placement on would fail the "is this
    # legal" check for the wrong reason.
    first: int = 0
    # Trades, sparse by step: `(step, a, b, received)` for every exchange the
    # engine cleared inside the action at `step`, `received` signed towards
    # `a`. Trading is not an action (`hexset.trading`), so it cannot ride in
    # the action triple, and it is not a function of the actions either --
    # it depends on what the seated bots published and accepted. Recording
    # it is therefore what makes a record replayable at all.
    trades: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    # Who took the action, sparse by step: `(step, seat)`. Only recorded for
    # the steps where the position does not already say -- in practice the
    # `Phase.DISCARD` ones. A seven's discards are simultaneous: every seat
    # over the limit owes cards at the same instant and any of them may pay
    # first (`hexset.game.may_act`), so "whose action was this" is a fact
    # about the game that happened, not something `to_move` can recover.
    # Without it a record could only ever replay a discard round in ascending
    # seat order, and a real table's order -- a server journal's, a
    # colonist.io log's -- could not be round-tripped at all.
    #
    # Absent (`()`) for every record written before this field existed and
    # for every game whose discards did happen in the engine's own order;
    # `replay` then falls back to `to_move` exactly as it always did, so no
    # stored record changes meaning.
    actors: tuple[tuple[int, int], ...] = ()

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


def recording(rng: random.Random) -> Recording:
    """The chance source a game has to be dealt with to be recordable.

    A function rather than something each loop inlines, because it is the
    one thing every recording loop must do identically: a `Record`'s
    `chance` is whatever this wrapper logged, so a loop that wrapped
    something else would file a stream nobody played. Shaped to hand
    straight to `hexset.arena.deal_game`'s `chance` seam, which passes in
    the generator the game would otherwise have drawn from -- so a recorded
    game is the same game, not a similar one.
    """
    return Recording(Live(rng))


@dataclass
class Tape:
    """A record under construction, collected as the game is played.

    Three loops record a game and they agree on nothing but this: one plays
    a whole game here in a `while`, one plays a whole game with a cleared-
    trade census riding along (`hexset.arena._play_and_record`), and one
    advances a single action per tick alongside every other lane's
    (`hexset.gym.lanes`). A second builder is precisely how a recorded game
    comes to differ from the game that was played, so the bookkeeping lives
    here once: a loop calls `step` where it applies an action and `sealed`
    where its game ends.
    """

    actions: list[tuple[int, int, int]] = field(default_factory=list)
    trades: list[tuple[int, int, int, tuple[int, ...]]] = field(default_factory=list)

    def step(self, action: Action, cleared: Sequence[Trade]) -> None:
        """File one applied action and the exchanges that cleared inside it.

        `cleared` is the `game.trades` slice taken across that action's own
        `apply`. The turn's first trade event fires eagerly, inside the ROLL
        or ROBBER action (`hexset.game.enter_main`), so its exchanges belong
        to that action's step -- which is exactly where `advance` replays
        them.
        """
        step = len(self.actions)
        for trade in cleared:
            self.trades.append((step, trade.a, trade.b, tuple(trade.received)))
        self.actions.append((int(action.type), action.a, action.b))

    def sealed(self, game: Game, *, seed: int | str | None = None) -> Record:
        """The finished `Record` of `game`, as this tape watched it played.

        The board, the setup snake's first seat, the winner and the turn
        count are read off the game rather than passed in, so a caller
        cannot hand back a record of a game other than the one it stepped.
        `game` must have been dealt with `recording` above: the chance
        stream is what that wrapper logged and there is nowhere else to get
        it from, so a game dealt without it is refused rather than sealed
        into a record that cannot replay.
        """
        chance = game.chance
        if not isinstance(chance, Recording):
            raise TypeError(
                "a recorded game must be dealt with hexset.record.recording as "
                f"its chance source, not {type(chance).__name__}"
            )
        # true state: the board is public either way, and `state(0)` is the
        # accessor every caller in this package already reads it through.
        board = game.state(0, hidden=False).board
        return Record(
            num_players=game.num_players,
            seed=seed,
            first=game.first,
            actions=tuple(self.actions),
            chance=tuple(chance.events),
            trades=tuple(self.trades),
            winner=game.won_by,
            turns=game.turns,
            **board_fields(board),
        )


def record_game(
    bots: Sequence[Bot],
    board: Board,
    seed: int,
    *,
    action_cap: int = MAX_ACTIONS,
) -> Record:
    """Play one game and record it. Each bot is seated at its own index.

    The bots are seated as the game's `gates` too, and a gate is a pure
    function of the position, asked fresh at every trade event -- there is
    nothing this loop has to publish, so it trades the way `arena.play`
    does: seat the gates and step the game.
    """
    rng = random.Random(seed)
    game = start(board, len(bots), rng, chance=recording(rng))
    game.gates = tuple(bots)
    tape = Tape()
    while not is_over(game) and len(tape.actions) < action_cap:
        seat = to_move(game)
        bot = bots[seat]
        before = len(game.trades)
        action = bot.choose(game)
        apply(game, action)
        tape.step(action, game.trades[before:])
    return tape.sealed(game, seed=seed)


def actions_of(record: Record) -> Iterator[Action]:
    """The recorded actions, in order."""
    for kind, a, b in record.actions:
        yield Action(ActionType(kind), a, b)


def moves(record: Record) -> Iterator[tuple[int | None, Action, tuple[Trade, ...]]]:
    """Each step as `(actor, action, trades)`: who took it, what they took,
    and the trades the engine cleared inside it.

    Everything that walks a record goes through here, so the reconstruction
    lives in one place and cannot drift between replaying, featurising and
    behaviour analysis. `actor` is `None` for every step the record does not
    name one for (see `Record.actors`), which means "whoever the position
    says" -- `hexset.game.to_move`.

    Shaped `(actor, action, trades)` to match `hexset.server.journal.steps`,
    the other replayable stream in this package.
    """
    by_step: dict[int, list[Trade]] = {}
    for step, a, b, received in record.trades:
        by_step.setdefault(step, []).append(Trade(a, b, tuple(received)))
    actors = dict(record.actors)
    for step, action in enumerate(actions_of(record)):
        yield actors.get(step), action, tuple(by_step.get(step, ()))


def steps(record: Record) -> Iterator[tuple[Action, tuple[Trade, ...]]]:
    """`moves` without the actor, for a caller that never needed it.

    Kept because it is what every consumer of a record outside this package
    already unpacks. A caller that replays the record with `advance` wants
    `moves` instead: a record whose discard round did not happen in ascending
    seat order (`Record.actors`) cannot be re-applied without knowing whose
    each discard was.
    """
    for _, action, trades in moves(record):
        yield action, trades


def advance(
    game: Game, action: Action, trades: Sequence[Trade], seat: int | None = None
) -> None:
    """Apply one recorded step: the action, then the trades it cleared.

    A replayed game has no seated bots (`game.gates` stays `None`), so its
    own trade event never fires -- `run_trade_event`'s own no-op guard --
    and the recorded exchanges are re-executed here instead. Applying them
    immediately after `apply` returns is exactly where they happened: the
    turn's first event fires eagerly, inside the roll or robber action's own
    `apply` (`enter_main`), so `record_game` already attributes it to that
    same step.

    `seat` is the recorded actor (`moves`), passed on to
    `hexset.actions.apply`, which only ever reads it for a `DISCARD`. `None`
    -- the historical call -- resolves as it always did, to the lowest-indexed
    seat still owing.
    """
    apply(game, action, seat)
    apply_trades(game, trades)


class _SeedChecked(Chance):
    """Drives replay from `record.chance` (`Scripted`) while cross-checking
    every outcome against a `Live` source seeded the same way the game
    originally was.

    This is the tripwire `Record`'s pre-v2 docstring described -- "a change
    to how the engine draws randomness shows up as a replay mismatch" --
    kept as an explicit check now that the recorded stream, not the seed,
    is what actually drives the replay. Only built when `record.seed` is
    not `None`; a seedless record replays from `Scripted` alone.
    """

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

    def steal(self, hand):
        got = self._scripted.steal(hand)
        want = self._seeded.steal(hand)
        self._check("steal", got, want)
        return got

    def discard(self, hand, n: int) -> list[int]:
        got = self._scripted.discard(hand, n)
        want = self._seeded.discard(hand, n)
        self._check("discard", got, want)
        return got


def open_record(record: Record) -> Game:
    """Open a record at its initial position, preserving chance and first seat.

    Incremental consumers must use this instead of reseeding a fresh game.
    Use `steps`/`advance` to walk it, or `replay` for full validation.
    """
    scripted = Scripted(record.chance)
    chance = scripted if record.seed is None else _SeedChecked(scripted, Live(random.Random(record.seed)))
    return start(board_of(record), record.num_players, random.Random(record.seed),
                 first=record.first, chance=chance)


def replay(record: Record) -> Game:
    """Re-play a record, checking it still describes the game it claims to.

    Drives the game from `record.chance` (`chance.Scripted`), not from the
    seed: a seedless record (`record.seed is None` -- the porting surface,
    `from_journal` below) replays exactly the same way. When `record.seed`
    *is* present, every scripted outcome is additionally checked against
    what that seed's stream would have produced (`_SeedChecked`), and a
    divergence raises `ReplayError` naming the event -- the tripwire the
    pre-v2 `Record` relied on implicitly, kept as an explicit check instead
    of a silent dependency.

    Legality is checked against the seat that actually took the action, not
    against `to_move`. The two differ only in `Phase.DISCARD`, where several
    seats owe cards at once and any of them may pay first: a record that names
    its actors (`Record.actors` -- a server journal converted by
    `from_journal`, a colonist.io log) therefore round-trips its true discard
    order instead of being refused for not being in ascending seat order. A
    record that names none replays exactly as before, since `may_act` is
    always true of `to_move` and `legal_actions(game, to_move(game))` is
    `legal_actions(game)`.
    """
    game = open_record(record)
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
            advance(game, action, trades, seat)
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

    `ply` counts actions applied, so `replay_to(record, 0)` is the opening
    position (`open_record`) and `replay_to(record, len(record.actions))` is
    the final one -- the same game `replay` ends on, without its legality
    and outcome checks. A driver that wants a stored position back -- to
    re-search it, to re-encode it under a new feature layout -- wants this
    and nothing else: a record is the one replay path, and re-deriving a
    position from a seed and an action stream is a second one that can
    disagree with it.

    Trades are applied as recorded (`advance`), because a replayed game
    seats no gates and its own trade event therefore never fires; a walk
    that only re-applied the actions would play a trade-free game however
    the table actually traded.

    A `ply` outside the record is refused rather than clamped: asking for a
    position a record does not hold is a bug in the caller's indexing, and
    silently handing back the last one it does hold would answer it with a
    wrong position that looks like a right one.
    """
    if ply < 0:
        raise ValueError("a ply is a non-negative number of actions applied")
    if ply > len(record.actions):
        raise ValueError(
            f"record holds {len(record.actions)} actions, ply {ply} asked for more"
        )
    game = open_record(record)
    for taken, (actor, action, trades) in enumerate(moves(record)):
        if taken >= ply:
            break
        advance(game, action, trades, actor)
    return game


def to_json(record: Record) -> str:
    data = asdict(record)
    data["version"] = VERSION
    return json.dumps(data, separators=(",", ":"))


def from_json(line: str) -> Record:
    raw = json.loads(line)
    version = raw.get("version")
    if version != VERSION:
        raise ValueError(
            f"record is version {version!r}, not {VERSION}: version 1 records "
            "(no chance stream, a required seed) are refused -- see "
            "agents/reference/game-records.md. Re-emit through "
            "record_game/write."
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
        actors=tuple((step, seat) for step, seat in raw.get("actors", ())),
        winner=raw["winner"],
        turns=raw["turns"],
        seed=raw.get("seed"),
        first=raw.get("first", 0),
    )


def from_journal(path) -> Record:
    """Convert a server journal (`hexset.server.journal`) into a `Record` --
    the porting surface `chance` was built for, proved on the one external
    format this project owns.

    The journal already spells out everything a `Record` needs to replay
    without the engine's seed at all: the shuffled deck is the header's
    `deck` (bottom of the deck first, matching `chance.Recording`'s own
    convention), and every roll and every steal that took a card is on its
    own action line (`Journal.action`'s `effects`) -- nothing here re-runs
    the engine to recover what happened, unlike a v1 record's replay. A
    discard is never a chance event on this path: the journal's own
    `Phase.DISCARD` actions are always a seat's explicit, one-card-at-a-time
    choice (`hexset.game.submit_discard`/`discard_one`), never
    `chance.discard`. It is also the one action whose actor the journal has
    to be believed about rather than recomputed: the server serves a seven's
    discards simultaneously (`hexset.game.may_act`), so a journalled round can
    have any interleaving of the owing seats and its own `actor` field is the
    only record of which. Every `Phase.DISCARD` step's actor is carried across
    into `Record.actors` for that reason; every other action belongs to
    `to_move` by construction and is left out, keeping the field empty for the
    overwhelming majority of games. The header's own `first` (the setup snake's start
    seat, needed on resume for the same reason -- `Journal.start`'s
    docstring) is read here too, not assumed 0: a journal dealt with a
    rotated snake would otherwise replay a different setup order than the
    one actually played.

    A trade the table executed outside the automatic event -- a round's
    accepted offer or counter (`Journal.manual_trade`, its own journal step
    with no action) -- is folded into the preceding action's `trades`: a
    `Record` applies a step's trades right after its action, and nothing
    else happened between the two live either. Journal step numbers count
    those trade steps, so an `undo`'s `back_to` is mapped through them
    rather than indexing the action list directly. Round *notes*
    (`Journal.note`) move nothing and are skipped.

    `path` must reach a game with a `result` line (`Journal.finish`) -- an
    abandoned, resultless journal has no `winner`/`turns` to record and is
    refused. Import of `hexset.server.journal` is local to keep that
    (server-only) module off this one's import graph for callers who never
    convert one.
    """
    from .server import journal as game_journal

    events = game_journal.read(path)
    if not events or events[0].get("kind") != "game":
        raise ValueError(f"not a journal (no header): {path}")
    header = events[0]

    steps: list[list] = []  # [action, trades, chance_events, actor], mutable for a folded trade
    # Journal step number -> index into `steps` of the action at or before it,
    # so an undo's `back_to` (a journal step) can be applied to the action list.
    at_step: dict[int, int] = {}
    result: dict | None = None
    for event in events[1:]:
        kind = event.get("kind")
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
                raise ValueError(f"journal executes a trade before any action: {path}")
            a, b, received = event["trade"]
            steps[-1][1] = tuple(steps[-1][1]) + (Trade(a, b, tuple(received)),)
            at_step[int(event["step"])] = len(steps)  # an undo back to here keeps the action before
        elif kind == "undo":
            back_to = int(event["back_to"])
            cut = at_step.get(back_to, len(steps))
            del steps[cut:]
            at_step = {step: index for step, index in at_step.items() if index < cut}
        elif kind == "result":
            result = event

    if result is None:
        raise ValueError(f"journal has no result line, cannot record: {path}")

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
        winner=result["winner"],
        turns=result["turns"],
        seed=header.get("seed"),
        first=header.get("first", 0),
    )


def write(path: str, records: Iterable[Record]) -> int:
    """Append records as JSON lines. Returns how many were written."""
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(to_json(record) + "\n")
            written += 1
    return written


def read(path: str) -> Iterator[Record]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield from_json(line)
