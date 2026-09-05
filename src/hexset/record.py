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
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
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
from .game import Game, is_over, start, to_move
from .trading import Trade, apply_trades, publish_valuation

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
    # below; colonist.io's own converter lives in dev-HexNet's private
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


def record_game(
    bots: Sequence[Bot],
    board: Board,
    seed: int,
    *,
    action_cap: int = MAX_ACTIONS,
) -> Record:
    """Play one game and record it. Each bot is seated at its own index.

    The bots are seated as the game's `gates` too, and each one publishes
    once a turn, exactly as `arena.play` does both (`Game.publish_due`, the
    PI amendment "publish points and the event trigger"), so a recorded
    game trades the way a played one does -- and the exchanges the engine
    cleared are written down, because nothing in the action list implies
    them.

    A publish can itself fire this turn's *first* trade event
    (`Game.event_pending`), before the seat due to publish has even chosen
    its action -- reached, in this loop, right after `to_move` names that
    seat and before `choose`. Those trades are attributed to the
    *previous* action's step (`len(actions) - 1`), never the upcoming one:
    they are still "the event on the way into `MAIN`" that `advance`
    already knows how to replay -- `apply(the ROLL or ROBBER action);
    apply_trades(...)` -- just fired lazily now instead of eagerly inside
    that action's own `apply`. Recording them under the *next* step instead
    would replay them after that step's action applies, which can leave a
    build the trade was meant to fund looking unaffordable on replay.
    """
    rng = random.Random(seed)
    recording = Recording(Live(rng))
    game = start(board, len(bots), rng, chance=recording)
    game.gates = tuple(bots)
    actions: list[tuple[int, int, int]] = []
    trades: list[tuple[int, int, int, tuple[int, ...]]] = []
    while not is_over(game) and len(actions) < action_cap:
        seat = to_move(game)
        bot = bots[seat]
        if game.publish_due(seat):
            before = len(game.trades)
            publish_valuation(game, seat, bot)
            for trade in game.trades[before:]:
                trades.append((len(actions) - 1, trade.a, trade.b, tuple(trade.received)))
        before = len(game.trades)
        action = bot.choose(game)
        apply(game, action)
        for trade in game.trades[before:]:
            trades.append((len(actions), trade.a, trade.b, tuple(trade.received)))
        actions.append((int(action.type), action.a, action.b))

    return Record(
        num_players=len(bots),
        seed=seed,
        first=game.first,
        actions=tuple(actions),
        chance=tuple(recording.events),
        trades=tuple(trades),
        winner=game.won_by,
        turns=game.turns,
        **board_fields(board),
    )


def actions_of(record: Record) -> Iterator[Action]:
    """The recorded actions, in order."""
    for kind, a, b in record.actions:
        yield Action(ActionType(kind), a, b)


def steps(record: Record) -> Iterator[tuple[Action, tuple[Trade, ...]]]:
    """Each action with the trades the engine cleared inside it.

    Everything that walks a record goes through here, so the reconstruction
    lives in one place and cannot drift between replaying, featurising and
    behaviour analysis.
    """
    by_step: dict[int, list[Trade]] = {}
    for step, a, b, received in record.trades:
        by_step.setdefault(step, []).append(Trade(a, b, tuple(received)))
    for step, action in enumerate(actions_of(record)):
        yield action, tuple(by_step.get(step, ()))


def advance(game: Game, action: Action, trades: Sequence[Trade]) -> None:
    """Apply one recorded step: the action, then the trades it cleared.

    A replayed game has no seated bots (`game.gates` stays `None`), so its
    own trade event never fires -- `legal_actions`/`game.state`'s trigger is
    a true no-op with no gates seated -- and the recorded exchanges are
    re-executed here instead. Applying them immediately after `apply`
    returns is exactly where they happened: `record_game` attributes a
    turn's first event (fired lazily, at the first publish or observation
    for the new current player, not eagerly inside `roll_dice`/
    `move_robber_to` any more) to the *roll or robber* step it logically
    belongs to, precisely so this ordering -- one action, then the trades
    that followed it -- stays correct to replay.
    """
    apply(game, action)
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
    """
    game = open_record(record)
    for step, (action, trades) in enumerate(steps(record)):
        if action not in legal_actions(game):
            raise ReplayError(
                f"step {step}: {action} is not legal in {game.phase.name}"
            )
        try:
            advance(game, action, trades)
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
    `chance.discard`. The header's own `first` (the setup snake's start
    seat, needed on resume for the same reason -- `Journal.start`'s
    docstring) is read here too, not assumed 0: a journal dealt with a
    rotated snake would otherwise replay a different setup order than the
    one actually played.

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

    steps: list[tuple[Action, tuple[Trade, ...], tuple[tuple[str, int], ...]]] = []
    result: dict | None = None
    for event in events[1:]:
        kind = event.get("kind")
        if kind == "action":
            action = game_journal.action_of(event)
            trades = game_journal.trades_of(event)
            chance_events: list[tuple[str, int]] = []
            if action.type is ActionType.ROLL:
                chance_events.append(("roll", event["roll"]))
            if action.type is ActionType.MOVE_ROBBER:
                stole = event.get("stole")
                if stole is not None and stole.get("resource") is not None:
                    chance_events.append(("steal", int(Resource[stole["resource"]])))
            steps.append((action, trades, tuple(chance_events)))
        elif kind == "undo":
            del steps[event["back_to"] :]
        elif kind == "result":
            result = event

    if result is None:
        raise ValueError(f"journal has no result line, cannot record: {path}")

    deck = tuple(("deck", int(DevCard[name])) for name in header["deck"])
    chance = deck + tuple(event for _, _, events in steps for event in events)
    actions = tuple((int(action.type), action.a, action.b) for action, _, _ in steps)
    trades = tuple(
        (step, trade.a, trade.b, tuple(trade.received))
        for step, (_, step_trades, _) in enumerate(steps)
        for trade in step_trades
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
