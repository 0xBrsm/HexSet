"""The server's own account of a game, written as the game happens.

One JSON line per event, written the instant it is applied, with hidden
information spelled out: the shuffled deck in the header, the dice, the card
drawn, the resource stolen, and every seat's full hand and development cards
afterwards. Nothing is held back, unlike the sidebar transcript. It is also
what a game is resumed from (`replayable`, `GameSession.restore`).

One file per game, since sessions are concurrent, each line written and closed
as it happens, so a game abandoned mid-turn leaves a complete account up to
that point. On by default; `HEXSET_UI_GAMES_DIR` empty switches it off, and an
unwritable directory disables it without stopping the game.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from hexset.actions import Action, ActionType
from hexset.board.board import Board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import NUM_DEV_CARDS, DevCard
from hexset.devcards import holdings
from hexset.game import Game
from hexset.trading import Trade
from hexset.victory import victory_points

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hexset.record import Record

    from ._webplay import RoundNote  # noqa: F401 -- `journal_of`'s annotation

ENV_DIR = "HEXSET_UI_GAMES_DIR"
DEFAULT_DIR = "games"

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name for c in DevCard)


def board_fields(board: Board) -> dict[str, tuple]:
    """The board, spelled out for the header, so the file can be read without
    re-running the generator. Resuming rebuilds from the seed and never reads
    these back. Port vertices are omitted; they follow from the edge."""
    return {
        "layout": tuple(tuple(h) for h in board.topology.hexes),
        "terrain": tuple(int(t) for t in board.terrain),
        "tokens": tuple(board.tokens),
        "ports": tuple(
            (p.edge, p.ratio, None if p.resource is None else int(p.resource))
            for p in board.ports
        ),
    }


def configured_dir() -> str | None:
    """Where games are journalled, or `None` when journalling is off. On by
    default; setting `HEXSET_UI_GAMES_DIR` to the empty string turns it off."""
    value = os.environ.get(ENV_DIR, DEFAULT_DIR).strip()
    return value or None


def new_game_id(seed: int | None) -> str:
    """A per-game filename stem: deal time, seed, and random suffix. The seed
    is in the name as well as the header, so a file can be matched without
    opening it; a game with none (`journal_of`) is named a record."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dealt = "record" if seed is None else f"seed{seed}"
    return f"{stamp}-{dealt}-{uuid.uuid4().hex[:6]}"


def effects(
    game: Game,
    actor: int,
    action: Action,
    before_hands: list[list[int]],
    before_held: list[list[int]],
) -> dict:
    """The parts of an action's outcome the action itself does not carry: what
    the engine's random stream decided — the dice (`roll`), the card drawn
    (`drew`), and the card a steal took (`stole`)."""
    # true state: the journal is the record of everything that
    # happened, including outcomes no view would expose.
    state = game.state(0, hidden=False)
    out: dict = {}

    if action.type is ActionType.ROLL:
        out["roll"] = game.last_roll

    if action.type is ActionType.BUY_DEV_CARD:
        held = holdings(state, actor)
        drawn = [c for c in range(NUM_DEV_CARDS) if held[c] > before_held[actor][c]]
        if drawn:
            out["drew"] = DEV_CARD_NAMES[drawn[0]]

    if action.type is ActionType.MOVE_ROBBER:
        victim = action.b if action.b < state.num_players else None
        if victim is not None:
            taken = [
                r
                for r in range(NUM_RESOURCES)
                if state.hands[victim][r] < before_hands[victim][r]
            ]
            # A victim with an empty hand is legal and moves nothing:
            # recorded with `resource: null`, distinct from a robber move
            # that named no victim at all (no `stole` key).
            out["stole"] = {
                "from": victim,
                "resource": RESOURCE_NAMES[taken[0]] if taken else None,
            }

    return out


@dataclass
class Journal:
    """One game's file. Created open, and appended to until the game ends."""

    directory: str
    game_id: str
    # Flipped by the first write that fails, so a read-only directory
    # complains once instead of once per action.
    _off: bool = field(default=False, repr=False)
    # Set once this object has made sure the file ends on a line break.
    _mended: bool = field(default=False, repr=False)

    @property
    def path(self) -> Path:
        return Path(self.directory) / f"{self.game_id}.jsonl"

    def _emit(self, event: dict) -> None:
        if self._off:
            return
        line = json.dumps(event, separators=(",", ":")) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+b") as handle:
                if not self._mended:
                    # A file reopened after a crash can end part-way through
                    # a line. Written straight onto it, this line would be torn
                    # too, and `read` could not tell either apart from noise.
                    self._mended = True
                    if handle.seek(0, os.SEEK_END) > 0:
                        handle.seek(-1, os.SEEK_END)
                        if handle.read(1) != b"\n":
                            handle.write(b"\n")
                handle.write(line.encode("utf-8"))
        except OSError as error:
            self._off = True
            print(f"game journal disabled ({self.path}): {error}")

    def start(
        self,
        game: Game,
        *,
        seed: int | None,
        first: int,
        human_seats: list[int],
        bot_names: dict[int, str],
        bot_specs: dict[int, str],
        player_names: dict[int, str] | None = None,
        clients: dict[int, dict] | None = None,
        code: str | None = None,
    ) -> None:
        """The header: everything true before the first action.

        `deck` is the shuffle, the one piece of hidden state fixed at deal
        time. `code` and each bot's `spec` are what rebuild the table on
        resume, and `first` is the seat the setup snake started at, a free
        per-table choice not derivable from `seed`/`num_players`. A `seed`
        of `None` says the board below and the recorded effects are the
        whole of the game's chance, which is how a resume reads it.
        `human_seats` are the seats occupied at deal time, of any kind,
        `player_names` whatever they registered as, and `clients`
        `api.parse_client`'s record for those that claimed with one. A seat
        claimed later, or closed, is its own event kind instead.
        """
        # true state: the journal is the record of everything that
        # happened, including outcomes no view would expose.
        state = game.state(0, hidden=False)
        self._emit(
            {
                "kind": "game",
                "id": self.game_id,
                "at": _now(),
                "seed": seed,
                "first": first,
                "code": code,
                "num_players": state.num_players,
                "human_seats": list(human_seats),
                "player_names": {str(s): n for s, n in sorted((player_names or {}).items())},
                "clients": {str(s): c for s, c in sorted((clients or {}).items())},
                "bots": {
                    str(seat): {"name": name, "spec": bot_specs.get(seat, name)}
                    for seat, name in sorted(bot_names.items())
                },
                # Bottom of the deck first: `devcards.buy` pops off the end.
                "deck": [DEV_CARD_NAMES[c] for c in state.deck],
                "robber": state.robber,
                "rules": asdict(state.rules),
                **{k: list(v) for k, v in board_fields(state.board).items()},
            }
        )

    def action(
        self,
        game: Game,
        *,
        step: int,
        round_num: int,
        actor: int,
        action: Action,
        before_hands: list[list[int]],
        before_held: list[list[int]],
        trades: Sequence[Trade] = (),
    ) -> None:
        """One applied action, with every seat's holdings after it."""
        # true state: the journal is the record of everything that
        # happened, including outcomes no view would expose.
        state = game.state(0, hidden=False)
        event = {
            "kind": "action",
            "step": step,
            "round": round_num,
            "actor": actor,
            "type": action.type.name,
            "a": action.a,
            "b": action.b,
            **effects(game, actor, action, before_hands, before_held),
            # Every seat, not just the actor's (a roll pays several at
            # once), and absolute counts rather than deltas, so any one
            # line stands on its own.
            "hands": [hand[:] for hand in state.hands],
            "dev": [holdings(state, p) for p in range(state.num_players)],
            "deck_left": len(state.deck),
        }
        if trades:
            # Exchanges the engine cleared inside this action
            # (`hexset.trading`). Recorded because they depend on what every
            # seat had published, so a replay cannot re-derive them.
            event["trades"] = [[t.a, t.b, list(t.received)] for t in trades]
        self._emit(event)

    def manual_trade(self, game: Game, *, step: int, round_num: int, trade: Trade) -> None:
        """A bundle executed outside the automatic event: its own line, since
        nothing but two hands changed and there is no action to attach it to.
        Replayed as its own step, re-executed directly rather than re-gated."""
        # true state: the journal is the record of everything that
        # happened, including outcomes no view would expose.
        state = game.state(0, hidden=False)
        self._emit(
            {
                "kind": "trade",
                "step": step,
                "round": round_num,
                "trade": [trade.a, trade.b, list(trade.received)],
                "hands": [hand[:] for hand in state.hands],
            }
        )

    def note(self, *, step: int, round_num: int, note) -> None:
        """One step of a trade round (`webplay.RoundNote`): the offer, a
        seat's answer, the actor declining, everyone having passed. Nothing
        moved, so no hands are written; `step` is the step this preceded,
        which is where `notes_of` puts it back."""
        self._emit(
            {
                "kind": "note",
                "step": step,
                "round": round_num,
                "note": note.kind,
                "seat": note.seat,
                "actor": note.actor,
                "bundle": None if note.bundle is None else list(note.bundle),
                **({"any": int(note.any)} if getattr(note, "any", 0) else {}),
            }
        )

    def undo(self, game: Game, *, back_to: int) -> None:
        """The human took a placement back. The file is append-only, so the
        undone action stays written and this line says everything from
        `back_to` onwards did not happen."""
        # true state: the journal is the record of everything that
        # happened, including outcomes no view would expose.
        state = game.state(0, hidden=False)
        self._emit(
            {
                "kind": "undo",
                "back_to": back_to,
                "hands": [hand[:] for hand in state.hands],
                "dev": [holdings(state, p) for p in range(state.num_players)],
                "deck_left": len(state.deck),
            }
        )

    def seated(self, *, seat: int, name: str, spec: str, client: dict | None = None) -> None:
        """A seat's occupant, named after the deal: a bot swapped in mid-game,
        or an open seat somebody joined. `spec` is empty for a person, which is
        how a resumed table tells the two apart and re-seats only bots.
        `client` is the header's `clients` record, `None` for a bot swap."""
        self._emit(
            {"kind": "seated", "at": _now(), "seat": seat, "name": name, "spec": spec, "client": client}
        )

    def locked(self, seat: int, *, at_step: int) -> None:
        """`seat` was closed while still empty (`api.Tables.close_seat`) or
        left (`api.Tables.leave_seat`), and is retired for the rest of the
        game. Where the line falls among the action lines is what a replay
        reads (`retirements`); `at_step` is diagnostic only."""
        self._emit({"kind": "locked", "at": _now(), "seat": seat, "at_step": at_step})

    def unlocked(self, seat: int, *, at_step: int) -> None:
        """`seat` was reopened (`api.Tables.open_seat`, or a bot seated there
        by `api.Tables.seat_bot`), only ever before the first move, so
        `at_step` is always 0. `retirements` reads the two kinds in order."""
        self._emit({"kind": "unlocked", "at": _now(), "seat": seat, "at_step": at_step})

    def renamed(self, *, seat: int, name: str) -> None:
        """A seat's occupant took a new display name (`api.Tables.rename`);
        `players` folds it in."""
        self._emit({"kind": "renamed", "at": _now(), "seat": seat, "name": name})

    def reopened(self, *, at_step: int) -> None:
        """This game was put back together from the lines above — a server
        restart, or a session evicted for going quiet. Marks the seam, which
        matters because the bots' random streams do not survive a resume."""
        self._emit({"kind": "reopened", "at": _now(), "at_step": at_step})

    def abandoned(self) -> None:
        """The game stopped without being played out: New Game, or a table
        evicted for going quiet (`api.Table.close`). Distinct from `finish`,
        which says the game *ended*; only the replayed game itself
        (`is_over`) says which happened."""
        self._emit({"kind": "abandoned", "at": _now()})

    def finish(self, game: Game) -> None:
        """The closing line of a game played out to an end. A journal without
        one was abandoned rather than finished."""
        # true state: the journal is the record of everything that
        # happened, including outcomes no view would expose.
        state = game.state(0, hidden=False)
        self._emit(
            {
                "kind": "result",
                "at": _now(),
                "winner": game.won_by,
                "turns": game.turns,
                "points": [victory_points(state, p) for p in range(state.num_players)],
            }
        )


def open_journal(seed: int, directory: str | None = None) -> Journal | None:
    """A journal for a game about to be dealt, or `None` when journalling is
    off. `directory` overrides the environment."""
    where = directory if directory is not None else configured_dir()
    if not where:
        return None
    return Journal(directory=where, game_id=new_game_id(seed))


def journal_of(
    record: "Record",
    directory: str,
    *,
    code: str,
    names: dict[int, str] | None = None,
    notes: Sequence[tuple[int, int, "RoundNote"]] = (),
) -> Path:
    """Write `record` as a journal filed under join code `code`, so the page's
    replay steps through it like a game this server dealt. Returns the file.

    The game is played through a `GameSession` with the journal attached, so
    every line is the one a live table would have written. The header has no
    seed: resuming reads the board from the header and the chance from the
    lines (`api._opening_session`). `names` labels the seats; every seat is a
    claimed player, never a bot, so nothing reopened plays a move. Each
    exchange is a line of its own after the action it followed, as a manual
    trade is.

    `notes` is the trade rounds the record's game had, as
    `(step, traded, note)`: `note` came after the action at `step` and after
    `traded` of the exchanges filed on it. Each is journalled there, so the
    log reads offers, answers and trades in the order they happened.

    The record is replayed first, and refused if it does not replay. So is one
    with a negative value anywhere in its chance, even in a deck slot nobody
    drew, and one with a note placed after an action or an exchange the record
    does not have.
    """
    from hexset.game import to_move
    from hexset.record import moves, open_record, replay

    from ._webplay import GameSession  # local at run time: webplay imports this module

    if any(value < 0 for _, value in record.chance):
        # An undrawn card is still written into the header's deck.
        raise ValueError("the record's chance names a card or resource that is not one")
    replay(record)
    placed: dict[tuple[int, int], list] = {}
    for step, traded, note in notes:
        placed.setdefault((int(step), int(traded)), []).append(note)
    journal = Journal(directory=directory, game_id=new_game_id(None))
    session = GameSession(
        game=open_record(record),
        claimed_seats=set(range(record.num_players)),
        seed=None,
        journal=journal,
        player_names=dict(names or {}),
        code=code.lower(),
    )
    for step, (actor, action, trades) in enumerate(moves(record)):
        seat = to_move(session.game) if actor is None else actor
        session.apply_action(seat, action, replay=())
        for traded, trade in enumerate(trades):
            for note in placed.pop((step, traded), ()):
                session.note(note)
            session.replay_trades(trade.a, (trade,))
        for note in placed.pop((step, len(trades)), ()):
            session.note(note)
    if placed:
        raise ValueError(f"notes placed where the record has no step: {sorted(placed)[:3]}")
    if not session.ended:
        journal.abandoned()
    return journal.path


# --- Reading one back ---------------------------------------------------------


# Lines meaning nothing further will be appended: played out (`result`) or
# walked away from (`abandoned`). Which of the two it was, `is_closed` does
# not say — `is_finished` does.
CLOSING_KINDS = frozenset({"result", "abandoned"})


def read(path: Path | str) -> list[dict]:
    """Every event in a journal, in the order it was written. A line that
    will not parse is skipped rather than raised on or stopped at: only a
    crash tears one, and every line after it was written by the game reopened
    from the lines before, which never saw it."""
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events


def header_of(path: Path | str) -> dict | None:
    """A journal's opening line alone. Files run to tens of thousands of
    lines, and `most_recent` scans every file in the directory."""
    try:
        with open(path, encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None
    try:
        event = json.loads(first)
    except json.JSONDecodeError:
        return None
    return event if event.get("kind") == "game" else None


def is_closed(events: list[dict]) -> bool:
    return any(event.get("kind") in CLOSING_KINDS for event in events)


def is_finished(events: list[dict]) -> bool:
    """Played out to an end, as against abandoned part-way — the distinction
    `is_closed` does not draw. `api.Tables.replay_view` asks this to decide how
    much of a past round to disclose, which a stand-in stopped mid-game cannot
    work out for itself."""
    return any(event.get("kind") == "result" for event in events)


def action_of(event: dict) -> Action:
    """The `Action` an action line describes. The recorded effects (dice, card
    drawn, card stolen) are deliberately not read back: the engine decides them
    again from the same seed, so a replay that disagrees shows up."""
    return Action(ActionType[event["type"]], event["a"], event["b"])


def trades_of(event: dict) -> tuple[Trade, ...]:
    """The exchanges an action line cleared. Unlike the effects above these
    *are* read back and re-executed: they depend on what every seat had
    published, which a replay cannot reconstruct."""
    return tuple(Trade(a, b, tuple(received)) for a, b, received in event.get("trades", ()))


def replayable_rounds(
    events: list[dict],
) -> tuple[list[tuple[int, Action | None, tuple[Trade, ...]]], list[int]]:
    """`replayable`'s steps, plus the round each was played in, built and
    truncated together. The rounds let a reader ask for a *position* rather
    than a step count: the last step carrying round N is where the board stood
    when that round finished."""
    steps: list[tuple[int, Action | None, tuple[Trade, ...]]] = []
    rounds: list[int] = []
    for event in events:
        kind = event.get("kind")
        if kind == "action":
            steps.append((event["actor"], action_of(event), trades_of(event)))
        elif kind == "trade":
            a, b, received = event["trade"]
            steps.append((a, None, (Trade(a, b, tuple(received)),)))
        elif kind == "undo":
            del steps[event["back_to"] :]
            del rounds[event["back_to"] :]
            continue
        else:
            continue
        rounds.append(int(event.get("round", 0)))
    return steps, rounds


def replayable(events: list[dict]) -> list[tuple[int, Action | None, tuple[Trade, ...]]]:
    """Every (actor, action, trades) to re-apply, in order. `action` is `None`
    for a `"trade"` line, which replays as its own step. Undo lines are
    honoured by dropping what they took back, the file being append-only."""
    return replayable_rounds(events)[0]


def notes_of(events: list[dict]) -> dict[int, list[tuple[int, RoundNote]]]:
    """The trade-round steps (`Journal.note`) as `(round, RoundNote)`, keyed
    by the step each preceded. An undo drops the notes it took back: those
    after the step it returned to, not those before it, which still
    happened."""
    from ._webplay import RoundNote  # local at run time: webplay imports this module

    notes: dict[int, list[tuple[int, RoundNote]]] = {}
    for event in events:
        kind = event.get("kind")
        if kind == "note":
            bundle = event.get("bundle")
            notes.setdefault(int(event["step"]), []).append(
                (
                    int(event["round"]),
                    RoundNote(
                        str(event["note"]), int(event["seat"]), int(event["actor"]),
                        None if bundle is None else tuple(int(n) for n in bundle),
                        int(event.get("any") or 0),
                    ),
                )
            )
        elif kind == "undo":
            for step in [s for s in notes if s > event["back_to"]]:
                del notes[step]
    return notes


def seating(events: list[dict]) -> dict[int, tuple[str, str]]:
    """Seat -> the (name, spec) of the bot on it: the dealt lineup with every
    later swap or bot claim folded on top. Bots only — a `seated` event with
    an empty `spec` is a person and removes the seat. `players` is the other
    half of the split."""
    header = events[0] if events else {}
    seats = {
        int(seat): (bot["name"], bot["spec"])
        for seat, bot in header.get("bots", {}).items()
    }
    for event in events:
        if event.get("kind") == "seated":
            if event["spec"]:
                seats[event["seat"]] = (event["name"], event["spec"])
            else:
                seats.pop(event["seat"], None)
    return seats


def players(events: list[dict]) -> dict[int, str]:
    """Seat -> the name of the person (or LLM) holding it: `seating`'s exact
    complement, folded the same way with the `spec` test inverted. The seats
    are the header's `human_seats` less the ones its `bots` map claims, named
    from `player_names`; a seat nobody named comes back with an empty string.
    `api.Tables._reopen` rebuilds them as claimed-but-untokened, so a reopened
    game's seats do not look open to `Table.join`."""
    header = events[0] if events else {}
    names = header.get("player_names", {})
    bots = set(header.get("bots", {}))
    seats = {
        int(seat): names.get(str(seat), "")
        for seat in header.get("human_seats", [])
        if str(seat) not in bots
    }
    for event in events:
        if event.get("kind") == "seated":
            if event["spec"]:
                seats.pop(event["seat"], None)
            else:
                seats[event["seat"]] = event.get("name") or ""
        elif event.get("kind") == "renamed" and event["seat"] in seats:
            seats[event["seat"]] = event.get("name") or ""
    return seats


def clients(events: list[dict]) -> dict[int, dict]:
    """Seat -> its client record (`{"id", "kind"}`), from the header's
    `clients` map with every later `seated` event folded on top, so a seat's
    identity for `POST /api/reclaim` survives a restart. A seat with no client
    anywhere is left out."""
    header = events[0] if events else {}
    result = {int(seat): client for seat, client in header.get("clients", {}).items() if client}
    for event in events:
        if event.get("kind") == "seated" and event.get("client"):
            result[event["seat"]] = event["client"]
    return result


def retirements(events: list[dict]) -> dict[int, list[int]]:
    """The seats to lock as a game replays, keyed by the step each was
    locked before: where each `Journal.locked` line falls among the steps
    `replayable_rounds` counts. Seats closed and reopened before the first
    move net out at step 0. An undo back past a retirement moves it to the
    step undone to, since an undo takes back moves, not a seat's leaving."""
    at: dict[int, list[int]] = {}
    steps = 0
    for event in events:
        kind = event.get("kind")
        if kind in ("action", "trade"):
            steps += 1
        elif kind == "undo":
            steps = min(steps, int(event["back_to"]))
            for later in sorted(s for s in at if s > steps):
                at.setdefault(steps, []).extend(at.pop(later))
        elif kind == "locked":
            at.setdefault(steps, []).append(int(event["seat"]))
        elif kind == "unlocked" and int(event["seat"]) in at.get(steps, []):
            at[steps].remove(int(event["seat"]))
    return {step: seats for step, seats in at.items() if seats}


def most_recent(directory: str | None, code: str) -> Path | None:
    """The most recent file filed under join code `code`, closed or not, or
    `None`. Closed files count, `api.reopen_session` deciding from the replayed
    game which it is; only the most recent file bearing the code is ever a
    candidate; codes are matched case-insensitively."""
    if not directory:
        return None
    wanted = code.lower()
    try:
        paths = sorted(Path(directory).glob("*.jsonl"), reverse=True)
    except OSError:
        return None
    for path in paths:
        header = header_of(path)
        if header is None or str(header.get("code") or "").lower() != wanted:
            continue
        return path
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
