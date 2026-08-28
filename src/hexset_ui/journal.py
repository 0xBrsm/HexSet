"""The server's own account of a game, written as the game happens.

One line per action, written the instant it is applied, with the hidden
information spelled out. The shuffled deck goes in the header, the dice on every
roll, the card drawn on every purchase, the resource taken on every steal, and
every seat's full hand and development cards afterwards. Nothing here needs the
engine to interpret it and nothing is held back — unlike the sidebar transcript,
which hides exactly what a player at the table could not see (see
`hexset_ui.webplay._describe`).

It is the only thing written, and it is also what a game is resumed from:
`replayable` reads these lines back into actions and `GameSession.restore`
re-applies them. A second, more compact file beside this one would be a subset
that could only ever disagree with it.

One file per game, not one shared file appended to: sessions are concurrent (the
web server deals a session per browser), and lines from three games in flight
would have to be de-interleaved before any one of them could be read. Every line
is written and closed as it happens, so a game abandoned mid-turn — or a server
killed outright — leaves a complete account up to that point instead of nothing.

Writing is on by default and switched off by setting `HEXSET_UI_GAMES_DIR`
empty. A directory that cannot be written to disables this journal and lets the
game carry on: a game nobody can log is still a game, and a player mid-turn
should not lose it to a full disk.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .actions import Action, ActionType
from .board.board import Board
from .board.terrain import NUM_RESOURCES, Resource
from .cards import NUM_DEV_CARDS, DevCard
from .devcards import holdings
from .game import Game
from .victory import victory_points

ENV_DIR = "HEXSET_UI_GAMES_DIR"
DEFAULT_DIR = "games"

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name for c in DevCard)


def board_fields(board: Board) -> dict[str, tuple]:
    """The board, spelled out for the header.

    Resuming rebuilds the board from the seed and never reads these back; they
    are here so the file can be understood without re-running the generator
    that made it, which is what this journal is for. Port vertices are left
    out — they follow from the edge for anyone rebuilding the topology.
    """
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
    """Where games are journalled, or `None` when that is switched off.

    On by default, so a server started with no arguments at all still keeps
    a full account of every game it deals. Pointing the variable somewhere
    else moves the directory; setting it to the empty string is how you turn
    the journal off, which is a thing to say explicitly rather than the
    accident of having forgotten a flag.
    """
    value = os.environ.get(ENV_DIR, DEFAULT_DIR).strip()
    return value or None


def new_game_id(seed: int) -> str:
    """A per-game filename stem: when it was dealt, its seed, and enough
    randomness to keep two browsers that started the same second apart. The
    seed is in the name (as well as in the header) so the file matching a
    record can be found without opening any of them."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-seed{seed}-{uuid.uuid4().hex[:6]}"


def effects(
    game: Game,
    actor: int,
    action: Action,
    before_hands: list[list[int]],
    before_held: list[list[int]],
) -> dict:
    """The parts of an action's outcome the action itself does not carry.

    Everything an action decides is already in its own operands, and every
    resource that moved can be read off the hands recorded either side of it.
    What neither of those covers is what the engine's random stream decided:
    the dice, which card came off the deck, and which card a steal took. Those
    are the three things named here, and they are the three things a reader
    would otherwise have to re-run the engine to learn.
    """
    state = game.state
    out: dict = {}

    if action.type is ActionType.ROLL:
        out["roll"] = game.last_roll

    if action.type is ActionType.BUY_DEV_CARD:
        held = holdings(state, actor)
        drawn = [c for c in range(NUM_DEV_CARDS) if held[c] > before_held[actor][c]]
        if drawn:
            out["drew"] = DEV_CARD_NAMES[drawn[0]]

    if action.type in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
        victim = action.b if action.b < state.num_players else None
        if victim is not None:
            taken = [
                r
                for r in range(NUM_RESOURCES)
                if state.hands[victim][r] < before_hands[victim][r]
            ]
            # A victim with an empty hand is a legal target, and nothing
            # moves: recorded as the steal that took nothing rather than
            # left out, so the log distinguishes it from a robber move that
            # named no victim at all.
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
    # Flipped by the first write that fails, so a journal on a directory that
    # turns out to be read-only complains once instead of once per action.
    _off: bool = field(default=False, repr=False)

    @property
    def path(self) -> Path:
        return Path(self.directory) / f"{self.game_id}.jsonl"

    def _emit(self, event: dict) -> None:
        if self._off:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        except OSError as error:
            self._off = True
            print(f"game journal disabled ({self.path}): {error}")

    def start(
        self,
        game: Game,
        *,
        seed: int,
        human_seat: int,
        bot_names: dict[int, str],
        bot_specs: dict[int, str],
        identity: str | None = None,
    ) -> None:
        """The header: everything true before the first action.

        The deck is the reason this line exists. It is the one piece of hidden
        state that is fixed at deal time and consumed silently thereafter —
        with the shuffle written down, every later purchase is checkable
        against it, and the whole game's development cards are known without
        the engine's random stream being involved at all.

        `identity` is the browser's `hexset_id` cookie and `spec` the string
        that built each bot, neither of which the game itself needs: they are
        here so `resume` can put this exact session back together for the
        person whose it was.
        """
        state = game.state
        self._emit(
            {
                "kind": "game",
                "id": self.game_id,
                "at": _now(),
                "seed": seed,
                "identity": identity,
                "num_players": state.num_players,
                "human_seat": human_seat,
                "bots": {
                    str(seat): {"name": name, "spec": bot_specs.get(seat, name)}
                    for seat, name in sorted(bot_names.items())
                },
                # Bottom of the deck first: `devcards.buy` pops off the end.
                "deck": [DEV_CARD_NAMES[c] for c in state.deck],
                "robber": state.robber,
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
    ) -> None:
        state = game.state
        event = {
            "kind": "action",
            "step": step,
            "round": round_num,
            "actor": actor,
            "type": action.type.name,
            "a": action.a,
            "b": action.b,
            **effects(game, actor, action, before_hands, before_held),
            # After, for every seat, not just the actor's: a roll pays several
            # players at once and a monopoly takes from all of them, so a
            # per-actor hand would make those actions unreadable. Absolute
            # counts rather than deltas, so any one line stands on its own
            # instead of being a correction to the lines before it.
            "hands": [hand[:] for hand in state.hands],
            "dev": [holdings(state, p) for p in range(state.num_players)],
            "deck_left": len(state.deck),
        }
        if action.give or action.want:
            event["give"] = list(action.give)
            event["want"] = list(action.want)
            event["ask"] = list(action.ask)
        self._emit(event)

    def undo(self, game: Game, *, back_to: int) -> None:
        """The human took a placement back (see `webplay.undo_last_build`).

        Written down rather than erased. The file is append-only and read
        forwards, so a reader that silently reused a step number would have two
        different actions claiming to be step N with nothing to say which one
        counted. This says which: everything from `back_to` onwards did not
        happen, and the steps that follow start again from there.
        """
        state = game.state
        self._emit(
            {
                "kind": "undo",
                "back_to": back_to,
                "hands": [hand[:] for hand in state.hands],
                "dev": [holdings(state, p) for p in range(state.num_players)],
                "deck_left": len(state.deck),
            }
        )

    def seated(self, *, seat: int, name: str, spec: str) -> None:
        """A different bot took `seat` mid-game (see the web server's
        `_handle_swap_bot`). The header names who sat down at the deal; a game
        put back together later has to seat whoever is there now, or resuming
        would quietly hand the human back a different set of opponents."""
        self._emit({"kind": "seated", "at": _now(), "seat": seat, "name": name, "spec": spec})

    def reopened(self, *, at_step: int) -> None:
        """This game was put back together from the lines above — a server
        restart, or a session evicted for going quiet (see `webplay.resume`).

        Written so the join shows. Without it the file reads as uninterrupted
        play, and the bots' own random streams do not survive a resume, so a
        reader comparing two halves of one file needs to know where the seam
        is.
        """
        self._emit({"kind": "reopened", "at": _now(), "at_step": at_step})

    def abandoned(self) -> None:
        """The human pressed New Game, which ends this one as surely as
        winning does. `resumable` hands back any game whose file has no
        closing line, so without this the game they chose to walk away from
        would be waiting for them on their next visit."""
        self._emit({"kind": "abandoned", "at": _now()})

    def finish(self, game: Game) -> None:
        """The closing line. A journal without one is a game that was abandoned
        rather than played out, which is the distinction anything counting
        results has to make."""
        state = game.state
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
    off. `directory` overrides the environment, for callers (tests, mostly)
    that would rather say where than set a variable."""
    where = directory if directory is not None else configured_dir()
    if not where:
        return None
    return Journal(directory=where, game_id=new_game_id(seed))


# --- Reading one back ---------------------------------------------------------


# The lines that mean this game is over and is not to be handed back: played
# out, or walked away from. Anything else leaves the file open.
CLOSING_KINDS = frozenset({"result", "abandoned"})


def read(path: Path | str) -> list[dict]:
    """Every event in a journal, in the order it was written.

    Stops at the first line that will not parse instead of raising. Lines are
    appended one at a time, so the only line that can be torn is the last one
    — a server killed mid-write — and everything before it is a complete
    account that should still be readable.
    """
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except OSError:
        return []
    return events


def header_of(path: Path | str) -> dict | None:
    """A journal's opening line alone, without reading the rest of it.

    `resumable` looks at every file in the directory to find one browser's
    game, and these run to tens of thousands of lines; only the one that
    matches is worth reading in full.
    """
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


def action_of(event: dict) -> Action:
    """The `Action` an action line describes. The recorded effects — the dice,
    the card drawn, the card stolen — are deliberately not used: they are the
    engine's to decide again from the same seed, and reading them back would
    turn a check that the replay agrees into a way of papering over that it
    does not."""
    offer = {}
    if "give" in event:
        offer = {
            "give": tuple(event["give"]),
            "want": tuple(event["want"]),
            "ask": tuple(event.get("ask", ())),
        }
    return Action(ActionType[event["type"]], event["a"], event["b"], **offer)


def replayable(events: list[dict]) -> list[tuple[int, Action]]:
    """Every (actor, action) to re-apply, in order.

    Undo lines are honoured by dropping what they took back. The file is
    append-only — an undone placement is still written down, by design (see
    `Journal.undo`) — so replaying the lines as they come would build the
    board the human explicitly rejected.
    """
    steps: list[tuple[int, Action]] = []
    for event in events:
        kind = event.get("kind")
        if kind == "action":
            steps.append((event["actor"], action_of(event)))
        elif kind == "undo":
            del steps[event["back_to"] :]
    return steps


def seating(events: list[dict]) -> dict[int, tuple[str, str]]:
    """Seat -> the (name, spec) of the bot on it: the lineup the game was
    dealt with, with every later swap applied in the order they happened."""
    header = events[0] if events else {}
    seats = {
        int(seat): (bot["name"], bot["spec"])
        for seat, bot in header.get("bots", {}).items()
    }
    for event in events:
        if event.get("kind") == "seated":
            seats[event["seat"]] = (event["name"], event["spec"])
    return seats


def resumable(directory: str | None, identity: str) -> Path | None:
    """The game `identity` left unfinished, or `None` to deal them a fresh one.

    Only their most recent game is ever a candidate. An older unfinished file
    is a game they already walked away from once — handing it back because a
    newer one happens to have ended would be reaching further into the past
    than the player ever asked for.
    """
    if not directory:
        return None
    try:
        paths = sorted(Path(directory).glob("*.jsonl"), reverse=True)
    except OSError:
        return None
    for path in paths:
        header = header_of(path)
        if header is None or header.get("identity") != identity:
            continue
        return None if is_closed(read(path)) else path
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
