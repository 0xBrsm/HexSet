"""The server's own account of a game, written as the game happens.

`catan.record` stores a game as a seed plus an action sequence: compact, and
enough to reproduce a game exactly — but only by re-running the engine that
wrote it. Dice, steals and the development deck's order are deliberately not in
it (see that module's docstring), so a record can only say what was drawn or
stolen by replaying, and a change to how the engine draws randomness turns every
older record into a `ReplayError` rather than into an answer.

This is the whole account instead: one line per action, written the instant it
is applied, with the hidden information spelled out. The shuffled deck goes in
the header, the dice on every roll, the card drawn on every purchase, the
resource taken on every steal, and every seat's full hand and development cards
afterwards. Nothing here needs the engine to interpret it and nothing is held
back — unlike the sidebar transcript, which hides exactly what a player at the
table could not see (see `catan.webplay._describe`).

It is the only thing written. Every field a `Record` carries is in here already
— board, seed, actions, offers, winner, turns — so a second file holding a
subset of this one would be a copy that could only ever disagree with it. A
consumer that wants records builds them from these lines (`test_webplay.py`
does exactly that to check a journalled game still replays clean).

One file per game, not one shared file appended to: sessions are concurrent (the
web server deals a session per browser), and lines from three games in flight
would have to be de-interleaved before any one of them could be read. Every line
is written and closed as it happens, so a game abandoned mid-turn — or a server
killed outright — leaves a complete account up to that point instead of nothing,
which is the case `catan.record` cannot cover: it only ever holds a game that
reached an ending.

Writing is on by default and switched off by setting `CATAN_WEB_GAMES_DIR`
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
from .board.terrain import NUM_RESOURCES, Resource
from .cards import NUM_DEV_CARDS, DevCard
from .devcards import holdings
from .game import Game
from .record import board_fields
from .victory import victory_points

ENV_DIR = "CATAN_WEB_GAMES_DIR"
DEFAULT_DIR = "games"

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name for c in DevCard)


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
    ) -> None:
        """The header: everything true before the first action.

        The deck is the reason this line exists. It is the one piece of hidden
        state that is fixed at deal time and consumed silently thereafter —
        with the shuffle written down, every later purchase is checkable
        against it, and the whole game's development cards are known without
        the engine's random stream being involved at all.
        """
        state = game.state
        self._emit(
            {
                "kind": "game",
                "id": self.game_id,
                "at": _now(),
                "seed": seed,
                "num_players": state.num_players,
                "human_seat": human_seat,
                "bots": {str(seat): name for seat, name in sorted(bot_names.items())},
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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
