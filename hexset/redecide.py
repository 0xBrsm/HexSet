# SPDX-License-Identifier: GPL-3.0-only
"""Decide a run's logged search decisions again, under other seats.

A run whose seats search keeps every searched decision in its journal
(`hexset.arena.compete`, each game's `searches`). This replays each game's
record and, at every searched decision the chosen entrant logged, decides
the position again with each of the named seats ("arms"): their searcher
(`decide`), not their whole bot, so a trader, closer or dealer the seat
carries plays no part. Each arm's searcher is reseeded before each
decision from the arm's seed and the decision's place in the run, so two
arms that read the position alike decide it alike. An arm named twice with
different seeds measures how often the search's own randomness changes a
move.

Plays no game. Appends each game's rows as it finishes, fsynced: the
decisions it re-decided (`kind` "decision") and then the game's tally
(`kind` "game"), after a "header" row in a new file; a rerun skips the games
already tallied.

    python -m hexset.redecide <journal.jsonl> <out.jsonl> \\
        --arm A=<lineup name> --arm B=<lineup name> [--arm-seed A2=<seed>] \\
        [--entrant 0] [--where module:function] [--runtime module] [--workers N]

`--where` names a function `(game, seat) -> bool` that picks the positions
worth deciding again; the others are counted and skipped. Seeds are the
seed policy's analysis seeds (`hexset.seeds`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from . import record as R
from . import seeds
from .decisions import option_json
from .arena import RUNTIME_HELP, entrant_from_name, load_runtime, searchers, spawn
from .game import Game, to_move


@dataclass(frozen=True)
class Arm:
    """A seat to decide with: its `name` in the output, its lineup `spec`
    (`hexset.arena.entrant_from_name`) and the `seed` its searcher is
    reseeded from."""

    name: str
    spec: str
    seed: int


def action_json(action: object) -> object:
    """`action` as a decision log writes a candidate
    (`hexset.decisions.option_json`): `[type, a, b]`, or a whole decision of
    several moves as `[kind, [[type, a, b], ...]]`."""
    return json.loads(json.dumps(option_json(action)))


def logged_move(decision: dict) -> object | None:
    """The move a logged decision chose, as its log writes it; `None` for a
    root it did not search, whose log keeps no moves."""
    if "actions" not in decision:
        return None
    return decision["actions"][decision["candidates"].index(decision["chosen"])]


def decision_seed(seed: int, game: int, ply: int) -> int:
    """The seed a searcher is reset to before deciding ply `ply` of game
    `game`: the same for every arm that names `seed`."""
    digest = hashlib.sha256(f"{seed}:{game}:{ply}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def searcher_of(bot: object) -> object:
    """The one searcher behind a seat (`hexset.arena.searchers`)."""
    found = searchers(bot)
    if len(found) != 1:
        raise ValueError(f"a seat to decide with needs exactly one searcher behind it, not {len(found)}")
    return found[0]


def hold_meta(searcher: object) -> dict | None:
    """The reading of hidden cards a searcher deals its worlds by, as a
    checkpoint declares it (`meta`), or its type where it declares none."""
    hold = getattr(searcher, "hold", None)
    if hold is None:
        return None
    meta = getattr(hold, "meta", None)
    return meta() if callable(meta) else {"type": type(hold).__name__}


def matched(line: dict, entrant: int) -> Iterator[tuple[Game, int, object, dict | None]]:
    """Replay a journal game line's record: each step as (the position before
    it, its ply, the move played, the entrant's logged decision for it or
    `None`). A logged decision is matched to the first step of its turn and
    phase where the entrant moves and plays the move it chose (any move, for
    a root it did not search); one passed by without a match is dropped."""
    seat = line["seating"][entrant]
    queue = list(line["searches"][entrant])
    rec = R.from_json(json.dumps(line["record"]))
    game = R.open_record(rec)
    shown = R.shown_by_step(rec)
    for ply, (actor, action, trades) in enumerate(R.moves(rec)):
        while queue and queue[0]["turn"] < game.turns:
            queue.pop(0)
        decision = None
        if (queue and to_move(game) == seat and queue[0]["turn"] == game.turns
                and queue[0]["phase"] == game.phase.name
                and logged_move(queue[0]) in (None, action_json(action))):
            decision = queue.pop(0)
        yield game, ply, action, decision
        R.advance(game, action, trades, actor, shown.get(ply, ()))


def _where(name: str | None) -> Callable[[Game, int], bool] | None:
    if not name:
        return None
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(module), function)


def redecide_game(line: dict, entrant: int, arms: Sequence[Arm], where: str | None = None) -> list[dict]:
    """The rows for one journal game: a "decision" row for each searched
    decision of `entrant` that `where` keeps, with every arm's decision of
    it, then the game's "game" row."""
    pick = _where(where)
    seat = line["seating"][entrant]
    bots: dict[str, object] = {}
    rows: list[dict] = []
    tally = {"logged": len(line["searches"][entrant]), "matched": 0, "searched": 0, "kept": 0}
    for game, ply, action, logged in matched(line, entrant):
        if logged is None:
            continue
        tally["matched"] += 1
        if not logged.get("searched"):
            continue
        tally["searched"] += 1
        if pick is not None and not pick(game, seat):
            continue
        tally["kept"] += 1
        decided = {}
        for arm in arms:
            if arm.name not in bots:
                rng = random.Random(decision_seed(arm.seed, line["index"], -1))
                bots[arm.name] = searcher_of(spawn(entrant_from_name(arm.spec), game._state.board, rng))
            searcher = bots[arm.name]
            searcher.rng = random.Random(decision_seed(arm.seed, line["index"], ply))
            start = time.process_time()
            decision = searcher.decide(game)
            cpu = time.process_time() - start
            searcher.decision_log.clear()
            out = decision.to_json()
            out["move"] = action_json(decision.options[decision.chosen])
            out["cpu"] = round(cpu, 4)
            decided[arm.name] = out
        rows.append({
            "kind": "decision", "game": line["index"], "ply": ply, "turn": game.turns,
            "phase": game.phase.name, "seat": seat, "played": action_json(action),
            "logged": logged_move(logged), "logged_value": logged["value"][seat], "arms": decided,
        })
    rows.append({"kind": "game", "game": line["index"], "seat": seat, **tally})
    return rows


def _games(journal: str, done: set[int]) -> Iterator[dict]:
    with open(journal) as f:
        for text in f:
            if '"kind":"game"' not in text[:40] and '"kind": "game"' not in text[:40]:
                continue
            line = json.loads(text)
            if line["index"] not in done:
                yield line


def _done(out: str) -> set[int]:
    done: set[int] = set()
    if os.path.exists(out):
        with open(out) as f:
            for text in f:
                row = json.loads(text)
                if row.get("kind") == "game":
                    done.add(row["game"])
    return done


def _append(out, rows: Sequence[dict]) -> None:
    for row in rows:
        out.write(json.dumps(row) + "\n")
    out.flush()
    os.fsync(out.fileno())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _worker(line: dict, entrant: int, arms: Sequence[Arm], where: str | None) -> list[dict]:
    return redecide_game(line, entrant, arms, where)


def run(journal: str, out: str, arms: Sequence[Arm], *, entrant: int = 0, where: str | None = None,
        runtime: Sequence[str] = (), workers: int = 1, header: dict | None = None) -> int:
    """Decide every game of `journal` again (`redecide_game`), appending each
    game's rows to `out` as it finishes; the games `out` already tallies are
    skipped. A new `out` opens with a "header" row. Returns the games done."""
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("every arm needs its own name")
    load_runtime(*runtime)
    done = _done(out)
    count = 0
    with open(out, "a") as f:
        if not done and f.tell() == 0:
            _append(f, [{"kind": "header", "journal": os.path.abspath(journal), "journal_sha256": _sha256(journal),
                         "entrant": entrant, "where": where, "runtime": list(runtime),
                         "arms": [{"name": a.name, "spec": a.spec, "seed": a.seed} for a in arms],
                         **(header or {})}])
        games = _games(journal, done)
        if workers <= 1:
            for line in games:
                _append(f, redecide_game(line, entrant, arms, where))
                count += 1
            return count
        with ProcessPoolExecutor(workers, initializer=load_runtime, initargs=tuple(runtime)) as pool:
            pending = set()
            for line in games:
                pending.add(pool.submit(_worker, line, entrant, arms, where))
                if len(pending) >= 4 * workers:
                    finished = next(as_completed(pending))
                    pending.discard(finished)
                    _append(f, finished.result())
                    count += 1
            for finished in as_completed(pending):
                _append(f, finished.result())
                count += 1
    return count


def arm_holds(arms: Sequence[Arm], runtime: Sequence[str], board) -> dict[str, dict | None]:
    """Each arm's reading of hidden cards (`hold_meta`), read off a seat
    spawned on `board`: what the header records, so a run shows which
    treatment each arm carried."""
    load_runtime(*runtime)
    return {arm.name: hold_meta(searcher_of(spawn(entrant_from_name(arm.spec), board, random.Random(0))))
            for arm in arms}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hexset.redecide", description=__doc__.splitlines()[0])
    ap.add_argument("journal", help="a journal whose games carry their search logs")
    ap.add_argument("out", help="the rows, one JSON line each; an existing file is resumed")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SPEC",
                    help="a seat to decide with, as a lineup names it (repeatable; the first is the reference)")
    ap.add_argument("--arm-seed", action="append", default=[], metavar="NAME=SEED",
                    help="an arm's seed, an analysis seed of the seed policy (default: the policy's first)")
    ap.add_argument("--entrant", type=int, default=0, help="the entrant whose logged decisions are decided again")
    ap.add_argument("--where", default=None, metavar="MODULE:FUNCTION",
                    help="keep only the positions where FUNCTION(game, seat) is true")
    ap.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    ap.add_argument("--workers", type=int, default=1)
    seeds.add_arguments(ap, help="the arms' default seed")
    args = ap.parse_args(argv)
    notes = [seeds.hold(args, use="analysis")]
    chosen = {}
    for item in args.arm_seed:
        name, _, value = item.partition("=")
        chosen[name] = int(value)
        notes.append(seeds.check(int(value), use="analysis", seats=4, bypass=args.bypass_seed_policy))
    arms = []
    for item in args.arm:
        name, sep, spec = item.partition("=")
        if not sep or not name or not spec:
            raise SystemExit(f"--arm takes NAME=SPEC, not {item!r}")
        arms.append(Arm(name, spec, chosen.get(name, args.seed)))
    first = next(_games(args.journal, set()), None)
    holds = arm_holds(arms, args.runtime, R.open_record(R.from_json(json.dumps(first["record"])))._state.board) \
        if first is not None else {}
    done = run(args.journal, args.out, arms, entrant=args.entrant, where=args.where, runtime=args.runtime,
               workers=args.workers, header={"holds": holds, "seed_notes": notes})
    print(f"DONE redecide: {done} games", flush=True)


if __name__ == "__main__":
    main()
