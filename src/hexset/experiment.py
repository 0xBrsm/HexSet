# SPDX-License-Identifier: GPL-3.0-only
"""Versioned experiment documents with settings and raw arena outcomes.

These documents describe comparisons; hexset.record stores the action/chance
sequences needed to replay individual games. Keep both for an experiment that
must support replay as well as independent statistical analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import __version__
from ._source import git_value
from .arena import MAX_ACTIONS, Entrant, Tournament

SCHEMA = "hexset.experiment"
SCHEMA_VERSION = 1


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("experiment mappings require string keys")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"experiment settings cannot describe {type(value).__name__}; use data, not a runtime object")


def provenance(entrants: Sequence[Entrant] = ()) -> dict:
    """Identify the source checkout and installed runtime versions.

    A dirty checkout is explicitly marked; its commit alone cannot reproduce
    the run. Package installs without Git retain their package version.
    """
    def git(*args):
        return git_value(__file__, *args)

    dependencies = {}
    revisions = {}
    for name in ("numpy", "onnxruntime", "torch", "catanatron", "gymnasium", "pettingzoo"):
        try:
            distribution = metadata.distribution(name)
            dependencies[name] = distribution.version
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            revision = direct.get("vcs_info", {}).get("commit_id")
            if revision:
                revisions[name] = revision
        except metadata.PackageNotFoundError:
            pass
    changes = git("status", "--porcelain")
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hexset": __version__,
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if changes is None else bool(changes),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "dependency_revisions": revisions,
        "execution_environment": {
            name: os.environ.get(name)
            for name in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "checkpoints": _checkpoints(entrants),
    }


def _checkpoints(entrants: Sequence[Entrant]) -> dict:
    checkpoints = {}
    for entrant in entrants:
        if entrant.kind not in ("network", "mcts") or not isinstance(entrant.weights, (str, Path)):
            continue
        path = Path(entrant.weights)
        key = str(path)
        if key in checkpoints:
            continue
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            checkpoints[key] = {"sha256": digest, "bytes": path.stat().st_size}
        else:
            checkpoints[key] = {"sha256": None, "bytes": None}
    return checkpoints


def result_document(
    tournament: Tournament,
    entrants: Sequence[Entrant],
    *,
    seed: int,
    workers: int = 1,
    action_cap: int = MAX_ACTIONS,
    antithetic: bool = True,
    run_provenance: dict | None = None,
) -> dict:
    """Describe an arena result without discarding per-game observations.

    Winner and point-vector indices refer to the supplied entrant order;
    ``seating[e]`` gives entrant e's board seat. ``board_index`` identifies
    the shared board/random-stream unit used by antithetic pairs.
    Capture ``provenance(entrants)`` before execution and pass it as run_provenance
    when the document is assembled afterward.
    """
    rows = (tournament.winners, tournament.points, tournament.turns, tournament.seating)
    if any(len(row) != tournament.games for row in rows):
        raise ValueError("experiment requires winner, points, turns and seating for every game")
    settings = _plain({
        "entrants": entrants, "games": tournament.games, "seed": seed,
        "workers": workers, "action_cap": action_cap, "antithetic": antithetic,
    })
    current_checkpoints = _checkpoints(entrants)
    before = None if run_provenance is None else run_provenance.get("checkpoints")
    captured_before = before is not None and set(before) == set(current_checkpoints)
    checkpoints = before if captured_before else current_checkpoints
    outcomes = []
    for i in range(tournament.games):
        row = {
            "index": i, "board_index": i // 2 if antithetic else i,
            "winner": tournament.winners[i], "points": tournament.points[i],
            "turns": tournament.turns[i], "seating": tournament.seating[i],
        }
        for name in ("roads", "settlements", "cities"):
            values = getattr(tournament, name)
            if values:
                if len(values) != tournament.games:
                    raise ValueError(f"{name} must contain one row per game")
                row[name] = values[i]
        outcomes.append(row)
    document = _plain({
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "provenance": provenance() if run_provenance is None else run_provenance,
        "checkpoints": checkpoints,
        "checkpoint_capture": "before_run" if captured_before else "after_run",
        "checkpoints_changed": before != current_checkpoints if captured_before else None,
        "summary": {"games": tournament.games, "unfinished": tournament.unfinished,
                    "seconds": tournament.seconds, "standings": tournament.standings},
        "outcomes": outcomes,
    })
    # Reject NaN/Infinity rather than emitting nonstandard JSON for research tools.
    json.dumps(document, allow_nan=False)
    _validate(document)
    return document


def write(path: str | Path, document: dict) -> None:
    """Write one experiment document as strict JSON."""
    _validate(document)
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")


def read(path: str | Path) -> dict:
    """Read a document, rejecting incompatible schema versions."""
    document = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    _validate(document)
    return document


def _check_version(document: dict) -> None:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA} schema version {SCHEMA_VERSION}")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _validate(document: dict) -> None:
    """Validate the index relationships required to interpret raw outcomes."""
    _check_version(document)
    try:
        settings, summary, outcomes = (document[k] for k in ("settings", "summary", "outcomes"))
        entrants, games = settings["entrants"], settings["games"]
        n = len(entrants)
        if not 2 <= n <= 6 or type(games) is not int or games <= 0:
            raise ValueError("experiment requires 2–6 entrants and a positive game count")
        if len(outcomes) != games or summary["games"] != games:
            raise ValueError("experiment game counts disagree")
        if [s["name"] for s in summary["standings"]] != [e["name"] for e in entrants]:
            raise ValueError("standings must match the supplied entrant order")
        wins = [0] * n
        unfinished = 0
        for i, row in enumerate(outcomes):
            if row["index"] != i or row["board_index"] != (i // 2 if settings["antithetic"] else i):
                raise ValueError("outcome index or board pairing disagrees with settings")
            seating = row["seating"]
            if any(type(seat) is not int for seat in seating) or sorted(seating) != list(range(n)):
                raise ValueError("seating must be a permutation of entrant seat indices")
            for key in ("points", "roads", "settlements", "cities"):
                if key in row and (len(row[key]) != n or any(type(v) is not int or v < 0 for v in row[key])):
                    raise ValueError(f"{key} must contain one nonnegative integer per entrant")
            if "points" not in row or type(row["turns"]) is not int or row["turns"] < 0:
                raise ValueError("outcomes require points and a nonnegative turn count")
            winner = row["winner"]
            if winner is None:
                unfinished += 1
            elif type(winner) is int and 0 <= winner < n:
                wins[winner] += 1
            else:
                raise ValueError("winner must be an entrant index or null")
        if summary["unfinished"] != unfinished or [s["wins"] for s in summary["standings"]] != wins:
            raise ValueError("summary does not match raw game outcomes")
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete or malformed experiment document") from exc
