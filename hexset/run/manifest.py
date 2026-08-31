# SPDX-License-Identifier: GPL-3.0-only
"""A run is a directory with a frozen manifest, and the manifest is the input.

Every result this project has lost was lost the same way: the configuration
lived in a shell script under `tmp/`, which is gitignored, so the run that
produced a checkpoint could not be reconstructed from the repository. Three
concrete costs, each of them paid on this project:

- **Lineage had to be reverse-engineered.** Which checkpoint `vlam099`
  continued from was recovered by reading the first iteration number in its
  `log.jsonl` and matching it against other runs' last ones.
- **A run's SHA was not recorded**, so a result could not be tied to the code
  that produced it except by the date on the log file.
- **A parameter that selects the data was invisible.** `benchmarks.duel` seeds
  one shard per worker, so `--workers` decides which games get played; two runs
  at the same `--seed` and different worker counts play different games. One
  checkpoint read 44.0% and 40.0% for exactly that reason.

So: `hexset.run.init` writes a directory with `run.json` and a `config/`
holding **every** parameter explicitly, and the trainers read that and nothing
else. The freeze is what makes "no defaults, no fallbacks" true rather than
aspirational — `load` refuses a config whose keys are not exactly the
parameter set, so a flag added to a parser cannot silently be absent from a
manifest, and a manifest written before that flag existed fails loudly instead
of picking up a new default.

The parameter set is read from the mode's own `build_parser()` rather than
duplicated here, so this module cannot drift from the trainers it freezes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = 1

# A manifest is one of two things and they must not be confused.
#
# "run" is a freeze: every parameter resolved through the mode's parser, so it
# can be launched and will mean exactly what it meant the first time.
#
# "record" is a reconstruction of a run that predates this machinery. Its
# configuration is whatever the checkpoint recorded about itself, which is not
# the full parameter set -- old runs were launched before some flags existed,
# and inventing values for those from today's defaults would be a fabrication
# dressed as provenance. A record is therefore readable and NOT launchable, and
# `load` says so rather than letting one be run.
KIND_RUN = "run"
KIND_RECORD = "record"

# mode -> the module that launches it. Not derivable from the mode name:
# "distill" runs from `hexset.distill_train`, so an error message that
# interpolated the mode would name a module that does not exist.
MODULES = {"train": "hexset.train", "league": "hexset.league", "distill": "hexset.distill_train"}

# mode -> the parser whose dests define a complete config for that mode.
# Imported lazily: `hexset.train` pulls in torch, and this module is imported by
# tooling that runs where torch is not installed.
PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {}


def _parser_for(mode: str) -> argparse.ArgumentParser:
    if mode not in PARSERS:
        if mode == "train":
            from ..train import build_parser
        elif mode == "league":
            from ..league import build_parser
        elif mode == "distill":
            from ..distill_train import build_parser
        else:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of train, league, distill"
            )
        PARSERS[mode] = build_parser
    return PARSERS[mode]()


def parameters(mode: str) -> set[str]:
    """Every dest the mode's parser defines -- the exact keys a config carries."""
    return {a.dest for a in _parser_for(mode)._actions if a.dest not in ("help",)}


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ("git", *args), cwd=cwd, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(repo: Path) -> dict[str, Any]:
    """The commit a run was launched from, and whether the tree was dirty.

    A dirty tree means the result cannot be cited: there is no SHA that
    reproduces it. Recorded rather than refused, because a smoke run on a dirty
    tree is legitimate and only a *reported* number needs the guarantee.
    """
    commit = _git("rev-parse", "HEAD", cwd=repo)
    status = _git("status", "--porcelain", cwd=repo)
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo),
    }


@dataclass(frozen=True)
class Manifest:
    """`run.json` plus the frozen config it points at."""

    name: str
    mode: str
    directory: Path
    config: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def parent(self) -> str | None:
        """The checkpoint this run continued from, or None if from scratch."""
        return self.meta.get("parent")

    def namespace(self) -> argparse.Namespace:
        """The config as the trainers already expect to receive it.

        Deliberately not re-serialised into argv and re-parsed: `init` resolved
        these values through the parser already, so the types are the parser's
        own. Re-parsing would only add a place for a round trip to go wrong.
        """
        return argparse.Namespace(**self.config)


def _run_id(now: datetime, entropy: bytes | None = None) -> str:
    stamp = now.strftime("%Y%m%d")
    tail = (entropy or os.urandom(3)).hex()[:6]
    return f"{stamp}-{tail}"


def freeze(
    mode: str,
    name: str,
    directory: Path,
    argv: list[str],
    *,
    repo: Path,
    description: str = "",
    plan: str | None = None,
    parent: str | None = None,
    now: datetime | None = None,
) -> Manifest:
    """Resolve `argv` through the mode's parser and write the run directory.

    Resolution is the point. Whatever the caller passes, what lands on disk is
    every parameter with an explicit value, so training never consults a
    default and a later change to a default cannot silently alter what a
    recorded run meant.
    """
    parser = _parser_for(mode)
    resolved = vars(parser.parse_args(argv))
    expected = parameters(mode)
    missing = expected - set(resolved)
    if missing:
        raise ValueError(f"parser did not resolve {sorted(missing)}")

    now = now or datetime.now(timezone.utc)
    # BEFORE the directory exists. `provenance` reads `git status --porcelain`,
    # which counts untracked paths, and a run directory is untracked at the
    # moment it is created -- so calling this afterwards reported `git_dirty`
    # true for every run, whatever the tree actually looked like. That was
    # invisible while `.gitignore` still held a blanket `/runs/` rule (the new
    # directory was ignored, so the status came back clean); tracking run
    # records as the record they are turned the field into a constant. A signal
    # that is always true cannot say "this result cannot be cited".
    prov = provenance(repo)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config").mkdir(exist_ok=True)
    (directory / "config" / f"{mode}.json").write_text(
        json.dumps(resolved, indent=1, sort_keys=True, default=str) + "\n"
    )
    meta = {
        "schema": SCHEMA,
        "kind": KIND_RUN,
        "name": name,
        "mode": mode,
        "description": description,
        "plan": plan,
        "parent": parent,
        "config": {mode: f"config/{mode}.json"},
        "run_id": _run_id(now),
        "created": now.isoformat(),
        "argv": argv,
        **prov,
    }
    (directory / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return Manifest(name=name, mode=mode, directory=directory, config=resolved, meta=meta)


def load(directory: str | Path) -> Manifest:
    """Read a run directory, refusing anything that is not a complete freeze."""
    directory = Path(directory)
    run_json = directory / "run.json"
    if not run_json.exists():
        raise SystemExit(
            f"{directory} has no run.json. Runs are created by "
            f"`python -m hexset.run.init`; a bare checkpoint directory is not a run."
        )
    meta = json.loads(run_json.read_text())
    if meta.get("schema") != SCHEMA:
        raise SystemExit(
            f"{run_json} is schema {meta.get('schema')!r}, this build reads {SCHEMA}"
        )
    kind = meta.get("kind", KIND_RUN)
    if kind == KIND_RECORD:
        raise SystemExit(
            f"{directory} is a reconstructed record, not a launchable run: its "
            f"configuration is what the checkpoint recorded about itself, which "
            f"is not the full parameter set. Read it with `hexset.run.read_record`; "
            f"to continue this line, freeze a new run with `hexset.run.init "
            f"--parent {meta.get('endpoint') or directory}`."
        )
    if kind != KIND_RUN:
        raise SystemExit(f"{directory} has unknown kind {kind!r}")
    mode = meta["mode"]
    relative = meta["config"][mode]
    config = json.loads((directory / relative).read_text())

    expected = parameters(mode)
    got = set(config)
    if missing := expected - got:
        raise SystemExit(
            f"{directory / relative} is missing {sorted(missing)}. A frozen config "
            f"carries every parameter explicitly -- this manifest predates those "
            f"flags, and filling them from today's defaults would silently change "
            f"what the run means. Re-freeze it with `hexset.run.init` instead."
        )
    if extra := got - expected:
        raise SystemExit(
            f"{directory / relative} carries {sorted(extra)}, which this build's "
            f"{mode} parser does not define. The flags were removed or renamed."
        )
    return Manifest(
        name=meta["name"], mode=mode, directory=directory, config=config, meta=meta
    )


def record(
    name: str,
    directory: Path,
    *,
    mode: str,
    stored: dict[str, Any],
    config: dict[str, Any],
    parent: str | None,
    endpoint: str | None,
    iterations: int | None,
    description: str = "",
    parent_source: str = "derived",
    parent_derived: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write a reconstructed manifest for a run that predates the freeze.

    `stored` and `config` are copied verbatim out of the run's own checkpoint —
    `hexset.train` already writes both into every `.pt` — so this is a reading of
    the artefact rather than a guess about it. What cannot be recovered is
    stated as null rather than filled in: a record with `git_commit: null` is
    telling the truth about a run launched from a gitignored script.

    Lands in `stored/` rather than `config/` so the directory layout itself says
    which kind it is, and so no tool can mistake one for a frozen config.
    """
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stored").mkdir(exist_ok=True)
    (directory / "stored" / "args.json").write_text(
        json.dumps(stored, indent=1, sort_keys=True, default=str) + "\n"
    )
    (directory / "stored" / "ppo_config.json").write_text(
        json.dumps(config, indent=1, sort_keys=True, default=str) + "\n"
    )
    meta = {
        "schema": SCHEMA,
        "kind": KIND_RECORD,
        "name": name,
        "mode": mode,
        "description": description,
        "parent": parent,
        # How the parent was established. "derived" means the artefacts say so;
        # "ledger" means the written record does and the artefacts cannot -- a
        # child that forked from a checkpoint its parent later wrote past leaves
        # nothing on disk to match. `parent_derived` keeps what the rule
        # concluded so a future disagreement is visible rather than overwritten.
        "parent_source": parent_source,
        "parent_derived": parent_derived,
        "endpoint": endpoint,
        "iterations": iterations,
        "stored": {"args": "stored/args.json", "config": "stored/ppo_config.json"},
        "reconstructed": now.isoformat(),
        "reconstructed_from": "the run's own checkpoint",
        "git_commit": None,
        "git_dirty": None,
        "note": (
            "Launched before hexset.run existed, from a script under gitignored "
            "tmp/. The SHA is unrecoverable. Not launchable -- see manifest.KIND_RECORD."
        ),
    }
    (directory / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_record(directory: str | Path) -> dict[str, Any]:
    """A record's metadata plus the checkpoint values it was rebuilt from."""
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    if meta.get("kind") != KIND_RECORD:
        raise SystemExit(f"{directory} is a {meta.get('kind')!r}, not a record")
    return {
        **meta,
        "args": json.loads((directory / meta["stored"]["args"]).read_text()),
        "config": json.loads((directory / meta["stored"]["config"]).read_text()),
    }
