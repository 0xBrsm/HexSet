# SPDX-License-Identifier: GPL-3.0-only
"""A project's seed policy: the seeds its runs may play and train on.

A run's games are a pure function of its seed and their indices
(`hexset.arena`), so two runs on one seed are paired game for game, and
two runs on different seeds are not paired at all. A project that compares
many runs can name a short list of seeds in a `seed-policy.json` and have
every run held to it:

    {"eval":  {"default": {"2": 2000, "*": 1000},
               "screen": [1000, 2000],
               "confirm": [3000, 4000]},
     "train": {"rl": [0, 1, 2], "bc": [10, 11, 12]},
     "analysis": [20, 21, 22]}

- `eval.default` is the seed a run takes when it names none, by the number
  of seats playing, `"*"` for any other count;
- `eval.screen` seeds may be named freely;
- `eval.confirm` seeds are for a test registered before its data: a run on
  one names its registration, a file that names the seed, and the file's
  digest is kept with the run;
- `train` lists the seeds each kind of training may use. A training seed is
  never an evaluation seed: a run scored on the games it trained on is not
  scored at all. A run that plays games to train or fit on (a dataset)
  names its kind as its `use` and is held to that list instead;
- `analysis` lists the seeds of randomness that plays no game: resampling,
  sampling records, a statistical null, a data split (`use="analysis"`).

Any other seed is refused. A run can still be made to play one by naming a
reason (`seed_bypass=`, `--bypass-seed-policy REASON`, or the
`HEXSET_SEED_BYPASS` environment variable for code that takes neither); the
reason is printed as a warning and kept in the run's journal.

The policy is found once per process: the file `HEXSET_SEED_POLICY` names,
else the nearest `seed-policy.json` above this package, else the nearest one
above the working directory (`find_policy`). A host
package can also install one (`register_seed_policy`), and can `require` one,
so that a tree staged without its policy file refuses to play rather than
playing unchecked. With no policy and no requirement nothing is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FILENAME", "SeedPolicy", "SeedRefused", "add_arguments", "check", "check_eval",
    "check_train", "default_eval", "default_seed", "find_policy", "policy",
    "hold", "register_seed_policy", "require", "resolve", "suspend",
]

FILENAME = "seed-policy.json"
POLICY_ENV = "HEXSET_SEED_POLICY"
BYPASS_ENV = "HEXSET_SEED_BYPASS"
REGISTRATION_ENV = "HEXSET_SEED_REGISTRATION"
#: `HEXSET_SEED_POLICY=off`: what `suspend` leaves for the processes a suite starts.
OFF = "off"


class SeedRefused(ValueError):
    """A seed the policy does not allow, named without a bypass reason."""


@dataclass(frozen=True)
class SeedPolicy:
    """A parsed `seed-policy.json`; `sha256` is the file's own digest."""

    path: str
    sha256: str
    default: dict[str, int]
    screen: tuple[int, ...]
    confirm: tuple[int, ...]
    train: dict[str, tuple[int, ...]] = field(default_factory=dict)
    analysis: tuple[int, ...] = ()

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "SeedPolicy":
        raw = Path(path).read_bytes()
        doc = json.loads(raw)
        ev = doc.get("eval") or {}
        default = {str(k): int(v) for k, v in (ev.get("default") or {}).items()}
        screen = tuple(int(s) for s in ev.get("screen") or ())
        confirm = tuple(int(s) for s in ev.get("confirm") or ())
        train = {str(k): tuple(int(s) for s in v) for k, v in (doc.get("train") or {}).items()}
        analysis = tuple(int(s) for s in doc.get("analysis") or ())
        policy = cls(str(Path(path).resolve()), hashlib.sha256(raw).hexdigest(),
                     default, screen, confirm, train, analysis)
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.default or "*" not in self.default:
            raise ValueError(f"{self.path}: eval.default needs a \"*\" entry")
        approved = set(self.screen) | set(self.confirm)
        if loose := sorted(set(self.default.values()) - set(self.screen)):
            raise ValueError(f"{self.path}: default seeds {loose} are not screen seeds")
        if both := sorted(set(self.screen) & set(self.confirm)):
            raise ValueError(f"{self.path}: seeds {both} are both screen and confirm")
        if "analysis" in self.train:
            raise ValueError(f"{self.path}: analysis seeds have their own list, not a training kind")
        if shared := sorted(set(self.analysis) & approved):
            raise ValueError(f"{self.path}: analysis seeds {shared} are also evaluation seeds")
        for kind, seeds in self.train.items():
            if shared := sorted(set(seeds) & approved):
                raise ValueError(f"{self.path}: {kind} training seeds {shared} are also evaluation seeds")

    def default_for(self, seats: int) -> int:
        return self.default.get(str(seats), self.default["*"])

    def kind(self, kind: str) -> tuple[int, ...] | None:
        """The seeds of a training kind, or of `"analysis"`; None for a kind
        the policy does not name."""
        return self.analysis or None if kind == "analysis" else self.train.get(kind)

    @property
    def training(self) -> set[int]:
        return {s for seeds in self.train.values() for s in seeds}


_state: dict = {"loaded": False, "policy": None, "required": None, "suspended": False, "warned": set()}


def find_policy(start: str | os.PathLike | None = None) -> Path | None:
    """The `seed-policy.json` in `start` or the nearest directory above it,
    or in a `hexset/` directory beside one (a checkout holding this package
    keeps it there), or None. `start` is this package's own directory by
    default."""
    here = Path(start or Path(__file__).resolve().parent).resolve()
    for directory in (here, *here.parents):
        for candidate in (directory / FILENAME, directory / "hexset" / FILENAME):
            if candidate.is_file():
                return candidate
    return None


def policy() -> SeedPolicy | None:
    """The policy in force, or None."""
    if _state["suspended"] or (os.environ.get(POLICY_ENV) == OFF and _state["policy"] is None):
        return None
    if not _state["loaded"]:
        named = os.environ.get(POLICY_ENV)
        path = Path(named) if named else (find_policy() or find_policy(Path.cwd()))
        if named and not path.is_file():
            raise FileNotFoundError(f"{POLICY_ENV}={named} names no file")
        if path is not None and _state["policy"] is None:
            _state["policy"] = SeedPolicy.from_file(path)
        _state["loaded"] = True
    return _state["policy"]


def register_seed_policy(source: "SeedPolicy | str | os.PathLike") -> SeedPolicy:
    """Install a policy for this process, from a file or a parsed one."""
    loaded = source if isinstance(source, SeedPolicy) else SeedPolicy.from_file(source)
    _state.update(policy=loaded, loaded=True, suspended=False)
    return loaded


def require(who: str) -> None:
    """Declare that `who` plays only under a policy: with none found, every
    checked run refuses to start instead of playing unchecked. Ignored once
    the policy is suspended."""
    if not _state["suspended"] and os.environ.get(POLICY_ENV) != OFF:
        _state["required"] = _state["required"] or who


def suspend() -> None:
    """Turn the policy off for the rest of the process: nothing is found or
    required after it, and only an explicit `register_seed_policy` turns one
    on again. For test suites, whose games are fixtures rather than runs;
    nothing else calls it. The processes it starts inherit it through
    `HEXSET_SEED_POLICY=off`."""
    _state.update(suspended=True, policy=None, loaded=True, required=None)
    os.environ[POLICY_ENV] = OFF


def default_eval(seats: int, fallback: int = 0) -> int:
    """The seed a run of `seats` playing seats takes when it names none."""
    current = policy()
    return fallback if current is None else current.default_for(seats)


def default_seed(use: str, seats: int, fallback: int = 0) -> int:
    """The seed a run takes when it names none: for `use="eval"` the
    policy's default for `seats` playing seats, for a training kind that
    kind's first seed; `fallback` with no policy."""
    current = policy()
    if current is None:
        return fallback
    if use == "eval":
        return current.default_for(seats)
    if not current.kind(use):
        raise SeedRefused(f"{current.path} names no {use!r} seeds")
    return current.kind(use)[0]


def check(
    seed: int, *, use: str = "eval", seats: int, bypass: str | None = None,
    registration: str | os.PathLike | None = None,
) -> dict:
    """`check_eval` for `use="eval"`, else `check_train` for that kind."""
    if use == "eval":
        return check_eval(seed, seats=seats, bypass=bypass, registration=registration)
    return check_train(seed, kind=use, bypass=bypass)


def _missing() -> None:
    if _state["required"] and not _state["suspended"] and os.environ.get(POLICY_ENV) != OFF:
        raise SeedRefused(
            f"{_state['required']} plays only under a seed policy, and none was found: no "
            f"{FILENAME} above {Path(__file__).resolve().parent} and no {POLICY_ENV}. Stage the "
            f"policy file with the code (it sits beside the package's source tree), or name it "
            f"in {POLICY_ENV}."
        )


def _warn(seed: int, reason: str, problem: str) -> None:
    key = (seed, reason, problem)
    if key in _state["warned"]:
        return
    _state["warned"].add(key)
    bar = "!" * 78
    print(f"\n{bar}\n!! SEED POLICY BYPASSED: seed {seed}\n!! {problem}\n!! reason given: {reason}\n"
          f"!! This run is NOT paired with runs on the approved seeds. The bypass is kept in its journal.\n"
          f"{bar}\n", file=sys.stderr, flush=True)


def _registration_note(seed: int, registration: str | os.PathLike) -> dict:
    path = Path(registration)
    if not path.is_file():
        raise SeedRefused(f"registration {path} is not a file")
    raw = path.read_bytes()
    text = raw.decode("utf-8", "replace").replace(",", "").replace("_", "")
    if not re.search(rf"(?<!\d){seed}(?!\d)", text):
        raise SeedRefused(
            f"registration {path} does not name seed {seed}: a confirmation seed is used by a test "
            f"registered for it, and the registration says which seed it runs on"
        )
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _refuse(seed: int, problem: str, allowed: str, bypass: str | None) -> dict:
    if not bypass:
        raise SeedRefused(
            f"seed {seed} refused: {problem}. {allowed + ' ' if allowed else ''}To play it anyway, name a reason with "
            f"--bypass-seed-policy REASON (seed_bypass= in code, {BYPASS_ENV} for code that takes "
            f"neither); the reason is kept with the run."
        )
    _warn(seed, bypass, problem)
    return {"status": "bypassed", "reason": bypass, "problem": problem}


def check_eval(
    seed: int, *, seats: int, bypass: str | None = None,
    registration: str | os.PathLike | None = None,
) -> dict:
    """Hold an evaluation's seed to the policy and return the note its
    journal keeps: the policy's digest and how the seed passed. Raises
    `SeedRefused` for a seed the policy does not allow, unless `bypass`
    (or `HEXSET_SEED_BYPASS`) names a reason."""
    bypass = bypass or os.environ.get(BYPASS_ENV) or None
    registration = registration or os.environ.get(REGISTRATION_ENV) or None
    current = policy()
    if current is None:
        _missing()
        return {"policy": None}
    note = {"policy": current.sha256}
    seed = int(seed)
    if seed in current.screen:
        return {**note, "status": "screen"}
    if seed in current.confirm:
        if registration:
            return {**note, "status": "confirm", "registration": _registration_note(seed, registration)}
        return {**note, **_refuse(seed, "a confirmation seed needs the run's registration (--registration PATH)",
                                  "", bypass)}
    problem = ("a training seed: these games are the ones training plays" if seed in current.training
               else "not an approved evaluation seed")
    allowed = (f"Screen on {', '.join(map(str, current.screen))} (the default for {seats} seats is "
               f"{current.default_for(seats)}); confirm a registered test on "
               f"{', '.join(map(str, current.confirm))}. Need more games? Play more indices on the same seed.")
    return {**note, **_refuse(seed, problem, allowed, bypass)}


def check_train(seed: int, *, kind: str, bypass: str | None = None) -> dict:
    """Hold a training run's seed to the policy's `train[kind]` list, or an
    analysis's to its `analysis` list (`kind="analysis"`)."""
    bypass = bypass or os.environ.get(BYPASS_ENV) or None
    current = policy()
    if current is None:
        _missing()
        return {"policy": None}
    note = {"policy": current.sha256}
    seeds = current.kind(kind)
    if seeds is None:
        raise SeedRefused(f"{current.path} names no {kind!r} seeds")
    word = "analysis" if kind == "analysis" else f"{kind} training"
    if int(seed) in seeds:
        return {**note, "status": "analysis" if kind == "analysis" else "train", "kind": kind}
    return {**note, **_refuse(int(seed), f"not an approved {word} seed",
                              f"Use {', '.join(map(str, seeds))}.", bypass)}


def add_arguments(parser: argparse.ArgumentParser, *, flag: str = "--seed", fallback: int = 0,
                  help: str = "the run's seed") -> None:
    """`flag` with no default (the policy's, by `resolve` or `hold`, else
    `fallback`), `--bypass-seed-policy REASON` and `--registration PATH`."""
    parser.add_argument(flag, type=int, default=None,
                        help=f"{help}; by default the seed policy's ({fallback} with no policy)")
    parser.add_argument("--bypass-seed-policy", metavar="REASON", default=None,
                        help="play a seed the seed policy refuses; the reason is kept in the journal")
    parser.add_argument("--registration", metavar="PATH", default=None,
                        help="the registration a confirmation seed's test was frozen in")


def resolve(
    args: argparse.Namespace, seats: int, *, dest: str = "seed", fallback: int = 0, use: str = "eval",
) -> int:
    """`args.<dest>` from `add_arguments`, or `default_seed(use, seats)` when
    it names none (`fallback` with no policy), written back so a journal
    named for the seed names the one played."""
    if getattr(args, dest) is None:
        setattr(args, dest, default_seed(use, seats, fallback))
    return getattr(args, dest)


def hold(
    args: argparse.Namespace, seats: int = 4, *, use: str = "eval", dest: str = "seed", fallback: int = 0,
) -> dict:
    """`resolve` then `check` in one: fill `args.<dest>` from the policy when
    it names none, hold it to the policy as a `use` seed with the arguments'
    `--bypass-seed-policy` and `--registration`, and return the note to keep
    with the run's output. For a tool that plays outside the arena (whose
    runs `compete` already holds) or seeds no game at all."""
    seed = resolve(args, seats, dest=dest, fallback=fallback, use=use)
    return check(seed, use=use, seats=seats, bypass=getattr(args, "bypass_seed_policy", None),
                 registration=getattr(args, "registration", None))
