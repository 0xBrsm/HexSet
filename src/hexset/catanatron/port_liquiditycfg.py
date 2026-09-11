"""Explicit, opt-in six-arm port-liquidity screen plan.

Importing this module registers the candidate entrant kinds through
``hexset.bots.heximax.port_liquidity``. It does not run games or alter the
ordinary ``heximax-notrade`` preset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import hexset.bots
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.bots.heximax.port_liquidity import ARM_LABELS, arm_manifest, entrant_for

GATES = ("ab2", "shipped")
SCREEN_GAMES = 240
SCREEN_WORKERS = 30
# Reserved future family; this module only plans jobs and never launches one.
SCREEN_SEEDS = {"ab2": 600_000_000, "shipped": 600_100_000}


def _name(arm: str) -> str:
    return "heximax-notrade" if arm == "control" else entrant_for(arm).name


def lineup(arm: str, gate: str) -> str:
    if arm not in ARM_LABELS or gate not in GATES:
        raise ValueError("unknown arm or gate")
    name = _name(arm)
    opponents = (
        "AB:2,AB:2,AB:2" if gate == "ab2" else
        "DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"
    )
    return f"DC:{name},{opponents}"


def source_fingerprint() -> str:
    here = Path(__file__).resolve()
    root = here.parents[1]
    files = (
        here, root / "arena.py", here.with_name("duel.py"),
        root / "bots" / "heximax" / "port_liquidity.py",
        here.with_name("port_liquidity_runner.py"),
        root / "bots" / "heximax" / "evaluate.py",
        root / "bots" / "heximax" / "search.py",
        root / "bots" / "heximax" / "presets.py",
        root / "state.py", root / "board" / "ports.py",
        here.with_name("player.py"), here.with_name("state.py"),
        here.with_name("actions.py"), here.with_name("bot.py"),
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def screen_jobs() -> list[dict]:
    jobs = []
    for arm_index, arm in enumerate(ARM_LABELS):
        for gate in GATES:
            jobs.append({
                "arm": arm,
                "role": "control" if arm == "control" else "candidate",
                "gate": gate, "games": SCREEN_GAMES, "workers": SCREEN_WORKERS,
                "seed": SCREEN_SEEDS[gate],
                "players": lineup(arm, gate),
                "phenotype": arm_manifest(arm),
            })
    return jobs


def manifest() -> dict:
    return {
        "schema": 1, "family": "heximax-port-liquidity-future",
        "source_hash": source_fingerprint(), "arms": list(ARM_LABELS),
        "gates": {
            "ab2": "DC:<arm>,AB:2,AB:2,AB:2",
            "shipped": "DC:<arm>,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade",
        },
        "screen": {
            "games": SCREEN_GAMES, "workers": SCREEN_WORKERS,
            "seeds": dict(SCREEN_SEEDS), "jobs": screen_jobs(),
            "controls_never_promote": True,
        },
        "weights": asdict(NO_TRADE_WEIGHTS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan:
        parser.error("this planning module only supports --plan")
    print(json.dumps(manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
