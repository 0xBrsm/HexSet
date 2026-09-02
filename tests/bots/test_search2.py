# SPDX-License-Identifier: GPL-3.0-only
"""The behaviour-preservation gate for `hexset.bots.search2` (and the shared
`hexset.bots.evaluate`), written before `hexset/bots.py` and `hexset/evaluate.py`
move into the `hexset.bots` package -- see `test_heximax.py`'s own census gate,
which this mirrors. Written and greened against the pre-move tree so the move
itself is provably exact: any later diff that flips one of these hashes has
changed what `search2` chooses somewhere in that game, not merely moved code.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from hexset.actions import apply
from hexset.arena import PRESETS, Entrant, spawn
from hexset.board.board import random_base_board
from hexset.game import is_over, start, to_move

CENSUS_FIXTURE = Path(__file__).parent / "fixtures" / "search2_census.json"

# name -> (Entrant, seeds). `search2` is the registered preset; `search2-offers0`
# is not registered anywhere (today's `PRESETS` has no zero-offer search2
# entry) and is built here directly -- `spawn` works on any `Entrant`, and
# adding a preset to `hexset.arena.PRESETS` is no part of a pure move.
CENSUS_ENTRANTS: dict[str, Entrant] = {
    "search2": PRESETS["search2"],
    "search2-offers0": Entrant(
        "search2-offers0", kind="search", depth=2, width=6, max_offers=0
    ),
}
CENSUS_SEEDS = range(300, 320)


def _census_game(entrant: Entrant, seed: int, players: int = 4) -> str:
    """Play one seeded game, every seat the same entrant, and hash the choices.

    Same construction as `test_heximax._census_game`: one rng off `seed`
    builds the board and starts the game, each seat's bot gets its own rng
    deterministic per seat, and the hash covers the full `(seat, action)`
    trace rather than just the outcome.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    bots = [
        spawn(entrant, board, random.Random(f"{seed}:{seat}"))
        for seat in range(players)
    ]
    trace = []
    moves = 0
    while not is_over(game):
        seat = to_move(game)
        action = bots[seat].choose(game)
        trace.append(
            (
                seat,
                int(action.type),
                action.a,
                action.b,
                list(action.give),
                list(action.want),
                list(action.ask),
            )
        )
        apply(game, action)
        moves += 1
        if moves > 60000:
            raise AssertionError(f"{entrant.name} seed {seed} did not finish")
    return hashlib.sha256(repr(trace).encode()).hexdigest()


def test_choices_are_byte_identical_to_the_recorded_census(request):
    """Plays 20 seeded games each for `search2` and `search2-offers0` and
    hashes every game's full `(seat, action)` sequence, the same gate
    `test_heximax` runs for heximax. `--write-census` (registered once,
    repo-wide, in `tests/conftest.py`) regenerates the fixture."""
    computed = {
        name: {str(seed): _census_game(entrant, seed) for seed in CENSUS_SEEDS}
        for name, entrant in CENSUS_ENTRANTS.items()
    }
    if request.config.getoption("--write-census"):
        CENSUS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        CENSUS_FIXTURE.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n")

        pytest.skip(f"wrote {CENSUS_FIXTURE}")
    recorded = json.loads(CENSUS_FIXTURE.read_text())
    assert computed == recorded
