# SPDX-License-Identifier: GPL-3.0-only
"""The behaviour-preservation gate for `hexset.bots.search2` (and the shared
`hexset.bots.evaluate`) -- see `test_heximax.py`'s own census gate, which
this mirrors. Any diff that flips one of these hashes has changed what
`search2` chooses somewhere in that game, not merely moved code.

**Re-baselined deliberately for the one-event trade mechanic.** Both arms
changed, including the no-trade one, and the reason is worth recording
because the registration predicted otherwise: `search2-notrade` suppressed
*proposing*, but `SearchBot._value` enumerated the engine's offer sample
inside the tree regardless, so the no-trade referent still searched
hypothetical offers. Stubbing `actions._offer_actions` to `[]` on the old
tree reproduces every hash below exactly, which is the attribution: the
offer sample's removal is the whole of the difference, and nothing else in
the mechanic moves a no-trade game.
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
from hexset.trading import publish_valuation

CENSUS_FIXTURE = Path(__file__).parent / "fixtures" / "search2_census.json"

# name -> Entrant. Both are registered presets now: `search2-notrade` is the
# no-trade referent the strength gates duel against, so it belongs in
# `hexset.arena.PRESETS` rather than being built here.
CENSUS_ENTRANTS: dict[str, Entrant] = {
    "search2": PRESETS["search2"],
    "search2-notrade": PRESETS["search2-notrade"],
}
CENSUS_SEEDS = range(300, 320)


def _census_game(entrant: Entrant, seed: int, players: int = 4) -> str:
    """Play one seeded game, every seat the same entrant, and hash the choices.

    Same construction as `test_heximax._census_game`: one rng off `seed`
    builds the board and starts the game, each seat's bot gets its own rng
    deterministic per seat, and the hash covers the full `(seat, action)`
    trace rather than just the outcome. The bots are seated as the game's
    `gates` too, and each publishes right after its own action, exactly as
    `arena.play` does both, so the census covers what they trade as well as
    what they choose.
    """
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    bots = [
        spawn(entrant, board, random.Random(f"{seed}:{seat}"))
        for seat in range(players)
    ]
    game.gates = tuple(bots)
    trace = []
    moves = 0
    while not is_over(game):
        seat = to_move(game)
        action = bots[seat].choose(game)
        cleared = len(game.trades)
        apply(game, action)
        publish_valuation(game, seat, bots[seat])
        trace.append(
            (
                seat,
                int(action.type),
                action.a,
                action.b,
                [(t.a, t.b, t.received) for t in game.trades[cleared:]],
            )
        )
        moves += 1
        if moves > 60000:
            raise AssertionError(f"{entrant.name} seed {seed} did not finish")
    return hashlib.sha256(repr(trace).encode()).hexdigest()


def test_choices_are_byte_identical_to_the_recorded_census(request):
    """Plays 20 seeded games each for `search2` and `search2-notrade` and
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
