# SPDX-License-Identifier: GPL-3.0-only
"""`Game.state(seat, *, hidden=True)`: the engine's information-set access
path (P0, `agents/reference/trading-design.md`, "Registration -- the
one-event trade mechanic").

`hidden=True` (the default) returns the seat's `View` (`hexset.view`,
moved here from `hexset.bots.heximax.belief.Belief` -- `Belief` is kept as
an alias); `hidden=False` returns the true `GameState`, the same object
every time, and is the only sanctioned way to read it from outside the
engine. This is a pure move: the census in `tests/bots/heximax` and
`tests/bots/test_search2.py` is the exactness guard for the bots that
actually depend on this path; this file pins the access path itself.
"""

from __future__ import annotations

import random
from pathlib import Path

from hexset.actions import apply, legal_actions
from hexset.board.board import random_base_board
from hexset.game import Phase, imagine, start
from hexset.ledger import SeatLedger
from hexset.view import View

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Only engine-internal modules may read `Game`'s private field directly.
# These four directories hold everything "outside the engine" per the
# registration's split -- bots, bench, server and clients all read the true
# state, when they genuinely need it, through `game.state(seat,
# hidden=False)` instead.
_OUTSIDE_ENGINE_DIRS = ("hexset/bots", "hexset/bench", "hexset/server", "hexset/clients")


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def after_setup(seed: int = 0, players: int = 4):
    game = a_game(players, seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def _set_known_hand(game, player: int, counts: list[int]) -> None:
    """Give `player` exactly `counts` from the bank, keeping `game.ledger`
    in sync -- mirrors `test_ledger_engine._set_known_hand`."""
    state = game._state
    for r, n in enumerate(state.hands[player]):
        if n:
            state.bank[r] += n
            state.hands[player][r] = 0
    game.ledger.seats[player] = SeatLedger()
    for r, n in enumerate(counts):
        if n:
            state.bank[r] -= n
            state.hands[player][r] += n
            game.ledger.receive(player, r, n)


# -- (a) game.state(seat) is a View matching the ledger and the truth -------


def test_game_state_hidden_returns_a_view():
    game = after_setup()
    view = game.state(0)
    assert isinstance(view, View)


def test_the_views_known_and_unknown_match_the_ledger():
    game = after_setup()
    perspective = 0
    other = 1
    _set_known_hand(game, other, [2, 0, 1, 0, 0])
    view = game.state(perspective)
    seat_ledger = game.ledger.seats[other]
    assert view.known[other] == seat_ledger.known
    assert view.unknown[other] == seat_ledger.unknown


def test_the_views_hand_sizes_match_the_truth_for_every_seat():
    game = after_setup()
    perspective = 0
    view = game.state(perspective)
    for seat in range(game._state.num_players):
        true_size = sum(game._state.hands[seat])
        view_size = sum(view.known[seat]) + view.unknown[seat]
        assert view_size == true_size, f"seat {seat}: {view_size} != {true_size}"


def test_the_perspective_seats_own_hand_is_exact():
    game = after_setup()
    perspective = 2
    view = game.state(perspective)
    assert view.known[perspective] == game._state.hands[perspective]
    assert view.unknown[perspective] == 0


# -- (b) game.state(seat, hidden=False) is the raw state, identity-stable ---


def test_game_state_not_hidden_is_the_raw_gamestate():
    game = after_setup()
    assert game.state(0, hidden=False) is game._state


def test_game_state_not_hidden_is_identical_across_calls():
    game = after_setup()
    first = game.state(1, hidden=False)
    second = game.state(3, hidden=False)
    assert first is second
    assert first is game._state


def test_game_state_not_hidden_reflects_a_mutation_through_the_reference():
    """`hidden=False` never copies, so mutating what it returns mutates the
    game itself -- the property `arena.py`'s and heximax's imagined
    children (and the search's own PIMC swap, `Game.set_state`) depend on."""
    game = after_setup()
    state = game.state(0, hidden=False)
    state.hands[0][0] += 7
    assert game._state.hands[0][0] == state.hands[0][0]


def test_a_views_own_state_attribute_is_the_same_true_state():
    """A `View`'s own `.state` (its constructor argument, unrelated to
    `Game.state`'s new method) is the same object `hidden=False` returns --
    unchanged by this move, since honesty was always enforced by what
    `View` exposes (`known`/`unknown`/`expected_hand`), never by denying
    the object to whoever built the view."""
    game = after_setup()
    view = game.state(0)
    assert view.state is game.state(0, hidden=False)


# -- (c) no `._state` outside the engine ------------------------------------


def test_no_bot_bench_server_or_client_module_reaches_into_game_state():
    offenders = []
    for rel in _OUTSIDE_ENGINE_DIRS:
        directory = SRC / rel
        assert directory.is_dir(), f"expected a directory at {directory}"
        for path in directory.rglob("*.py"):
            text = path.read_text()
            if "._state" in text:
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "these files read Game's private state directly instead of going "
        f"through game.state(seat, hidden=...): {offenders}"
    )


# -- (d) imagine carries the view semantics ----------------------------------


def test_a_view_built_on_an_imagined_copy_diverges_after_a_mutation():
    game = after_setup()
    perspective = 0
    other = 1
    _set_known_hand(game, other, [1, 0, 0, 0, 0])

    child = imagine(game, random.Random(0))
    # Mutate the child's true hand and its ledger record for `other`,
    # independently of the parent -- `imagine` copies both.
    child._state.bank[1] += child._state.hands[other][1]
    child._state.hands[other][1] = 0
    child._state.bank[2] -= 1
    child._state.hands[other][2] += 1
    child.ledger.seats[other] = SeatLedger(known=[0, 0, 1, 0, 0], unknown=0)

    parent_view = game.state(perspective)
    child_view = child.state(perspective)

    assert parent_view.known[other] == [1, 0, 0, 0, 0]
    assert child_view.known[other] == [0, 0, 1, 0, 0]
    assert game._state.hands[other] != child._state.hands[other]
