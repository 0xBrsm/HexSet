# SPDX-License-Identifier: GPL-3.0-only
"""A catanatron `Player` sitting at a HexSet table -- `hexset.catanatron.bot`.

The direction the rest of this suite covers reads a live catanatron game;
this one mirrors a live HexSet game back into catanatron, so the guard has to
be that the mirror is the *same position*: the board translation run backwards
and forwards must land on the board it started from, and a position mirrored
into catanatron and read back out through `state.translate` must agree with
the original on everything both engines represent.
"""

from __future__ import annotations

import random

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

import hexset.bots  # noqa: F401 -- registers the "heximax" presets
from hexset.actions import ActionType, apply, legal_actions
from hexset.arena import entrant_from_name, spawn
from hexset.board.board import random_base_board
from hexset.game import Phase, is_over, start, to_move
from hexset.victory import victory_points

from catanatron.models.player import Color

from hexset.catanatron.board import catanatron_map, translate_board
from hexset.catanatron.bot import CatanatronBot
from hexset.catanatron.state import seating, to_catanatron, translate


def _positions(seeds, seats=4):
    """Real positions from heximax self-play, sampled at every decision."""
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        board = random_base_board(rng)
        game = start(board, seats, rng)
        bots = [
            spawn(entrant_from_name("heximax"), board, random.Random(seed * 10 + i))
            for i in range(seats)
        ]
        depth = random.Random(seed).randrange(20, 140)
        for _ in range(depth):
            if is_over(game):
                break
            out.append(game)
            apply(game, bots[to_move(game)].choose(game))
    return out


@pytest.fixture(scope="module")
def positions():
    return _positions(range(3))


def test_the_map_is_the_boards_own_translation_run_backwards():
    """`catanatron_map` is `translate_board`'s inverse, ports included.

    Ports are the half that could plausibly be wrong and still look right:
    HexSet spaces them evenly around the coast rather than at catanatron's
    official positions, so every one of them is re-seated on a coastal edge
    the template has no port at.
    """
    for seed in range(3):
        board = random_base_board(random.Random(seed))
        mirrored = translate_board(catanatron_map(board)).board
        assert mirrored.topology == board.topology
        assert mirrored.terrain == board.terrain
        assert mirrored.tokens == board.tokens
        # `vertices` is an unordered pair on either side; everything else is
        # the port itself.
        assert {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in mirrored.ports
        } == {
            (p.edge, p.resource, p.ratio, frozenset(p.vertices)) for p in board.ports
        }


def test_a_position_survives_the_round_trip(positions):
    """Mirrored into catanatron and read back out, a position is unchanged.

    The one field that cannot round-trip exactly is the split between matured
    and freshly-bought development cards: catanatron records maturity as one
    boolean per card type where HexSet counts copies, so only the per-type
    *total* is preserved (see `state.to_catanatron`). Everything else is
    compared as-is.
    """
    assert len(positions) > 50
    for game in positions:
        state = game.state(0, hidden=False)
        seats = seating(tuple(list(Color)[: state.num_players]))
        mapping = translate_board(catanatron_map(state.board))
        mirror = to_catanatron(game, mapping, seats)
        back, back_seats = translate(mirror, mapping, random.Random(0))
        again = back.state(0, hidden=False)

        assert back_seats == seats
        assert again.hands == state.hands
        assert again.bank == state.bank
        assert again.vertex_owner == state.vertex_owner
        assert again.vertex_building == state.vertex_building
        assert again.edge_owner == state.edge_owner
        assert again.robber == state.robber
        assert again.deck == state.deck
        assert again.knights_played == state.knights_played
        assert again.longest_road_holder == state.longest_road_holder
        assert again.largest_army_holder == state.largest_army_holder
        assert [
            [held + fresh for held, fresh in zip(*seat)]
            for seat in zip(again.dev_cards, again.new_dev_cards)
        ] == [
            [held + fresh for held, fresh in zip(*seat)]
            for seat in zip(state.dev_cards, state.new_dev_cards)
        ]
        assert back.current_player == game.current_player
        assert back.phase is game.phase
        assert back.turns == game.turns
        assert back.discard_quota == game.discard_quota
        assert back.free_roads == game.free_roads
        assert back.dev_card_played == game.dev_card_played
        if game.phase is Phase.SETUP_ROAD:
            assert back.last_settlement == game.last_settlement


def test_every_offered_action_maps_back_to_exactly_one_of_ours(positions):
    """The offer is a bijection between the two engines' action sets.

    `CatanatronBot` never hands catanatron's own `playable_actions` to the
    player: it offers HexSet's legal actions translated forwards, so the
    inverse is a dict lookup. What this pins is that the translation forwards
    is injective over that legal set -- two of our actions collapsing onto one
    of catanatron's would silently drop a move -- with exactly one documented
    exception, `PLAY_KNIGHT`, which catanatron asks as two decisions and so
    offers as one `PLAY_KNIGHT_CARD` however many robber targets we have.
    """
    bot = CatanatronBot()
    checked = 0
    for game in positions:
        state = game.state(0, hidden=False)
        bot._mapping = translate_board(catanatron_map(state.board))
        bot._seats = seating(tuple(list(Color)[: state.num_players]))
        mirror = to_catanatron(game, bot._mapping, bot._seats)
        raw = list(mirror.playable_actions)
        offered = bot._offer(game, mirror)

        ours = legal_actions(game)
        assert len(set(offered.values())) == len(offered), "two keys, one of our actions"
        for their, our in offered.items():
            assert their in raw, "offered an action catanatron never had"
            assert our in ours

        # The exception list, in full: the knights past the first, which all
        # collapse onto the single `PLAY_KNIGHT_CARD` catanatron asks first.
        knights = [a for a in ours if a.type is ActionType.PLAY_KNIGHT]
        dropped = [a for a in ours if a not in offered.values()]
        assert dropped == knights[1:] or (not knights and not dropped)
        checked += 1
    assert checked > 50


def test_a_knight_is_reassembled_from_catanatrons_two_decisions():
    """The one case where one HexSet action is two catanatron ones.

    Reached by handing the seat a matured knight in `Phase.ROLL` -- the phase
    where HexSet offers the roll and the knight and nothing else, and
    catanatron's single `PLAY_TURN` prompt would offer every dev card.
    """
    from hexset.cards import DevCard

    rng = random.Random(3)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    bots = [spawn(entrant_from_name("heximax"), board, random.Random(i)) for i in range(4)]
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bots[to_move(game)].choose(game))
    state = game.state(0, hidden=False)
    state.dev_cards[game.current_player][DevCard.KNIGHT] = 1
    assert game.phase is Phase.ROLL

    action = CatanatronBot().choose(game)
    assert action.type is ActionType.PLAY_KNIGHT
    assert action in legal_actions(game)
    assert action.a != state.robber


@pytest.mark.slow
def test_a_catanatron_seat_plays_out_full_games():
    """A couple of four-seat games, `catanatron` against three `heximax`.

    A first read on the seat, not a bar: what it has to do here is finish
    eight whole games without a translation failing anywhere in them.
    """
    import hexset.catanatron.bot  # noqa: F401 -- registers the preset

    wins = 0
    games = 2
    for seed in range(games):
        rng = random.Random(1000 + seed)
        board = random_base_board(rng)
        game = start(board, 4, rng)
        seat = seed % 4
        bots = [
            spawn(
                entrant_from_name("catanatron" if i == seat else "heximax"),
                board,
                random.Random(seed * 4 + i),
            )
            for i in range(4)
        ]
        while not is_over(game):
            apply(game, bots[to_move(game)].choose(game))
        points = [victory_points(game.state(0, hidden=False), p) for p in range(4)]
        wins += points.index(max(points)) == seat
    print(f"\ncatanatron won {wins}/{games} against three heximax")
    assert 0 <= wins <= games


# --- Seated at the served table -----------------------------------------------


def test_the_picker_offers_catanatron_and_a_seat_takes_it():
    """`/api/models` and `POST /api/bot`, the two the browser actually uses.

    The order is the picker's order: `heximax` first (the default opponent),
    then `catanatron`, then `search2`, then whatever checkpoints are on disk.
    """
    from conftest import new_tables

    registry = new_tables()
    models = registry.handle("GET", "/api/models", {}, None)["models"]
    assert models[:3] == ["heximax", "catanatron", "search2"]

    data = registry.handle("POST", "/api/games", {"bots": []}, None)
    code, token = data["code"], data["token"]
    table = registry.get(code)
    open_seats = [i for i, s in enumerate(table.seats) if s.kind.name == "EMPTY"]
    seated = registry.handle(
        "POST", "/api/bot", {"seat": open_seats[0], "model": "catanatron"}, token
    )
    assert seated["seats"][open_seats[0]]["name"] == "catanatron"
