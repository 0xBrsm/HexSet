# SPDX-License-Identifier: GPL-3.0-only
"""`robber_hits`: a checkpoint that declares it puts the robber only on a hex
with another seat's building and none of its own, wherever one is legal --
one forward, by rollouts or by tree search, from the file's metadata. It is
never forced onto its own hexes."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from hexset.actions import Action, ActionSpace, ActionType, build_space, options_for
from hexset.clients._modelmeta import robber_hits_config, robber_hits_of
from hexset.clients.netbot import NetworkBot, bot_for, hit_robber, rollouts_for, searcher_for
from hexset.robber import occupants
from helpers import clear_hand
from tests.test_deals import a_robber_move

ROB = ActionType.MOVE_ROBBER
PLAYERS = 4
NOBODY = PLAYERS
VALUE = (0.4, 0.2, 0.2, 0.2)


def hits(game, option: Action) -> bool:
    return any(p != 0 for p in occupants(game._state, option.a))


def clean(game, option: Action) -> bool:
    return hits(game, option) and 0 not in occupants(game._state, option.a)


@dataclass
class EmptyHexPolicy:
    """A prior that favours robber moves robbing nobody, uniform elsewhere;
    `VALUE` everywhere; the action the prior's argmax."""

    space: ActionSpace

    def _prior(self, options):
        raw = [(1.0 if o.b == NOBODY else 0.01) if o.type is ROB else 1.0 for o in options]
        total = sum(raw)
        return [w / total for w in raw]

    def act_rows(self, rows):
        return [max(options, key=lambda o, p=self._prior(options): p[options.index(o)])
                for _, _, options in rows]

    def value_rows(self, rows):
        return [VALUE for _ in rows]

    def score_rows(self, rows):
        return [(self._prior(options), VALUE) for _, _, options in rows]


@dataclass(frozen=True)
class Checkpoint:
    policy: EmptyHexPolicy
    space: ActionSpace
    players: int = PLAYERS
    robber_hits: bool = False


def checkpoint(game, on: bool) -> Checkpoint:
    topology = game._state.board.topology
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, PLAYERS)
    return Checkpoint(EmptyHexPolicy(space), space, robber_hits=on)


@pytest.fixture
def game():
    game = a_robber_move()
    options = options_for(game)
    assert any(hits(game, o) for o in options) and any(not hits(game, o) for o in options)
    return game


def test_only_moves_onto_another_seats_hex_without_the_movers_are_left(game):
    options = tuple(options_for(game))
    kept = hit_robber(game, options)
    assert kept == tuple(o for o in options if clean(game, o))


def shared(game, option: Action) -> bool:
    return hits(game, option) and 0 in occupants(game._state, option.a)


def test_a_hex_shared_with_the_mover_goes_beside_one_it_is_not_on(game):
    options = tuple(options_for(game))
    assert any(shared(game, o) for o in options)
    assert not any(shared(game, o) for o in hit_robber(game, options))


def test_where_every_hex_another_seat_is_on_is_the_movers_too_every_move_stands(game):
    allowed = frozenset(o.a for o in options_for(game) if shared(game, o) or not hits(game, o))
    game.robber_allowed = allowed   # a host's rule: no hex hits another seat alone
    options = tuple(options_for(game))
    assert any(shared(game, o) for o in options) and any(not hits(game, o) for o in options)
    assert hit_robber(game, options) == options


def test_a_seat_holding_no_cards_still_counts_as_hit(game):
    # Its hex is a move that robs nobody, but the robber there blocks it.
    clear_hand(game._state, 1)
    alone = [o for o in options_for(game) if set(occupants(game._state, o.a)) == {1}]
    assert alone and all(o.b == NOBODY for o in alone)
    assert set(alone) <= set(hit_robber(game, options_for(game)))


def test_with_no_other_seat_to_hit_every_move_stands(game):
    empty = frozenset(o.a for o in options_for(game) if not hits(game, o))
    game.robber_allowed = empty     # a host's rule: only hexes nobody else is on
    options = tuple(options_for(game))
    assert options and hit_robber(game, options) == options


def test_any_other_decision_is_untouched(game):
    roads = (Action(ActionType.BUILD_ROAD, 1), Action(ActionType.BUILD_ROAD, 2))
    assert hit_robber(game, roads) == roads


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("True", True), ("0", False), ("", False), (None, False), ("yes", False),
])
def test_the_metadata_key_reads_as_on_or_off(value, expected):
    assert robber_hits_config({} if value is None else {"robber_hits": value}) is expected


def test_a_checkpoint_without_the_key_does_not_filter():
    assert robber_hits_of(object()) is False
    assert NetworkBot.__dataclass_fields__["robber_hits"].default is False


def test_one_forward_hits_a_seat_only_where_the_checkpoint_declares_it(game):
    assert bot_for(checkpoint(game, False)).choose(game).b == NOBODY
    on = bot_for(checkpoint(game, True))
    assert on.robber_hits and clean(game, on.choose(game))
    (batched,) = NetworkBot.choose_many([(on, game)])
    assert clean(game, batched)


def test_rollouts_search_only_the_moves_that_hit(game):
    seat = rollouts_for(checkpoint(game, True), worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert seat.root_options(game) == hit_robber(game, options_for(game))
    move = seat.choose(game)
    assert clean(game, move)
    (decision,) = seat.decision_log
    assert all(clean(game, o) for o in decision.options)
    off = rollouts_for(checkpoint(game, False), worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert off.root_options(game) == tuple(options_for(game))


def test_a_tree_search_plays_its_most_visited_move_that_hits(game):
    seat = searcher_for(checkpoint(game, True), simulations=16, wave=4, rng=random.Random(3))
    assert clean(game, seat.choose(game))


@dataclass
class PlanKeepingPolicy(EmptyHexPolicy):
    """`act_rows` answers from the game's own legal moves, not the row's
    options, as a policy that keeps plans does (`hexset.decisions.Plans`)."""

    def act_rows(self, rows):
        return super().act_rows([(game, seat, tuple(options_for(game))) for game, seat, _ in rows])


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_a_policy_that_answers_from_the_game_still_hits(game, temperature):
    space = checkpoint(game, True).space
    planned = Checkpoint(PlanKeepingPolicy(space), space, robber_hits=True)
    bot = bot_for(planned, temperature=temperature)
    assert clean(game, bot.choose(game))
    (batched,) = NetworkBot.choose_many([(bot, game)])
    assert clean(game, batched)
    seat = rollouts_for(planned, worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert clean(game, seat.choose(game))
