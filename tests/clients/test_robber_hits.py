# SPDX-License-Identifier: GPL-3.0-only
"""`robber_hits`: a checkpoint that declares it puts the robber only on a hex
where another seat has a building, wherever one is legal -- one forward, by
rollouts or by tree search, from the file's metadata. A hex it shares with
another seat stays; it is never forced onto its own hexes, and a tie goes to
a hex it is not on."""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hexset.actions import Action, ActionSpace, ActionType, build_space, options_for
from hexset.board.terrain import Terrain
from hexset.clients._modelmeta import robber_hits_config, robber_hits_of
from hexset.clients.netbot import (
    NetworkBot, bot_for, hit_robber, prefer_off_own, rollouts_for, searcher_for,
)
from hexset.robber import occupants
from hexset.rollout import PairedRollouts
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


def shared(game, option: Action) -> bool:
    return hits(game, option) and 0 in occupants(game._state, option.a)


@dataclass
class EmptyHexPolicy:
    """A prior that favours robber moves robbing nobody, uniform elsewhere;
    `VALUE` everywhere; the action the prior's argmax."""

    space: ActionSpace

    def _weight(self, option) -> float:
        return (1.0 if option.b == NOBODY else 0.01) if option.type is ROB else 1.0

    def _prior(self, options):
        raw = [self._weight(o) for o in options]
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


def test_only_moves_onto_a_hex_another_seat_is_on_are_left(game):
    options = tuple(options_for(game))
    kept = hit_robber(game, options)
    assert kept == tuple(o for o in options if hits(game, o))


def test_a_hex_shared_with_another_seat_stays(game):
    options = tuple(options_for(game))
    together = [o for o in options if shared(game, o)]
    assert together and any(clean(game, o) for o in options)
    assert set(together) <= set(hit_robber(game, options))


def test_a_hex_only_the_mover_is_on_goes(game):
    options = tuple(options_for(game))
    alone = [o for o in options if occupants(game._state, o.a) == (0,)]
    assert alone
    assert not set(alone) & set(hit_robber(game, options))


def test_the_desert_goes(game):
    state = game._state
    desert = next(h for h, t in enumerate(state.board.terrain) if t is Terrain.DESERT)
    state.robber = next(o.a for o in options_for(game) if o.a != desert)
    options = tuple(options_for(game))
    assert any(o.a == desert for o in options)
    assert all(o.a != desert for o in hit_robber(game, options))


def test_where_every_hex_another_seat_is_on_is_the_movers_too_only_those_stay(game):
    allowed = frozenset(o.a for o in options_for(game) if shared(game, o) or not hits(game, o))
    game.robber_allowed = allowed   # a host's rule: no hex hits another seat alone
    options = tuple(options_for(game))
    assert any(shared(game, o) for o in options) and any(not hits(game, o) for o in options)
    assert hit_robber(game, options) == tuple(o for o in options if shared(game, o))


def test_a_seat_holding_no_cards_still_counts_as_hit(game):
    # Its hex is a move that robs nobody, but the robber there blocks it.
    clear_hand(game._state, 1)
    alone = [o for o in options_for(game) if set(occupants(game._state, o.a)) == {1}]
    assert alone and all(o.b == NOBODY for o in alone)
    assert set(alone) <= set(hit_robber(game, options_for(game)))


def test_with_no_other_seat_to_hit_every_move_stands(game):
    # Empty hexes and the mover's own alike: the rule only takes moves away.
    empty = frozenset(o.a for o in options_for(game) if not hits(game, o))
    game.robber_allowed = empty     # a host's rule: only hexes nobody else is on
    options = tuple(options_for(game))
    assert any(occupants(game._state, o.a) == (0,) for o in options)
    assert any(not occupants(game._state, o.a) for o in options)
    assert hit_robber(game, options) == options


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
    assert on.robber_hits and hits(game, on.choose(game))
    (batched,) = NetworkBot.choose_many([(on, game)])
    assert hits(game, batched)


def test_rollouts_search_only_the_moves_that_hit(game):
    seat = rollouts_for(checkpoint(game, True), worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert seat.root_options(game) == hit_robber(game, options_for(game))
    move = seat.choose(game)
    assert hits(game, move)
    (decision,) = seat.decision_log
    assert all(hits(game, o) for o in decision.options)
    off = rollouts_for(checkpoint(game, False), worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert off.root_options(game) == tuple(options_for(game))


def test_a_tree_search_plays_its_most_visited_move_that_hits(game):
    seat = searcher_for(checkpoint(game, True), simulations=16, wave=4, rng=random.Random(3))
    assert hits(game, seat.choose(game))


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
    assert hits(game, bot.choose(game))
    (batched,) = NetworkBot.choose_many([(bot, game)])
    assert hits(game, batched)
    seat = rollouts_for(planned, worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert hits(game, seat.choose(game))


# --- ties: a hex the mover is not on ------------------------------------------------


def tied_pair(game) -> tuple[Action, Action]:
    """A move onto a hex the mover shares, then one onto a hex it is not on,
    the shared one first among the legal moves."""
    options = tuple(options_for(game))
    first = next(o for o in options if shared(game, o))
    later = next(o for o in options[options.index(first) + 1:] if clean(game, o))
    return first, later


@dataclass
class TiedPolicy(EmptyHexPolicy):
    """The prior's top robber moves are `top` (weight 1 each), then `next`
    (weight `below`); every other move 0.01."""

    top: tuple = ()
    next: tuple = ()
    below: float = 1.0

    def _weight(self, option) -> float:
        if option.type is not ROB:
            return 1.0
        return 1.0 if option in self.top else self.below if option in self.next else 0.01


def tied_checkpoint(game, on: bool, **policy) -> Checkpoint:
    space = checkpoint(game, on).space
    return Checkpoint(TiedPolicy(space, **policy), space, robber_hits=on)


def test_the_preference_keeps_a_mixed_tie_off_the_movers_hexes_and_leaves_others(game):
    options = tuple(options_for(game))
    on, off = (options.index(o) for o in tied_pair(game))
    assert prefer_off_own(game, options, [on, off]) == (off,)
    assert prefer_off_own(game, options, [on]) == (on,)
    both_off = [i for i, o in enumerate(options) if clean(game, o)][:2]
    assert prefer_off_own(game, options, both_off) == tuple(both_off)
    roads = (Action(ActionType.BUILD_ROAD, 1), Action(ActionType.BUILD_ROAD, 2))
    assert prefer_off_own(game, roads, [0, 1]) == (0, 1)


def test_one_forward_breaks_an_exact_tie_to_a_hex_it_is_not_on(game):
    first, later = tied_pair(game)
    assert bot_for(tied_checkpoint(game, False, top=(first, later))).choose(game) == first
    bot = bot_for(tied_checkpoint(game, True, top=(first, later)))
    assert bot.choose(game) == later
    assert NetworkBot.choose_many([(bot, game)]) == [later]


def test_one_forward_without_a_tie_plays_the_shared_hex(game):
    first, later = tied_pair(game)
    bot = bot_for(tied_checkpoint(game, True, top=(first,), next=(later,), below=0.9))
    assert bot.choose(game) == first


def flat_lanes(seat):
    """Every playout scores and leads alike: every candidate is tied."""
    real = seat._lanes

    def lanes(tasks, moves, mover, start):
        out = real(tasks, moves, mover, start)
        return dataclasses.replace(out, outcomes=np.full_like(out.outcomes, 0.5),
                                   leads=np.zeros_like(out.outcomes))
    seat._lanes = lanes


def test_rollouts_break_a_tie_to_a_hex_it_is_not_on(game):
    first, later = tied_pair(game)
    ckpt = tied_checkpoint(game, True, top=(first,), next=(later,), below=0.9)
    kwargs = dict(worlds=4, top=3, horizon=1, margin=1.5, rng=random.Random(1))
    seat = rollouts_for(ckpt, **kwargs)
    flat_lanes(seat)
    assert seat.choose(game) == later
    (decision,) = seat.decision_log
    assert decision.options[decision.candidates[0]] == first and decision.tied
    # The same root with no preference keeps the policy's own move.
    plain = rollouts_for(ckpt, **kwargs)
    plain.prefer = lambda g, o, tied: PairedRollouts.prefer(plain, g, o, tied)
    flat_lanes(plain)
    assert plain.choose(game) == first


def test_rollouts_without_a_margin_break_an_exact_score_tie_the_same_way(game):
    first, later = tied_pair(game)
    ckpt = tied_checkpoint(game, True, top=(first,), next=(later,), below=0.9)
    seat = rollouts_for(ckpt, worlds=2, top=3, horizon=1, rng=random.Random(1))
    flat_lanes(seat)
    assert seat.choose(game) == later


def test_a_tree_search_breaks_a_tie_on_visits_to_a_hex_it_is_not_on(game):
    first, later = tied_pair(game)
    options = tuple(options_for(game))
    i, j = options.index(first), options.index(later)
    visits = np.ones(len(options))
    visits[i] = visits[j] = 9.0
    decision = SimpleNamespace(options=options, chosen=i, visits=visits)
    seat = searcher_for(checkpoint(game, True), simulations=16, wave=4, rng=random.Random(3))
    assert seat._pick(game, decision) == j
    visits[i] = 10.0
    assert seat._pick(game, decision) == i


def only_hits(game) -> None:
    """A host's rule leaving only hexes another seat is on: `hit_robber` removes nothing."""
    game.robber_allowed = frozenset(o.a for o in options_for(game) if hits(game, o))
    options = tuple(options_for(game))
    assert hit_robber(game, options) == options


def test_with_nothing_removed_an_exact_tie_still_goes_off_the_movers_hex(game):
    only_hits(game)
    first, later = tied_pair(game)
    bot = bot_for(tied_checkpoint(game, True, top=(first, later)))
    assert bot.choose(game) == later
    assert NetworkBot.choose_many([(bot, game)]) == [later]


@dataclass
class PlannedPolicy(TiedPolicy):
    """`act_rows` plays `planned`, a move below the prior's best, as a policy
    continuing a plan it holds does."""

    planned: object = None

    def act_rows(self, rows):
        return [self.planned for _ in rows]


def test_with_nothing_removed_and_no_tie_the_policys_own_move_stands(game):
    only_hits(game)
    first, later = tied_pair(game)
    space = checkpoint(game, True).space
    policy = PlannedPolicy(space, top=(later,), next=(first,), below=0.9, planned=first)
    planned = Checkpoint(policy, space, robber_hits=True)
    bot = bot_for(planned)
    assert bot.choose(game) == first
    assert NetworkBot.choose_many([(bot, game)]) == [first]
    seat = rollouts_for(planned, worlds=2, top=3, horizon=1, rng=random.Random(1))
    assert seat._own(game, 0, tuple(seat.root_options(game))) == first
