# SPDX-License-Identifier: GPL-3.0-only
"""`robber_tip`: a checkpoint's robber move leans to the leader its own value
head names, where its policy rates robbing that seat nearly as highly as the
move it picked -- one forward or searched, from the file's metadata."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from hexset.actions import Action, ActionSpace, ActionType, build_space, options_for
from hexset.clients._modelmeta import robber_tip_config, robber_tip_of
from hexset.clients.netbot import bot_for, searcher_for, tip_robber
from tests.test_deals import a_robber_move

ROB = ActionType.MOVE_ROBBER
PLAYERS = 4
# Seat 2 is the leader by win chance; seat 1 is the policy's favourite victim.
VALUE = (0.1, 0.2, 0.5, 0.2)


def rob(hex_: int, victim: int) -> Action:
    return Action(ROB, hex_, victim)


OPTIONS = (rob(3, 1), rob(3, 2), rob(5, 3), rob(7, 2))


@pytest.mark.parametrize("tip,expected", [(0.5, 1), (0.8, 0)])
def test_the_leaders_best_option_replaces_the_pick_only_within_the_tip(tip, expected):
    prior = (0.40, 0.30, 0.20, 0.10)
    assert tip_robber(OPTIONS, prior, VALUE, 0, 0, tip) == expected


def test_a_pick_that_already_robs_the_leader_stands():
    prior = (0.40, 0.05, 0.20, 0.35)
    assert tip_robber(OPTIONS, prior, VALUE, 0, 3, 0.01) == 3


def test_the_leader_is_the_value_heads_not_the_mover_itself():
    # The mover rates itself highest; the leader is the best of the others.
    value = (0.7, 0.1, 0.05, 0.15)
    prior = (0.40, 0.30, 0.25, 0.05)
    assert tip_robber(OPTIONS, prior, value, 0, 0, 0.5) == 2


def test_no_option_on_the_leader_leaves_the_pick():
    options = (rob(3, 1), rob(5, 3), rob(8, PLAYERS))
    assert tip_robber(options, (0.5, 0.3, 0.2), VALUE, 0, 0, 0.01) == 0


@pytest.mark.parametrize("value,expected", [
    ("0.5", 0.5), ("1", 1.0), ("2", 1.0), ("-0.3", 0.0), ("", 0.0), (None, 0.0),
    ("nope", 0.0), ("nan", 0.0), ("inf", 0.0),
])
def test_the_metadata_key_reads_as_a_bounded_fraction(value, expected):
    meta = {} if value is None else {"robber_tip": value}
    assert robber_tip_config(meta) == expected


def test_a_checkpoint_without_the_key_does_not_tip():
    assert robber_tip_of(object()) == 0.0


@dataclass
class VictimPolicy:
    """Prior by the seat each option robs (`weights`), uniform elsewhere;
    `VALUE` everywhere; the action the prior's argmax. Counts forwards."""

    space: ActionSpace
    weights: dict
    forwards: int = 0

    def _prior(self, options):
        raw = [self.weights.get(o.b, 0.01) if o.type is ROB else 1.0 for o in options]
        total = sum(raw)
        return [w / total for w in raw]

    def act_rows(self, rows):
        self.forwards += 1
        return [max(options, key=lambda o, p=self._prior(options): p[options.index(o)])
                for _, _, options in rows]

    def value_rows(self, rows):
        self.forwards += 1
        return [VALUE for _ in rows]

    def score_rows(self, rows):
        self.forwards += 1
        return [(self._prior(options), VALUE) for _, _, options in rows]


@dataclass(frozen=True)
class Checkpoint:
    policy: VictimPolicy
    space: ActionSpace
    players: int = PLAYERS
    robber_tip: float = 0.0


def checkpoint(game, tip: float, weights: dict) -> Checkpoint:
    topology = game._state.board.topology
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, PLAYERS)
    return Checkpoint(VictimPolicy(space, weights), space, robber_tip=tip)


def test_one_forward_robs_the_leader_within_the_tip_in_the_forward_it_already_makes():
    game = a_robber_move()
    assert {o.b for o in options_for(game)} >= {1, 2}
    near = checkpoint(game, 0.5, {1: 0.55, 2: 0.45})
    bot = bot_for(near)
    assert bot.robber_tip == 0.5
    assert bot.choose(game).b == 2
    assert near.policy.forwards == 1

    far = checkpoint(game, 0.5, {1: 0.55, 2: 0.2})
    assert bot_for(far).choose(game).b == 1


def test_without_the_key_one_forward_plays_the_policys_own_pick():
    game = a_robber_move()
    plain = checkpoint(game, 0.0, {1: 0.55, 2: 0.45})
    assert bot_for(plain).choose(game).b == 1


def test_a_search_tips_on_its_roots_prior_and_value():
    game = a_robber_move()

    def searched(tip):
        ckpt = checkpoint(game, tip, {1: 0.55, 2: 0.45})
        return searcher_for(ckpt, simulations=16, wave=4, rng=random.Random(3)).choose(game)

    assert searched(0.5).b == 2
    # Off, the search's own most-visited move, whatever it is.
    off = checkpoint(game, 0.0, {1: 0.55, 2: 0.45})
    plain = searcher_for(off, simulations=16, wave=4, rng=random.Random(3))
    assert searched(0.0) == plain.choose(game)
