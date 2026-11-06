# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.rollout.PairedRollouts` with a policy that chooses whole decisions
(`hexset.decisions`): a root it does not search plays the policy's own move,
so a seat that searches nothing plays as the policy alone; a searched root
puts that move first and wins it every tie; a plan the search overrides is
dropped; and where the policy lists whole options, a decision of several
moves is searched as one."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from helpers import clear_hand, give

from hexset import decisions as D
from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, is_over, roll_dice, start, to_move
from hexset.rollout import Lanes, PairedRollouts


def _score(option) -> float:
    """A fixed score per whole option, the same in every process: an
    arbitrary but stable preference, so the most probable whole option and
    the prior's favourite first move often differ."""
    return (hash(D.key(option)) % 1_000_003) / 1_000_003 * 6.0


class Whole:
    """Chooses whole decisions by `_score` and plays them a move at a time
    (`plans`). `score_rows` is the whole decisions' softmax marginalised onto
    their first moves; values are each seat's hand size over 10. `options`
    false: it lists no whole options (`has_options`), as an older file."""

    def __init__(self, *, options: bool = True, double_turn: bool = False) -> None:
        self.plans = D.Plans()
        self.has_options = options
        self.double_turn = double_turn
        self.acted = 0

    def _listed(self, game, seat):
        listed = tuple(D.options(game, seat, double_turn=self.double_turn))
        scores = np.array([_score(o) for o in listed])
        probs = np.exp(scores - scores.max())
        return listed, probs / probs.sum()

    def option_rows(self, rows):
        assert self.has_options, "asked for whole options it does not list"
        return [self._listed(game, seat) for game, seat in rows]

    def best(self, game, seat):
        listed, probs = self._listed(game, seat)
        return listed[int(np.argmax(probs))]

    def act_rows(self, rows):
        out = []
        for game, seat, options in rows:
            self.acted += 1
            out.append(self.plans.act(game, seat, options, self.best,
                                      lambda g, s, o: D.first_move(self.best(g, s))))
        return out

    def score_rows(self, rows):
        out = []
        for game, seat, options in rows:
            listed, probs = self._listed(game, seat)
            mass = {}
            for option, p in zip(listed, probs):
                first = D.first_move(option)
                mass[first] = mass.get(first, 0.0) + p
            prior = np.array([mass.get(o, 0.0) for o in options])
            out.append((prior / prior.sum(), self.values(game)))
        return out

    def value_rows(self, rows):
        return [self.values(game) for game, _ in rows]

    @staticmethod
    def values(game):
        return tuple(sum(hand) / 10 for hand in game.state(0, hidden=False).hands)


class Atomic:
    """A policy of one move at a time (as a record-contract file): plays its
    prior's favourite, keeps no plans, lists no whole options."""

    def __init__(self) -> None:
        self.inner = Whole(options=False)

    def act_rows(self, rows):
        return [options[int(np.argmax(prior))] for (_, _, options), (prior, _) in zip(rows, self.score_rows(rows))]

    def score_rows(self, rows):
        return self.inner.score_rows(rows)

    def value_rows(self, rows):
        return self.inner.value_rows(rows)


def opening(seed: int = 4):
    return start(random_base_board(random.Random(seed)), 4, random.Random(seed))


def discarding(seed: int = 0):
    """Seat 0 owing half of ten cards to a seven, nobody else owing."""
    game = opening(seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    for seat in range(4):
        clear_hand(game._state, seat)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    game.phase, game.current_player = Phase.ROLL, 0
    roll_dice(game, 7)
    assert game.phase is Phase.DISCARD and game.discard_quota[0] == 5
    return game


def alone(policy):
    return lambda game: policy.act_rows([(game, to_move(game), tuple(legal_actions(game)))])[0]


def moves(choose, seed: int, count: int):
    game, out = opening(seed), []
    while len(out) < count and not is_over(game):
        action = choose(game)
        out.append(action)
        apply(game, action)
    return out


@pytest.mark.parametrize("options", [True, False], ids=["whole options", "a move at a time"])
@pytest.mark.parametrize("double_turn", [False, True], ids=["two setup turns", "double turn"])
def test_a_seat_that_searches_no_root_plays_as_the_policy_alone(options, double_turn):
    """Every root decisive: the searcher plays game for game what the
    policy plays alone, its whole setup turns and double turn included, at
    roots where the prior's favourite first move is another one."""
    for seed in (4, 7):
        played = moves(alone(Whole(options=options, double_turn=double_turn)), seed, 120)
        search = PairedRollouts(Whole(options=options, double_turn=double_turn), worlds=2, decisive=1e-6,
                                rng=random.Random(seed))
        assert moves(search.choose, seed, 120) == played
        assert all(d.worlds == 0 for d in search.decision_log)
    assert any(d.chosen != int(np.argmax(d.prior)) for d in search.decision_log), "the defect was not exercised"


def test_a_searched_setup_turn_is_one_decision_led_by_the_policys_own():
    policy = Whole()
    search = PairedRollouts(policy, worlds=2, top=3, horizon=1, marks=(), rng=random.Random(5))
    game = opening()
    own = policy.best(game, to_move(game))
    decision = search.decide(game)
    assert decision.whole and decision.worlds == 2 and len(decision.candidates) == 3
    assert decision.options[decision.candidates[0]] == own
    assert all(isinstance(decision.options[i], D.Compound) for i in decision.candidates)
    # The others by probability, highest first.
    rest = [decision.prior[i] for i in decision.candidates[1:]]
    assert rest == sorted(rest, reverse=True) and rest[0] == max(
        p for i, p in enumerate(decision.prior) if i != decision.candidates[0])
    chosen = decision.options[decision.chosen]
    assert decision.move == chosen.parts[0]
    line = json.loads(json.dumps(decision.to_json()))
    assert line["whole"] is True and line["actions"][0][0] == "SETUP"
    assert line["actions"][0][1] == [[p.type.name, p.a, p.b] for p in own.parts]
    # The road is the chosen turn's own, asked of nobody.
    apply(game, decision.move)
    acted = policy.acted
    road = search.decide(game)
    assert road.worlds == 0 and not road.whole and road.move == chosen.parts[1]
    assert policy.acted == acted
    assert policy.plans.peek(game, to_move(game)) == ()


def test_full_setup_searches_the_whole_turn_once():
    """`setup=full` with a decisive prior: the setup turn is searched as one
    decision at its settlement, and its road follows from it."""
    policy = Whole()
    search = PairedRollouts(policy, worlds=2, top=2, horizon=1, marks=(), decisive=1e-6, full_setup=True,
                            rng=random.Random(6))
    game = opening()
    first = search.decide(game)
    apply(game, first.move)
    second = search.decide(game)
    assert first.whole and first.worlds == 2 and second.worlds == 0


def test_a_whole_discard_is_searched_and_played_out_of_its_plan():
    policy = Whole()
    search = PairedRollouts(policy, worlds=2, top=3, horizon=1, marks=(), rng=random.Random(2))
    game = discarding()
    decision = search.decide(game)
    assert decision.whole and decision.options[decision.candidates[0]] == policy.best(game, 0)
    chosen = decision.options[decision.chosen]
    assert chosen.kind is D.CompoundKind.DISCARD and len(chosen.parts) == 5
    played = [decision.move]
    apply(game, decision.move, 0)
    while game.phase is Phase.DISCARD:
        step = search.decide(game)
        assert step.worlds == 0
        played.append(step.move)
        apply(game, step.move, 0)
    assert tuple(played) == chosen.parts


class Rigged(PairedRollouts):
    """Every playout scores candidate `winner` 1 and the rest 0."""

    winner = 1

    def _lanes(self, tasks, moves, seat, start):
        row = np.array([1.0 if k == self.winner else 0.0 for k in range(len(moves))])
        n = len(tasks) * len(moves)
        return Lanes(outcomes=np.tile(row, len(tasks)), after=None, reads=np.zeros((len(self.marks), n)),
                     steps=0, row_steps=0, mark_row_steps=tuple(0 for _ in self.marks), truncated=0)


@pytest.mark.parametrize("winner", [0, 1])
def test_a_plan_the_search_overrides_is_dropped(winner):
    """A move at a time (no whole options): the policy opens its setup turn
    and keeps the road as its plan. Where the search plays another
    settlement, the plan is gone; where it plays the policy's own, it stays."""
    policy = Whole(options=False)
    search = Rigged(policy, worlds=2, top=2, marks=(), rng=random.Random(1))
    search.winner = winner
    game = opening()
    own = policy.best(game, to_move(game))
    decision = search.decide(game)
    assert not decision.whole and decision.options[decision.candidates[0]] == own.parts[0]
    assert (decision.move == own.parts[0]) == (winner == 0)
    assert policy.plans.peek(game, to_move(game)) == (own.parts[1:] if winner == 0 else ())


def test_a_discard_overridden_a_card_at_a_time_is_never_finished_from_the_old_plan():
    """A discard part of the old plan is still legal after another card is
    given: the plan is dropped all the same, and the next card is the
    policy's choice from the position as it stands."""
    policy = Whole(options=False)
    search = Rigged(policy, worlds=2, top=2, marks=(), rng=random.Random(1))
    game = discarding()
    own = policy.best(game, 0)
    decision = search.decide(game)
    assert decision.move != own.parts[0]
    assert policy.plans.peek(game, 0) == ()
    apply(game, decision.move, 0)
    asked = policy.acted
    rest = policy.act_rows([(game, 0, tuple(legal_actions(game)))])[0]
    assert policy.acted == asked + 1 and rest == D.first_move(policy.best(game, 0))


def test_a_tie_goes_to_the_policys_own_move():
    policy = Whole(options=False)
    search = Rigged(policy, worlds=2, top=3, marks=(), rng=random.Random(1))
    search.winner = -1          # every candidate scores 0
    game = opening()
    decision = search.decide(game)
    own = policy.best(game, to_move(game)).parts[0]
    assert decision.chosen == decision.candidates[0] == decision.options.index(own)


def test_a_policy_of_one_move_at_a_time_searches_its_prior_order():
    """A policy that keeps no plans and plays its prior's favourite: the
    candidates are its prior's top moves, in order, as before."""
    game = opening()
    policy = Atomic()
    search = PairedRollouts(policy, worlds=2, top=3, horizon=1, marks=(), rng=random.Random(3))
    decision = search.decide(game)
    options = tuple(legal_actions(game))
    ((prior, _),) = policy.score_rows([(game, to_move(game), options)])
    order = sorted(range(len(options)), key=lambda i: -prior[i])
    assert not decision.whole and decision.candidates == tuple(order[:3])
    skip = PairedRollouts(Atomic(), worlds=2, decisive=1e-6, rng=random.Random(3)).decide(game)
    assert skip.chosen == order[0]


def test_plans_can_be_read_dropped_and_kept():
    game = opening()
    plans = D.Plans()
    options = tuple(legal_actions(game))
    plans.keep(game, 0, options[:2])
    assert plans.peek(game, 0) == options[:2] and plans.peek(game, 1) == ()
    assert plans.planned(game, 0, options)
    plans.drop(game)
    assert plans.peek(game, 0) == () and not plans.planned(game, 0, options)
    plans.keep(game, 0, ())
    assert plans.peek(game, 0) == ()
