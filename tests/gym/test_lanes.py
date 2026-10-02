# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes`: many games in flight, stepped in lockstep.

A lane is not a second engine: for the same `(seed, index)` it must play the
*same* game `hexset.arena` does, however many lanes are in flight.
"""

from __future__ import annotations

import pytest

from hexset import arena
from hexset.arena import Entrant, _play_one
from hexset.rules import STANDARD_GAME
from hexset.game import UNSTRUCTURED_TURN_CAP

#: Lanes are exercised with random play, which takes about four times the
#: turns real play does and so runs past `hexset.game.MAX_TURNS`.
TURN_CAP = UNSTRUCTURED_TURN_CAP


def LaneEnv(*args, **kwargs):
    """`lanes.LaneEnv` with the horizon unstructured play needs. Every lane
    here is driven by random or scripted bots, which run past the engine's
    default cap -- that cap is read off agents that are trying to win."""
    kwargs.setdefault("turn_cap", TURN_CAP)
    return _LaneEnv(*args, **kwargs)
from hexset.trading import TradeParams
from hexset.actions import options_for
from hexset.chance import Live, Recording
from hexset.gym.lanes import LaneEnv as _LaneEnv
from hexset.record import advance, moves, open_record, replay, replay_to
from hexset.victory import victory_points

SEED = 11
CAP = 2000


class Scripted:
    """A deterministic bot with a real trade gate: determinism lets one object
    serve every seat while `hexset.arena` spawns one per entrant. The gate
    wants one resource chosen by the *view's* perspective, so seats sharing the
    object still want different things.

    It declares one offer a turn, because a bot that declares nothing has no
    limit and will keep offering while it has an offer left -- which is a
    great deal of trading to push a whole game through, and not what these
    tests are about.
    """

    trade_floor = 0.0
    trade_params = TradeParams(max_offers=1)

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


class Refuses:

    trade_floor = 0.0

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def spawn(board):
    return Scripted()


def actions_of(episode):
    return [
        (int(d.action.type), d.action.a, d.action.b) for d in episode.stream()
    ]


#: A game the scripted bots finish under the cap (index 3 no longer does).
FINISHED = 4


@pytest.fixture(scope="module")
def played_both_ways():
    """Game `FINISHED`, played once by `hexset.arena` and once by a lane."""
    arena.register_entrant_kind("test-scripted", lambda entrant, board, rng: Scripted())
    try:
        theirs = _play_one(
            ((Entrant("scripted", "test-scripted"),) * 4, FINISHED, SEED, CAP, False,
             True, "round", STANDARD_GAME, TURN_CAP)
        )
    finally:
        arena.unregister_entrant_kind("test-scripted")
    env = LaneEnv(4, SEED, 3, deal=1, action_cap=CAP, first_game=FINISHED, bots={0: spawn})
    (episode,) = env.drain()
    return theirs, episode


def test_a_lane_plays_the_arenas_game_for_the_same_seed_and_index(played_both_ways):
    theirs, episode = played_both_ways

    assert actions_of(episode) == list(theirs.record.actions)
    assert episode.outcome.winner == theirs.seat
    assert episode.outcome.turns == theirs.turns
    assert episode.outcome.truncated is False
    assert episode.index == FINISHED and episode.seed == SEED


def test_lane_count_does_not_change_the_games():
    """Two lanes for three games, so one lane is refilled mid-drain. Capped:
    lockstep interference shows in the first actions as surely as the last."""
    played = {}
    for lanes in (1, 2):
        env = LaneEnv(4, SEED, lanes, deal=3, action_cap=100, bots={0: spawn})
        played[lanes] = {
            e.index: (actions_of(e), e.trades, e.outcome) for e in env.drain()
        }
        assert sorted(played[lanes]) == [0, 1, 2]
        assert env.running is False

    assert played[1] == played[2]
    assert any(trades for _, trades, _ in played[1].values())


def test_the_trade_census_matches_the_engines_own(played_both_ways):
    theirs, episode = played_both_ways

    assert episode.trades  # a vacuous pass would not do
    assert tuple(episode.trades) == tuple(theirs.record.trades)
    assert episode.outcome.trades == len(episode.trades)
    assert all(step < episode.outcome.actions for step, _, _, _ in episode.trades)


def test_a_caster_routes_seats_and_seats_every_seats_gate():
    bots: dict[int, Scripted] = {}

    def one_bot(board):
        return bots.setdefault(id(board), Scripted())

    refusals = Refuses()
    cast = (0, 1, 1, 1)
    env = LaneEnv(
        4,
        SEED,
        2,
        deal=2,
        action_cap=300,
        caster=lambda index: cast,
        bots={1: one_bot},
        gates={0: lambda game, seat: refusals},
    )

    episodes = []
    while env.running:
        requests = env.requests()
        for request in requests:
            assert request.policy == cast[request.seat]
            assert request.view.perspective == request.seat
            assert request.options
            gates = request.game.gates
            assert gates[0] is refusals
            assert all(gate in bots.values() for gate in gates[1:])
        episodes.extend(
            env.step(
                {r.lane: r.options[0] for r in requests if r.policy == 0}
            )
        )

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.cast == cast
        assert episode.decisions[0], "the caller's seat took no decisions"
        assert all(d.policy == 0 for d in episode.decisions[0])
        assert episode.trades, "the bots' gates cleared nothing"
        assert all({a, b} <= {1, 2, 3} for _, a, b, _ in episode.trades)


def test_a_board_law_pairs_boards_by_index_and_a_cohort_re_arms_the_bound():
    """Games `2k` and `2k+1` share `deal_board(seed, 2k)` while each keeps its own
    dice; `cohort(n)` deals `n` more from where the counter stands.
    """
    from hexset.arena import deal_board
    from hexset.gym.lanes import LaneEnv as _LaneEnv

    seed = 5
    def law(index):
        return deal_board(seed, index - index % 2)
    env = LaneEnv(players=3, seed=seed, lanes=2, deal=4, board=law, bots={0: spawn}, action_cap=60)
    first = env.drain()
    assert sorted(e.index for e in first) == [0, 1, 2, 3]
    assert law(0).tokens == law(1).tokens and law(0).tokens != law(2).tokens
    assert not env.running and env.games_started() == 4

    env.cohort(2)
    second = env.drain()
    assert sorted(e.index for e in second) == [4, 5]
    assert env.games_started() == 6 and env.games == 6


def test_mask_of_is_legal_mask_over_the_options_in_hand():
    from hexset.actions import legal_actions, legal_mask, mask_of, space_for
    from hexset.arena import deal_game

    game = deal_game(1, 0, 4)
    space = space_for(game)
    assert mask_of(space, legal_actions(game)) == legal_mask(game, space)


def drain_watching(env):
    games, episodes = {}, []
    while env.running:
        requests = env.requests()
        for request in requests:
            games[request.index] = request.game
        episodes.extend(env.step([None] * len(requests)))
    return games, episodes


def test_a_lane_record_replays_to_the_game_the_lane_played():
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=200, bots={0: spawn}, records=True)
    games, episodes = drain_watching(env)

    assert len(episodes) == 2
    for episode in episodes:
        played = games[episode.index].state(0, hidden=False)
        replayed = replay(episode.record).state(0, hidden=False)
        assert replayed.hands == played.hands
        assert tuple(victory_points(replayed, s) for s in range(4)) == (
            episode.outcome.points
        )
        assert episode.record.trades == episode.trades
        assert len(episode.record.actions) == episode.outcome.actions


def test_replay_to_is_the_record_stepped_that_far():
    env = LaneEnv(4, SEED, 1, deal=1, action_cap=200, bots={0: spawn}, records=True)
    (episode,) = env.drain()
    record = episode.record
    walk = list(moves(record))

    for ply in (0, 1, 17, len(record.actions)):
        stepped = open_record(record)
        for _, action, trades in walk[:ply]:
            advance(stepped, action, trades)
        got = replay_to(record, ply)
        assert got.state(0, hidden=False).hands == stepped.state(0, hidden=False).hands
        assert (got.turns, got.phase, got.current_player) == (
            stepped.turns,
            stepped.phase,
            stepped.current_player,
        )

    with pytest.raises(ValueError):
        replay_to(record, len(record.actions) + 1)
    with pytest.raises(ValueError):
        replay_to(record, -1)


def test_an_environment_not_asked_for_records_does_not_record_chance():
    plain = LaneEnv(4, SEED, 1, deal=1, action_cap=40, bots={0: spawn})
    assert [type(game.chance) for game in plain.in_flight()] == [Live]
    (episode,) = plain.drain()
    assert episode.record is None

    recorded = LaneEnv(4, SEED, 1, deal=1, action_cap=40, bots={0: spawn}, records=True)
    assert [type(game.chance) for game in recorded.in_flight()] == [Recording]


@pytest.mark.parametrize("bad_response", ["missing", "illegal", "inactive"])
def test_invalid_batch_does_not_advance_any_lane_and_can_be_retried(bad_response):
    from hexset.actions import Action, ActionType

    env = LaneEnv(players=2, lanes=2, deal=2, action_cap=1, records=True)
    requests = env.requests()
    answers = {r.lane: r.options[0] for r in requests}
    if bad_response == "missing":
        del answers[requests[-1].lane]
    elif bad_response == "illegal":
        answers[requests[-1].lane] = Action(ActionType.END_TURN)
    else:
        answers[99] = requests[0].options[0]
    with pytest.raises(ValueError):
        env.step(answers)
    assert env.requests() is requests
    assert (env.steps, env.ticks, env.games) == (0, 0, 0)
    episodes = env.step([r.options[0] for r in requests])
    assert len(episodes) == 2
    assert all(len(e.record.actions) == len(e) == 1 for e in episodes)


def test_illegal_scripted_bot_action_does_not_advance_other_lanes():
    from hexset.actions import Action, ActionType

    class InvalidBot:
        def choose(self, game):
            return Action(ActionType.END_TURN)

    env = LaneEnv(players=2, lanes=2, deal=2, bots={1: lambda board: InvalidBot()},
                  caster=lambda index: (index, index), action_cap=1)
    requests = env.requests()
    with pytest.raises(ValueError, match="illegal bot action"):
        env.step([requests[0].options[0], None])
    assert env.requests() is requests
    assert env.steps == 0
    assert len(env.step([r.options[0] for r in requests])) == 2


def test_lightweight_policy_collects_batched_training_rows_and_replayable_episodes():
    import numpy as np
    from hexset.actions import mask_of, space_for
    from hexset.clients.netbot import NetworkBot
    from hexset.encoding import encode, encode_batch

    class Policy:
        def __init__(self, space):
            self.space = space
            self.batch_sizes = []
            self.samples = []

        def act_rows(self, rows):
            self.batch_sizes.append(len(rows))
            observations = encode_batch([g for g, _, _ in rows], [s for _, s, _ in rows])
            actions = []
            for (game, seat, options), obs in zip(rows, observations, strict=True):
                mask = np.asarray(mask_of(self.space, options))
                action = options[0]
                assert mask[self.space.index(action)]
                assert mask.sum() == len(options)
                # Copy: the arrays must outlive the lane's next advance.
                self.samples.append((seat, obs.globals.copy(), mask.copy()))
                actions.append(action)
            return actions

        def value_rows(self, rows):
            return [(0.5, 0.5) for _ in rows]

        def score_rows(self, rows):
            return [([1 / len(options)] * len(options), (0.5, 0.5))
                    for _, _, options in rows]

    space = space_for(arena.deal_game(19, 0, 2))
    policy = Policy(space)
    gates = []

    def gate(game, seat):
        bot = NetworkBot(policy, players=2, seat=seat)
        bot.seat_at(game)
        gates.append(bot)
        return bot

    env = LaneEnv(players=2, seed=19, lanes=3, deal=4, action_cap=48,
                  gates={0: gate}, records=True)
    episodes, retained, collector_batches = [], {}, []
    while env.running:
        requests = env.requests()
        collector_batches.append(len(requests))
        actions = policy.act_rows([(r.game, r.seat, r.options) for r in requests])
        for r, sample in zip(requests, policy.samples[-len(requests):], strict=True):
            assert r.view.perspective == r.seat
            retained[r.index, r.step] = sample
        episodes.extend(env.step(actions))

    assert max(collector_batches) == 3 and min(collector_batches) == 1
    # The gate reads `value_rows`, so `act_rows` is called only by the
    # collector loop above, once per tick.
    assert len(policy.batch_sizes) == len(collector_batches)
    assert len(gates) == 8
    assert sorted(e.index for e in episodes) == list(range(4))
    assert env.steps == len(retained) == 4 * 48
    for episode in episodes:
        assert episode.outcome.truncated and episode.outcome.winner is None
        assert len(episode.outcome.points) == 2
        assert episode.record.trades == episode.trades
        for decision in episode.stream()[::11]:
            game = replay_to(episode.record, decision.step)
            seat, observation, mask = retained[episode.index, decision.step]
            assert seat == decision.seat
            np.testing.assert_array_equal(observation, encode(game, seat).globals)
            assert mask[space.index(decision.action)]
        final = replay(episode.record)
        assert tuple(victory_points(final.state(s, hidden=False), s) for s in range(2)) == episode.outcome.points


def test_a_game_that_runs_out_of_turns_is_exhausted_not_truncated():
    """Both are undecided, `winner` `None`; `truncated` is the action cap
    stopping a game still in play, `exhausted` the engine ending it at the
    turn cap."""
    (capped,) = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, turn_cap=4,
                        bots={0: spawn}).drain()
    assert capped.outcome.winner is None and capped.outcome.turns == 4
    assert capped.outcome.exhausted and not capped.outcome.truncated

    (cut,) = LaneEnv(4, SEED, 1, deal=1, action_cap=30, bots={0: spawn}).drain()
    assert cut.outcome.winner is None
    assert cut.outcome.truncated and not cut.outcome.exhausted
