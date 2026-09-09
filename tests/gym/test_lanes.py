# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.gym.lanes`: many games in flight, stepped in lockstep.

The claim under test is that a lane is not a second engine. A game played
through the lane environment must be the *same* game `hexset.arena` plays for
the same `(seed, index)` -- same actions, same cleared trades, same winner --
however many lanes are in flight, because both deal from the one game law
(`hexset.arena.deal_game`) and both seat a gate at every seat.

Imports nothing from `pettingzoo`/`gymnasium` and skips nothing: the lane
environment is engine-only, so it runs on a plain `pip install hexset`.
"""

from __future__ import annotations

import pytest

from hexset import arena
from hexset.arena import Entrant, _play_one
from hexset.actions import options_for
from hexset.chance import Live, Recording
from hexset.gym.lanes import LaneEnv
from hexset.record import advance, moves, open_record, replay, replay_to
from hexset.victory import victory_points

SEED = 11
CAP = 2000


class Scripted:
    """A deterministic bot with a real trade gate, and no generator at all.

    Determinism is what lets one bot object serve every seat of a lane while
    `hexset.arena` spawns one per entrant and the two still play the same game:
    a bot that tie-broke from its own `random.Random` would play differently
    depending on how many of it there were.

    The gate wants one resource, chosen by the *view's* perspective, so the
    seats of a table sharing one bot object still want different things and
    trades clear between them. Its gain is strictly increasing in what it
    wants, which is what `hexset.trading.trade_event` requires of a gate.
    """

    trade_floor = 0.0

    def choose(self, game):
        return options_for(game)[0]

    def gains_many(self, view, received, counterparties):
        wants = view.perspective % 5
        return [float(bundle[wants]) for bundle in received]


class Refuses:
    """A gate that declines everything, so its seat never trades."""

    trade_floor = 0.0

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def spawn(board):
    return Scripted()


def actions_of(episode):
    """The episode's decision stream in `hexset.record.Record`'s action shape."""
    return [
        (int(d.action.type), d.action.a, d.action.b) for d in episode.stream()
    ]


@pytest.fixture
def scripted_kind():
    """`Scripted` as an `arena` entrant kind, so `_play_one` can seat it."""
    arena.register_entrant_kind("test-scripted", lambda entrant, board, rng: Scripted())
    try:
        yield Entrant("scripted", "test-scripted")
    finally:
        arena._ENTRANT_KIND_FACTORIES.pop("test-scripted", None)


def played_by_the_arena(entrant, index, players=4, cap=CAP):
    """The same `(seed, index)` game, played by `hexset.arena` with a record."""
    return _play_one(((entrant,) * players, index, SEED, cap, False, True))


def test_a_lane_plays_the_arenas_game_for_the_same_seed_and_index(scripted_kind):
    """The whole point of the shared game law: one lane, one game, no drift."""
    index = 3
    theirs = played_by_the_arena(scripted_kind, index)

    env = LaneEnv(4, SEED, 3, deal=1, action_cap=CAP, first_game=index, bots={0: spawn})
    (episode,) = env.drain()

    assert actions_of(episode) == list(theirs.record.actions)
    assert episode.outcome.winner == theirs.seat
    assert episode.outcome.turns == theirs.turns
    assert episode.outcome.truncated is False
    assert episode.index == index and episode.seed == SEED


def test_lane_count_does_not_change_the_games():
    """A game is a function of `(seed, index)`, not of how it was scheduled."""
    played = {}
    for lanes in (1, 4, 16):
        env = LaneEnv(4, SEED, lanes, deal=4, action_cap=CAP, bots={0: spawn})
        played[lanes] = {
            e.index: (actions_of(e), e.trades, e.outcome) for e in env.drain()
        }
        assert sorted(played[lanes]) == [0, 1, 2, 3]
        assert env.running is False

    assert played[1] == played[4] == played[16]


def test_the_action_cap_truncates_and_says_so():
    """A capped game is stopped, not finished, and the outcome distinguishes them."""
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=40, bots={0: spawn})
    episodes = env.drain()

    assert [e.outcome.actions for e in episodes] == [40, 40]
    assert all(e.outcome.truncated for e in episodes)
    assert all(e.outcome.winner is None for e in episodes)
    assert all(len(e) == 40 for e in episodes)

    uncapped = LaneEnv(4, SEED, 2, deal=1, action_cap=CAP, bots={0: spawn}).drain()
    assert uncapped[0].outcome.truncated is False


def test_the_trade_census_matches_the_engines_own(scripted_kind):
    """Trades are not actions, so the census is the only record the stream has.

    Compared against `hexset.arena`'s independently written census of the same
    game (`_play_and_record`'s before/after diff around its own `apply`), not
    merely against itself.
    """
    index = 3
    theirs = played_by_the_arena(scripted_kind, index)

    env = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, first_game=index, bots={0: spawn})
    (episode,) = env.drain()

    assert episode.trades  # the gate above trades; a vacuous pass would not do
    assert tuple(episode.trades) == tuple(theirs.record.trades)
    assert episode.outcome.trades == len(episode.trades)
    assert all(step < episode.outcome.actions for step, _, _, _ in episode.trades)


def test_a_caster_routes_seats_and_seats_every_seats_gate():
    """Policy 0 is the caller's; 1..3 are bots, gated by the bots themselves."""
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
        action_cap=600,
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
        # The mapping form: the caller answers only the seats it holds, and the
        # bot seats are played by their own bots.
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
        # Seat 0 refuses everything, so every exchange is between two bot seats.
        assert all({a, b} <= {1, 2, 3} for _, a, b, _ in episode.trades)


def test_a_board_law_pairs_boards_by_index_and_a_cohort_re_arms_the_bound():
    """`board(index)` is the board half of a paired evaluation: games `2k`
    and `2k+1` share `deal_board(seed, 2k)` while each keeps its own dice.
    `cohort(n)` deals `n` more games from where the counter stands without
    rebuilding the environment, so its counters carry across cohorts."""
    from hexset.arena import deal_board
    from hexset.gym.lanes import LaneEnv

    seed = 5
    def law(index):
        return deal_board(seed, index - index % 2)
    env = LaneEnv(players=3, seed=seed, lanes=2, deal=4, board=law, bots={0: spawn}, action_cap=60)
    first = env.drain()
    assert sorted(e.index for e in first) == [0, 1, 2, 3]
    # Games 2k and 2k+1 were dealt on one board; 2k and 2k+2 on different ones.
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
    """`env.drain()`, holding on to each lane's live `Game` by game index.

    A finished lane is refilled on the spot, so an `Episode` is all a caller
    normally keeps -- and comparing a record against the game it recorded
    needs the game itself.
    """
    games, episodes = {}, []
    while env.running:
        requests = env.requests()
        for request in requests:
            games[request.index] = request.game
        episodes.extend(env.step([None] * len(requests)))
    return games, episodes


def test_a_lane_record_replays_to_the_game_the_lane_played():
    """The point of `records=True`: the episode carries the game, replayable.

    Checked against the live game the lane actually played, not against the
    record's own claims -- `hexset.record.replay` already re-checks the
    winner and the turn count internally, so what is left to prove is that
    the position it lands on is the position the lane left behind.
    """
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=CAP, bots={0: spawn}, records=True)
    games, episodes = drain_watching(env)

    assert len(episodes) == 2
    for episode in episodes:
        played = games[episode.index].state(0, hidden=False)
        replayed = replay(episode.record).state(0, hidden=False)
        assert replayed.hands == played.hands
        assert tuple(victory_points(replayed, s) for s in range(4)) == (
            episode.outcome.points
        )
        # A record's own trade log is the episode's, since both are the one
        # before/after diff around the same `apply`.
        assert episode.record.trades == episode.trades
        assert len(episode.record.actions) == episode.outcome.actions


def test_replay_to_is_the_record_stepped_that_far():
    """`replay_to` is `open_record` + `advance`, and refuses a ply it lacks."""
    env = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, bots={0: spawn}, records=True)
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
    """Off by default, and skipped rather than discarded: no `Recording` at all."""
    plain = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, bots={0: spawn})
    assert [type(game.chance) for game in plain.in_flight()] == [Live]
    (episode,) = plain.drain()
    assert episode.record is None

    recorded = LaneEnv(4, SEED, 1, deal=1, action_cap=CAP, bots={0: spawn}, records=True)
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
    """Exercise the collection contract without a model or optional runtime."""
    import random
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
                # Saved arrays belong to this decision even after the lane advances.
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
        bot = NetworkBot(policy, players=2, seat=seat,
                         rng=random.Random(19 + len(gates)))
        bot.seat_at(game)  # The collector calls the policy directly, not bot.choose.
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
    # The same runtime also services batched imagined continuations for gates.
    assert len(policy.batch_sizes) > len(collector_batches)
    assert len(gates) == 8  # A new, correctly seated gate for every game/seat.
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
