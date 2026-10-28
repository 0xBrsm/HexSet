"""Rule draws survive sharding and reach the engine and its replay record."""
import numpy as np
import pytest

from hexset import contract9
from hexset.arena import deal_game
from hexset.actions import apply, legal_actions
from hexset.game import Phase, observe, roll_odds
from hexset.gym.distribution import Distribution
from hexset.gym.lanes import LaneEnv
from hexset.record import open_record


def specification():
    return {"random": {"weight": 1, "players": [[2, 1], [3, 1], [4, 1]],
                       "winning_points": [[8, 1], [13, 2], [16, 1]], "discard_limit": [[6, 1], [8, 2], [10, 1]],
                       "friendly_robber": [[False, 1], [True, 1]], "balanced_dice": [[False, 1], [True, 1]],
                       "board_mode": [["random", 1], ["spiral", 1]]},
            "exclude": [{"players": 3, "winning_points": 13, "discard_limit": 8}]}


def test_draws_are_indexed_exclude_held_out_cells_and_use_every_slot():
    law = Distribution(specification(), 4, 9)
    sequential = [law(i) for i in range(300)]
    assert sequential == [dict((i, law(i)) for w in range(5) for i in range(w, 300, 5))[i] for i in range(300)]
    assert {4 - len(d.locked) for d in sequential} == {2, 3, 4}
    assert {d.board_mode for d in sequential} == {"random", "spiral"}
    assert set.union(*(set(d.locked) for d in sequential)) == set(range(4))
    assert all((4 - len(d.locked), d.game_type.rules.winning_points, d.game_type.rules.discard_limit)
               != (3, 13, 8) for d in sequential)


def test_rule_and_board_draws_reach_the_lanes_and_their_replay_records():
    law = Distribution(specification(), 4, 9)
    env = LaneEnv(4, 9, lanes=4, deal=8, action_cap=20, deal_law=law)
    episodes = []
    while env.running:
        requests = env.requests()
        episodes.extend(env.step([r.options[0] for r in requests]))
    assert len(episodes) == 8
    for episode in episodes:
        drawn = law(episode.index)
        record = episode.record
        assert record.rules == drawn.game_type.rules and frozenset(record.locked) == drawn.locked
        rebuilt = open_record(record)
        want = deal_game(9, episode.index, 4, game_type=drawn.game_type, locked=drawn.locked,
                         board_mode=drawn.board_mode)
        assert rebuilt._state.board == want._state.board
        assert all(not episode.decisions[s] for s in drawn.locked)


def test_rules_and_live_flags_are_public_and_old_observations_keep_their_width():
    law = Distribution(specification(), 4, 9)
    for i in range(12):
        drawn = law(i)
        game = deal_game(9, i, 4, game_type=drawn.game_type, locked=drawn.locked)
        while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            apply(game, legal_actions(game)[0])
        for seat in set(range(4)) - drawn.locked:
            new = contract9.encode9(game, seat, "9.1")
            np.testing.assert_array_equal(contract9.flatten(new),
                                          contract9.flatten(contract9.encode9(observe(game, seat), seat, "9.1")))
            assert new.seats[:, -1].tolist() == [float((seat + k) % 4 not in drawn.locked) for k in range(4)]
            t = contract9.TABLE_COLUMNS_91
            odds = dict(roll_odds(game))
            np.testing.assert_allclose(new.table[t["roll_odds"]], [odds.get(n, 0) for n in range(2, 13)])
            np.testing.assert_allclose(new.seats[:, contract9.SEAT_COLUMNS["points"]],
                                       contract9.encode9(game, seat).seats[:, contract9.SEAT_COLUMNS["points"]]
                                       * 10 / drawn.game_type.rules.winning_points)
            assert new.table[t["winning_points"]] == np.float32(drawn.game_type.rules.winning_points / 10)
            assert new.table[t["discard_limit"]] == np.float32(drawn.game_type.rules.discard_limit / 10)
            assert new.table[t["friendly_robber"]] == drawn.game_type.rules.friendly_robber
            assert new.table[t["balanced_dice"]] == drawn.game_type.rules.balanced_dice
            np.testing.assert_allclose(new.hexes[:, -1] * 12, game._state.board.tokens, atol=1e-6)
            assert contract9.flatten(new).shape == (contract9.layout(4, 19, 54, 72, "9.1").width,)
            assert contract9.encode9(game, seat).seats.shape == (4, 39)


def test_bad_distributions_fail_before_a_game_is_dealt():
    spec = specification()
    spec["exclude"] = [{"players": 2}, {"players": 3}, {"players": 4}]
    with pytest.raises(ValueError, match="every random format"):
        Distribution(spec, 4, 1)
    for field, value in (("players", 5), ("winning_points", 0), ("discard_limit", -1),
                         ("balanced_dice", 1), ("board_mode", "missing")):
        spec = specification()
        spec["random"][field] = [[value, 1]]
        with pytest.raises(ValueError):
            Distribution(spec, 4, 1)
