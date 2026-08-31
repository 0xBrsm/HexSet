from __future__ import annotations

import pickle
import random

import pytest

from catan import arena
from catan.arena import (
    MAX_ACTIONS,
    PRESETS,
    Entrant,
    Estimate,
    Standing,
    Tournament,
    compete,
    entrant_from_name,
    lineup_from_names,
    mean_interval,
    play,
    pooled,
    seat_of,
    spawn,
    wilson,
)
from catan.board.board import random_base_board
from catan.bots import RandomBot, options_for
from catan.game import is_over


@pytest.mark.parametrize("seats", [2, 3, 4])
def test_rotation_gives_every_entrant_every_seat_once(seats):
    for entrant in range(seats):
        taken = [seat_of(entrant, game, seats) for game in range(seats)]
        assert sorted(taken) == list(range(seats))
    for game in range(seats):
        lineup = [seat_of(entrant, game, seats) for entrant in range(seats)]
        assert sorted(lineup) == list(range(seats))


def test_wilson_brackets_the_observed_rate():
    low, high = wilson(50, 100)
    assert low < 0.5 < high


def test_wilson_narrows_as_games_accumulate():
    narrow = wilson(500, 1000)
    wide = wilson(50, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == 1.0
    assert wilson(0, 100)[1] > 0.0
    assert wilson(100, 100)[0] < 1.0


def test_wilson_says_nothing_without_games():
    assert wilson(0, 0) == (0.0, 1.0)


def test_a_standing_reports_its_rate():
    assert Standing("greedy", 3, 4).win_rate == 0.75


def test_play_seats_each_bot_at_its_own_index():
    rng = random.Random(0)
    board = random_base_board(rng)
    bots = [RandomBot(random.Random(i)) for i in range(4)]
    game = play(bots, board, rng)
    assert is_over(game)
    assert game.state.num_players == 4


def test_the_action_cap_stops_a_game_that_will_not_end():
    rng = random.Random(0)
    board = random_base_board(rng)
    bots = [RandomBot(random.Random(i)) for i in range(4)]
    game = play(bots, board, rng, action_cap=5)
    assert not is_over(game)


def test_every_game_is_won_or_counted_unfinished():
    result = compete(lineup_from_names(["random"] * 4), 4, seed=1)
    assert sum(s.wins for s in result.standings) + result.unfinished == result.games
    assert all(s.games == 4 for s in result.standings)
    assert result.mean_turns > 0


def test_the_same_seed_runs_the_same_tournament():
    lineup = lineup_from_names(["random", "greedy", "random", "greedy"])
    first = compete(lineup, 4, seed=2)
    second = compete(lineup, 4, seed=2)
    assert first.standings == second.standings


def test_a_lineup_the_rotation_cannot_balance_is_refused():
    with pytest.raises(ValueError, match="divide evenly"):
        compete(lineup_from_names(["random"] * 4), 6)


def test_a_tournament_needs_opponents():
    with pytest.raises(ValueError, match="at least two"):
        compete(lineup_from_names(["random"]), 4)


def test_repeated_bots_are_numbered_and_unknown_ones_rejected():
    named = [e.name for e in lineup_from_names(["greedy", "random", "greedy"])]
    assert named == ["greedy#0", "random", "greedy#1"]
    with pytest.raises(ValueError, match="unknown bots: mcts"):
        lineup_from_names(["mcts", "random"])


def test_a_checkpoint_path_names_an_entrant_wherever_a_preset_would():
    lineup = lineup_from_names(
        ["network:/tmp/latest.pt", "network:/tmp/latest.pt", "greedy", "greedy"]
    )
    assert [e.name for e in lineup] == [
        "network#0",
        "network#1",
        "greedy#0",
        "greedy#1",
    ]
    assert lineup[0].kind == "network"
    assert lineup[0].weights == "/tmp/latest.pt"
    # The whole reason entrants are descriptions: this has to reach a worker.
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_a_network_entrant_leaves_its_offer_budget_to_the_checkpoint():
    """`None` here means what it trained under, not the engine's eight.

    Scoring a policy on a horizon it never played is the mistake this default
    exists to make hard, so it is pinned here rather than left to `netbot`.
    """
    assert entrant_from_name("network:/tmp/latest.pt").max_offers is None


def test_an_mcts_entrant_names_its_simulation_and_wave_budgets():
    plain = entrant_from_name("mcts:/tmp/x.pt")
    assert (plain.kind, plain.weights, plain.simulations, plain.wave) == (
        "mcts",
        "/tmp/x.pt",
        128,
        16,
    )

    sized = entrant_from_name("mcts:/tmp/x.pt@32")
    assert (sized.name, sized.simulations, sized.wave) == ("mcts32", 32, 16)

    batched = entrant_from_name("mcts:/tmp/x.pt@256w64")
    assert (batched.name, batched.simulations, batched.wave) == (
        "mcts256w64",
        256,
        64,
    )


def test_pooling_adds_up_the_seats_a_side_took():
    standings = (
        Standing("network#0", 30, 100),
        Standing("network#1", 25, 100),
        Standing("greedy#0", 22, 100),
        Standing("greedy#1", 23, 100),
    )
    assert pooled(standings, 100) == [
        Standing("network", 55, 100),
        Standing("greedy", 45, 100),
    ]


def test_every_preset_can_be_built():
    board = random_base_board(random.Random(0))
    for entrant in PRESETS.values():
        assert spawn(entrant, board, random.Random(0)) is not None


def test_an_unknown_bot_kind_is_refused():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="unknown bot kind"):
        spawn(Entrant("bogus", kind="oracle"), board, random.Random(0))


def test_entrants_are_picklable_so_they_can_cross_a_process():
    """The reason entrants are data and not closures."""
    lineup = lineup_from_names(["greedy", "search2"])
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_workers_do_not_change_the_result():
    """Parallelism may change the clock and nothing else."""
    lineup = lineup_from_names(["greedy", "random", "greedy", "random"])
    serial = compete(lineup, 8, seed=11, workers=1)
    parallel = compete(lineup, 8, seed=11, workers=4)
    assert serial.standings == parallel.standings
    assert serial.unfinished == parallel.unfinished
    assert serial.mean_turns == parallel.mean_turns


def test_greedy_beats_random_play():
    result = compete(lineup_from_names(["greedy", "greedy", "random", "random"]), 4)
    by_name = {s.name: s.wins for s in result.standings}
    assert by_name["greedy#0"] + by_name["greedy#1"] == result.games - result.unfinished
    assert by_name["random#0"] == by_name["random#1"] == 0


def test_the_default_action_cap_is_not_reached_by_ordinary_play():
    result = compete(lineup_from_names(["greedy"] * 4), 4)
    assert result.unfinished == 0
    assert MAX_ACTIONS > result.mean_turns


def test_the_winning_entrants_points_reach_the_victory_threshold():
    """Points are indexed by entrant, so the winner's row must show a win.

    Seat-order points would put an arbitrary seat's score at the winner's
    index, which is the one way this mapping can silently go wrong.
    """
    result = compete(lineup_from_names(["greedy"] * 4), 8, seed=3)

    assert result.unfinished == 0
    for winner, row in zip(result.winners, result.points):
        assert row[winner] >= 10
        assert row[winner] == max(row)


def test_stronger_entrants_score_more_points_than_the_seats_they_beat():
    result = compete(
        lineup_from_names(["greedy", "greedy", "random", "random"]), 8, seed=5
    )

    differences = [row[0] - row[2] for _, row in result.decided()]
    assert mean_interval(differences).lower > 0


def test_unfinished_games_are_left_out_of_the_decided_rows():
    result = Tournament(
        standings=(),
        games=3,
        unfinished=1,
        mean_turns=0.0,
        seconds=0.0,
        winners=(0, None, 0),
        points=((10, 6), (0, 99), (9, 5)),
    )

    assert result.decided() == [(0, (10, 6)), (0, (9, 5))]


def test_a_constant_paired_advantage_has_no_sampling_error():
    assert mean_interval([2.0] * 20) == Estimate(2.0, 2.0, 2.0, 20)


def test_too_few_point_samples_say_nothing_either_way():
    estimate = mean_interval([2.0])
    assert estimate.mean == 2.0
    assert estimate.lower == float("-inf")
    assert estimate.upper == float("inf")

    empty = mean_interval([])
    assert empty.samples == 0
    assert empty.lower == float("-inf")
    assert empty.upper == float("inf")


def test_a_network_spec_can_self_impose_an_offer_budget():
    """`network:<path>@<offers>`, mirroring `mcts:<path>@<simulations>`.

    The budget exists so a duel can price the policy's proposing behaviour
    against itself; without a suffix a network entrant keeps `max_offers=None`,
    which means the budget its checkpoint trained under rather than the engine's
    whole cap.
    """
    from catan.arena import entrant_from_name

    plain = entrant_from_name("network:/tmp/x.pt")
    assert plain.max_offers is None
    assert plain.name == "network"
    assert plain.weights == "/tmp/x.pt"

    capped = entrant_from_name("network:/tmp/x.pt@1")
    assert capped.max_offers == 1
    assert capped.name == "network-offers1"
    assert capped.weights == "/tmp/x.pt"


class Streamless:
    """A bot that plays the first legal action and holds no random stream.

    The antithetic pairing's exact answer only exists for two sides that play
    the *same* game, and every scripted bot here samples hidden information
    through an rng keyed to its lineup slot -- so two identical `greedy` sides
    diverge on the first determinization that differs, and their split is paired
    but not exact (48 games of `greedy` at seed 7 read 23/25, not 24/24). The
    case the guarantee was measured on is a checkpoint played straight:
    `network:` argmax touches no rng at all, and a self-duel of one came out
    24/24 against 25/23 unpaired. Torch cannot be installed where this file
    runs, so the determinism comes from a stub rather than from a checkpoint.
    """

    def choose(self, game):
        return options_for(game)[0]


def test_a_pair_is_one_game_played_twice_with_the_seat_pairs_exchanged(monkeypatch):
    """The property the dice-keying trap broke, stated so it cannot come back.

    The first attempt at this keyed the board to the pair and left the dice
    keyed to the raw game index, so a pair's two halves played the same board
    with *different dice* -- two unrelated games, and nothing cancelled. It read
    20/28 where the design guarantees 24/24, and only a known exact answer
    caught it.
    """
    monkeypatch.setattr(arena, "spawn", lambda entrant, board, rng: Streamless())
    lineup = tuple(lineup_from_names(["random"] * 4))

    first = arena._play_one((lineup, 0, 20000, MAX_ACTIONS, True))
    second = arena._play_one((lineup, 1, 20000, MAX_ACTIONS, True))

    # One game, relabelled: the same board, the same dice, the same play. Only
    # which entrant sat where differs, and it differs by the `seats // 2` shift.
    assert second[3] == first[3][2:] + first[3][:2]
    assert second[1] == first[1]  # the winning *seat* is the same seat
    # And the win lands on the other side of an [a, a, b, b] lineup.
    assert second[0] == (first[0] + 2) % 4


def test_antithetic_pairing_splits_an_identical_pair_exactly(monkeypatch):
    monkeypatch.setattr(arena, "spawn", lambda entrant, board, rng: Streamless())
    lineup = lineup_from_names(["random"] * 4)

    paired = compete(lineup, 48, seed=20000)
    wins = [standing.wins for standing in paired.standings]
    # Every decided game is decided twice, once for each side, so the split is
    # exact rather than close. Counted over decided games because a stub that
    # always takes the first legal action does not always reach ten points, and
    # an unfinished pair is unfinished in both halves.
    decided = paired.games - paired.unfinished
    assert wins[0] + wins[1] == wins[2] + wins[3] == decided // 2

    # The control: the same boards without the pairing. The rotation already
    # gives every entrant every seat, so the *mean* seat effect is gone -- what
    # is left is the parity-correlated residual, and it does not average out.
    # Any one 48-game block can land on an even split by chance (it did, at
    # seed 20000, the day the piece supply changed which games get played), so
    # the claim is made over several blocks: the unpaired split is exact in at
    # most a minority of them, where the paired split is exact in all.
    uneven = 0
    for seed in range(20000, 20005):
        plain = compete(lineup, 48, seed=seed, antithetic=False)
        loose = [standing.wins for standing in plain.standings]
        uneven += loose[0] + loose[1] != loose[2] + loose[3]
        again = compete(lineup, 48, seed=seed)
        tight = [standing.wins for standing in again.standings]
        assert tight[0] + tight[1] == tight[2] + tight[3]
    assert uneven >= 3


def test_a_sweep_gets_an_interval_containing_its_own_estimate():
    """`wilson(4, 4)` used to return an upper bound of 0.9999999999999999.

    Analytically the bound at `p = 1` is exactly 1; in floating point the
    algebraic form lands an ulp short, so a clean sweep reported an interval
    excluding its own point estimate and `wilson_low <= win_rate <=
    wilson_high` failed in `test_train`.
    """
    for wins, games in ((4, 4), (0, 4), (400, 400), (0, 800), (1, 2)):
        low, high = wilson(wins, games)
        assert 0.0 <= low <= wins / games <= high <= 1.0
