# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

from benchmarks import rollout
from benchmarks.throughput import environment, main, run


def test_a_small_run_reports_sane_numbers():
    result = run(games=3, players=4, seed=0, workers=1)

    assert result.games == 3
    assert result.seconds > 0
    assert result.games_per_second > 0
    assert result.mean_turns > 0
    assert result.mean_actions > result.mean_turns
    assert result.finished_by_win == 3


def test_the_same_seed_measures_the_same_games():
    a = run(games=3, players=4, seed=1, workers=1)
    b = run(games=3, players=4, seed=1, workers=1)
    assert (a.mean_turns, a.mean_actions) == (b.mean_turns, b.mean_actions)


def test_environment_is_recorded_for_reproducibility():
    env = environment()
    assert set(env) == {"commit", "dirty", "python", "platform", "machine"}
    # A SHA without this is not enough to reproduce a figure from.
    assert env["dirty"] in {"true", "false", "unknown"}
    assert env["python"]


def test_the_cli_runs(capsys):
    assert main(["--games", "2", "--json"]) == 0
    assert "games_per_second" in capsys.readouterr().out


def test_the_rollout_benchmark_counts_every_lane():
    result = rollout.run(lanes=4, players=4, ticks=20, seed=0, warmup=2)

    assert result.actions == 80
    assert result.ticks_per_second > 0
    # Lanes buy batch size, not ticks: a tick costs what its widest lane costs.
    assert result.actions_per_second == pytest.approx(
        result.ticks_per_second * 4, rel=1e-3
    )
    # A trivial policy is meant to be invisible, so the figure is the plumbing.
    assert 0.0 <= result.policy_share < 0.5


def test_the_rollout_cli_runs(capsys):
    assert rollout.main(["--lanes", "2", "--ticks", "5", "--warmup", "1", "--json"]) == 0
    assert "actions_per_second" in capsys.readouterr().out


def test_mix_cost_prices_both_sides_of_a_cast_shard():
    """`benchmarks.mix_cost`, on a shard small enough to be a test.

    The property that matters is that the stopwatch lands on the right side: a
    cast shard must attribute decisions to the opponent, or the affordability
    table it produces is the learner's cost twice.
    """
    torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")
    assert torch is not None
    from benchmarks.mix_cost import shard

    point = shard(
        "greedy=1.0",
        games=1,
        lanes=1,
        players=4,
        seed=0,
        width=8,
        rounds=1,
        max_offers=3,
        action_cap=400,
        parent="",
    )

    assert point.games == 1
    assert point.seconds > 0
    assert point.actions_per_game > 0
    assert point.cast_share == 1.0
    learner, opponent = point.sides
    assert (learner["name"], opponent["name"]) == ("learner", "greedy")
    assert learner["decisions"] > 0 and opponent["decisions"] > 0
    assert learner["ms_per_decision"] > 0 and opponent["ms_per_decision"] > 0


def test_mix_cost_interpolates_between_the_endpoints_it_measures():
    """The claim the table rests on: `S(f) = (1-f) S(0) + f S(1)` needs only the
    endpoints, because the cast is drawn per game and cost is additive over
    games. Checked on the cast *share*, which is the same draw and is free."""
    pytest.importorskip("torch", reason="hexset.collect imports torch at module load")
    from hexset.collect import mixed_caster

    caster = mixed_caster([0.25], players=4, seed=11)
    cast = sum(1 for index in range(2000) if any(caster(index)))
    assert 0.22 < cast / 2000 < 0.28, cast / 2000
