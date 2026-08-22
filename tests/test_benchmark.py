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
