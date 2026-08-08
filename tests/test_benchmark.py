from __future__ import annotations

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
    assert set(env) == {"commit", "python", "platform", "machine"}
    assert env["python"]


def test_the_cli_runs(capsys):
    assert main(["--games", "2", "--json"]) == 0
    assert "games_per_second" in capsys.readouterr().out
