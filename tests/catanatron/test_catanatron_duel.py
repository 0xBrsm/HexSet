# SPDX-License-Identifier: GPL-3.0-only
"""Sharding semantics, provenance stamping, and the PYTHONHASHSEED pin.

The first two tests are the original `catanatron_bridge` duel-runner smoke
tests, ported unchanged in substance. The rest exercise
`_ensure_pythonhashseed_zero` and `provenance()` without ever spawning a real
game -- see `agents/reference/heximax.md`, "R-H1c take 2", for why the pin
exists: catanatron's own tie-breaks (`prune_robber_actions`'s
`max(set_of_actions, key=impact)`) resolve via hash-order-sensitive set
iteration, so a reproducibility check across two launches is meaningless
unless both fixed the same hash seed at interpreter start-up.
"""

from __future__ import annotations

import pytest

# A submodule, not bare "catanatron": this directory is itself named
# `catanatron`, and once pytest's default import mode puts `tests/` on
# sys.path (for the sibling top-level test modules), a bare `catanatron`
# import can resolve to *this directory* as an empty namespace package
# instead of failing -- silently skipping nothing and then blowing up on
# the first real submodule access. `catanatron.game` only exists in the
# real distribution.
pytest.importorskip("catanatron.game")

from hexset.catanatron.duel import _ensure_pythonhashseed_zero, provenance, run_duel


def test_sharded_duel_matches_single_process_semantics():
    result = run_duel("DC:search2-notrade,R,R,R", num_games=8, workers=4, seed=1)
    assert result.games == 8
    assert sum(result.wins.values()) == 8  # every game has exactly one winner
    dc_color = next(iter(result.labels))
    # search2 against three random players should win close to every game;
    # this is a smoke test for the runner, not a strength claim.
    assert result.wins[dc_color] >= 6


def test_shards_are_capped_at_the_requested_game_count_even_when_uneven():
    result = run_duel("DC:search2-notrade,R,R,R", num_games=5, workers=4, seed=2)
    assert result.games == 5
    assert sum(result.wins.values()) == 5


def test_hashseed_already_pinned_does_not_reexec():
    calls = []

    def fake_execve(path, argv, env):
        calls.append((path, argv, env))

    reexecd = _ensure_pythonhashseed_zero(
        argv=["duel.py", "--players=DC:search2-notrade,R,R,R"],
        env={"PYTHONHASHSEED": "0", "OTHER": "kept"},
        execve=fake_execve,
    )

    assert reexecd is False
    assert calls == []


@pytest.mark.parametrize("ambient", [{}, {"PYTHONHASHSEED": "random"}, {"PYTHONHASHSEED": "1"}])
def test_hashseed_unpinned_reexecs_with_the_module_form_and_seed_zero(ambient):
    calls = []

    def fake_execve(path, argv, env):
        calls.append((path, argv, env))

    argv = ["/some/path/duel.py", "--players=DC:search2-notrade,R,R,R", "--num=8"]
    env = {**ambient, "OTHER": "kept"}

    reexecd = _ensure_pythonhashseed_zero(argv=argv, env=env, execve=fake_execve)

    assert reexecd is True
    assert len(calls) == 1
    path, new_argv, new_env = calls[0]
    # Re-exec through `-m hexset.catanatron.duel`, not the original argv[0]
    # (a bare file path) -- duel.py uses relative imports (`.player`), which
    # only resolve when it is loaded as part of the `hexset.catanatron`
    # package, i.e. via `-m`.
    assert new_argv[1:3] == ["-m", "hexset.catanatron.duel"]
    assert new_argv[3:] == argv[1:]
    assert new_env["PYTHONHASHSEED"] == "0"
    assert new_env["OTHER"] == "kept"  # the rest of the environment survives


def test_provenance_reports_the_pythonhashseed_actually_in_effect(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert "PYTHONHASHSEED=0" in provenance()

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    assert "PYTHONHASHSEED=unset" in provenance()
