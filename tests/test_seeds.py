"""The seed policy: approved seeds pass, others are refused or bypassed with
a reason the journal keeps, and the arena holds every run to it."""

from __future__ import annotations

import copy
import json
import os

import pytest

from hexset import seeds
from hexset.arena import PRESETS, compete, read_journal
from hexset.game import UNSTRUCTURED_TURN_CAP

POLICY = {
    "eval": {"default": {"2": 2000, "*": 1000}, "screen": [1000, 2000], "confirm": [3000, 4000]},
    "train": {"rl": [0, 1, 2], "bc": [10, 11]},
    "analysis": [20, 21],
}


def _write(path, doc=POLICY):
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def policy(tmp_path, monkeypatch):
    """A policy installed for one test; the suite's suspension comes back after."""
    saved = copy.deepcopy(seeds._state)
    monkeypatch.delenv(seeds.POLICY_ENV, raising=False)
    monkeypatch.delenv(seeds.BYPASS_ENV, raising=False)
    monkeypatch.delenv(seeds.REGISTRATION_ENV, raising=False)
    installed = seeds.register_seed_policy(_write(tmp_path / seeds.FILENAME))
    yield installed
    seeds._state.clear()
    seeds._state.update(saved)


@pytest.fixture
def no_policy(monkeypatch):
    saved = copy.deepcopy(seeds._state)
    monkeypatch.delenv(seeds.POLICY_ENV, raising=False)
    seeds._state.update(loaded=True, policy=None, required=None, suspended=False)
    yield
    seeds._state.clear()
    seeds._state.update(saved)


def _random_run(tmp_path, name="run", **kwargs):
    journal = tmp_path / f"{name}.jsonl"
    compete([PRESETS["random"]] * 4, 4, turn_cap=UNSTRUCTURED_TURN_CAP, journal=journal, **kwargs)
    return journal


def _source(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()
            if json.loads(line).get("kind") == "source"]


def test_a_policy_parses_and_digests_its_file(policy):
    assert policy.default_for(2) == 2000 and policy.default_for(4) == 1000 and policy.default_for(3) == 1000
    assert policy.training == {0, 1, 2, 10, 11}
    assert len(policy.sha256) == 64


@pytest.mark.parametrize("doc, problem", [
    ({**POLICY, "eval": {**POLICY["eval"], "default": {"2": 2000}}}, '"*"'),
    ({**POLICY, "eval": {**POLICY["eval"], "default": {"*": 3000}}}, "not screen seeds"),
    ({**POLICY, "eval": {**POLICY["eval"], "confirm": [1000]}}, "both screen and confirm"),
    ({**POLICY, "train": {"rl": [1000]}}, "also evaluation seeds"),
    ({**POLICY, "analysis": [3000]}, "analysis seeds .* are also evaluation seeds"),
    ({**POLICY, "train": {"analysis": [5]}}, "their own list"),
])
def test_an_inconsistent_policy_is_refused(tmp_path, doc, problem):
    with pytest.raises(ValueError, match=problem):
        seeds.SeedPolicy.from_file(_write(tmp_path / "p.json", doc))


def test_the_policy_file_is_found_above_a_directory(tmp_path):
    _write(tmp_path / seeds.FILENAME)
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert seeds.find_policy(deep) == tmp_path / seeds.FILENAME


def test_screen_seeds_pass_and_defaults_follow_the_table(policy):
    assert seeds.check_eval(1000, seats=4)["status"] == "screen"
    assert seeds.check_eval(2000, seats=4)["status"] == "screen"
    assert seeds.default_eval(2) == 2000 and seeds.default_eval(4) == 1000
    assert seeds.default_seed("rl", 4) == 0 and seeds.default_seed("bc", 4) == 10


def test_a_confirmation_seed_needs_a_registration_that_names_it(policy, tmp_path):
    with pytest.raises(seeds.SeedRefused, match="registration"):
        seeds.check_eval(3000, seats=4)
    other = tmp_path / "other.md"
    other.write_text("this test runs on seed 4000")
    with pytest.raises(seeds.SeedRefused, match="does not name seed 3000"):
        seeds.check_eval(3000, seats=4, registration=other)
    mine = tmp_path / "mine.md"
    mine.write_text("4,000 games on seed 3,000, indices 0-3,999")
    note = seeds.check_eval(3000, seats=4, registration=mine)
    assert note["status"] == "confirm" and len(note["registration"]["sha256"]) == 64


def test_a_training_seed_is_never_an_evaluation_seed(policy):
    with pytest.raises(seeds.SeedRefused, match="training seed"):
        seeds.check_eval(0, seats=4)


def test_any_other_seed_is_refused_and_says_how_to_bypass(policy):
    with pytest.raises(seeds.SeedRefused, match="--bypass-seed-policy"):
        seeds.check_eval(12345, seats=4)


def test_a_bypass_warns_once_and_is_noted(policy, capsys):
    first = seeds.check_eval(12345, seats=4, bypass="replaying a frozen kit")
    seeds.check_eval(12345, seats=4, bypass="replaying a frozen kit")
    assert first["status"] == "bypassed" and first["reason"] == "replaying a frozen kit"
    assert capsys.readouterr().err.count("SEED POLICY BYPASSED") == 1


def test_the_environment_bypasses_code_that_takes_no_flag(policy, monkeypatch):
    monkeypatch.setenv(seeds.BYPASS_ENV, "frozen kit")
    assert seeds.check_eval(12345, seats=4)["reason"] == "frozen kit"


def test_training_seeds_are_held_to_their_kind(policy):
    assert seeds.check_train(1, kind="rl")["status"] == "train"
    with pytest.raises(seeds.SeedRefused, match="approved bc training seed"):
        seeds.check_train(1, kind="bc")
    with pytest.raises(seeds.SeedRefused, match="no 'sl' seeds"):
        seeds.check_train(1, kind="sl")


def test_analysis_seeds_are_their_own_list(policy):
    assert seeds.check_train(20, kind="analysis")["status"] == "analysis"
    assert seeds.default_seed("analysis", 4) == 20
    with pytest.raises(seeds.SeedRefused, match="approved analysis seed"):
        seeds.check_train(0, kind="analysis")


def test_hold_fills_and_checks_a_tools_arguments(policy):
    import argparse

    parser = argparse.ArgumentParser()
    seeds.add_arguments(parser, flag="--null-seed", fallback=11)
    args = parser.parse_args([])
    assert seeds.hold(args, use="analysis", dest="null_seed", fallback=11)["status"] == "analysis"
    assert args.null_seed == 20
    args = parser.parse_args(["--null-seed", "11"])
    with pytest.raises(seeds.SeedRefused):
        seeds.hold(args, use="analysis", dest="null_seed", fallback=11)
    args = parser.parse_args(["--null-seed", "11", "--bypass-seed-policy", "an old null"])
    assert seeds.hold(args, use="analysis", dest="null_seed")["reason"] == "an old null"
    eval_args = parser.parse_args([])
    eval_args.seed = None
    assert seeds.hold(eval_args, 2)["status"] == "screen" and eval_args.seed == 2000


def test_compete_takes_the_default_and_keeps_the_note(policy, tmp_path):
    journal = _random_run(tmp_path)
    header, _ = read_journal(journal)
    assert header["seed"] == 1000
    assert _source(journal)[0]["seeds"] == {"policy": policy.sha256, "status": "screen"}


def test_compete_refuses_before_its_journal_exists(policy, tmp_path):
    with pytest.raises(seeds.SeedRefused):
        _random_run(tmp_path, seed=7)
    assert not (tmp_path / "run.jsonl").exists()


def test_a_bypassed_run_keeps_its_reason(policy, tmp_path):
    journal = _random_run(tmp_path, seed=7, seed_bypass="a probe")
    assert _source(journal)[0]["seeds"]["reason"] == "a probe"


def test_a_dataset_run_is_held_to_training_seeds(policy, tmp_path):
    journal = _random_run(tmp_path, seed_use="bc")
    assert read_journal(journal)[0]["seed"] == 10
    with pytest.raises(seeds.SeedRefused):
        _random_run(tmp_path, name="eval-seed", seed=1000, seed_use="bc")


def test_no_policy_checks_nothing_unless_one_is_required(no_policy, tmp_path):
    assert read_journal(_random_run(tmp_path, seed=7))[0]["seed"] == 7
    seeds.require("a host package")
    with pytest.raises(seeds.SeedRefused, match="a host package plays only under a seed policy"):
        _random_run(tmp_path, name="required", seed=7)


def test_suspend_turns_the_policy_off_here_and_below(policy, monkeypatch):
    seeds.suspend()
    assert seeds.policy() is None and os.environ[seeds.POLICY_ENV] == seeds.OFF
    seeds.require("a host package")
    assert seeds.check_eval(12345, seats=4) == {"policy": None}
