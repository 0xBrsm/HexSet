from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan import train  # noqa: E402
from catan.selfplay import Collector, RandomPolicy  # noqa: E402


TINY = [
    "--device",
    "cpu",
    "--width",
    "8",
    "--rounds",
    "1",
    # One lane, one game: `--collect-mode cohort` plays its whole cohort out,
    # so lanes above the cohort size would only sit empty.
    "--lanes",
    "1",
    "--games-per-iteration",
    "1",
    "--action-cap",
    "600",
    "--minibatch",
    "256",
    "--epochs",
    "1",
]


def run(directory, iterations, extra=()):
    return train.main(
        TINY
        + ["--iterations", str(iterations), "--checkpoint-dir", str(directory)]
        + list(extra)
    )


def test_crippled_flags_cpu_device_regardless_of_core_count():
    assert train._crippled("cpu", collect_workers=8, cores=32) is True


def test_crippled_flags_a_lone_collector_on_a_many_core_box():
    assert train._crippled("cuda", collect_workers=0, cores=32) is True


def test_crippled_is_false_off_cpu_with_workers_or_too_few_cores_to_shard():
    assert train._crippled("cuda", collect_workers=4, cores=32) is False
    assert train._crippled("cuda", collect_workers=0, cores=4) is False


def test_a_run_prints_the_effective_device_and_worker_counts(tmp_path, capsys):
    """Loud rather than a changed default -- see the module docstring's rule."""
    assert run(tmp_path, 1) == 0

    err = capsys.readouterr().err
    assert "device=cpu" in err
    assert "collect-workers=0" in err
    assert "update-workers=0" in err


def test_the_default_configuration_warns_that_it_is_crippled(tmp_path, capsys):
    assert run(tmp_path, 1) == 0

    assert "WARNING" in capsys.readouterr().err


def test_a_run_writes_a_checkpoint_carrying_the_weights_and_the_game_counter(tmp_path):
    assert run(tmp_path, 1, ["--checkpoint-every", "1"]) == 0

    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 1
    assert state["games_started"] > 0
    assert state["net"], "no weights in the checkpoint"
    assert "state" in state["optimiser"]
    # Anti-vacuity: a checkpoint that saved nothing would still have the keys.
    assert any(v.numel() for v in state["net"].values())


def test_numbered_checkpoints_are_kept_alongside_the_one_that_gets_overwritten(
    tmp_path,
):
    assert run(tmp_path, 4, ["--checkpoint-every", "1", "--keep-every", "2"]) == 0

    kept = sorted(p.name for p in tmp_path.glob("iter-*.pt"))
    assert kept == ["iter-00002.pt", "iter-00004.pt"]
    assert (tmp_path / "latest.pt").exists()
    # The point of keeping them is that they differ; identical copies of the
    # final weights would answer nothing about when training stopped helping.
    early = torch.load(tmp_path / "iter-00002.pt", weights_only=False)
    late = torch.load(tmp_path / "iter-00004.pt", weights_only=False)
    assert early["iteration"] == 2 and late["iteration"] == 4
    assert any(
        not torch.equal(early["net"][k], late["net"][k]) for k in early["net"]
    )


def test_keeping_can_be_switched_off(tmp_path):
    assert run(tmp_path, 2, ["--checkpoint-every", "1", "--keep-every", "0"]) == 0
    assert list(tmp_path.glob("iter-*.pt")) == []


def test_a_resumed_run_carries_on_from_the_iteration_it_reached(tmp_path):
    run(tmp_path, 1, ["--checkpoint-every", "1"])
    first = torch.load(tmp_path / "latest.pt", weights_only=False)

    run(tmp_path, 3, ["--checkpoint-every", "1", "--resume"])
    second = torch.load(tmp_path / "latest.pt", weights_only=False)

    assert first["iteration"] == 1
    assert second["iteration"] == 3
    # It continued rather than restarted: the game counter only moves forward.
    assert second["games_started"] > first["games_started"]

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [record["iteration"] for record in lines] == [0, 1, 2]


def test_resuming_with_nothing_to_resume_is_an_error_rather_than_a_fresh_start(tmp_path):
    """It used to fall through and silently begin at iteration 0.

    On a 150-iteration GPU block that is hours spent discarding the campaign,
    with a log that starts at 0 and looks perfectly healthy. A typo'd
    `--checkpoint-dir` or an unseeded directory is all it takes. Same shape as
    the learning rate `--resume` used to throw away: a flag that quietly does
    nothing.
    """
    with pytest.raises(SystemExit):
        run(tmp_path / "empty", 1, ["--checkpoint-every", "1", "--resume"])


def test_a_resumed_run_steps_at_the_learning_rate_it_was_given(tmp_path):
    """The bug that silently voided a whole 150-iteration block.

    `Optimizer.load_state_dict` rebuilds `param_groups` from the *saved* groups,
    keeping only `params` from the live ones, so every hyperparameter comes back
    from the checkpoint and the command line is discarded. ppo4 launched with
    `--learning-rate 6e-4`, wrote 6e-4 into both its `args` and `config` blobs,
    and stepped Adam at the 3e-4 baked into the checkpoint it resumed from. Every
    gauge matched the previous run to within noise, because the configuration
    *was* the previous run's. Nothing in `log.jsonl` could show it, which is why
    `lr` is asserted here too.
    """
    assert run(tmp_path, 1, ["--checkpoint-every", "1", "--learning-rate", "3e-4"]) == 0
    first = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert first["optimiser"]["param_groups"][0]["lr"] == pytest.approx(3e-4)

    assert (
        run(
            tmp_path,
            2,
            ["--checkpoint-every", "1", "--resume", "--learning-rate", "6e-4"],
        )
        == 0
    )
    resumed = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert resumed["optimiser"]["param_groups"][0]["lr"] == pytest.approx(6e-4)

    # And the rate the update actually used is in the log, not just the args.
    rows = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[-1]["lr"] == pytest.approx(6e-4)


def test_the_adaptive_schedule_raises_the_rate_when_the_kl_runs_cold(tmp_path):
    """A cold gauge must move the knob, and the knob must reach the log.

    At `--target-kl 10` no realistic update can fill the band, so the controller
    is obliged to multiply the rate every iteration. That direction is the one
    that matters here: the whole campaign ran a third of the way into the
    conventional KL band and never moved the rate at all.
    """
    assert (
        run(
            tmp_path,
            2,
            [
                "--checkpoint-every",
                "1",
                "--learning-rate",
                "3e-4",
                "--lr-schedule",
                "adaptive",
                "--target-kl",
                "10",
            ],
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["lr"] == pytest.approx(3e-4)
    assert rows[1]["lr"] > rows[0]["lr"], "a cold KL left the rate where it was"


def test_a_resumed_run_plays_new_games_rather_than_the_ones_it_learned_from(tmp_path):
    # A game is a pure function of the seed and its index, so restarting the
    # counter would replay the training set — and it would look like it worked.
    first = Collector(RandomPolicy(random.Random(0)), lanes=4, seed=3, action_cap=600)
    first.collect(2)
    reached = first.games_started()
    assert reached > 0

    resumed = Collector(
        RandomPolicy(random.Random(0)),
        lanes=4,
        seed=3,
        action_cap=600,
        first_game=reached,
    )
    played = {e.index for e in resumed.collect(2)}
    assert played, "the resumed collector finished nothing"
    assert not (played & set(range(reached))), f"replayed {played & set(range(reached))}"
    assert min(played) >= reached


def test_a_failed_save_leaves_the_last_good_checkpoint_readable(tmp_path):
    # The whole point of writing to a temporary file and renaming: a crash
    # during `torch.save` must not take the previous checkpoint with it.
    path = tmp_path / "latest.pt"
    train.save(path, {"iteration": 1, "marker": "good"})

    real = torch.save

    def explode(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("crashed mid-save")

    torch.save = explode
    try:
        with pytest.raises(RuntimeError):
            train.save(path, {"iteration": 2, "marker": "bad"})
    finally:
        torch.save = real

    survived = torch.load(path, weights_only=False)
    assert survived["marker"] == "good"
    assert survived["iteration"] == 1


def test_a_save_never_leaves_a_partial_file_where_the_checkpoint_belongs(tmp_path):
    path = tmp_path / "latest.pt"
    train.save(path, {"iteration": 7})
    assert path.exists()
    assert not list(tmp_path.glob("*.partial"))


class Fixed:
    """A policy that answers with a marker, so dispatch order can be checked."""

    def __init__(self, marker):
        self.marker = marker
        self.batches = []

    def act(self, requests):
        from catan.selfplay import Choice

        self.batches.append([r.seat for r in requests])
        return [Choice(action=r.options[0], log_prob=float(self.marker)) for r in requests]


def test_a_mixed_policy_asks_each_side_once_and_answers_in_request_order():
    network, other = Fixed(1), Fixed(2)
    mixed = train.MixedPolicy(network, other, network_seats=(0, 2))

    collector = Collector(mixed, lanes=8, seed=4)
    # Every lane opens on seat 0's setup placement and the lanes stay in step
    # through the snake draft, so the first tick puts the whole batch on one
    # side and would test the split vacuously. Run on until the seats disagree,
    # which is the only case that exercises both branches.
    for _ in range(200):
        requests = [
            collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)
        ]
        if len({r.seat in (0, 2) for r in requests}) == 2:
            break
        collector.tick()
    else:
        pytest.fail(f"lanes never split across the seats: {[r.seat for r in requests]}")

    network.batches.clear()
    other.batches.clear()
    choices = mixed.act(requests)

    assert len(choices) == len(requests)
    for request, choice in zip(requests, choices):
        expected = 1.0 if request.seat in (0, 2) else 2.0
        assert choice.log_prob == expected
        assert choice.action in request.options

    # One call each, not one per position — the dispatch toll is per call.
    assert len(network.batches) == 1
    assert len(other.batches) == 1


def test_a_mixed_policy_still_answers_when_one_side_has_no_lanes():
    network, other = Fixed(1), Fixed(2)
    mixed = train.MixedPolicy(network, other, network_seats=(0, 1, 2, 3))
    collector = Collector(mixed, lanes=4, seed=6)
    requests = [
        collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)
    ]

    choices = mixed.act(requests)
    assert all(c.log_prob == 1.0 for c in choices)
    assert other.batches == []


def test_alternating_swaps_the_pairs_by_game_parity():
    caster = train.alternating(4)
    assert caster(0) == (0, 1, 0, 1)
    assert caster(1) == (1, 0, 1, 0)
    assert caster(2) == (0, 1, 0, 1)


def test_the_mix_caster_is_pure_and_respects_its_fractions():
    caster = train.mixed_caster([0.5], players=4, seed=3)
    casts = [caster(index) for index in range(400)]

    # Pure in the index: a resumed run must cast the same games the same way.
    assert casts == [caster(index) for index in range(400)]

    mixed = [cast for cast in casts if any(cast)]
    assert 140 <= len(mixed) <= 260, f"{len(mixed)} of 400 at a nominal 200"
    for cast in mixed:
        seats = tuple(seat for seat, pid in enumerate(cast) if pid == 1)
        assert seats in ((1, 3), (0, 2))


def test_parse_mix_rejects_overcommitted_or_empty_shares():
    assert train.parse_mix("") == []
    assert train.parse_mix("greedy=0.25,parent=0.25") == [
        ("greedy", 0.25),
        ("parent", 0.25),
    ]
    with pytest.raises(ValueError):
        train.parse_mix("greedy=0.7,parent=0.7")
    with pytest.raises(ValueError):
        train.parse_mix("greedy=0")


def test_versus_scores_the_learner_across_rotating_seats():
    result = train.versus(
        RandomPolicy(random.Random(0)),
        RandomPolicy(random.Random(1)),
        games=4,
        lanes=4,
        players=4,
        seed=21,
        max_offers=3,
    )
    assert result["games"] == 4
    assert 0 <= result["wins"] <= 4
    assert result["wilson_low"] <= result["win_rate"] <= result["wilson_high"]
    assert result["paired_vp_low"] <= result["paired_vp"] <= result["paired_vp_high"]


def test_a_run_with_a_greedy_mix_trains_and_logs_the_canary(tmp_path):
    assert run(tmp_path, 1, ["--checkpoint-every", "1", "--mix", "greedy=1.0"]) == 0

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    record = lines[-1]
    # The canary is in the log, counted over learner seats only.
    assert "accepts_per_seat_game" in record
    assert "proposes_per_seat_game" in record
    assert record["accepts_per_seat_game"] >= 0.0
    # The update really consumed learner transitions from cast games.
    assert record["positions"] > 0


def test_the_ladder_reports_both_rungs_against_a_parent(tmp_path):
    run(tmp_path / "first", 1, ["--checkpoint-every", "1"])
    parent = tmp_path / "first" / "latest.pt"

    assert (
        run(
            tmp_path / "second",
            1,
            [
                "--checkpoint-every",
                "1",
                "--parent",
                str(parent),
                "--mix",
                "parent=0.5",
                "--eval-every",
                "1",
                "--eval-games",
                "2",
            ],
        )
        == 0
    )

    lines = [
        json.loads(l)
        for l in (tmp_path / "second" / "log.jsonl").read_text().splitlines()
    ]
    ladder = lines[-1]["ladder"]
    assert set(ladder) == {"parent", "greedy"}
    for rung in ladder.values():
        assert rung["games"] == 2
        assert rung["paired_vp_low"] <= rung["paired_vp"] <= rung["paired_vp_high"]


def test_a_named_entrant_can_be_added_to_the_ladder_as_a_rung(tmp_path):
    """`--search-rung` was specified in the run-2 design and never wired in.

    The design called for rungs against `greedy-offers3`, `search2-offers3` and
    the run-1 checkpoint; `train.py` only ever built `parent` and `greedy`, so
    every ladder reading in ppo2, ppo3 and ppo4 is missing the search rung. The
    entrant here is a cheap one — what is under test is the wiring, not the bot,
    and `search2-offers3` costs a 2-ply search per move which a unit test should
    not pay. That it resolves at all is pinned separately below.

    The existing two rungs must keep their names, because the campaign's trend is
    only comparable across runs if `parent` and `greedy` keep reading the same
    thing.
    """
    assert (
        run(
            tmp_path,
            1,
            [
                "--checkpoint-every",
                "1",
                "--eval-every",
                "1",
                "--eval-games",
                "2",
                "--search-rung",
                "greedy-offers3",
            ],
        )
        == 0
    )
    lines = [
        json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    ladder = lines[-1]["ladder"]
    assert set(ladder) == {"greedy", "greedy-offers3"}
    assert ladder["greedy-offers3"]["games"] == 2


def test_the_search_rung_resolves_and_a_mistyped_one_fails_with_a_sentence():
    """The entrant the eval target actually wants, and the typo path.

    `search2-offers3` is measured at parity with catanatron's `AB:2`, lives in
    this repo, and plays the trading game catanatron does not model at all — so it
    can read the dimension the external benchmark is blind to. Constructed here
    but never played: `BotPolicy` builds its bots per board on demand, so this
    stays cheap.
    """
    from catan.collect import named_opponent

    assert named_opponent("search2-offers3", seed=0, lanes=4) is not None
    with pytest.raises(SystemExit):
        named_opponent("search2-offer3", seed=0, lanes=4)


def test_a_run_with_collect_workers_trains_and_checkpoints(tmp_path):
    assert (
        run(
            tmp_path,
            1,
            ["--checkpoint-every", "1", "--collect-workers", "2", "--mix", "greedy=0.5"],
        )
        == 0
    )

    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 1
    assert state["games_started"] > 0
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert lines[-1]["positions"] > 0


def test_async_collection_requires_parallel_workers(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, 0, ["--async-collect"])


def test_async_collection_prefetches_each_batch_once(tmp_path):
    assert (
        run(
            tmp_path,
            2,
            [
                "--checkpoint-every",
                "1",
                "--collect-workers",
                "2",
                # One lane per worker: a tick then finishes at most one game,
                # so the exact cohort counts below cannot be blurred by two
                # capped lanes ending on the same tick.
                "--lanes",
                "2",
                # Prefetching *is* the off-policy path, so it has to say so.
                "--collect-mode",
                "stream",
                "--async-collect",
            ],
        )
        == 0
    )

    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 2
    lines = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert [line["iteration"] for line in lines] == [0, 1]
    # One game per iteration, with no unused final prefetch.
    assert lines[-1]["games"] == 2
    assert all(line["positions"] > 0 for line in lines)


def _rows(directory):
    return [
        json.loads(line)
        for line in (directory / "log.jsonl").read_text().splitlines()
    ]


def test_a_cohort_run_trains_on_policy(tmp_path):
    """The number this whole change exists to move.

    `approx_kl_first_minibatch` is the divergence between the policy that
    played the batch and the policy about to learn from it, read *before* any
    gradient step — so on a batch the current weights actually played it is
    zero by construction. It must stay zero on every iteration, not merely the
    first: iteration 1 read zero in production too, because the collector had
    just been built, and the campaign's four generations of staleness only
    appeared once lanes started carrying games across the weight sync.
    """
    assert run(tmp_path, 3) == 0

    rows = [row for row in _rows(tmp_path) if "approx_kl_first_minibatch" in row]
    assert len(rows) == 3
    assert all(row["collect_mode"] == "cohort" for row in rows)
    assert all(row["positions"] > 0 for row in rows), "vacuously on-policy"
    assert all(abs(row["approx_kl_first_minibatch"]) < 1e-6 for row in rows), [
        row["approx_kl_first_minibatch"] for row in rows
    ]


def test_streaming_collection_ignores_the_batch_size_it_was_asked_for(tmp_path):
    """The contrast, which is what makes the test above worth having.

    `--games-per-iteration` reads like a batch size and under `stream` it is
    not one: `collect` stops on the tick that finishes the *first* game, and
    every other lane that ended on the same tick comes along. Asking for one
    game on four lanes trains on four. A cohort delivers what it was asked for.

    The staleness half of the same defect is pinned in `test_selfplay`, where
    games vary in length. It cannot be shown here: `TINY` caps every game at
    600 actions, so the lanes start together, truncate together, and never
    desynchronise enough to leave one mid-play across a weight sync.
    """
    assert run(tmp_path, 2, ["--collect-mode", "stream", "--lanes", "4"]) == 0
    streamed = [row for row in _rows(tmp_path) if "positions" in row]
    assert all(row["collect_mode"] == "stream" for row in streamed)
    assert all(row["games"] % 4 == 0 for row in streamed), "one game was asked for"

    fresh = tmp_path / "cohort"
    assert run(fresh, 2, ["--lanes", "1"]) == 0
    cohort = [row for row in _rows(fresh) if "positions" in row]
    assert [row["games"] for row in cohort] == [1, 2]
    assert all(
        row["positions"] < other["positions"] for row, other in zip(cohort, streamed)
    ), "the cohort trained on no less data than the four-lane stream"


def test_a_cohort_cannot_be_asked_for_more_lanes_than_it_deals(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, 1, ["--lanes", "8", "--games-per-iteration", "4"])


def test_prefetching_has_to_opt_out_of_the_on_policy_guarantee(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, 1, ["--collect-workers", "2", "--async-collect"])
