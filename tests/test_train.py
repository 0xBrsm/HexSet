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
    "--lanes",
    "4",
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
