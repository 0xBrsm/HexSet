# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from hexset.bench.human_agreement import (
    Decision,
    Tally,
    grouped,
    option_key,
    positions,
    progress_bucket,
    report,
    score,
    summarise,
)
from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.game import Phase, start, to_move
from hexset.record import Record, advance, board_fields, board_of, record_game, steps

# Torch is never imported here. `hexset.bench.human_agreement` keeps it inside
# `main`, and the scoring core is duck-typed on `hexset.mcts.Evaluator`, so every
# number in this file is arithmetic over a stub whose prior is known in advance.


class Uniform:
    """A prior that says nothing, so the policy *is* its own null."""

    def evaluate(self, leaves):
        return [
            (np.full(len(leaf.options), 1.0 / len(leaf.options)), (0.0,) * 4)
            for leaf in leaves
        ]


class Peaked:
    """`weight` on the first option, the rest spread evenly over the others.

    The taken action's probability is then exactly `weight` on any position
    whose recorded action is the first legal one, which is what makes the
    log-loss a closed form rather than something to read off a run.
    """

    def __init__(self, weight: float = 0.7) -> None:
        self.weight = weight
        self.calls = 0

    def evaluate(self, leaves):
        out = []
        for leaf in leaves:
            count = len(leaf.options)
            prior = np.full(count, (1.0 - self.weight) / (count - 1))
            prior[0] = self.weight
            self.calls += 1
            out.append((prior, (0.0,) * 4))
        return out


class First:
    """A bot that always takes the first legal action, so `Peaked` is an oracle."""

    def choose(self, game):
        return legal_actions(game)[0]


def some_records(count: int = 4, bot: str = "greedy", players: int = 4):
    out = []
    for seed in range(count):
        board = random_base_board(random.Random(2000 + seed))
        bots = [
            spawn(PRESETS[bot], board, random.Random(seed * 16 + seat))
            for seat in range(players)
        ]
        out.append(record_game(bots, board, seed))
    return out


def a_first_option_record(seed: int = 0, cap: int = 400) -> Record:
    board = random_base_board(random.Random(2000 + seed))
    return record_game([First()] * 4, board, seed, action_cap=cap)


def a_two_action_record() -> tuple[Record, tuple[Action, Action]]:
    """A settlement and the road after it, taken as the first legal option each.

    Two decisions, both with the recorded action at index 0, so `Peaked`'s
    numbers are the whole answer and nothing about a real game enters them.
    """
    board = random_base_board(random.Random(0))
    game = start(board, 4, random.Random(0))
    first = legal_actions(game)[0]
    apply(game, first)
    second = legal_actions(game)[0]
    record = Record(
        num_players=4,
        seed=0,
        actions=(
            (int(first.type), first.a, first.b),
            (int(second.type), second.a, second.b),
        ),
        trades=(),
        winner=None,
        turns=0,
        **board_fields(board),
    )
    return record, (first, second)


def a_decision(options: int, *, agreed: bool = True, game: int = 0, kind=None) -> Decision:
    return Decision(
        game=game,
        step=0,
        seat=0,
        progress=0.5,
        phase="MAIN",
        kind=int(kind if kind is not None else ActionType.END_TURN),
        options=options,
        agreed=agreed,
        log_prob=-1.0,
    )


# ---------------------------------------------------------------------------
# The arithmetic case


def test_two_hand_built_decisions_score_to_their_closed_form():
    record, (first, second) = a_two_action_record()
    scored, tally = score([record], Peaked(0.7), batch=8)

    assert len(scored) == 2
    assert [d.kind for d in scored] == [int(first.type), int(second.type)]
    assert [d.phase for d in scored] == [
        Phase.SETUP_SETTLEMENT.name,
        Phase.SETUP_ROAD.name,
    ]
    assert tally.games == 1
    assert tally.actions == 2

    # Both recorded actions sit at index 0, so `Peaked` puts 0.7 on each.
    counts = [d.options for d in scored]
    assert all(d.agreed for d in scored)
    block = summarise(scored)
    assert block["top1"] == 1.0
    assert block["log_loss"] == pytest.approx(-math.log(0.7))

    # And the null is the mean of the two positions' own uniform baselines.
    assert block["top1_null"] == pytest.approx(sum(1 / n for n in counts) / 2)
    assert block["log_loss_null"] == pytest.approx(
        sum(math.log(n) for n in counts) / 2
    )
    assert block["mean_options"] == pytest.approx(sum(counts) / 2, abs=5e-3)


def test_a_uniform_policy_scores_exactly_its_own_null():
    """The tightest available check that the null is the right null.

    A policy that is uniform over the legal options has, by construction, the
    log-loss of the uniform baseline at every single position. If the two
    differ at all, the null is being computed somewhere other than the position
    it belongs to.
    """
    scored, _ = score(some_records(2), Uniform(), batch=32)
    block = summarise(scored)
    assert block["decisions"] > 100
    assert block["log_loss"] == pytest.approx(block["log_loss_null"], abs=1e-9)
    assert block["log_loss_gain"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The null is per position


def test_the_null_is_computed_per_position_not_from_a_global_constant():
    """`mean(1/n)` and `mean(log n)`, never `1/mean(n)` or `log(mean n)`.

    Two positions, two options and thirty-two, is enough to separate them: the
    per-position null is 26.6% and 2.079 nats, the global-constant version is
    5.9% and 2.833. Both wrong readings are also *flattering* here, which is
    why this is pinned rather than eyeballed.
    """
    scored = [a_decision(2), a_decision(32)]
    block = summarise(scored)

    assert block["top1_null"] == pytest.approx((1 / 2 + 1 / 32) / 2)
    assert block["log_loss_null"] == pytest.approx(
        (math.log(2) + math.log(32)) / 2
    )
    assert block["top1_null"] != pytest.approx(1 / 17)
    assert block["log_loss_null"] != pytest.approx(math.log(17))


def test_a_bucket_carries_its_own_null_and_not_the_aggregate_s():
    """Stratification is pointless if every stratum quotes one baseline."""
    scored = [
        a_decision(2, kind=ActionType.ROLL),
        a_decision(32, kind=ActionType.BANK_TRADE),
    ]
    blocks = grouped(scored, lambda d: ActionType(d.kind).name)
    assert blocks["ROLL"]["log_loss_null"] == pytest.approx(math.log(2))
    assert blocks["BANK_TRADE"]["log_loss_null"] == pytest.approx(math.log(32))


# ---------------------------------------------------------------------------
# What is excluded, and the accounting that has to close


def test_trivial_decisions_are_excluded_and_counted():
    scored, tally = score(some_records(2), Peaked(), batch=32)
    assert tally.trivial > 0
    assert all(d.options >= 2 for d in scored)
    # Every considered position is scored, trivial, or unrepresented — nothing
    # may fall out of the accounting unnamed.
    assert tally.considered == len(scored) + tally.trivial + tally.unrepresented


def test_a_forced_roll_is_exactly_the_kind_of_decision_excluded():
    """The exclusion is not an abstraction: `Phase.ROLL` without a knight in
    hand offers one action, and counting it would score the policy on a
    decision nobody made."""
    record = a_first_option_record()
    scored, tally = score([record], Peaked(), batch=32)
    assert tally.trivial > 0
    assert not any(d.phase == Phase.ROLL.name and d.options < 2 for d in scored)


def test_off_seat_decisions_are_skipped_and_counted():
    records = some_records(2)
    everyone, all_tally = score(records, Peaked(), batch=32)
    one, one_tally = score(records, Peaked(), seats={0}, batch=32)

    assert all_tally.off_seat == 0
    assert one_tally.off_seat > 0
    assert {d.seat for d in one} == {0}
    assert len(one) < len(everyone)


# ---------------------------------------------------------------------------
# The option key


def test_the_option_key_is_the_whole_action():
    """An `Action` is exactly its type and two operands now that a trade
    offer is not one, so `option_key` is the identity on it -- kept as a
    function because every comparison in the module goes through one place."""
    action = Action(ActionType.BANK_TRADE, 0, 4)
    assert option_key(action) == (int(ActionType.BANK_TRADE), 0, 4)
    assert option_key(action) != option_key(Action(ActionType.BANK_TRADE, 0, 3))








# ---------------------------------------------------------------------------
# The breakdown partitions the aggregate


def test_the_per_action_type_breakdown_sums_to_the_aggregate():
    scored, tally = score(some_records(3), Peaked(0.6), batch=32)
    payload = report(scored, tally)
    overall = payload["overall"]

    for key in ("by_action", "by_phase", "by_progress"):
        blocks = payload[key].values()
        total = sum(b["decisions"] for b in blocks)
        assert total == overall["decisions"], key
        for metric in ("top1", "top1_null", "log_loss", "log_loss_null"):
            pooled = sum(b["decisions"] * b[metric] for b in blocks) / total
            # Exact up to float summation order. `summarise` rounds nothing, so
            # a bucket that quoted a baseline it did not own would show here.
            assert pooled == pytest.approx(overall[metric], rel=1e-9), (key, metric)


def test_the_progress_breakdown_is_ordered_and_covers_the_whole_game():
    scored, tally = score([a_first_option_record()], Peaked(), batch=32)
    buckets = report(scored, tally)["by_progress"]
    assert list(buckets) == sorted(buckets)
    assert progress_bucket(a_decision(4)) in buckets


# ---------------------------------------------------------------------------
# Determinism, and that the replay is not disturbed


def test_two_runs_at_one_seed_score_the_same_decisions():
    records = some_records(3)
    first, first_tally = score(records, Peaked(), sample=0.3, seed=11, batch=16)
    again, again_tally = score(records, Peaked(), sample=0.3, seed=11, batch=16)

    assert first == again
    assert first_tally == again_tally
    assert 0 < len(first)
    assert first_tally.unsampled > 0


def test_a_different_seed_subsamples_different_decisions():
    records = some_records(3)
    first, _ = score(records, Peaked(), sample=0.3, seed=11, batch=16)
    other, _ = score(records, Peaked(), sample=0.3, seed=12, batch=16)
    assert [d.step for d in first] != [d.step for d in other]


def test_the_batch_size_cannot_change_the_answer():
    records = some_records(2)
    small, _ = score(records, Peaked(), batch=1)
    large, _ = score(records, Peaked(), batch=512)
    assert small == large


def test_snapshotting_a_position_does_not_disturb_the_replay():
    """`imagine` gets its own stream, and the scored copy is never stepped. If
    either were untrue the replay would desync from the recorded dice and the
    positions it walks would stop matching a plain replay."""
    record = some_records(1)[0]

    walked = []
    game = start(board_of(record), record.num_players, random.Random(record.seed))
    for action, trades in steps(record):
        walked.append((to_move(game), game.phase.name, len(legal_actions(game))))
        advance(game, action, trades)

    seen = []
    for item in positions(record, 0, Tally()):
        seen.append((item.seat, item.phase, len(item.leaf.options)))

    trimmed = [row for row in walked if row[2] >= 2]
    assert seen == trimmed
