"""The iso-KL solve, pinned against a response whose answer is already known.

The block this backs sets its treatment rate from `iso_kl_rate`, so the
arithmetic decides what gets trained rather than merely what gets reported.
The recorded `lr` -> end-to-end-KL response from the ppo4 learning-rate probe
is the fixture: it is real, it straddles the point where the response stops
being linear, and two of its points are exact arms whose answer is the arm.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from benchmarks.minibatch_iso_kl import _grid, iso_kl_rate  # noqa: E402

# lr -> finished-update KL, measured on one fixed batch off ppo3-500 (status.md).
# Linear to ~1.2e-3, superlinear past it.
RESPONSE = [
    {"lr": 3.0e-4, "end_to_end_kl": 0.0051, "entropy": 0.60, "steps": 95},
    {"lr": 6.0e-4, "end_to_end_kl": 0.0105, "entropy": 0.61, "steps": 95},
    {"lr": 1.2e-3, "end_to_end_kl": 0.0170, "entropy": 0.63, "steps": 95},
    {"lr": 2.4e-3, "end_to_end_kl": 0.0563, "entropy": 0.68, "steps": 95},
    {"lr": 4.8e-3, "end_to_end_kl": 0.1639, "entropy": 0.78, "steps": 95},
]


@pytest.mark.parametrize("arm", RESPONSE[:-1])
def test_a_target_that_is_an_arm_solves_to_that_arm(arm):
    got = iso_kl_rate(arm["end_to_end_kl"], RESPONSE)

    assert got is not None
    assert got["lr"] == pytest.approx(arm["lr"])
    assert got["entropy"] == pytest.approx(arm["entropy"])


def test_it_interpolates_inside_the_bracket_it_reports():
    got = iso_kl_rate(0.0080, RESPONSE)

    assert got["bracket"] == [3.0e-4, 6.0e-4]
    assert 3.0e-4 < got["lr"] < 6.0e-4
    assert 0.60 < got["entropy"] < 0.61


def test_it_interpolates_in_log_log_rather_than_linearly():
    """The knee is why: across it the two readings differ by 4%."""
    got = iso_kl_rate(0.0300, RESPONSE)  # between 1.2e-3 and 2.4e-3, over the knee

    linear = 1.2e-3 + (0.0300 - 0.0170) / (0.0563 - 0.0170) * (2.4e-3 - 1.2e-3)
    assert got["lr"] == pytest.approx(1.6671e-3, rel=1e-4)
    assert linear == pytest.approx(1.5969e-3, rel=1e-4)
    assert got["lr"] > linear


def test_a_target_outside_the_swept_range_is_not_extrapolated():
    assert iso_kl_rate(0.5, RESPONSE) is None
    assert iso_kl_rate(0.0001, RESPONSE) is None


def test_an_arm_that_ran_out_of_memory_is_not_a_bracket():
    starved = [dict(RESPONSE[1], oom=True)] + [RESPONSE[0], RESPONSE[2]]

    got = iso_kl_rate(0.0080, starved)

    assert got["bracket"] == [3.0e-4, 1.2e-3]


def test_the_grid_reads_sizes_and_the_full_batch_sentinel():
    assert _grid("4096:3e-4,6e-4") == [(4096, [3.0e-4, 6.0e-4])]
    assert _grid("full:1.2e-3") == [(-1, [1.2e-3])]
    assert [size for size, _ in _grid("4096:3e-4|16384:6e-4|full:1.2e-3")] == [
        4096,
        16384,
        -1,
    ]
