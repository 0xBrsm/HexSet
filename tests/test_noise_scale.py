# SPDX-License-Identifier: GPL-3.0-only
"""The gradient noise scale estimator, against gradients whose answer is known.

The measurement it backs is a claim about where this problem's training sits
relative to its critical batch size, so the arithmetic that produces the number
is worth pinning independently of the model it is run on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from benchmarks.noise_scale import estimate  # noqa: E402


def _measured(signal_sq: float, trace: float, batch: int) -> float:
    """`E|G_B|^2 = |G|^2 + tr(S)/B`, the identity the estimator inverts."""
    return signal_sq + trace / batch


def test_it_recovers_the_signal_and_noise_it_was_built_from():
    signal_sq, trace = 0.004, 200.0
    small, big = 256, 4096

    got = estimate(
        _measured(signal_sq, trace, small), _measured(signal_sq, trace, big), small, big
    )

    assert got["signal_g_squared"] == pytest.approx(signal_sq)
    assert got["noise_trace_sigma"] == pytest.approx(trace)
    assert got["b_simple"] == pytest.approx(trace / signal_sq)


def test_a_noiseless_gradient_has_no_scale_to_speak_of():
    got = estimate(_measured(0.5, 0.0, 64), _measured(0.5, 0.0, 1024), 64, 1024)

    assert got["noise_trace_sigma"] == pytest.approx(0.0)
    assert got["b_simple"] == pytest.approx(0.0)
    assert got["signal_share_at_b_big"] == pytest.approx(1.0)


def test_the_signal_share_is_the_fraction_of_the_gradient_that_is_real():
    """At B == B_simple, noise and signal contribute equally to the squared
    norm, so the true gradient is `sqrt(1/2)` of the estimate's length."""
    signal_sq, trace = 0.01, 40.96
    big = int(trace / signal_sq)

    got = estimate(_measured(signal_sq, trace, 256), _measured(signal_sq, trace, big), 256, big)

    assert got["b_simple"] == pytest.approx(big)
    assert got["signal_share_at_b_big"] == pytest.approx(0.5**0.5)


def test_a_gradient_lost_in_noise_refuses_to_report_a_scale():
    """`|G|^2` estimated at or below zero means the true gradient is not
    resolved at either size, so the honest answer is no number."""
    got = estimate(200.0, 1.0, 256, 4096)

    assert got["signal_g_squared"] <= 0
    assert got["b_simple"] is None


def test_a_larger_draw_measuring_more_variance_refuses_too():
    """Sampling error in the estimate itself, not a property of the gradient."""
    got = estimate(0.05, 0.06, 256, 4096)

    assert got["noise_trace_sigma"] < 0
    assert got["b_simple"] is None


def test_the_paired_probe_refuses_an_odd_cohort():
    from benchmarks.noise_scale import main

    with pytest.raises(SystemExit, match="even"):
        main(["--paired", "--games", "3"])


def test_pair_correlations_pair_by_index_not_by_arrival_order():
    from benchmarks.noise_scale import _pair_correlations
    from hexset.selfplay import Episode, Outcome, Transition

    def episode(index: int, points: tuple[int, ...]) -> Episode:
        # Only the fields the correlations read are meaningful: the outcome,
        # the index, and the *last* transition's own-value estimate. A zero
        # estimate makes the residual the reward itself, so rho_v must equal
        # rho exactly on this fixture.
        trajectories = tuple(
            (
                Transition(
                    seat=seat,
                    step=seat,
                    observation=None,
                    mask=None,
                    action=None,
                    index=0,
                    log_prob=0.0,
                    value=(0.0, 0.0, 0.0, 0.0),
                ),
            )
            for seat in range(4)
        )
        return Episode(
            index=index,
            seed=0,
            players=4,
            trajectories=trajectories,
            outcome=Outcome(
                winner=None, points=points, turns=1, actions=4, truncated=False
            ),
        )

    # Two pairs whose halves ended identically: whatever order they arrive in,
    # pairing by `index ^ 1` correlates r with an identical r', so every
    # correlation is exactly 1. Pairing by list position would correlate the
    # two different pairs' outcomes instead and fall short of it.
    shuffled = [
        episode(3, (2, 8, 6, 4)),
        episode(0, (10, 4, 4, 4)),
        episode(2, (2, 8, 6, 4)),
        episode(1, (10, 4, 4, 4)),
    ]
    got = _pair_correlations(shuffled, players=4)

    for measure in ("rho", "rho_v"):
        assert got[measure]["pooled"] == pytest.approx(1.0)
        assert got[measure]["per_seat"] == pytest.approx([1.0] * 4)
    assert got["rho"] == got["rho_v"], "a zero estimate leaves the residual as r"
