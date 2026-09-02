# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

from benchmarks.production_curve import CurvePoint, correlation, proxy_summary, ranks


def point(
    production: float,
    win_rate: float,
    win_interval: tuple[float, float],
    point_difference: float,
    point_interval: tuple[float, float],
) -> CurvePoint:
    return CurvePoint(
        production=production,
        wins=round(win_rate * 100),
        decided=100,
        win_rate=win_rate,
        win_interval=win_interval,
        candidate_points=7.0,
        intact_points=7.0 - point_difference,
        point_difference=point_difference,
        point_interval=point_interval,
        outcome_point_correlation=0.8,
        seconds=1.0,
    )


def test_correlation_reports_perfect_agreement_and_disagreement():
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_correlation_of_a_constant_signal_is_zero():
    assert correlation([1, 1, 1], [1, 2, 3]) == 0.0


def test_ranks_average_ties():
    assert ranks([30, 10, 10, 20]) == [3.0, 0.5, 0.5, 2.0]


def test_proxy_summary_finds_the_closest_distinguishable_level():
    points = [
        point(0.0, 0.2, (0.1, 0.3), -2.0, (-2.2, -1.8)),
        point(3.0, 0.45, (0.42, 0.48), -0.2, (-0.3, -0.1)),
        point(5.0, 0.49, (0.46, 0.52), -0.1, (-0.2, 0.0)),
        point(7.067, 0.5, (0.47, 0.53), 0.0, (-0.1, 0.1)),
        point(10.0, 0.51, (0.48, 0.54), 0.1, (0.0, 0.2)),
    ]

    summary = proxy_summary(points)

    assert summary["closest_distinguishable_production"] == 3.0
    assert summary["closest_familywise_distinguishable_production"] == 0.0
    assert summary["significant_sign_agreements"] == 2
    assert summary["significant_sign_conflicts"] == 0
    assert summary["pearson_at_or_above_3"] > 0.9
    assert summary["spearman_at_or_above_3"] == pytest.approx(1.0)
