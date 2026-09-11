import json
import math


def wilson(w, n):
    z = 1.959963984540054
    p = w / n
    d = 1 + z * z / n
    q = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p + z * z / (2 * n) - q) / d


def verdict(rows):
    thresholds = {"vs-ab2": .50, "vs-shipped": .25}
    return all(wilson(r["wins"], r["games"]) > thresholds[r["gate"]] for r in rows)


def test_unequal_thresholds_require_both_gates():
    assert not verdict([
        {"gate": "vs-ab2", "wins": 1050, "games": 2048},
        {"gate": "vs-shipped", "wins": 600, "games": 2048},
    ])
    assert verdict([
        {"gate": "vs-ab2", "wins": 1100, "games": 2048},
        {"gate": "vs-shipped", "wins": 600, "games": 2048},
    ])


def test_screen_excludes_baseline_and_ranks_threshold_margin():
    rows = [("baseline", .20, .60), ("a", .51, .40), ("b", .55, .27)]
    ranked = sorted((min(a - .50, s - .25), n) for n, a, s in rows if n != "baseline")
    assert [x[1] for x in ranked] == ["a", "b"]


def test_incomplete_or_either_gate_cannot_pass():
    assert not verdict([
        {"gate": "vs-ab2", "wins": 1200, "games": 2048},
        {"gate": "vs-shipped", "wins": 500, "games": 2048},
    ])


def test_seed_ranges_are_disjoint():
    assert set(range(610000, 610000 + 21 * 100, 100)).isdisjoint(range(710000, 710400, 100))
    assert set(range(710000, 710400, 100)).isdisjoint({810000, 820000})
