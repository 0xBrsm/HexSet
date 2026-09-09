# SPDX-License-Identifier: GPL-3.0-only
"""Board-level summaries for arena experiments using adjacent paired games."""
from collections.abc import Sequence

from hexset.arena import Estimate, Tournament, mean_interval, wilson


def paired_mean(values: Sequence[float]) -> Estimate:
    """Average each board's two readings before estimating uncertainty."""
    if len(values) % 2:
        raise ValueError("paired measurements need two readings per board")
    return mean_interval([(values[i] + values[i + 1]) / 2 for i in range(0, len(values), 2)])


def side_metrics(tournament: Tournament, slots: Sequence[int]) -> dict:
    """Win share of all games; unfinished games contribute zero wins.

    The normal interval uses board-pair means as independent observations.
    Wilson bounds are retained separately as a descriptive game-level measure;
    their independent-Bernoulli assumption does not hold for shared boards.
    """
    won = [float(w in slots) for w in tournament.winners]
    estimate = paired_mean(won)
    wins = int(sum(won))
    return {
        "games": tournament.games,
        "boards": estimate.samples,
        "decided": tournament.games - tournament.unfinished,
        "unfinished": tournament.unfinished,
        "wins": wins,
        "win_rate": wins / tournament.games if tournament.games else 0.0,
        "interval_95": [max(0.0, estimate.lower), min(1.0, estimate.upper)],
        "interval_unit": "board",
        "win_rate_denominator": "all games, including unfinished",
        "wilson_interval_95": list(wilson(wins, tournament.games)),
        "wilson_assumption": "independent games; paired games are correlated",
    }


def json_metrics(value):
    """Represent unbounded estimates as null in JSON-facing metrics."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_metrics(item) for item in value]
    return value
