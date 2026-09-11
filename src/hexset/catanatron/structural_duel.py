# SPDX-License-Identifier: GPL-3.0-only
"""Spawn-safe duel driver for the opt-in structural family.

The worker imports ``structuralcfg`` before catanatron parses player specs, so
custom entrant kinds are registered in a fresh interpreter as well as a fork.
This module has no import-time catanatron dependency and remains an isolated
future-family runner.
"""
from __future__ import annotations

from multiprocessing import get_context
import time


def _play_chunk(args):
    # This import is intentionally inside the worker function: it is the
    # registration boundary for spawn, where the parent's arena is absent.
    from . import structuralcfg  # noqa: F401
    from .duel import _play_chunk as play_chunk
    return play_chunk(args)


def run_duel(players_spec: str, num_games: int, workers: int, seed: int = 0):
    from catanatron.models.player import Color
    from .duel import DuelResult, shard_plan
    if num_games < 1 or workers < 1:
        raise ValueError("num_games and workers must be positive")
    parts = players_spec.split(",")
    colors = list(Color)[: len(parts)]
    labels = {color: f"{i}:{part}" for i, (color, part) in enumerate(zip(colors, parts))}
    shard_size, _ = shard_plan(num_games, workers)
    chunks = []
    remaining = num_games
    index = 0
    while remaining:
        count = min(shard_size, remaining)
        chunks.append((players_spec, count, seed + index))
        remaining -= count
        index += 1
    start = time.perf_counter()
    with get_context("spawn").Pool(min(workers, len(chunks))) as pool:
        shard_results = pool.map(_play_chunk, chunks)
    wins = {color: 0 for color in colors}
    points = {color: [] for color in colors}
    for shard_wins, shard_points in shard_results:
        for color, count in shard_wins.items():
            wins[color] = wins.get(color, 0) + count
        for color, values in shard_points.items():
            points.setdefault(color, []).extend(values)
    elapsed = time.perf_counter() - start
    return DuelResult(
        players_spec=players_spec, games=num_games, seconds=elapsed,
        labels=labels, seed=seed, workers=workers, wins=wins, points=points,
    )
