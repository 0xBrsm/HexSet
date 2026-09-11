# SPDX-License-Identifier: GPL-3.0-only
"""Optional accelerations for the pinned Catanatron reference implementation.

The base evaluator below is derived from Catanatron players/value.py at
d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816 (GPL-3.0). Arithmetic, including
its fixed seven-card penalty and relative P1 opponent, is preserved exactly.
The scoped context is process-global and intended for single-threaded workers.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

from catanatron.features import (
    build_production_features, count_production, get_owned_or_buildable,
    get_zero_nodes, iter_level_nodes,
    reachability_features as native_reachability_features,
    resource_hand_features,
)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.models.board import Board
from catanatron.players import value as native_value
from catanatron.state_functions import (
    get_longest_road_length, get_played_dev_cards, player_key,
    player_num_dev_cards, player_num_resource_cards,
)

MODES = ("off", "basic", "cached")
_NATIVE_COPY = Board.copy
_NATIVE_BASE = native_value.base_fn
_active = False
_PRODUCTION = build_production_features(True)
value_production = native_value.value_production


@lru_cache(maxsize=1)
def verify_runtime() -> None:
    """Reject an unsupported upstream implementation before installing patches."""
    import catanatron.features as features
    from catanatron.models import board
    from catanatron.players import minimax
    expected = (
        (features, "f6ec7fd7f30f740a75c978c9a8166512c6514052a79802992216dbd812553746"),
        (native_value, "e25e6e15ff38141568119b912474761fee47a289ad3735a25913b8635f58a56b"),
        (board, "f22f004b5894d4002575dd2c5dad8feecec96d1557b140d057550701229f9d1a"),
        (minimax, "97a6bf10e5fac7f6655ad5b1614f268926ed86976ee59815d93f24a1b0e048dd"),
    )
    for module, digest in expected:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"unsupported Catanatron source for speedups: {module.__name__}")


def clone_connected_components(value: Any) -> Any:
    # Preserve defaultdict factory and insertion order.  Components are mutable
    # sets; node IDs and colors are immutable.
    if isinstance(value, defaultdict):
        out = defaultdict(value.default_factory)
    else:
        out = {}
    for color, components in value.items():
        out[color] = [set(list(component)) for component in components]
    return out


def clone_buildable_edges_cache(value: dict[Any, list[tuple[Any, ...]]]) -> dict[Any, list[tuple[Any, ...]]]:
    # Retain cached list order; do not regenerate from a set.
    return {color: list(edges) for color, edges in value.items()}


def clone_player_port_resources_cache(value: dict[Any, set[Any]]) -> dict[Any, set[Any]]:
    return {color: set(list(resources)) for color, resources in value.items()}


def clone_board_mutable_structures(board: Any) -> Any:
    """Copy mutable native board fields while sharing immutable map geometry."""
    out = type(board)(board.map, initialize=False)
    out.map = board.map
    out.buildings = board.buildings.copy()
    out.roads = board.roads.copy()
    out.connected_components = clone_connected_components(board.connected_components)
    out.board_buildable_ids = board.board_buildable_ids.copy()
    out.road_lengths = board.road_lengths.copy()
    out.road_color = board.road_color
    out.road_length = board.road_length
    out.robber_coordinate = board.robber_coordinate
    out.buildable_subgraph = board.buildable_subgraph
    out.buildable_edges_cache = clone_buildable_edges_cache(board.buildable_edges_cache)
    out.player_port_resources_cache = clone_player_port_resources_cache(board.player_port_resources_cache)
    return out


TARGET_LEVELS = 2


def p0_reachability_features(game, p0_color, levels=TARGET_LEVELS):
    """Return the exact ten reachability values consumed by pinned ``base_fn``.

    ``base_fn`` passes ``levels=2`` and subsequently reads only P0 levels 0 and
    1.  For all other calls, preserve the native full-helper behavior.
    """
    if levels != TARGET_LEVELS:
        return native_reachability_features(game, p0_color, levels)

    board = game.state.board
    board_buildable = board.buildable_node_ids(p0_color, True)
    color = p0_color
    owned_or_buildable = get_owned_or_buildable(game, color, board_buildable)
    zero_nodes = get_zero_nodes(game, color)
    features = {}

    production = count_production(
        frozenset(owned_or_buildable.intersection(zero_nodes)),
        board.map,
    )
    for resource in RESOURCES:
        features[f"P0_0_ROAD_REACHABLE_{resource}"] = production[resource]

    enemy_nodes = frozenset(
        k for k, v in board.buildings.items() if v is not None and v[0] != color
    )
    enemy_roads = frozenset(
        k for k, v in board.roads.items() if v is not None and v != color
    )
    # Native level 1 is independent of whether native level 2 is subsequently
    # generated, so levels=1 avoids computing the unused level-2 layer.
    for level, level_nodes, _paths in iter_level_nodes(
        enemy_nodes, enemy_roads, 1, frozenset(zero_nodes)
    ):
        production = count_production(
            frozenset(owned_or_buildable.intersection(level_nodes)),
            board.map,
        )
        for resource in RESOURCES:
            features[f"P0_{level}_ROAD_REACHABLE_{resource}"] = production[resource]

    return features


def _board_terms(game, color):
    # Native base_fn computes this same production sample twice.
    sample = _PRODUCTION(game, color)
    production = value_production(sample, "P0")
    enemy_production = value_production(sample, "P1", False)
    reach = p0_reachability_features(game, color, 2)
    zero = sum([reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES])
    one = sum([reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES])
    buildings = game.state.buildings_by_color[color]
    tiles = set()
    for node in buildings[SETTLEMENT] + buildings[CITY]:
        tiles.update(game.state.board.map.adjacent_tiles[node])
    return (production, enemy_production, zero, one, len(tiles),
            len(game.state.board.buildable_node_ids(color)))


class BoardFeatureCache:
    """Bounded cache shared by all native base evaluators in one worker.

    Map objects are retained and a map change clears entries. Ordered building
    lists preserve floating-point summation order. Component node iteration is
    retained because native reachability unions sets before summing production.
    Hands, VP, army, dev cards and longest-road length are read live, outside
    this cache. No game objects, final scores or search bounds are retained.
    """
    def __init__(self, max_entries=4096):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.map = None
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def terms(self, game, color):
        state = game.state
        board = state.board
        if board.map is not self.map:
            self.entries.clear()
            self.map = board.map
        key = (
            color, tuple(state.colors), board.robber_coordinate,
            tuple((c, tuple(state.buildings_by_color[c][SETTLEMENT]),
                   tuple(state.buildings_by_color[c][CITY])) for c in state.colors),
            tuple(board.buildings.items()), tuple(board.roads.items()),
            tuple(tuple(component) for component in board.connected_components[color]),
            tuple(board.board_buildable_ids),
        )
        value = self.entries.get(key)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        value = _board_terms(game, color)
        if len(self.entries) >= self.max_entries:
            # FIFO bounds memory without updating an LRU chain at every leaf.
            del self.entries[next(iter(self.entries))]
        self.entries[key] = value
        return value

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self.entries), "max_entries": self.max_entries}


@contextmanager
def catanatron_speedups(mode="cached"):
    """Apply and restore optimizations around a worker's native games.

    ``basic`` combines structural board copies, one production calculation and
    the consumed reachability subset. ``cached`` also reuses board terms across
    leaves/decisions for every player, with perspective included in the key.
    Native search, action ordering and the 20-second deadline are untouched.
    The patches also accelerate native F players using the same base evaluator.
    """
    global _active
    if mode not in MODES:
        raise ValueError(f"unknown Catanatron speedups: {mode!r}")
    if mode == "off":
        if _active:
            raise RuntimeError("cannot disable speedups inside an active context")
        yield None
        return
    verify_runtime()
    if _active or Board.copy is not _NATIVE_COPY or native_value.base_fn is not _NATIVE_BASE:
        raise RuntimeError("Catanatron speedups require an unmodified, non-nested worker")
    cache = BoardFeatureCache() if mode == "cached" else None
    _active = True
    try:
        Board.copy = clone_board_mutable_structures
        native_value.base_fn = lambda params=native_value.DEFAULT_WEIGHTS: optimized_base_fn(params, cache=cache)
        yield cache
    finally:
        native_value.base_fn = _NATIVE_BASE
        Board.copy = _NATIVE_COPY
        _active = False


def optimized_base_fn(params=native_value.DEFAULT_WEIGHTS, *, cache=None):
    def fn(game, p0_color):
        (production, enemy_production, reachable_production_at_zero,
         reachable_production_at_one, num_tiles, num_buildable_nodes) = (
            cache.terms(game, p0_color) if cache is not None
            else _board_terms(game, p0_color)
        )
        key = player_key(game.state, p0_color)
        longest_road_length = get_longest_road_length(game.state, p0_color)

        hand_sample = resource_hand_features(game, p0_color)
        distance_to_city = (
            max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
            + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
        ) / 5.0  # 0 means good. 1 means bad.
        distance_to_settlement = (
            max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
            + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
            + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
            + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
        ) / 4.0  # 0 means good. 1 means bad.
        hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

        num_in_hand = player_num_resource_cards(game.state, p0_color)
        discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

        longest_road_factor = (
            params["longest_road"] if num_buildable_nodes == 0 else 0.1
        )

        return float(
            game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
            + production * params["production"]
            + enemy_production * params["enemy_production"]
            + reachable_production_at_zero * params["reachable_production_0"]
            + reachable_production_at_one * params["reachable_production_1"]
            + hand_synergy * params["hand_synergy"]
            + num_buildable_nodes * params["buildable_nodes"]
            + num_tiles * params["num_tiles"]
            + num_in_hand * params["hand_resources"]
            + discard_penalty
            + longest_road_length * longest_road_factor
            + player_num_dev_cards(game.state, p0_color) * params["hand_devs"]
            + get_played_dev_cards(game.state, p0_color, "KNIGHT") * params["army_size"]
        )

    return fn
