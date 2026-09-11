# SPDX-License-Identifier: GPL-3.0-only
"""Optional accelerations for the pinned Catanatron reference implementation.

The base evaluator below is derived from Catanatron players/value.py at
ecf931181b9a65bb4116a2153fb78c16f1438e00 (GPL-3.0; `base_fn` is unchanged
from d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816). Arithmetic, including
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
from catanatron.state import State
from catanatron.players import value as native_value
from catanatron.state_functions import (
    get_longest_road_length, get_played_dev_cards, player_key,
    player_num_dev_cards, player_num_resource_cards,
)

MODES = ("off", "basic", "cached", "fast")
_NATIVE_COPY = Board.copy
_NATIVE_STATE_COPY = State.copy
_NATIVE_BASE = native_value.base_fn
_active = False
_PRODUCTION = build_production_features(True)
value_production = native_value.value_production


@lru_cache(maxsize=1)
def verify_runtime() -> None:
    """Reject an unsupported upstream implementation before installing patches."""
    import catanatron.features as features
    import catanatron.state as native_state
    from catanatron.models import board
    from catanatron.players import minimax
    expected = (
        (native_state, "be1e4da790b21589a171aed8068a847594cdde8da9818c35c89988b4ae5f2e15"),
        (features, "f6ec7fd7f30f740a75c978c9a8166512c6514052a79802992216dbd812553746"),
        (native_value, "22d207049f55247e314513adfa5a23c52e998578a81cdff44094f91429903e6b"),
        (board, "f22f004b5894d4002575dd2c5dad8feecec96d1557b140d057550701229f9d1a"),
        (minimax, "45499973edf091a7a7bf2c0fa02958dd39ca9bb3a7301d57c3bb7292e83f454d"),
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


def clone_state_mutable_structures(self):
    """Creates a copy of this State class that can be modified without
    repercusions to this one. Immutable values are just copied over.

    Returns:
        State: State copy.
    """
    state_copy = State([], None, initialize=False)
    # Shared by reference, like the native copy: simulations on copies must
    # advance the same per-game stream (upstream `State.copy` does this).
    state_copy.random = self.random
    state_copy.players = self.players
    state_copy.discard_limit = self.discard_limit  # immutable
    state_copy.friendly_robber = self.friendly_robber  # immutable

    state_copy.board = self.board.copy()

    state_copy.player_state = self.player_state.copy()
    state_copy.color_to_index = self.color_to_index
    state_copy.colors = self.colors  # immutable

    state_copy.resource_freqdeck = self.resource_freqdeck.copy()
    state_copy.development_listdeck = self.development_listdeck.copy()

    state_copy.buildings_by_color = {
        color: (defaultdict(buildings.default_factory,
                            ((kind, nodes.copy()) for kind, nodes in buildings.items()))
                if isinstance(buildings, defaultdict)
                else {kind: nodes.copy() for kind, nodes in buildings.items()})
        for color, buildings in self.buildings_by_color.items()
    }
    state_copy.action_records = self.action_records.copy()
    state_copy.num_turns = self.num_turns

    # Current prompt / player
    # Two variables since there can be out-of-turn plays
    state_copy.current_player_index = self.current_player_index
    state_copy.current_turn_index = self.current_turn_index

    state_copy.current_prompt = self.current_prompt
    state_copy.is_initial_build_phase = self.is_initial_build_phase
    state_copy.is_discarding = self.is_discarding
    state_copy.discard_counts = self.discard_counts.copy()
    state_copy.is_moving_knight = self.is_moving_knight
    state_copy.is_road_building = self.is_road_building
    state_copy.free_roads_available = self.free_roads_available

    state_copy.is_resolving_trade = self.is_resolving_trade
    state_copy.current_trade = self.current_trade
    state_copy.acceptees = self.acceptees

    return state_copy

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
    def __init__(self, max_entries=4096, *, compact=False):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.compact = compact
        self.colors = None
        self.map = None
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def terms(self, game, color):
        state = game.state
        board = state.board
        if board.map is not self.map or state.colors != self.colors:
            self.entries.clear()
            self.map = board.map
            self.colors = state.colors
        if self.compact:
            # Cache the inputs actually consumed by the board terms. Enemy
            # ownership identity/type is irrelevant to P0 reachability; its
            # node/edge blockers are sufficient. Ordered per-seat buildings
            # still identify production for every color and relative P1.
            key = (
                color.value, board.robber_coordinate,
                tuple((tuple(state.buildings_by_color[c][SETTLEMENT]),
                       tuple(state.buildings_by_color[c][CITY])) for c in self.colors),
                frozenset(n for n, owner in board.buildings.items()
                          if owner is not None and owner[0] != color),
                frozenset(e for e, owner in board.roads.items()
                          if owner is not None and owner != color),
                tuple(tuple(component) for component in board.connected_components[color]),
                tuple(board.board_buildable_ids),
            )
        else:
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
    ``fast`` additionally uses structural state copies, direct own-hand reads,
    and a smaller dependency-complete cache key without repeated enum hashing.
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
    if (_active or Board.copy is not _NATIVE_COPY
            or State.copy is not _NATIVE_STATE_COPY or native_value.base_fn is not _NATIVE_BASE):
        raise RuntimeError("Catanatron speedups require an unmodified, non-nested worker")
    cache = BoardFeatureCache(compact=mode == "fast") if mode in ("cached", "fast") else None
    _active = True
    try:
        Board.copy = clone_board_mutable_structures
        if mode == "fast":
            State.copy = clone_state_mutable_structures
        native_value.base_fn = lambda params=native_value.DEFAULT_WEIGHTS: optimized_base_fn(
            params, cache=cache, direct_hand=mode == "fast"
        )
        yield cache
    finally:
        native_value.base_fn = _NATIVE_BASE
        Board.copy = _NATIVE_COPY
        State.copy = _NATIVE_STATE_COPY
        _active = False


def optimized_base_fn(params=native_value.DEFAULT_WEIGHTS, *, cache=None, direct_hand=False):
    def fn(game, p0_color):
        (production, enemy_production, reachable_production_at_zero,
         reachable_production_at_one, num_tiles, num_buildable_nodes) = (
            cache.terms(game, p0_color) if cache is not None
            else _board_terms(game, p0_color)
        )
        key = player_key(game.state, p0_color)
        longest_road_length = get_longest_road_length(game.state, p0_color)

        if direct_hand:
            player_state = game.state.player_state
            wheat = player_state[key + "_WHEAT_IN_HAND"]
            ore = player_state[key + "_ORE_IN_HAND"]
            sheep = player_state[key + "_SHEEP_IN_HAND"]
            brick = player_state[key + "_BRICK_IN_HAND"]
            wood = player_state[key + "_WOOD_IN_HAND"]
        else:
            hand_sample = resource_hand_features(game, p0_color)
            wheat, ore, sheep, brick, wood = (
                hand_sample["P0_WHEAT_IN_HAND"], hand_sample["P0_ORE_IN_HAND"],
                hand_sample["P0_SHEEP_IN_HAND"], hand_sample["P0_BRICK_IN_HAND"],
                hand_sample["P0_WOOD_IN_HAND"],
            )
        distance_to_city = (
            max(2 - wheat, 0)
            + max(3 - ore, 0)
        ) / 5.0  # 0 means good. 1 means bad.
        distance_to_settlement = (
            max(1 - wheat, 0)
            + max(1 - sheep, 0)
            + max(1 - brick, 0)
            + max(1 - wood, 0)
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
