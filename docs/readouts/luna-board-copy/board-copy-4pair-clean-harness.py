import datetime
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import hexset.catanatron.duel  # registers DC
from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import play_batch
from catanatron.models.board import Board

SPEC = "DC:heximax-notrade,AB:2,AB:2,AB:2"
SEEDS = list(range(291000000, 291000004))
OUT_DIR = os.environ.get("OUT_DIR", ".")
PROGRESS_PATH = os.path.join(OUT_DIR, "luna-board-copy-clean-progress.json")
RESULT_PATH = os.path.join(OUT_DIR, "luna-board-copy-clean-result.json")
ORIGINAL_COPY = Board.copy


def structural_copy(self):
    out = Board(self.map, initialize=False)
    out.map = self.map
    out.buildings = self.buildings.copy()
    out.roads = self.roads.copy()
    cc = defaultdict(self.connected_components.default_factory)
    for color, components in self.connected_components.items():
        cc[color] = [set(list(component)) for component in components]
    out.connected_components = cc
    out.board_buildable_ids = self.board_buildable_ids.copy()
    out.road_lengths = self.road_lengths.copy()
    out.road_color = self.road_color
    out.road_length = self.road_length
    out.robber_coordinate = self.robber_coordinate
    out.buildable_subgraph = self.buildable_subgraph
    out.buildable_edges_cache = {
        color: list(edges) for color, edges in self.buildable_edges_cache.items()
    }
    out.player_port_resources_cache = {
        color: set(list(resources))
        for color, resources in self.player_port_resources_cache.items()
    }
    return out


def board_shape(board):
    cc = board.connected_components
    return {
        "connected_components_type": type(cc).__name__,
        "connected_components_factory": repr(getattr(cc, "default_factory", None)),
        "component_list_types": sorted({type(v).__name__ for v in cc.values()}),
        "component_member_types": sorted({type(x).__name__ for vs in cc.values() for v in vs for x in v}),
        "edge_cache_type": type(board.buildable_edges_cache).__name__,
        "port_cache_type": type(board.player_port_resources_cache).__name__,
        "port_value_types": sorted({type(v).__name__ for v in board.player_port_resources_cache.values()}),
    }


def run_one(seed, cloned):
    random.seed(seed)
    Board.copy = structural_copy if cloned else ORIGINAL_COPY
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        players = parse_cli_string(SPEC)
        wins, points, games = play_batch(1, players, quiet=True)
    finally:
        elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start
        Board.copy = ORIGINAL_COPY
    game = games[0]
    return {
        "seed": seed,
        "clone": cloned,
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu_elapsed,
        "winner": repr(game.winning_color()),
        "wins": {repr(k): int(v) for k, v in wins.items()},
        "points": {repr(k): list(v) for k, v in points.items()},
        "action_trace": [repr(record) for record in game.state.action_records],
        "fallbacks": [getattr(player, "fallbacks", None) for player in players],
        "actions": len(game.state.action_records),
        "board_shape": board_shape(game.state.board),
    }


def atomic_write(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def source_hashes():
    root = Path(os.environ.get("SOURCE_ROOT", "/study/src"))
    names = [
        "hexset/catanatron/player.py",
        "hexset/catanatron/duel.py",
        "hexset/catanatron/board.py",
        "hexset/catanatron/actions.py",
    ]
    result = {}
    for name in names:
        path = root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


started = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows = []
qualified = True
mismatch = None
for seed in SEEDS:
    if (seed - SEEDS[0]) % 2 == 0:
        first = run_one(seed, False)
        second = run_one(seed, True)
    else:
        second = run_one(seed, True)
        first = run_one(seed, False)
    baseline = first if not first["clone"] else second
    clone = second if not first["clone"] else first
    rows.append({"baseline": baseline, "clone": clone})
    for field in ("winner", "wins", "points", "action_trace", "fallbacks", "board_shape"):
        if baseline[field] != clone[field]:
            qualified = False
            mismatch = {"seed": seed, "field": field, "baseline": baseline[field], "clone": clone[field]}
            break
    atomic_write(PROGRESS_PATH, {
        "started_utc": started,
        "spec": SPEC,
        "seed": seed,
        "completed_pairs": len(rows),
        "qualified_so_far": qualified,
        "mismatch": mismatch,
        "last_baseline_seconds": baseline["elapsed_seconds"],
        "last_clone_seconds": clone["elapsed_seconds"],
        "last_baseline_cpu_seconds": baseline["cpu_seconds"],
        "last_clone_cpu_seconds": clone["cpu_seconds"],
    })
    if mismatch is not None:
        break

finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = {
    "started_utc": started,
    "finished_utc": finished,
    "spec": SPEC,
    "seeds": SEEDS,
    "qualified": qualified,
    "mismatch": mismatch,
    "completed_pairs": len(rows),
    "rows": rows,
    "runtime": {
        "python": sys.version,
        "platform": platform.platform(),
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "source_root": os.environ.get("SOURCE_ROOT", "/study/src"),
        "source_hashes": source_hashes(),
        "image": os.environ.get("BENCH_IMAGE"),
        "cpus": os.environ.get("BENCH_CPUS"),
    },
}
atomic_write(RESULT_PATH, result)
print(json.dumps({"qualified": qualified, "mismatch": mismatch, "completed_pairs": len(rows), "result_path": RESULT_PATH}), flush=True)
