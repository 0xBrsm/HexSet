import json, os, random, time
from collections import defaultdict
import hexset.catanatron.duel  # register DC adapter
from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import play_batch
from catanatron.models.board import Board
from catanatron.players import minimax

SPEC = "DC:heximax-notrade,AB:2,AB:2,AB:2"
SEEDS = list(range(290000000, 290000008))
OUT_DIR = os.environ.get("OUT_DIR", ".")
PROGRESS_PATH = os.path.join(OUT_DIR, "luna-board-copy-progress.json")
RESULT_PATH = os.path.join(OUT_DIR, "luna-board-copy-result.json")
ORIGINAL_COPY = Board.copy
ORIGINAL_EXPAND = minimax.expand_spectrum

def structural_copy(self):
    board = Board(self.map, initialize=False)
    board.map = self.map
    board.buildings = self.buildings.copy()
    board.roads = self.roads.copy()
    cc = defaultdict(self.connected_components.default_factory)
    for color, components in self.connected_components.items():
        cc[color] = [set(list(component)) for component in components]
    board.connected_components = cc
    board.board_buildable_ids = self.board_buildable_ids.copy()
    board.road_lengths = self.road_lengths.copy()
    board.road_color = self.road_color
    board.road_length = self.road_length
    board.robber_coordinate = self.robber_coordinate
    board.buildable_subgraph = self.buildable_subgraph
    board.buildable_edges_cache = {
        color: list(edges)
        for color, edges in self.buildable_edges_cache.items()
    }
    board.player_port_resources_cache = {
        color: set(list(resources))
        for color, resources in self.player_port_resources_cache.items()
    }
    return board

def board_shape(board):
    cc = board.connected_components
    edge = board.buildable_edges_cache
    ports = board.player_port_resources_cache
    return {
        "connected_components_type": type(cc).__name__,
        "connected_components_factory": repr(getattr(cc, "default_factory", None)),
        "component_list_types": sorted({type(v).__name__ for v in cc.values()}),
        "component_member_types": sorted({type(item).__name__ for vals in cc.values() for item in vals}),
        "edge_cache_type": type(edge).__name__,
        "edge_value_types": sorted({type(v).__name__ for v in edge.values()}),
        "port_cache_type": type(ports).__name__,
        "port_value_types": sorted({type(v).__name__ for v in ports.values()}),
        "connected_component_aliases": sum(
            id(a) == id(b)
            for vals in cc.values() for i, a in enumerate(vals)
            for b in vals[i + 1:]
        ),
    }

def run_one(seed, cloned):
    random.seed(seed)
    spectrum = []
    def record_spectrum(game, actions):
        outcomes = ORIGINAL_EXPAND(game, actions)
        spectrum.append([
            (repr(action), tuple(float(probability) for _state, probability in values))
            for action, values in outcomes.items()
        ])
        return outcomes
    minimax.expand_spectrum = record_spectrum
    Board.copy = structural_copy if cloned else ORIGINAL_COPY
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        players = parse_cli_string(SPEC)
        wins, points, games = play_batch(1, players, quiet=True)
    finally:
        elapsed = time.perf_counter() - wall_started
        cpu_elapsed = time.process_time() - cpu_started
        Board.copy = ORIGINAL_COPY
        minimax.expand_spectrum = ORIGINAL_EXPAND
    game = games[0]
    trace = [repr(record) for record in game.state.action_records]
    fallbacks = [getattr(player, "fallbacks", None) for player in players]
    return {
        "seed": seed,
        "clone": cloned,
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu_elapsed,
        "winner": repr(game.winning_color()),
        "wins": {repr(k): int(v) for k, v in wins.items()},
        "points": {repr(k): list(v) for k, v in points.items()},
        "action_trace": trace,
        "spectrum": spectrum,
        "fallbacks": fallbacks,
        "actions": len(trace),
        "board_shape": board_shape(game.state.board),
    }

def atomic_write(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


rows = []
qualified = True
mismatch = None
for seed in SEEDS:
    # Alternate order to reduce warm-up and host-load bias.
    if (seed - SEEDS[0]) % 2 == 0:
        first = run_one(seed, False)
        second = run_one(seed, True)
    else:
        second = run_one(seed, True)
        first = run_one(seed, False)
    baseline = first if not first["clone"] else second
    clone = second if not first["clone"] else first
    rows.append({"baseline": baseline, "clone": clone})
    for field in ("winner", "wins", "points", "action_trace", "spectrum", "fallbacks", "board_shape"):
        if baseline[field] != clone[field]:
            qualified = False
            mismatch = {"seed": seed, "field": field,
                        "baseline": baseline[field], "clone": clone[field]}
            break
    atomic_write(PROGRESS_PATH, {
        "spec": SPEC, "seed": seed, "completed_pairs": len(rows),
        "qualified_so_far": qualified, "mismatch": mismatch,
        "last_baseline_seconds": baseline["elapsed_seconds"],
        "last_clone_seconds": clone["elapsed_seconds"],
        "last_baseline_cpu_seconds": baseline["cpu_seconds"],
        "last_clone_cpu_seconds": clone["cpu_seconds"],
    })
    if mismatch is not None:
        break

result = {"spec": SPEC, "seeds": SEEDS, "qualified": qualified,
          "mismatch": mismatch, "completed_pairs": len(rows), "rows": rows}
atomic_write(RESULT_PATH, result)
print(json.dumps({"qualified": qualified, "mismatch": mismatch,
                  "completed_pairs": len(rows), "result_path": RESULT_PATH},
                 sort_keys=True), flush=True)
