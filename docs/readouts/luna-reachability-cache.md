# Reachability cache review (pinned Catanatron)

Pinned source: `catanatron/features.py`, certified image `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`. The relevant function is `reachability_features(game, p0_color, levels=2)`.

Literal body shape:

```python
board_buildable = game.state.board.buildable_node_ids(p0_color, True)
for i, color in iter_players(game.state.colors, p0_color):
    owned_or_buildable = get_owned_or_buildable(game, color, board_buildable)
    zero_nodes = get_zero_nodes(game, color)
    production = count_production(
        frozenset(owned_or_buildable.intersection(zero_nodes)),
        game.state.board.map,
    )
    for resource in RESOURCES:
        features[f"P{i}_0_ROAD_REACHABLE_{resource}"] = production[resource]
    enemy_nodes = frozenset(k for k, v in game.state.board.buildings.items()
                             if v is not None and v[0] != color)
    enemy_roads = frozenset(k for k, v in game.state.board.roads.items()
                            if v is not None and v != color)
    for level, level_nodes, paths in iter_level_nodes(
        enemy_nodes, enemy_roads, levels, frozenset(zero_nodes)
    ):
        production = count_production(
            frozenset(owned_or_buildable.intersection(level_nodes)),
            game.state.board.map,
        )
        for resource in RESOURCES:
            features[f"P{i}_{level}_ROAD_REACHABLE_{resource}"] = production[resource]
```

With four players, `levels=2`, and `RESOURCES=['WOOD','BRICK','SHEEP','WHEAT','ORE']`, the output has 60 keys: for each rotated `P0..P3`, levels `0,1,2`, five resources. The value function consumes only its own `P0_0_*` and `P0_1_*` ten keys, after computing all 60.

Existing lower caches are `iter_players` (1024), `map_tile_features` (`NUM_TILES*2`), `map_port_features` (1), `initialize_graph_features_template` (4), `get_node_hot_encoded` (8192), `get_node_production` (1000), `iter_level_nodes` (2000), `count_production` (1000), and `get_feature_ordering` (12). `reachability_features`, `get_zero_nodes`, and `get_owned_or_buildable` themselves are uncached. `iter_level_nodes` caches frozen enemy-node/road/zero-node inputs and returns sets/path dictionaries; `count_production` caches frozenset/map pairs and returns a Counter that this function does not mutate.

Profile `/data/data/com.termux/files/usr/tmp/profile-heximax-notrade-ab2-190000000.prof` (75,563,370 calls, 33.125s cumulative game): `reachability_features` was 65,152 calls, 3.356s self / 7.136s cumulative. `get_zero_nodes` was 260,608 calls, 0.591s self / 0.843s cumulative; `get_owned_or_buildable` was 260,608 calls, 0.462s / 0.696s; `iter_level_nodes` was 4,257 calls, 0.101s / 0.142s; `count_production` was 1,395 calls, 0.003s / 0.059s. The same profile put production extraction at 130,304 calls and 6.707s cumulative, so reachability has a potentially larger addressable ceiling than production, but no speedup is claimed from profile attribution alone.

A future exact cache could canonicalize absolute-color reachability values and lazily project `P{i}` rotation. Its sparse key must include map identity, ordered colors, `board.buildings`, `board.roads`, each `buildings_by_color` settlement/city list, each `connected_components[color]` list/set structure, `board_buildable_ids`, and `levels`. `board_buildable_ids` changes on settlement placement; connected components and roads change on road placement; cities change `board.buildings` and the per-color city/settlement lists. A key omitting any of these can return stale reachability. Canonical values must be copied before exposure, and cached projections must preserve native key insertion order.

This could save more than the production cache in principle: reachability accounts for 7.136s cumulative versus 6.707s for production in this profile. The practical ceiling is lower because existing `iter_level_nodes`/`count_production` caches already remove much repeated work and constructing a full mutable-board key has its own cost. The next meaningful test is a clean native equivalence arm comparing exact traces and 60-key ordered digests before timing; no implementation is included here.
