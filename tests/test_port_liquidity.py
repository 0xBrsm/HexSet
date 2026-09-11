from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.bots.heximax.evaluate import HonestEvaluator, NO_TRADE_WEIGHTS
from hexset.bots.heximax.port_liquidity import (
    ARM_LABELS,
    FLAT_PORT_4X,
    PortLiquidityEvaluator, PortLiquidityHeximax,
    arm_manifest,
    candidate_weights,
    liquidity_from_rates,
    production_rates,
)
from hexset.state import new_game, place_settlement, upgrade_to_city


def _state(seed=7):
    board = random_base_board(random.Random(seed))
    state = new_game(board, 3, random.Random(seed + 1))
    # Pick separated vertices so the real Survey walk sees occupied nodes.
    place_settlement(state, 0, 0, connected=False)
    for vertex in range(1, len(state.vertex_owner)):
        if state.vertex_building[vertex] == 0 and all(
            state.vertex_building[n] == 0 for n in state.board.topology.vertex_neighbors[vertex]
        ):
            place_settlement(state, 0, vertex, connected=False)
            break
    return state


def test_liquidity_formula_is_off_resource_and_ratio_monotone():
    rates = (4.0, 0.0, 0.0, 0.0, 0.0)
    # A named wood port helps only wood; a generic port is weaker for this
    # concentrated income, and improving 4 -> 3 -> 2 is monotone.
    specific = liquidity_from_rates(rates, (2, 4, 4, 4, 4))
    generic = liquidity_from_rates(rates, (3, 3, 3, 3, 3))
    assert specific > generic > 0.0
    assert liquidity_from_rates(rates, (4, 4, 4, 4, 4)) == 0.0
    assert liquidity_from_rates(rates, (3, 4, 4, 4, 4)) < specific
    assert liquidity_from_rates(rates, (2, 4, 4, 4, 4)) > liquidity_from_rates(rates, (3, 4, 4, 4, 4))


def test_rates_match_survey_city_and_robber_conventions():
    state = _state()
    evaluator = HonestEvaluator(state.board, NO_TRADE_WEIGHTS)
    rates_before = production_rates(evaluator, state, 0)
    # Upgrade a real owned, producing settlement and verify city multiplicity.
    owned = next(
        v for v, owner in enumerate(state.vertex_owner)
        if owner == 0 and any(
            h != state.robber and resource is not None
            for h, resource, _ in evaluator.inner.yields[v]
        )
    )
    resource = next(
        resource for h, resource, _ in evaluator.inner.yields[owned]
        if h != state.robber and resource is not None
    )
    upgrade_to_city(state, 0, owned)
    rates_city = production_rates(evaluator, state, 0)
    assert rates_city[int(resource)] > rates_before[int(resource)]
    robbed_hex = next(
        h for h, resource_at_hex, _ in evaluator.inner.yields[owned]
        if h != state.robber and resource_at_hex == resource
    )
    state.robber = robbed_hex
    rates_robbed = production_rates(evaluator, state, 0)
    assert rates_robbed[int(resource)] < rates_city[int(resource)]


def test_scalar_batch_agree_and_cache_key_tracks_public_state():
    state = _state(17)
    weights = replace(NO_TRADE_WEIGHTS, port=0.0)
    evaluator = PortLiquidityEvaluator(state.board, weights, liquidity_coefficient=2.785)
    hands = np.asarray([
        [[0, 0, 1, 1, 3], [1, 1, 0, 0, 0], [0, 1, 1, 0, 1]],
        [[1, 0, 1, 1, 2], [0, 2, 0, 1, 0], [1, 0, 1, 0, 1]],
    ], dtype=float)
    batch = evaluator.score_many(state, 0, hands)
    for row in range(len(hands)):
        for seat in range(state.num_players):
            assert batch[row, seat] == pytest.approx(
                evaluator.score(state, seat, hands[row, seat], knower=0), abs=1e-12
            )
    evaluator._liquidity(state, 0)
    cache_size = len(evaluator._liquidity_cache)
    state.robber = next(h for h in range(state.board.num_hexes) if h != state.robber)
    evaluator._liquidity(state, 0)
    assert len(evaluator._liquidity_cache) > cache_size


def test_disabled_original_weights_are_stock_exact():
    state = _state(23)
    stock = HonestEvaluator(state.board, NO_TRADE_WEIGHTS)
    disabled = PortLiquidityEvaluator(state.board, NO_TRADE_WEIGHTS, liquidity_coefficient=0.0)
    hands = np.asarray([[[1, 0, 1, 1, 2]] * state.num_players], dtype=float)
    np.testing.assert_array_equal(disabled.score_many(state, 0, hands), stock.score_many(state, 0, hands))
    for seat in range(state.num_players):
        assert disabled.score(state, seat, hands[0, seat], knower=0) == stock.score(
            state, seat, hands[0, seat], knower=0
        )


def test_native_candidate_constructors_and_six_arm_manifest():
    # Explicit import is the registration boundary; ordinary preset remains native.
    import hexset.bots.heximax.port_liquidity  # noqa: F401
    board = random_base_board(random.Random(31))
    ordinary = spawn(PRESETS["heximax-notrade"], board, random.Random(32))
    assert type(ordinary).__name__ == "Heximax"
    assert ARM_LABELS == (
        "control", "drop-flat-port", "flat-port-4x",
        "liquidity-1.3925", "liquidity-2.785", "liquidity-5.57",
    )
    assert FLAT_PORT_4X == pytest.approx(.12252)
    for arm in ARM_LABELS[1:]:
        name = "heximax-port-liquidity-" + arm
        bot = spawn(PRESETS[name], board, random.Random(33))
        assert type(bot) is PortLiquidityHeximax
        weights, coefficient = candidate_weights(arm)
        assert bot.depth == 2 and bot.width == 6 and bot.max_nodes == 600 and bot.k == 1
        assert bot.max_trades == 0 and bot.mode == "notrade"
        assert bot.evaluator.weights == weights
        assert bot.evaluator.liquidity_coefficient == coefficient
        assert arm_manifest(arm)["weights"] == arm_manifest(arm)["weights"]


def test_screen_plan_has_six_arms_and_disjoint_gate_jobs():
    from hexset.catanatron.port_liquiditycfg import GATES, manifest, screen_jobs
    jobs = screen_jobs()
    assert len(jobs) == 12
    assert {(job["arm"], job["gate"]) for job in jobs} == {
        (arm, gate) for arm in ARM_LABELS for gate in GATES
    }
    assert {job["seed"] for job in jobs if job["gate"] == "ab2"} == {600000000}; assert {job["seed"] for job in jobs if job["gate"] == "shipped"} == {600100000}
    assert all(job["games"] == 240 and job["workers"] == 30 for job in jobs)
    assert all(
        job["role"] == ("control" if job["arm"] == "control" else "candidate")
        for job in jobs
    )
    doc = manifest()
    assert doc["screen"]["controls_never_promote"] is True
    assert doc["source_hash"]


def test_paired_runner_plan_and_fail_closed_records(tmp_path):
    from hexset.catanatron.port_liquidity_runner import paired_seed, run_screen, validate_record
    from hexset.catanatron.port_liquiditycfg import SCREEN_GAMES

    assert SCREEN_GAMES == 240
    assert paired_seed("ab2", 0) == paired_seed("ab2", 0)
    assert paired_seed("ab2", 239) == 600000239
    assert paired_seed("shipped", 0) == 600100000
    with pytest.raises(ValueError, match="source fingerprint"):
        validate_record({"family": "heximax-port-liquidity-future", "source_hash": "bad"}, "control", "ab2", 0)
    with pytest.raises(ValueError, match="workers must be 30"):
        run_screen("control", "ab2", tmp_path, workers=1)


def test_runner_resume_only_executes_missing_record(tmp_path):
    from hexset.catanatron.port_liquidity_runner import run_screen
    from hexset.catanatron.port_liquiditycfg import lineup, source_fingerprint, arm_manifest

    arm, gate = "liquidity-2.785", "ab2"
    def record(index):
        seed = 600_000_000 + index
        winner = "Color.WHITE"
        game = {"id": f"fixture-{index}", "seed": seed + 99,
                "requested_seed": seed,
                "seating": ["Color.RED", "Color.WHITE", "Color.BLUE", "Color.ORANGE"],
                "candidate_color": "Color.RED", "winner": winner,
                "points": {c: 2 for c in ["Color.RED", "Color.WHITE", "Color.BLUE", "Color.ORANGE"]}}
        return {"family": "heximax-port-liquidity-future", "source_hash": source_fingerprint(),
                "arm": arm, "gate": gate, "index": index, "seed": seed,
                "requested_games": 1, "games": [game], "workers": 30,
                "players": lineup(arm, gate), "candidate_color": "Color.RED",
                "candidate_wins": 0, "winner": winner, "wins": {winner: 1},
                "points": {}, "actual_points": {}, "decisions": 0, "fallbacks": 0,
                "phenotype": arm_manifest(arm)}
    for i in range(239):
        path = tmp_path / "games" / gate / arm / f"{i:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(record(i), sort_keys=True))
    preserved = (tmp_path / "games" / gate / arm / "0000.json").read_bytes()
    seen = []
    class Pool:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def imap_unordered(self, fn, jobs):
            jobs = list(jobs); seen.extend(jobs)
            for job in jobs: yield record(job[2])
    summary = run_screen(arm, gate, tmp_path, pool_factory=lambda n: Pool())
    assert [job[2] for job in seen] == [239]
    assert summary["games"] == summary["candidate_wins"] + 240
    assert (tmp_path / "games" / gate / arm / "0000.json").read_bytes() == preserved


def test_runner_validates_win_loss_draw_and_mismatch():
    from hexset.catanatron.port_liquidity_runner import validate_record, paired_seed
    from hexset.catanatron.port_liquiditycfg import lineup, source_fingerprint, arm_manifest
    arm, gate, index = "liquidity-2.785", "ab2", 0
    seed = paired_seed(gate, index)
    seating = ["Color.RED", "Color.WHITE", "Color.BLUE", "Color.ORANGE"]
    def base(winner, wins, candidate_wins, games=True):
        game = [{"id": "fixture", "seed": seed + 99, "requested_seed": seed,
                 "seating": seating, "candidate_color": "Color.RED", "winner": winner,
                 "points": {c: 2 for c in seating}}] if games else []
        return {"family": "heximax-port-liquidity-future", "source_hash": source_fingerprint(),
                "arm": arm, "gate": gate, "index": index, "seed": seed,
                "requested_games": 1, "games": game, "workers": 30,
                "players": lineup(arm, gate), "candidate_color": "Color.RED",
                "candidate_wins": candidate_wins, "winner": winner, "wins": wins,
                "points": {}, "actual_points": {}, "decisions": 0, "fallbacks": 0,
                "phenotype": arm_manifest(arm)}
    validate_record(base("Color.RED", {"Color.RED": 1}, 1), arm, gate, index)
    validate_record(base("Color.WHITE", {"Color.WHITE": 1}, 0), arm, gate, index)
    validate_record(base(None, {}, 0, games=False), arm, gate, index)
    with pytest.raises(ValueError, match="candidate win count mismatch"):
        validate_record(base("Color.WHITE", {"Color.WHITE": 1}, 1), arm, gate, index)
