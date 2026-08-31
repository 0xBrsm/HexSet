# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.actions import space_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.encoding import encode, encode_batch, static_graph  # noqa: E402
from hexset.game import is_over, start, to_move  # noqa: E402
from hexset.onnx_record import RECORD_FIELDS, RecordEncoder, record_batch, record_from_game  # noqa: E402
from hexset.play import step_randomly  # noqa: E402


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        if is_over(game):
            break
        step_randomly(game, rng)
    return game


def as_tensors(record: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value) for name, value in record.items()}


def run_encoder(encoder: RecordEncoder, record: dict[str, np.ndarray]):
    tensors = as_tensors(record)
    args = [tensors[name] for name in RECORD_FIELDS[:21]]  # everything but the two masks
    with torch.no_grad():
        return encoder(*args)


def assert_exact(got: tuple, want) -> None:
    names = ("hexes", "vertices", "edges", "globals")
    for name, g, w in zip(names, got, (want.hexes, want.vertices, want.edges, want.globals)):
        g_np = g.numpy()
        assert g_np.dtype == np.float32 == w.dtype, name
        assert g_np.shape == w.shape, name
        assert np.array_equal(g_np, w), f"{name} mismatch: {np.abs(g_np - w).max()}"


def test_record_encoder_matches_encode_exactly_on_a_seeded_playout():
    """Real positions from a random playout, not gaussians: `encode` is a
    construction from integers, so the torch mirror must match it bit for
    bit, with no tolerance."""
    rng = random.Random(7)
    games = [start(random_base_board(rng), 4, rng) for _ in range(6)]
    space = space_for(games[0])
    graph = static_graph(games[0].state.board.topology)
    encoder = RecordEncoder(graph, players=4)
    encoder.eval()

    checked = 0
    for _ in range(60):
        for game in games:
            if not is_over(game):
                step_randomly(game, rng)
        for game in games:
            if is_over(game):
                continue
            seat = to_move(game)
            record = record_from_game(game, seat, space)
            got = run_encoder(encoder, {k: v[None] for k, v in record.items()})
            want = encode(game, seat)
            assert_exact(tuple(g[0] for g in got), want)
            checked += 1

    assert checked > 200


def test_record_encoder_matches_encode_batch():
    """The batched numpy path is the one the graph actually replaces, since a
    search encodes a whole wave of leaves at once -- so it needs its own
    check against the torch encoder, not just the single-position path."""
    rng = random.Random(11)
    games = [a_game(seed=seed, steps=40 + seed) for seed in range(10)]
    live = [g for g in games if not is_over(g)]
    space = space_for(live[0])
    graph = static_graph(live[0].state.board.topology)
    encoder = RecordEncoder(graph, players=4)
    encoder.eval()

    perspectives = [to_move(g) for g in live]
    record = record_batch(list(zip(live, perspectives)), space)
    got = run_encoder(encoder, record)

    want = encode_batch(live, perspectives)
    for i, obs in enumerate(want):
        assert_exact(tuple(g[i] for g in got), obs)


def test_record_from_game_defaults_perspective_to_the_mover():
    game = a_game(seed=3, steps=50)
    space = space_for(game)
    default = record_from_game(game, None, space)
    explicit = record_from_game(game, to_move(game), space)
    for name in RECORD_FIELDS:
        assert np.array_equal(default[name], explicit[name])


def test_record_rejects_an_out_of_range_perspective():
    game = a_game(seed=4, steps=10)
    space = space_for(game)
    with pytest.raises(ValueError):
        record_from_game(game, 99, space)


def test_the_record_carries_the_offer_and_filters_answered_to_the_proposer():
    """Board-seat order in the record; the answered block is the proposer's
    information alone (`encoding._offer_parts`'s rule, applied record-side
    where every other information-set filter lives)."""
    import random

    from hexset.actions import space_for
    from hexset.board.board import random_base_board
    from hexset.board.terrain import Resource
    from hexset.game import Phase, decline_trade, propose_trade, start, to_move
    from hexset.play import step_randomly
    from hexset.state import NO_OWNER
    from hexset.trading import bundle

    rng = random.Random(9)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(400):
        if game.phase is Phase.MAIN:
            break
        step_randomly(game, rng)
    assert game.phase is Phase.MAIN
    space = space_for(game)

    row = record_from_game(game, game.current_player, space)
    assert int(row["offer_proposer"]) == NO_OWNER
    assert not row["offer_give"].any()
    assert not row["offer_want"].any()
    assert not row["offer_answered"].any()

    proposer = game.current_player
    others = [s for s in range(4) if s != proposer]
    game.state.hands[proposer][Resource.WOOD] = 2
    for s in others:
        game.state.hands[s][Resource.ORE] = 1
    propose_trade(game, bundle(wood=2), bundle(ore=1), ask=tuple(others))
    first = to_move(game)
    decline_trade(game, first)

    responder_row = record_from_game(game, to_move(game), space)
    assert responder_row["offer_give"][Resource.WOOD] == 2
    assert responder_row["offer_want"][Resource.ORE] == 1
    assert int(responder_row["offer_proposer"]) == proposer
    assert not responder_row["offer_answered"].any()

    proposer_row = record_from_game(game, proposer, space)
    assert proposer_row["offer_answered"][first] == 1
    assert proposer_row["offer_answered"].sum() == 1
