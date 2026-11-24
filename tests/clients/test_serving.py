"""Joining forwards' feeds (`hexset.clients.batching.join` and `split`) and
running ONNX files on one server (`hexset.clients.serving`)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

from hexset.clients import batching

FIXTURES = Path(__file__).parent / "fixtures"


# --- join and split -------------------------------------------------------------


def _feed(rng, rows: int, rb: int, disc: int, double: int) -> dict[str, np.ndarray]:
    """A decision feed of `rows` rows with `rb` Road Building candidates,
    `disc` discard candidates and `double` double-turn rows, owners drawn
    from its own rows."""
    return {
        "observation": rng.standard_normal((rows, 5)).astype(np.float32),
        "flat": rng.random((rows, 7)) < 0.5,
        "setup": rng.random((rows, 3)) < 0.5,
        "road_building": rng.integers(0, 72, (rb, 2)),
        "road_building_row": np.sort(rng.integers(0, rows, rb)),
        "discard": rng.integers(0, 4, (disc, 10)),
        "discard_row": np.sort(rng.integers(0, rows, disc)),
        "double_rows": np.sort(rng.choice(rows, double, replace=False)),
    }


def _graph(feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A stand-in for a decision graph whose rows do not read one another:
    each row's outputs from its own inputs, each candidate's from itself and
    its owner row, so a candidate joined to the wrong row answers wrongly."""
    row_sum = feed["observation"].sum(axis=1) + feed["flat"].sum(axis=1)
    return {
        "value": np.stack([row_sum, -row_sum], axis=1),
        "decision": np.stack([feed["setup"].sum(axis=1), feed["flat"].argmax(axis=1)], axis=1),
        "option_road_building": feed["road_building"].sum(axis=1) * 1000.0 + row_sum[feed["road_building_row"]],
        "option_discard": feed["discard"].sum(axis=1) * 1000.0 + row_sum[feed["discard_row"]],
        "option_double": row_sum[feed["double_rows"]][:, None, None] * np.ones((1, 3, 3)),
    }


def test_split_of_a_joined_run_gives_each_feed_what_it_gets_alone():
    rng = np.random.default_rng(20260909)
    feeds = [_feed(rng, 3, 2, 0, 1), _feed(rng, 1, 0, 3, 0), _feed(rng, 5, 4, 2, 2), _feed(rng, 2, 0, 0, 0)]
    names = ["value", "option_discard", "decision", "option_road_building", "option_double"]
    joined = batching.join(feeds)
    whole = _graph(joined.feed)
    parts = batching.split(names, [whole[name] for name in names], joined)
    assert len(parts) == len(feeds)
    for feed, part in zip(feeds, parts):
        alone = _graph(feed)
        for name, got in zip(names, part):
            assert np.array_equal(got, alone[name]), name


def test_join_moves_owner_rows_past_the_rows_before_them():
    rng = np.random.default_rng(20260910)
    a, b = _feed(rng, 3, 1, 1, 1), _feed(rng, 4, 2, 0, 1)
    joined = batching.join([a, b])
    assert np.array_equal(joined.feed["road_building_row"], np.concatenate([a["road_building_row"],
                                                                            b["road_building_row"] + 3]))
    assert np.array_equal(joined.feed["double_rows"], np.concatenate([a["double_rows"], b["double_rows"] + 3]))
    assert joined.counts == ({"rows": 3, "road_building_row": 1, "discard_row": 1, "double_rows": 1},
                             {"rows": 4, "road_building_row": 2, "discard_row": 0, "double_rows": 1})


def test_a_feed_without_candidates_joins_by_rows_alone():
    rng = np.random.default_rng(20260911)
    feeds = [{"x": rng.standard_normal((n, 4))} for n in (1, 3, 2)]
    joined = batching.join(feeds)
    parts = batching.split(["y"], [joined.feed["x"] * 2], joined)
    assert all(np.array_equal(part[0], feed["x"] * 2) for feed, part in zip(feeds, parts))


def test_join_refuses_feeds_naming_different_inputs_or_an_owner_outside_its_rows():
    rng = np.random.default_rng(20260909)
    a = _feed(rng, 2, 1, 0, 0)
    with pytest.raises(ValueError, match="different inputs"):
        batching.join([a, {k: v for k, v in a.items() if k != "setup"}])
    bad = dict(a, road_building_row=np.array([2]))
    with pytest.raises(ValueError, match="outside its feed"):
        batching.join([bad])


# --- the server -------------------------------------------------------------------


def _stub9(tmp_path) -> str:
    sys.path.insert(0, str(FIXTURES))
    try:
        from build_stub9 import build
    finally:
        sys.path.remove(str(FIXTURES))
    return str(build(tmp_path / "stub9.onnx"))


def _stub9_feeds(path: str, count: int):
    """Real decision feeds for the stub, from positions of a random game."""
    from hexset.actions import apply, options_for
    from hexset.board.board import random_base_board
    from hexset.clients import onnxbot
    from hexset.game import start, to_move

    board = random_base_board(random.Random(0))
    policy = onnxbot.load(path, board.topology).policy
    game, rng, feeds = start(board, 4, random.Random(9)), random.Random(9), []
    while len(feeds) < count and game.won_by is None:
        seat = to_move(game)
        feeds.append(policy._feed([(game, seat)], [policy._decision(game, seat)]))
        apply(game, rng.choice(options_for(game)))
    return feeds


def test_a_remote_session_answers_what_the_file_answers_in_this_process(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from hexset.clients import serving

    path = _stub9(tmp_path)
    feeds = _stub9_feeds(path, 12)
    local = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in local.get_outputs()]
    with serving.served("cpu") as stats:
        remote = serving.remote_session(path, serving.current())
        assert [a.name for a in remote.get_inputs()] == [a.name for a in local.get_inputs()]
        assert remote.get_modelmeta().custom_metadata_map == local.get_modelmeta().custom_metadata_map
        for feed in feeds:
            for got, want in zip(remote.run(names, feed), local.run(names, feed)):
                assert np.array_equal(got, want)
        # Several requests in one message are joined into one run there.
        replies = remote.joined.run_many([(remote, ["decision"], feed) for feed in feeds])
        for reply, feed in zip(replies, feeds):
            assert np.array_equal(reply[0], local.run(["decision"], feed)[0])
    assert stats["requests"] == 2 * len(feeds) and stats["forwards"] == len(feeds) + 1
    assert serving.current() is None


def test_a_request_the_server_cannot_run_raises_in_the_caller(tmp_path):
    pytest.importorskip("onnxruntime")
    from hexset.clients import serving

    path = _stub9(tmp_path)
    with serving.served("cpu"):
        remote = serving.remote_session(path, serving.current())
        with pytest.raises(RuntimeError, match="the inference server"):
            remote.run(["decision"], {"observation": np.zeros((1, 3), dtype=np.float32)})
        with pytest.raises(RuntimeError, match="could not load"):
            serving.remote_session(str(tmp_path / "missing.onnx"), serving.current())


def test_a_server_is_refused_an_unknown_device_and_a_second_open():
    from hexset.clients import serving

    with pytest.raises(ValueError, match="runs on one of"):
        with serving.served("tpu"):
            pass
    pytest.importorskip("onnxruntime")
    with serving.served("cpu"):
        with pytest.raises(RuntimeError, match="already open"):
            with serving.served("cpu"):
                pass


@pytest.mark.parametrize("lanes", [None, 3])
def test_a_served_run_plays_the_games_an_unserved_run_plays(tmp_path, monkeypatch, lanes):
    """The stub's answers are the same in any batch, so serving it may change
    nothing about the games: every forward, decisions with their candidate
    rows included, reaches the server and comes back as the file answers it."""
    pytest.importorskip("onnxruntime")
    from hexset.arena import Entrant, compete, compete_lanes, lineup_from_names, read_game_journal
    from hexset.arena import register_entrant_kind, unregister_entrant_kind
    from hexset.clients import onnxbot
    from hexset.record import to_json

    from hexset.clients import serving

    loaded_remote = []
    remote_session = serving.remote_session

    def counted(path, server):
        loaded_remote.append(path)
        return remote_session(path, server)

    monkeypatch.setattr(serving, "remote_session", counted)
    register_entrant_kind("serve-stub", lambda e, board, rng: onnxbot.spawn(e.weights, board, rng=rng))
    try:
        stub = Entrant("serve-stub", kind="serve-stub", weights=_stub9(tmp_path))
        lineup = (stub, *lineup_from_names(["random"] * 3))
        records = {}
        for serve in (None, "cpu"):
            journal = tmp_path / f"{serve}.jsonl"
            if lanes is None:
                compete(lineup, 4, action_cap=60, turn_cap=400, journal=journal, serve=serve)
            else:
                compete_lanes({i: (lineup, 97000, i) for i in range(4)}, journal, lanes=lanes,
                              action_cap=60, turn_cap=400, serve=serve)
            header, games = read_game_journal(journal)
            assert header.get("serve") == serve
            records[serve] = sorted(to_json(outcome.record) for _, outcome in games)
            assert bool(loaded_remote) == (serve is not None)
        assert records[None] == records["cpu"]
    finally:
        unregister_entrant_kind("serve-stub")


def _register_onnx() -> None:
    from hexset.clients import netbot, onnxbot

    netbot.register_entrants(onnxbot.load)


def test_worker_processes_started_inside_the_block_run_their_files_on_the_server(tmp_path):
    pytest.importorskip("onnxruntime")
    from hexset.arena import compete, lineup_from_names, read_game_journal
    from hexset.clients import serving
    from hexset.record import to_json

    lineup = lineup_from_names([f"network:{_stub9(tmp_path)}", "random", "random", "random"])
    records = {}
    for arm in ("alone", "served"):
        journal = tmp_path / f"{arm}.jsonl"
        if arm == "served":
            with serving.served("cpu") as stats:
                compete(lineup, 4, workers=2, action_cap=60, turn_cap=400, journal=journal,
                        worker_initializer=_register_onnx)
        else:
            compete(lineup, 4, workers=2, action_cap=60, turn_cap=400, journal=journal,
                        worker_initializer=_register_onnx)
        records[arm] = sorted(to_json(outcome.record) for _, outcome in read_game_journal(journal)[1])
    assert stats["requests"] > 0 and stats["rows"] >= stats["requests"]
    assert records["alone"] == records["served"]


def test_auto_serves_a_searching_onnx_seat_where_cuda_is_found_and_a_journal_resumes_across_routes(
        tmp_path, monkeypatch):
    """`"auto"` is the default of `compete`: it must serve only a lineup that
    searches with an ONNX file and only where CUDA is found, step aside for an
    open server or `HEXSET_SERVE_AUTO=0`, and a journal started on one route
    must resume on the other."""
    pytest.importorskip("onnxruntime")
    from hexset.arena import Entrant, _serve_device, compete, lineup_from_names
    from hexset.clients import serving
    from hexset import gamelog

    stub = _stub9(tmp_path)
    searched = [Entrant(f"rollout:{stub}", kind="rollout", weights=stub), *lineup_from_names(["random"])]
    forward = [Entrant(f"network:{stub}", kind="network", weights=stub), *lineup_from_names(["random"])]
    monkeypatch.setattr(serving, "cuda_available", lambda: True)
    assert _serve_device("auto", [searched]) == "cuda"
    assert _serve_device("auto", [forward]) is None
    assert _serve_device(None, [searched]) is None and _serve_device("cpu", [forward]) == "cpu"
    monkeypatch.setenv(serving.AUTO_ENV, "0")
    assert _serve_device("auto", [searched]) is None
    monkeypatch.delenv(serving.AUTO_ENV)
    monkeypatch.setenv(serving.ENV, "open-server")
    assert _serve_device("auto", [searched]) is None
    monkeypatch.delenv(serving.ENV)
    monkeypatch.setattr(serving, "cuda_available", lambda: False)
    assert _serve_device("auto", [searched]) is None

    lineup = lineup_from_names([f"network:{stub}", "random", "random", "random"])
    journal = tmp_path / "resumed.jsonl"
    compete(lineup, 4, action_cap=60, turn_cap=400, journal=journal, serve=None,
            worker_initializer=_register_onnx)
    compete(lineup, 8, action_cap=60, turn_cap=400, journal=journal, serve="cpu", resume=True,
            worker_initializer=_register_onnx)
    lines = gamelog.read_lines(journal)
    assert "serve" not in lines[0]
    assert [l.get("serve") for l in lines if l.get("kind") == "source"] == [None, "cpu"]
    assert sum(1 for l in lines if l.get("kind") == "game") == 8
