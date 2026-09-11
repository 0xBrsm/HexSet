import importlib.util
import json
import sys

import pytest


def test_searchcfg_single_gate_cli_emits_exactly_one_row(tmp_path, monkeypatch):
    # The real module imports the pinned catanatron package at module load.
    if importlib.util.find_spec("catanatron") is None:
        pytest.skip("requires pinned catanatron runtime (Docker validation image)")
    try:
        import hexset.catanatron.searchcfg as searchcfg
    except ModuleNotFoundError as exc:
        pytest.skip(f"requires complete pinned catanatron runtime: {exc}")

    seen = []

    class Result:
        wins = {"Color.RED": 69, "Color.BLUE": 30, "Color.ORANGE": 20, "Color.WHITE": 1}
        points = {}

        def report(self):
            return "mock"

    def fake_run_duel(players, games, workers, seed):
        seen.append((players, games, workers, seed))
        return Result()

    monkeypatch.setattr(searchcfg, "run_duel", fake_run_duel)
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["searchcfg", "--candidate", "nodes1200-width8", "--games", "2048", "--workers", "30", "--seed", "123", "--gate", "vs-ab2", "--out", str(out)])
    searchcfg.main()
    document = json.loads(out.read_text())
    assert len(document["rows"]) == 1 and document["rows"][0]["gate"] == "vs-ab2"
    assert seen == [("DC:heximax-search-nodes1200-width8,AB:2,AB:2,AB:2", 2048, 30, 123)]
