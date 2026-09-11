import sys
import types
from pathlib import Path
import importlib
from types import SimpleNamespace


def test_play_one_tracks_parsed_candidate_color_not_first_seat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "src"))
    import hexset.catanatron.evolve as evolve
    evolve = importlib.reload(evolve)
    assert Path(evolve.__file__).resolve() == Path(__file__).parents[1] / "src/hexset/catanatron/evolve.py"

    red = SimpleNamespace(value="RED")
    white = SimpleNamespace(value="WHITE")
    blue = SimpleNamespace(value="BLUE")
    orange = SimpleNamespace(value="ORANGE")
    player = SimpleNamespace(color=red, decisions=3, fallbacks=0)
    winner = {"value": "WHITE"}

    duel = types.ModuleType("hexset.catanatron.duel")
    def parse(spec):
        assert spec.startswith("DC:heximax-evolve-worker-test")
        return [player]
    def batch(n, players, quiet=True):
        game = SimpleNamespace(
            id="game-1", seed=17,
            state=SimpleNamespace(colors=(white, red, blue, orange)),
            winning_color=lambda: winner,
        )
        return ({"Color.RED": 1}, {"Color.RED": 8}, [game])
    duel.parse_cli_string = parse
    duel.play_batch = batch
    play = types.ModuleType("catanatron.cli.play")
    play.get_actual_victory_points = lambda state, color: 8
    monkeypatch.setitem(sys.modules, "hexset.catanatron.duel", duel)
    monkeypatch.setitem(sys.modules, "catanatron.cli.play", play)

    record = evolve._play_one(("worker-test", "vs-ab2", "discovery", 0, 0, 17))
    assert record["games"][0]["candidate_color"] == "RED"
    assert record["candidate_wins"] == 0

    winner = red
    record = evolve._play_one(("worker-test", "vs-ab2", "discovery", 0, 0, 17))
    assert record["games"][0]["candidate_color"] == "RED"
    assert record["candidate_wins"] == 1
