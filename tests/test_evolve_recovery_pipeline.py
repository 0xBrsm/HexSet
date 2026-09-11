from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("evolve_recovery_pipeline", ROOT / "scripts/evolve_recovery_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def _candidate(cid: str):
    return {"id": cid, "weights": {"production": 3.0, "victory_point": 1.0},
            "stance": "win", "depth": 2, "width": 6, "max_nodes": 600,
            "k": 1, "mode": "notrade", "max_trades": 0,
            "temperature": 2.0}


def _row(gate: str, games: int, wins: int):
    return {"gate": gate, "games": games, "candidate_wins": wins,
            "wins": {"Color.RED": wins, "Color.BLUE": games - wins}}


def _checkpoint(tmp_path: Path, *, incomplete: bool = False):
    low = _candidate("g00-p-low")
    discovery = _candidate("g00-p-discovery")
    later = _candidate("g01-p-later")
    def item(c, games, ab2, shipped):
        rows = [_row("vs-ab2", games, ab2), _row("vs-shipped", games, shipped)]
        return {"candidate": c["id"], "weights": c["weights"], "stance": c["stance"], "rows": rows}
    low_item = item(low, 300, 210, 120)
    if incomplete:
        low_item["rows"].pop()
    generations = [{"generation": 0, "complete": True, "promoted": [low["id"]],
                    "candidates": [low, discovery],
                    "summary": [low_item, item(discovery, 120, 299, 119)]}]
    for number in range(1, 5):
        generations.append({"generation": number, "complete": True, "promoted": [],
                            "candidates": [], "summary": []})
    generations.append({"generation": 5, "complete": True, "promoted": [later["id"]],
                         "candidates": [later], "summary": [item(later, 300, 240, 150)]})
    state = {"protocol": 1, "source_hash": pipeline.LEGACY_SOURCE,
             "config": {"total_games": 300}, "generations": generations}
    path = tmp_path / "checkpoint.json"; path.write_text(json.dumps(state)); return path


def test_choose_candidate_requires_promoted_full_rows_and_later_generation(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    chosen = pipeline.choose_candidate(checkpoint, Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py"))
    assert chosen["candidate"] == "g01-p-later"
    assert chosen["generation"] == 5


def test_discovery_only_or_incomplete_rows_fail_closed(tmp_path):
    checkpoint = _checkpoint(tmp_path, incomplete=True)
    state = json.loads(checkpoint.read_text())
    state["generations"][0]["summary"][0]["rows"] = []
    state["generations"][5]["summary"][0]["rows"] = []
    checkpoint.write_text(json.dumps(state))
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    with pytest.raises(ValueError, match="no complete promoted"):
        pipeline.choose_candidate(checkpoint, controller)


def test_excluded_generation_is_never_required_or_selected(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    state = json.loads(checkpoint.read_text())
    state["generations"][5]["excluded_from_selection"] = True
    checkpoint.write_text(json.dumps(state))
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    with pytest.raises(ValueError, match="six complete evolve generations"):
        pipeline.choose_candidate(checkpoint, controller)



def test_selection_rejects_summary_stance_or_temperature_mismatch(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    state = json.loads(checkpoint.read_text())
    for generation in (state["generations"][0], state["generations"][5]):
        item = generation["summary"][0]
        item["stance"] = "paranoid"
        item["temperature"] = 7.0
    checkpoint.write_text(json.dumps(state))
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    with pytest.raises(ValueError, match="no complete promoted"):
        pipeline.choose_candidate(checkpoint, controller)


def test_pipeline_calls_existing_fresh_validator_for_selected_phenotype(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    out = tmp_path / "fresh"
    calls = []
    def runner(cmd, check=True):
        calls.append(cmd)
        if "--run" in cmd:
            # The wrapper's one entrypoint must invoke all six generations
            # before it enters the fresh evaluator.
            assert cmd[cmd.index("--generations") + 1] == "6"
            assert cmd[cmd.index("--promote-top") + 1] == "4"
            return
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        games = int(args["--games"]); gate = args.get("--gate")
        gates = ("vs-ab2", "vs-shipped") if gate in (None, "both") else (gate,)
        rows = [_row(g, games, 800 if games == 1024 and g == "vs-ab2" else
                     600 if games == 1024 else 3000 if g == "vs-ab2" else 2500)
                for g in gates]
        target = Path(args["--out"]); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"candidate": args["--candidate-id"], "games": games,
                                      "workers": 30, "seed": int(args["--seed"]),
                                      "gate": gate or "both", "weights": {"production": 3.0, "victory_point": 1.0},
                                      "stance": "win", "temperature": 2.0, "rows": rows}))
    result = pipeline.run_pipeline(checkpoint, controller, out, runner=runner)
    assert result["selection"]["candidate"] == "g01-p-later"
    assert result["fresh"]["status"] == "PASS"
    assert all("g01-p-later" in call for call in calls)
    assert (out / "pipeline-verdict.json").exists()


def test_pipeline_single_entrypoint_runs_evolution_then_fresh(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    out = tmp_path / "fresh"
    calls = []

    def runner(cmd, check=True):
        calls.append(cmd)
        if "--run" in cmd:
            return
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        games = int(args["--games"]); gate = args.get("--gate")
        gates = ("vs-ab2", "vs-shipped") if gate == "both" else (gate,)
        wins = 800 if games == 1024 else 4090
        rows = [_row(g, games, wins) for g in gates]
        Path(args["--out"]).write_text(json.dumps({
            "candidate": args["--candidate-id"], "games": games, "workers": 30,
            "seed": int(args["--seed"]), "gate": gate,
            "weights": {"production": 3.0, "victory_point": 1.0},
            "stance": "win", "temperature": 2.0, "rows": rows}))

    result = pipeline.run_pipeline(checkpoint, controller, out,
                                   runner=runner, run_evolution=True)
    assert result["fresh"]["status"] == "PASS"
    assert calls[0][0] == pipeline.sys.executable
    assert calls[0][1] == str(controller)
    assert calls[0][calls[0].index("--generations") + 1] == "6"
    assert calls[1][calls[1].index("--candidate-id") + 1] == "g01-p-later"


def test_pipeline_persists_terminal_rejection_atomically(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    out = tmp_path / "fresh"

    def runner(cmd, check=True):
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        gate = args.get("--gate")
        gates = ("vs-ab2", "vs-shipped") if gate == "both" else (gate,)
        target = Path(args["--out"])
        target.write_text(json.dumps({
            "candidate": args["--candidate-id"], "games": int(args["--games"]),
            "workers": 30, "seed": int(args["--seed"]), "gate": gate,
            "weights": {"production": 3.0, "victory_point": 1.0},
            "stance": "win", "temperature": 2.0,
            "rows": [_row(g, int(args["--games"]), 1) for g in gates]}))

    with pytest.raises(RuntimeError, match="did not PASS"):
        pipeline.run_pipeline(checkpoint, controller, out, runner=runner)
    verdict = json.loads((out / "pipeline-verdict.json").read_text())
    assert verdict["status"] == "NEEDS_FRESH_VALIDATION"
    assert verdict["fresh"]["stage"] == "confirmation"


def test_pipeline_rejects_reused_artifact_missing_phenotype_identity(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    out = tmp_path / "fresh"; out.mkdir()
    out.joinpath("confirmation.json").write_text(json.dumps({
        "candidate": "g01-p-later", "games": 1024, "workers": 30,
        "seed": 50_000_000, "gate": "both",
        "weights": {"production": 3.0, "victory_point": 1.0},
        "rows": [_row("vs-ab2", 1024, 800), _row("vs-shipped", 1024, 600)],
    }))
    with pytest.raises(ValueError, match="missing identity fields"):
        pipeline.run_pipeline(checkpoint, controller, out, runner=lambda *args, **kwargs: pytest.fail("runner called"))
    verdict = json.loads((out / "pipeline-verdict.json").read_text())
    assert verdict["status"] == "ERROR"

def test_pipeline_rejects_reused_artifact_with_wrong_worker_binding(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    out = tmp_path / "fresh"; out.mkdir()
    out.joinpath("confirmation.json").write_text(json.dumps({
        "candidate": "g01-p-later", "games": 1024, "workers": 29,
        "seed": 50_000_000, "gate": "both",
        "weights": {"production": 3.0, "victory_point": 1.0},
        "stance": "win", "temperature": 2.0,
        "rows": [_row("vs-ab2", 1024, 800), _row("vs-shipped", 1024, 600)],
    }))
    chosen = pipeline.choose_candidate(checkpoint, controller)
    out.joinpath("manifest.json").write_text(json.dumps({
        "candidate": chosen["candidate"],
        "fingerprint": pipeline._fresh_fingerprint(chosen),
        "checkpoint_sha256": chosen["checkpoint_sha256"],
    }))
    with pytest.raises(ValueError, match="identity/source mismatch"):
        pipeline.run_pipeline(checkpoint, controller, out, runner=lambda *args, **kwargs: pytest.fail("runner called"))
    verdict = json.loads((out / "pipeline-verdict.json").read_text())
    assert verdict["status"] == "ERROR"


def test_pipeline_rejects_same_candidate_artifact_with_changed_source_manifest(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    controller = Path("/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py")
    chosen = pipeline.choose_candidate(checkpoint, controller)
    out = tmp_path / "fresh"; out.mkdir()
    out.joinpath("manifest.json").write_text(json.dumps({
        "candidate": chosen["candidate"], "fingerprint": "different-source",
        "checkpoint_sha256": chosen["checkpoint_sha256"],
    }))
    out.joinpath("confirmation.json").write_text(json.dumps({
        "candidate": chosen["candidate"], "games": 1024, "workers": 30,
        "seed": 50_000_000, "gate": "both",
        "weights": chosen["candidate_record"]["weights"], "stance": "win",
        "temperature": 2.0, "rows": [_row("vs-ab2", 1024, 800),
                                      _row("vs-shipped", 1024, 600)],
    }))
    with pytest.raises(ValueError, match="source/phenotype manifest mismatch"):
        pipeline.run_pipeline(checkpoint, controller, out,
                              runner=lambda *args, **kwargs: pytest.fail("runner called"))
