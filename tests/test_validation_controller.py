"""Executable contract tests for the Luna validation controller."""

from __future__ import annotations
import json, os, shutil, stat, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
CONTROLLER = ROOT / "docs/readouts/luna-search/validation.sh"

def _make_fixture(tmp_path: Path, *, screen_count: int = 9):
    study = tmp_path / "study"; results = study / "results"; results.mkdir(parents=True)
    source = tmp_path / "src"; shutil.copytree(ROOT / "src", source)
    scores = {"c1": (90,90), "c2": (88,95), "c3": (100,90), "c4": (75,90), "c5": (80,75), "c6": (70,90), "c7": (85,70), "c8": (60,90), "c9": (65,70)}
    for i, (candidate, (ab2, shipped)) in enumerate(scores.items()):
        if i >= screen_count: break
        (results / f"{i:02d}-{candidate}.json").write_text(json.dumps({"candidate": candidate, "games": 120, "rows": [{"gate":"vs-ab2", "games":120, "wins":{"Color.RED":ab2,"Color.BLUE":120-ab2}}, {"gate":"vs-shipped", "games":120, "wins":{"Color.RED":shipped,"Color.BLUE":120-shipped}}]}))
    log = tmp_path / "calls.jsonl"; bindir = tmp_path / "bin"; bindir.mkdir(); fake = bindir / "python"
    real = str(Path(sys.executable).resolve())
    fake.write_text(("#!" + real + r"""
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if len(args) >= 2 and args[0] == "-m" and args[1] == "hexset.catanatron.searchcfg":
    def val(flag): return args[args.index(flag) + 1]
    candidate, games, seed, out = val("--candidate"), int(val("--games")), int(val("--seed")), val("--out")
    gate = args[args.index("--gate") + 1] if "--gate" in args else None
    if games == 1024: counts = {"c1": {"vs-ab2":700,"vs-shipped":600}, "c2": {"vs-ab2":650,"vs-shipped":700}, "c3": {"vs-ab2":800,"vs-shipped":500}}[candidate]
    elif gate == "vs-ab2": counts = {"vs-ab2":1300}
    else: counts = {"vs-shipped":800}
    Path(os.environ["LUNA_CALL_LOG"]).open("a").write(json.dumps({"candidate":candidate,"games":games,"seed":seed,"gate":gate,"out":out})+"\n")
    d = {"candidate":candidate,"games":games,"seed":seed,"workers":30,"rows":[{"gate":g,"games":games,"wins":{"Color.RED":counts.get(g,0),"Color.BLUE":games-counts.get(g,0)}} for g in ("vs-ab2","vs-shipped")]}
    Path(out).write_text(json.dumps(d))
else: raise SystemExit(subprocess.call(["REAL_PYTHON_PLACEHOLDER"] + args))
""").replace("REAL_PYTHON_PLACEHOLDER", real))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return study, source

def _run(tmp_path: Path, *, screen_count=9):
    study, source = _make_fixture(tmp_path, screen_count=screen_count); script = tmp_path / "validation.sh"
    script.write_text(CONTROLLER.read_text().replace("/study", str(study))); script.chmod(script.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy(); env["PATH"] = str(tmp_path/"bin") + os.pathsep + env.get("PATH",""); env["PYTHONPATH"] = str(source); env["LUNA_CALL_LOG"] = str(tmp_path/"calls.jsonl")
    return subprocess.run(["bash",str(script)], text=True, capture_output=True, env=env), study

def test_controller_runs_mocked_pipeline_to_wilson_verdict(tmp_path: Path):
    proc, study = _run(tmp_path); assert proc.returncode == 0, proc.stderr
    calls = [json.loads(x) for x in (tmp_path/"calls.jsonl").read_text().splitlines()]
    assert len(calls) == 5
    assert [(c["games"],c["seed"],c["gate"]) for c in calls] == [(1024,1200000,None),(1024,1201000,None),(1024,1202000,None),(2048,1300000,"vs-ab2"),(2048,1400000,"vs-shipped")]
    assert sum(c["games"] * (2 if c["gate"] is None else 1) for c in calls) == 10240
    assert json.loads((study/"validation/finalist.json").read_text())["candidate"] == "c3"
    verdict = json.loads((study/"validation/verdict.json").read_text()); assert verdict["pass"] is True and verdict["attempts"] == 1 and verdict["z"] == 2.5758293035489004
    assert verdict["rows"] == {"vs-ab2":[1300,2048],"vs-shipped":[800,2048]}

def test_controller_rejects_partial_screen_before_mock_calls(tmp_path: Path):
    proc, _ = _run(tmp_path, screen_count=8); assert proc.returncode != 0; assert "screen requires exactly 9 files" in proc.stderr; assert not (tmp_path/"calls.jsonl").exists()
