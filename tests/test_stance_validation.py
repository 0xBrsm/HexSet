import json
from pathlib import Path
import pytest
from hexset.catanatron.stance_validation import aggregate

C = ("baseline-win-T2.4766", "win-T1", "win-T1.5", "win-T4", "win-T6", "own", "relative", "paranoid")
G = ("vs-ab2", "vs-shipped")

def make(tmp_path, *, bad=None):
    out=[]
    for i,c in enumerate(C):
        for g in G:
            a,s = (70,40) if c == "win-T1" else ((80,70) if c == "win-T1.5" else (60,30))
            if c == "baseline-win-T2.4766": a,s = 75,35
            if bad == (c,g): a = 121
            d={"candidate":c,"games":120,"points":{"Color.RED":[1]*120},"rows":[{"gate":g,"games":120,"wins":{"Color.RED":a,"Color.BLUE":120-a}}]}
            p=tmp_path/f"{i:02d}-{c}-{g}.json"; p.write_text(json.dumps(d)); out.append(p)
    return out

def test_aggregator_excludes_control_and_ranks_dual_margin(tmp_path):
    result=aggregate(make(tmp_path))
    assert result["candidate"] == "win-T1.5"
    assert result["manifest"]["control"] == "baseline-win-T2.4766"
    assert result["confidence"].startswith("screening-only")
    assert result["ranking"][0]["positive_both"] is True

def test_aggregator_rejects_incomplete_or_corrupt_matrix(tmp_path):
    paths=make(tmp_path)
    with pytest.raises(ValueError): aggregate(paths[:-1])
    paths=make(tmp_path, bad=("own", "vs-ab2"))
    with pytest.raises(ValueError): aggregate(paths)


def test_aggregator_accepts_stancecfg_row_points_schema(tmp_path):
    paths = make(tmp_path)
    for path in paths:
        d = json.loads(path.read_text())
        d["rows"][0]["points"] = d.pop("points")
        path.write_text(json.dumps(d))
    result = aggregate(paths)
    assert result["candidate"] == "win-T1.5"

def test_aggregator_rejects_empty_points_in_either_schema(tmp_path):
    paths = make(tmp_path)
    d = json.loads(paths[0].read_text())
    d.pop("points")
    d["rows"][0]["points"] = {}
    paths[0].write_text(json.dumps(d))
    with pytest.raises(ValueError, match="invalid row points"):
        aggregate(paths)
