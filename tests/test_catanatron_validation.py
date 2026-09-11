import json
import pytest
from hexset.catanatron.validation import rank_screen, decide_holdout

def doc(name, a, s):
    return {"candidate": name, "games": 120, "rows": [
        {"gate": "vs-ab2", "games": 120, "wins": {"Color.RED": a, "x": 120-a}},
        {"gate": "vs-shipped", "games": 120, "wins": {"Color.RED": s, "x": 120-s}},
    ]}

def test_rank_uses_asymmetric_threshold_margins(tmp_path):
    vals = [("a", 75, 35), ("b", 70, 60)] + [(f"x{i}", 60, 30) for i in range(7)]
    paths=[]
    for n,a,s in vals:
        p=tmp_path/(n+".json"); p.write_text(json.dumps(doc(n,a,s))); paths.append(p)
    assert rank_screen(paths)[:2] == ["b", "a"]

def test_rank_fails_closed_on_missing_or_partial_file(tmp_path):
    paths=[]
    for i in range(9):
        p=tmp_path/f"{i}.json"; p.write_text(json.dumps(doc(str(i),60,30))); paths.append(p)
    paths[0].write_text(json.dumps({"candidate":"0","games":120,"rows":[]}))
    with pytest.raises(ValueError): rank_screen(paths)

def test_holdout_requires_strict_99_percent_bounds_on_both_gates():
    assert not decide_holdout({"vs-ab2": (1100,2048), "vs-shipped": (530,2048)})
    assert decide_holdout({"vs-ab2": (1250,2048), "vs-shipped": (700,2048)})
