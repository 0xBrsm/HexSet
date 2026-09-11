from pathlib import Path
import pytest
from hexset.catanatron.fair_budget import plan as screen_plan, source_fingerprint
from hexset.catanatron.fair_budget_validation import (PROTOCOL, plan, run,
                                                       validate_screen)

def screen_fixture(tmp_path):
    manifest=screen_plan(tmp_path / "screen")
    rows=[]
    for job in manifest["jobs"]:
        wins=70 if job["role"]=="candidate" else 60
        rows.append({**{k:job[k] for k in ("config","role","entrant","gate","games","workers","seed","players","phenotype")},"wins":{"Color.RED":wins,"Color.BLUE":120-wins}})
    return {"manifest":manifest,"results":rows}

def test_validation_requires_complete_candidate_point_screen(tmp_path):
    s=screen_fixture(tmp_path)
    d=plan(tmp_path / "validation",s)
    assert d["candidate"]=="heximax-fair-k4" and d["phenotype"]["k"]==4 and d["phenotype"]["width"]==12 and d["phenotype"]["max_nodes"]==9600
    assert [(j["games"],j["seed"]) for j in d["jobs"][:2]]==[(1024,90000000),(1024,90100000)]
    assert [(j["games"],j["seed"]) for j in d["jobs"][2:]]==[(4096,100000000),(4096,110000000),(4096,100004096),(4096,110004096),(8192,100008192),(8192,110008192)]

def test_screen_without_manifest_or_both_roles_cannot_validate():
    with pytest.raises(ValueError): validate_screen({"results":[]})

def test_runner_resumes_atomic_artifacts_and_rejects_bad_identity(tmp_path):
    s=screen_fixture(tmp_path); root=tmp_path / "validation"; calls=[]
    def game(j):
        calls.append(j)
        return {"protocol":PROTOCOL,"source_hash":source_fingerprint(),"candidate":j["candidate"],"gate":j["gate"],"games":j["games"],"seed":j["seed"],"workers":j["workers"],"players":j["players"],"phenotype":j["phenotype"],"candidate_wins":600 if j["phase"]=="confirmation" else 3000,"wins":{"Color.RED":600 if j["phase"]=="confirmation" else 3000,"Color.BLUE":j["games"]-(600 if j["phase"]=="confirmation" else 3000)}}
    out=run(root,s,game); assert out["status"] in ("PASS","CONTINUE") and len(calls)>=4
    run(root,s,lambda j: (_ for _ in ()).throw(AssertionError()))
    bad=next(root.glob("confirmation-*.json")); text=bad.read_text().replace('"seed": 90000000','"seed": 1'); bad.write_text(text)
    with pytest.raises(ValueError): run(root,s,lambda j: (_ for _ in ()).throw(AssertionError()))
