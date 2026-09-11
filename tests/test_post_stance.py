import json
from pathlib import Path

from hexset.catanatron.post_stance import run_controller, wilson_lower
from hexset.catanatron.stance_validation import EXPECTED_CANDIDATES


def screen(root, good=True):
    d = root / "screen"; d.mkdir(parents=True)
    for c in EXPECTED_CANDIDATES:
        for gate in ("vs-ab2", "vs-shipped"):
            w = (70 if gate == "vs-ab2" else 40) if (good and c == "win-T1.5") else 40
            (d / f"{c}-{gate}.json").write_text(json.dumps({"candidate": c, "games": 120, "points": {"Color.RED": [1]}, "rows": [{"gate": gate, "games": 120, "wins": {"Color.RED": w, "Color.BLUE": 120-w}}]}))

def runner_factory(calls):
    def run(argv, check=True):
        calls.append(argv)
        if "--candidate" not in argv: return
        a = {argv[i]: argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}
        n, gate, out = int(a["--games"]), a["--gate"], Path(a["--out"])
        gates = ("vs-ab2", "vs-shipped") if gate == "both" else (("vs-ab2",) if gate == "ab2" else ("vs-shipped",))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"candidate": a["--candidate"], "games": n, "rows": [{"gate": g, "games": n, "wins": {"Color.RED": int(n*.75 if g == "vs-ab2" else n*.5), "Color.BLUE": n-int(n*.75 if g == "vs-ab2" else n*.5)}} for g in gates]}))
    return run

def test_good_screen_confirmation_holdouts_and_resume(tmp_path):
    screen(tmp_path); calls=[]
    result = run_controller(tmp_path, evolve=tmp_path/"LATEST_evolve.py", runner=runner_factory(calls))
    assert result["status"] == "PASS" and len(calls) == 3
    calls.clear(); assert run_controller(tmp_path, runner=runner_factory(calls))["status"] == "PASS" and not calls

def test_bad_screen_runs_evolution_once(tmp_path):
    screen(tmp_path, good=False); calls=[]
    result = run_controller(tmp_path, evolve=tmp_path/"LATEST_evolve.py", runner=runner_factory(calls))
    assert result["status"] == "FAIL"
    assert len(calls) == 1 and "--generations" in calls[0]

def test_missing_screen_fails_without_children(tmp_path):
    calls=[]; result=run_controller(tmp_path, runner=runner_factory(calls))
    assert result["status"] == "FAIL" and calls == []

def test_wilson_is_strict_and_99_9_z(tmp_path):
    assert wilson_lower(4096, 4096) > .5
    assert wilson_lower(2000, 4096) < .5

def test_evolve_selection_uses_dual_margin_and_skips_incomplete(tmp_path):
    from hexset.catanatron.evolve_validation import select_candidate
    state = {"protocol": 1, "source_hash": "x", "generations": [
        {"generation": 0, "complete": True, "summary": [
            {"candidate": "bad", "rows": [{"gate":"vs-ab2","games":300,"candidate_wins":299},{"gate":"vs-shipped","games":300,"candidate_wins":60}]},
            {"candidate": "good", "rows": [{"gate":"vs-ab2","games":300,"candidate_wins":190},{"gate":"vs-shipped","games":300,"candidate_wins":100}]},
        ]},
        {"generation": 1, "complete": False, "summary": [{"candidate":"future","rows":[]}]},
    ]}
    p=tmp_path/"checkpoint.json"; p.write_text(json.dumps(state))
    assert select_candidate(p)["candidate"] == "good"


def test_evolved_evaluator_real_main_boundary(tmp_path, monkeypatch):
    import sys, types
    from dataclasses import dataclass
    from hexset.catanatron import evolve_candidate_eval as ev
    class E:
        def __init__(self,*a,**k): self.kw=k
    reg=[]
    arena=types.ModuleType("hexset.arena"); arena.Entrant=E; arena.register_preset=lambda n,e: reg.append((n,e))
    @dataclass(frozen=True)
    class W:
        production: float = 1
        victory_point: float = 1
    weights=W()
    evaluate=types.ModuleType("hexset.bots.heximax.evaluate"); evaluate.NO_TRADE_WEIGHTS=weights
    @dataclass
    class R:
        wins: dict
        points: dict = None
        def report(self): return "mock"
    duel=types.ModuleType("hexset.catanatron.duel")
    duel.run_duel=lambda lineup,games,workers,seed: R({"Color.RED": games//2, "Color.BLUE": games-games//2}, {})
    monkeypatch.setitem(sys.modules,"hexset.arena",arena); monkeypatch.setitem(sys.modules,"hexset.bots.heximax.evaluate",evaluate); monkeypatch.setitem(sys.modules,"hexset.catanatron.duel",duel); monkeypatch.setitem(sys.modules,"hexset.bots",types.ModuleType("hexset.bots"))
    out=tmp_path/"x.json"; assert ev.main(["--candidate-id","g0-p0","--weights-json",json.dumps({"production":2}),"--stance","win","--temperature","1.5","--gate","both","--games","2","--workers","30","--seed","50000000","--out",str(out)]) == 0
    d=json.loads(out.read_text()); assert len(d["rows"])==2 and all(r["games"]==2 for r in d["rows"]); assert reg[0][1].kw["max_trades"]==0 and reg[0][1].kw["temperature"]==1.5

def test_bad_screen_evolution_continues_to_goal_verdict(tmp_path):
    from hexset.catanatron.post_stance import run_controller
    screen(tmp_path, good=False); calls=[]
    def run(argv, check=True):
        calls.append(argv)
        if "--run" in argv:
            cp=Path(argv[argv.index("--checkpoint")+1]); cp.write_text(json.dumps({"protocol":1,"source_hash":"x","generations":[{"generation":0,"complete":True,"summary":[{"candidate":"g0-p0","weights":{"production":2},"stance":"win","temperature":1.5,"rows":[{"gate":"vs-ab2","games":300,"candidate_wins":200},{"gate":"vs-shipped","games":300,"candidate_wins":100}]}]}]})); return
        a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}; n=int(a["--games"]); gate=a["--gate"]; gs=("vs-ab2","vs-shipped") if gate=="both" else (("vs-ab2",) if gate=="vs-ab2" else ("vs-shipped",)); out=Path(a["--out"]); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"candidate":a["--candidate-id"],"games":n,"rows":[{"gate":g,"games":n,"candidate_wins":n,"wins":{"Color.RED":n}} for g in gs]}))
    result=run_controller(tmp_path,evolve=tmp_path/"LATEST_evolve.py",runner=run)
    assert result["status"]=="PASS" and (tmp_path/"validation/evolved-fresh/goal-verdict.json").exists()
    assert any("evolve_candidate_eval" in x for c in calls for x in c)

def test_exact_continuation_decision_is_conservative_and_bounded():
    from hexset.catanatron.evolve_validation import continuation_decision, binomial_upper_tail
    assert binomial_upper_tail(4096, 4096, .5) < .001/3
    assert continuation_decision({"vs-ab2":(2158,4096),"vs-shipped":(1120,4096)}) == "PASS"
    assert continuation_decision({"vs-ab2":(2000,4096),"vs-shipped":(1200,4096)}) == "REJECTED_AT_LOOK"
    assert continuation_decision({"vs-ab2":(8400,16384),"vs-shipped":(4270,16384)}) == "UNRESOLVED_AT_CAP"

def _checkpoint(p):
    p.write_text(json.dumps({"protocol":1,"source_hash":"src","generations":[{"generation":0,"complete":True,"summary":[{"candidate":"evolved","weights":{"production":2},"stance":"win","temperature":1.5,"rows":[{"gate":"vs-ab2","games":300,"candidate_wins":200},{"gate":"vs-shipped","games":300,"candidate_wins":100}]}]}]}))

def test_run_fresh_three_looks_resume_and_disjoint_blocks(tmp_path):
    from hexset.catanatron.evolve_validation import run_fresh
    cp=tmp_path/"c.json"; _checkpoint(cp); calls=[]
    def runner(argv,check=True):
        calls.append(argv); a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}; n=int(a["--games"]); g=a["--gate"]; out=Path(a["--out"]); out.parent.mkdir(parents=True,exist_ok=True)
        w=n if g=="both" else (int(n*.52) if g=="vs-ab2" else int(n*.26)); gs=("vs-ab2","vs-shipped") if g=="both" else (g,); out.write_text(json.dumps({"candidate":"evolved","games":n,"rows":[{"gate":x,"games":n,"candidate_wins":(n if g=="both" else w),"wins":{"Color.RED":(n if g=="both" else w)}} for x in gs]}))
    result=run_fresh(cp,tmp_path/"fresh",runner=runner); assert result["status"]=="UNRESOLVED_AT_CAP"; blocks=[c for c in calls if "--games" in c and c[c.index("--games")+1] != "1024"]; assert [int(c[c.index("--games")+1]) for c in blocks]==[4096,4096,4096,4096,8192,8192]; seeds=[int(c[c.index("--seed")+1]) for c in blocks]; assert len(seeds)==len(set(seeds)); calls.clear(); run_fresh(cp,tmp_path/"fresh",runner=runner); assert not any('stancecfg' in x for c in calls for x in c)

def test_injected_cheap_candidate_does_not_pass(tmp_path):
    screen(tmp_path, good=False); calls=[]
    result=run_controller(tmp_path, evolve=tmp_path/"e.py", runner=runner_factory(calls), cheap_runner=lambda root,py:{"status":"CANDIDATE","candidate":"bankext","holdouts":{}})
    assert result["status"]=="FAIL" and result["stage"]=="evolve"

def test_injected_cheap_validated_is_accepted(tmp_path):
    screen(tmp_path, good=False); calls=[]
    result=run_controller(tmp_path, evolve=tmp_path/"e.py", runner=runner_factory(calls), cheap_runner=lambda root,py:{"status":"PASSED_VALIDATION","candidate":"bankext","gates":{"vs-ab2":{"wins":3000,"games":4096},"vs-shipped":{"wins":1200,"games":4096}},"source_hash":"x"})
    assert result["status"]=="PASS" and result["stage"]=="cheap-followon"
    assert not calls

def test_fresh_block_malformed_or_candidate_mismatch_rejected(tmp_path):
    from hexset.catanatron.evolve_validation import _read_fresh
    p=tmp_path/"bad.json"; p.write_text(json.dumps({"candidate":"other","games":4096,"rows":[]}))
    import pytest
    with pytest.raises(ValueError): _read_fresh(p,4096,{"vs-ab2"},"evolved")
    p.write_text("{}")
    with pytest.raises(ValueError): _read_fresh(p,4096,{"vs-ab2"},"evolved")

def test_run_fresh_continues_then_passes(tmp_path):
    from hexset.catanatron.evolve_validation import run_fresh
    cp=tmp_path/"c.json"; _checkpoint(cp); calls=[]
    def runner(argv,check=True):
        calls.append(argv); a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}; n=int(a["--games"]); g=a["--gate"]; seed=int(a["--seed"]); out=Path(a["--out"]); out.parent.mkdir(parents=True,exist_ok=True)
        if g=="both": rows=[{"gate":"vs-ab2","games":n,"candidate_wins":600,"wins":{"Color.RED":600}},{"gate":"vs-shipped","games":n,"candidate_wins":350,"wins":{"Color.RED":350}}]
        else:
            second=seed in (60004096,70004096); w=(2400 if second else 2130) if g=="vs-ab2" else (1400 if second else 1085); rows=[{"gate":g,"games":n,"candidate_wins":w,"wins":{"Color.RED":w}}]
        out.write_text(json.dumps({"candidate":"evolved","games":n,"rows":rows}))
    r=run_fresh(cp,tmp_path/"fresh",runner=runner); assert r["status"]=="PASS"; hs=[x for x in calls if "--games" in x and x[x.index("--games")+1]=="4096"]; assert len(hs)==4; assert all("8192" not in x for x in calls); calls.clear(); run_fresh(cp,tmp_path/"fresh",runner=runner); assert not calls

def test_real_cheap_schema_feeds_post_stance_alias_normalization(tmp_path, monkeypatch):
    import hexset.catanatron.cheap_followon as cheap
    screen(tmp_path, good=False); calls=[]
    def fake(argv, check=True):
        calls.append(argv); a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}; n=int(a["--games"]); g=a["--gate"]; out=Path(a["--out"]); w=(70 if g=="ab2" else 40) if n==120 else ((600 if g=="ab2" else 350) if n==1024 else (3500 if g=="ab2" else 1800)); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"candidate":a["--candidate"],"gate":"vs-"+g,"games":n,"workers":30,"seed":int(a["--seed"]),"max_trades":0,"points":{"Color.RED":[1]*n},"wins":{"Color.RED":w,"Color.BLUE":n-w}}))
    monkeypatch.setattr(cheap.subprocess,"run",fake)
    # Cheap's real flat-schema result is passed into the real post controller.
    def invoke(root, py): return cheap.run(root, py)
    result=run_controller(tmp_path, evolve=tmp_path/"e.py", runner=lambda *a,**k: (_ for _ in ()).throw(AssertionError("evolve")), cheap_runner=invoke)
    assert result["status"]=="PASS" and result["stage"]=="cheap-followon"

def test_cheap_subthreshold_never_passes(tmp_path):
    screen(tmp_path, good=False); calls=[]
    def evolve(*a,**k): calls.append(a); raise RuntimeError("evolve called")
    result=run_controller(tmp_path, evolve=tmp_path/"e.py", runner=evolve, cheap_runner=lambda r,p:{"status":"PASSED_VALIDATION","candidate":"bankext","gates":{"vs-ab2":{"wins":2000,"games":4096},"vs-shipped":{"wins":1800,"games":4096}}})
    assert result["status"]=="FAIL" and result["stage"]=="evolve"

def test_cheap_mismatched_gate_evidence_fails_closed(tmp_path):
    screen(tmp_path, good=False); result=run_controller(tmp_path, evolve=tmp_path/"e.py", runner=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("evolve")), cheap_runner=lambda r,p:{"status":"PASSED_VALIDATION","candidate":"bankext","gates":{"vs-ab2":{"wins":4096,"games":4096,"candidate":"other"},"vs-shipped":{"wins":4096,"games":4096,"candidate":"bankext"}}})
    assert result["status"]=="FAIL"

def test_actual_evolve_candidate_schema_feeds_selector(tmp_path):
    from hexset.catanatron.evolve import candidate
    from hexset.catanatron.evolve_validation import select_candidate
    from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
    from dataclasses import asdict
    c=candidate("g00-p00", {"production":asdict(NO_TRADE_WEIGHTS)["production"],"temperature":1.5})
    state={"protocol":1,"source_hash":"actual","generations":[{"generation":0,"complete":True,"candidates":[c],"summary":[{"candidate":c["id"],"weights":c["weights"],"stance":c["stance"],"temperature":c["temperature"],"rows":[{"gate":"vs-ab2","games":300,"candidate_wins":200},{"gate":"vs-shipped","games":300,"candidate_wins":100}]}]}]}
    p=tmp_path/"actual.json"; p.write_text(json.dumps(state)); picked=select_candidate(p)
    assert picked["candidate"]==c["id"] and picked["candidate_record"]["weights"]==c["weights"] and picked["candidate_record"]["temperature"]==1.5

def test_real_evolve_loop_emits_selector_compatible_checkpoint(tmp_path, monkeypatch):
    import sys
    import hexset.catanatron.evolve as e
    def fake(job):
        cid, gate, stage, gen, idx, seed = job
        return {"candidate":cid,"gate":gate,"stage":stage,"generation":gen,"game_index":idx,"seed":seed,"candidate_wins":1,"wins":{"Color.RED":1},"points":{}}
    class Pool:
        def __init__(self,n): pass
        def __enter__(self): return self
        def __exit__(self,*x): pass
        def imap_unordered(self,fn,jobs):
            for j in jobs: yield fake(j)
    monkeypatch.setattr(e,"_play_one",fake); monkeypatch.setattr(e,"Pool",Pool)
    ck=tmp_path/"state.json"; sys.argv=["evolve","--checkpoint",str(ck),"--count","1","--games","1","--total-games","1","--generations","1","--run"]; e.main()
    from hexset.catanatron.evolve_validation import select_candidate
    picked=select_candidate(ck); state=json.loads(ck.read_text()); assert set(state.keys()) >= {"protocol","source_hash","generations"}; assert picked["candidate"].startswith("g00-p00")

def test_stance_continues_then_passes_at_8192(tmp_path):
    screen(tmp_path, good=True); calls=[]
    def run(argv,check=True):
        calls.append(argv); a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}; n=int(a["--games"]); g=a["--gate"]; out=Path(a["--out"]); out.parent.mkdir(parents=True,exist_ok=True)
        if n==1024: rows=[{"gate":"vs-ab2","games":n,"wins":{"Color.RED":600,"Color.BLUE":424}},{"gate":"vs-shipped","games":n,"wins":{"Color.RED":350,"Color.BLUE":674}}]
        else:
            second=(int(a["--seed"]) > (2204096 if g=="ab2" else 2304096))
            w=(2400 if second else 2500) if g=="ab2" else (1400 if second else 1400)
            rows=[{"gate":"vs-ab2" if g=="ab2" else "vs-shipped","games":n,"wins":{"Color.RED":w,"Color.BLUE":n-w}}]
        out.write_text(json.dumps({"candidate":"win-T1.5","games":n,"rows":rows}))
    result=run_controller(tmp_path,evolve=tmp_path/"e.py",runner=run)
    assert result["status"]=="PASS"; assert [int(c[c.index("--games")+1]) for c in calls if "--games" in c].count(8192)==0; calls.clear(); run_controller(tmp_path,runner=run); assert not any('stancecfg' in x for c in calls for x in c)

def test_stance_reaches_cap16384_and_resumes(tmp_path):
    screen(tmp_path, good=True); calls=[]
    def run(argv,check=True):
        calls.append(argv); a={argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}; n=int(a['--games']); g=a['--gate']; out=Path(a['--out']); out.parent.mkdir(parents=True,exist_ok=True)
        if n==1024: rows=[{'gate':'vs-ab2','games':n,'wins':{'Color.RED':600,'Color.BLUE':424}},{'gate':'vs-shipped','games':n,'wins':{'Color.RED':350,'Color.BLUE':674}}]
        else:
            w=(2050 if n==4096 else 4100) if g=='ab2' else (1025 if n==4096 else 2050); rows=[{'gate':'vs-ab2' if g=='ab2' else 'vs-shipped','games':n,'wins':{'Color.RED':w,'Color.BLUE':n-w}}]
        out.write_text(json.dumps({'candidate':'win-T1.5','games':n,'rows':rows}))
    result=run_controller(tmp_path,evolve=tmp_path/'e.py',runner=run,cheap_runner=lambda *x:{'status':'NO_CANDIDATE'})
    assert result['status']=='FAIL'; assert (tmp_path/'validation/holdout-decision-16384.json').exists()
    calls.clear(); run_controller(tmp_path,runner=run,cheap_runner=lambda *x:{'status':'NO_CANDIDATE'}); assert not any('stancecfg' in x for c in calls for x in c)

def test_stance_manifest_source_change_rejects_before_dispatch(tmp_path):
    screen(tmp_path, good=True); run_controller(tmp_path,runner=runner_factory([])); m=tmp_path/'validation/stance-manifest.json'; d=json.loads(m.read_text()); d['fingerprint']='changed'; m.write_text(json.dumps(d)); calls=[]
    result=run_controller(tmp_path,runner=runner_factory(calls)); assert result['status']=='FAIL' and calls==[]
