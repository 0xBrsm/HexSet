from hexset.arena import PRESETS
from hexset.catanatron.fair_budget import (CANDIDATE, CONTROL, PHENOTYPES,
                                           entrant_config, plan, players_for)

def test_fair_plan_is_four_jobs_constant_width(tmp_path):
    d=plan(tmp_path)
    assert len(d["jobs"])==4
    assert [(j["config"],j["role"],j["gate"],j["width"] if "width" in j else j["phenotype"]["width"],j["phenotype"]["max_nodes"]) for j in d["jobs"]] == [("k1","control","ab2",12,2400),("k1","control","shipped",12,2400),("k4","candidate","ab2",12,9600),("k4","candidate","shipped",12,9600)]
    assert len({j["seed"] for j in d["jobs"]})==4 and all(j["workers"]==30 for j in d["jobs"])

def test_controls_never_promote(tmp_path):
    assert all(j["role"] in ("control","candidate") for j in plan(tmp_path)["jobs"])

def test_actual_entrant_parameters_are_native_and_frozen():
    for label,k,nodes in (("k1",1,2400),("k4",4,9600)):
        e=entrant_config(label)
        assert (e.kind,e.k,e.width,e.max_nodes,e.max_trades,e.mode)==("heximax",k,12,nodes,0,"notrade")
    assert PRESETS[CONTROL].kind == PRESETS[CANDIDATE].kind == "heximax"
    assert players_for(CONTROL,"shipped").endswith("DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
