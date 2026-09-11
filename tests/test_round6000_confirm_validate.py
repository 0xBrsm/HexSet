import importlib.util, json, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("round6000_confirm_validate",ROOT/"scripts/round6000_confirm_validate.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
main=mod.main

def _fixture(root):
    side={"execution":"NOT_EXECUTED_BY_THIS_PLANNER","selected_candidate":"g6000-p10","phenotype":{"id":"g6000-p10","stance":"win","temperature":2.476644394795811,"weights":{"x":1}},"source":{"source_hash":"s"},"input_binding_sha256":"b"}
    (root/"sidecar.json").write_text(json.dumps(side))
    for gate,seed in (("vs-ab2",620000000),("vs-shipped",620100000)):
        players="DC:heximax-evolve-g6000-p10,AB:2,AB:2,AB:2" if gate=="vs-ab2" else "DC:heximax-evolve-g6000-p10,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"
        row={"gate":gate,"games":1000,"workers":30,"seed":seed,"players":players,"wins":{"Color.RED":400,"Color.BLUE":200,"Color.ORANGE":200,"Color.WHITE":200},"points":{c:[1]*1000 for c in ("Color.RED","Color.BLUE","Color.ORANGE","Color.WHITE")}}
        (root/f"round6000-confirmation-{gate}-{seed}.json").write_text(json.dumps({"candidate":"g6000-p10","games":1000,"workers":30,"seed":seed,"gate":gate,"stance":"win","temperature":2.476644394795811,"weights":{"x":1},"rows":[row]}))

def test_real_shape_rejects_below_target_but_passes_identity():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); _fixture(root); assert main(["--sidecar",str(root/"sidecar.json"),"--artifact-dir",str(root),"--out",str(root/"v.json")])==0; assert json.loads((root/"v.json").read_text())["status"]=="REJECTED_AT_CONFIRMATION"

def test_wrong_lineup_and_extra_row_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); _fixture(root); p=root/"round6000-confirmation-vs-ab2-620000000.json"; x=json.loads(p.read_text()); x["rows"][0]["players"]="DC:wrong,AB:2,AB:2,AB:2"; x["rows"].append(dict(x["rows"][0])); p.write_text(json.dumps(x)); assert main(["--sidecar",str(root/"sidecar.json"),"--artifact-dir",str(root),"--out",str(root/"v.json")])==2; assert json.loads((root/"v.json").read_text())["status"]=="ERROR"
