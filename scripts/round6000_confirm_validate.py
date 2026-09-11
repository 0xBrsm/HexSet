#!/usr/bin/env python3
"""Fail-closed round6000 artifact identity/count validator with atomic verdict."""
import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
GATES=(("vs-ab2",620000000),("vs-shipped",620100000))
COLORS={"Color.RED","Color.BLUE","Color.ORANGE","Color.WHITE"}
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wilson(w,n,z=1.959963984540054):
 p=w/n; d=1+z*z/n
 return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
def atomic(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=".verdict-",dir=path.parent)
 with os.fdopen(fd,"w") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n")
 Path(tmp).replace(path)
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--sidecar",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--artifact-dir",type=Path,required=True); a=p.parse_args(argv)
 try:
  s=json.loads(a.sidecar.read_text()); cid=s["selected_candidate"]; ph=s["phenotype"]
  if s.get("execution")!="NOT_EXECUTED_BY_THIS_PLANNER": raise ValueError("sidecar execution identity changed")
  if ph.get("id")!=cid or ph.get("stance")!="win" or float(ph["temperature"])!=2.476644394795811: raise ValueError("phenotype identity mismatch")
  rows=[]
  for gate,seed in GATES:
   path=a.artifact_dir/(f"round6000-confirmation-{gate}-{seed}.json"); d=json.loads(path.read_text())
   for k,v in (("candidate",cid),("games",1000),("workers",30),("seed",seed),("gate",gate),("stance","win")):
    if d.get(k)!=v: raise ValueError("artifact identity mismatch: "+k)
   if float(d.get("temperature"))!=float(ph["temperature"]) or d.get("weights")!=ph["weights"]: raise ValueError("artifact phenotype mismatch")
   expected=("DC:heximax-evolve-g6000-p10,AB:2,AB:2,AB:2" if gate=="vs-ab2" else "DC:heximax-evolve-g6000-p10,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
   rs=d.get("rows")
   if not isinstance(rs,list) or len(rs)!=1 or rs[0].get("gate")!=gate: raise ValueError("artifact row matrix mismatch")
   r=rs[0]
   for k,v in (("games",1000),("workers",30),("seed",seed),("players",expected)):
    if r.get(k)!=v: raise ValueError("artifact row identity mismatch: "+k)
   wins=r.get("wins")
   if not isinstance(wins,dict) or set(wins)!=COLORS or any(type(v)is not int or v<0 for v in wins.values()) or sum(wins.values())>1000: raise ValueError("artifact wins counts mismatch")
   pts=r.get("points")
   if not isinstance(pts,dict) or set(pts)!=COLORS or any(not isinstance(v,list) or len(v)!=1000 or any(type(x)is not int or x<0 for x in v) for v in pts.values()): raise ValueError("artifact point rows mismatch")
   w=wins["Color.RED"]; rows.append({"gate":gate,"wins":w,"games":1000,"rate":w/1000,"wilson_lower":wilson(w,1000),"target":.50 if gate=="vs-ab2" else .25,"sha256":sha256(path)})
  verdict={"status":"PASS" if all(r["rate"]>r["target"] for r in rows) else "REJECTED_AT_CONFIRMATION","candidate":cid,"source_declared":s["source"],"input_binding_sha256":s["input_binding_sha256"],"sidecar_sha256":sha256(a.sidecar),"gates":rows}
 except Exception as e: verdict={"status":"ERROR","error":str(e),"sidecar":str(a.sidecar)}
 atomic(a.out,verdict); print(json.dumps(verdict,indent=2)); return 0 if verdict["status"]!="ERROR" else 2
if __name__=="__main__": raise SystemExit(main())
