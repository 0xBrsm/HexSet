#!/usr/bin/env python3
"""Bind completed round6000 artifacts to read-only Docker inspect receipts."""
import argparse, json, os, tempfile
from pathlib import Path
EXPECTED_IMAGE="sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be"
SOURCE="/home/bsm/tmp/hexset-luna-followon-certified"
def atomic(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix=".receipt-",dir=p.parent)
 with os.fdopen(fd,"w") as f: json.dump(v,f,indent=2,sort_keys=True); f.write("\n")
 Path(t).replace(p)
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument("--ab2",type=Path,required=True); ap.add_argument("--shipped",type=Path,required=True); ap.add_argument("--verdict",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(argv)
 try:
  v=json.loads(a.verdict.read_text()); receipts=[]
  for name,path,gate,seed in (("ab2",a.ab2,"vs-ab2",620000000),("shipped",a.shipped,"vs-shipped",620100000)):
   d=json.loads(path.read_text())[0]; st=d["State"]; cfg=d["Config"]; hc=d["HostConfig"]
   if d.get("Image")!=EXPECTED_IMAGE or st.get("ExitCode")!=0 or st.get("Status")!="exited": raise ValueError(name+" runtime state/image mismatch")
   if cfg.get("User")!="1000:1000" or cfg.get("WorkingDir")!="/study" or cfg.get("Entrypoint")!=["python"]: raise ValueError(name+" runtime user/workdir mismatch")
   if hc.get("NetworkMode")!="none" or hc.get("NanoCpus")!=30000000000: raise ValueError(name+" runtime network/cpu mismatch")
   if sorted(hc.get("Binds",[]))!=sorted([SOURCE+":/study:ro","/home/bsm/tmp/luna-round6000-confirmation:/out:rw"]): raise ValueError(name+" runtime mounts mismatch")
   env=set(cfg.get("Env",[]));
   if not {"PYTHONHASHSEED=0","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONPATH=/study/src"}.issubset(env): raise ValueError(name+" runtime env mismatch")
   args=cfg.get("Cmd",[])
   for x in ("--candidate-id","g6000-p10","--gate",gate,"--games","1000","--workers","30","--seed",str(seed)): 
    if x not in args: raise ValueError(name+" runtime command mismatch: "+x)
   receipts.append({"name":name,"container_id":d["Id"],"image":d["Image"],"exit_code":st["ExitCode"],"finished_at":st["FinishedAt"],"mounts":d["Mounts"],"env":sorted(x for x in env if x.split("=",1)[0] in {"PYTHONHASHSEED","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONPATH"}),"cpu_nano":hc["NanoCpus"],"verified":True})
  out={"status":v["status"],"artifact_verdict":str(a.verdict),"runtime_verified":True,"receipts":receipts,"gates":v["gates"],"candidate":v["candidate"]}
 except Exception as e: out={"status":"ERROR","runtime_verified":False,"error":str(e)}
 atomic(a.out,out); print(json.dumps(out,indent=2)); return 0 if out["status"]!="ERROR" else 2
if __name__=="__main__": raise SystemExit(main())
