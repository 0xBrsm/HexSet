import glob,json,math,os
def interval(w,n):
    if n<=0:return [0.0,0.0]
    z=1.959963984540054; p=w/n; d=1+z*z/n; q=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return [(p+z*z/(2*n)-q)/d,(p+z*z/(2*n)+q)/d]
def read(gate):
    fs=glob.glob(f"/study/results/holdout-{gate}-*.json")
    if len(fs)!=1:return {"gate":gate,"status":"incomplete","files":fs}
    d=json.load(open(fs[0])); row=next((x for x in d["rows"] if x["gate"]==gate),None)
    if not row:return {"gate":gate,"status":"incomplete","reason":"missing_gate_row"}
    n=int(row.get("games",d.get("games",0))); w=int(row["wins"].get("Color.RED",row["wins"].get("RED",0))); ci=interval(w,n); threshold=.50 if gate=="vs-ab2" else .25
    return {"gate":gate,"games":n,"wins":w,"win_rate":w/n if n else None,"wilson95":ci,"threshold":threshold,"pass":bool(n and ci[0]>threshold)}
rows=[read("vs-ab2"),read("vs-shipped")]; ok=all(x.get("status","ok")=="ok" and x.get("pass") for x in rows)
out={"status":"PASS" if ok else "FAIL","gates":rows,"completed_games":sum(x.get("games",0) for x in rows)}
json.dump(out,open("/study/holdout-verdict.json","w"),indent=2); print(json.dumps(out,indent=2),flush=True)
