import glob, json, subprocess

def focal(x):
    wins=x["wins"]
    return wins.get("Color.RED", wins.get("RED", 0))/x.get("games", 64)
def gatevals(d): return {x["gate"]: focal(x) for x in d["rows"]}
rows=[]
for fn in glob.glob("/study/results/*.json"):
    d=json.load(open(fn))
    if d["candidate"] == "baseline": continue
    v=gatevals(d)
    if set(("vs-ab2","vs-shipped")) <= set(v):
        rows.append((min(v["vs-ab2"]-.50, v["vs-shipped"]-.25), d["candidate"], v))
rows.sort(reverse=True)
chosen=[x[1] for x in rows[:4]]
json.dump({"screen_ranking_by_threshold_margin":rows,"chosen":chosen,"thresholds":{"vs-ab2":.50,"vs-shipped":.25}},open("/study/selection.json","w"),indent=2)
for rank,c in enumerate(chosen):
    subprocess.run(["python","-u","-m","hexset.catanatron.ablation","--candidate",c,"--games","256","--workers","30","--seed",str(710000+rank*100),"--out",f"/study/results/confirm-{rank:02d}-{c}.json"],check=True)
confirm=[]
for fn in glob.glob("/study/results/confirm-*.json"):
    d=json.load(open(fn)); v=gatevals(d); confirm.append((min(v["vs-ab2"]-.50,v["vs-shipped"]-.25),d["candidate"],v))
confirm.sort(reverse=True)
finalist=confirm[0][1]
json.dump({"confirm_ranking_by_threshold_margin":confirm,"finalist":finalist},open("/study/finalist.json","w"),indent=2)
for gate_seed,gate in ((810000,"vs-ab2"),(820000,"vs-shipped")):
    subprocess.run(["python","-u","-m","hexset.catanatron.ablation","--candidate",finalist,"--games","2048","--workers","30","--seed",str(gate_seed),"--out",f"/study/results/holdout-{gate}-{finalist}.json"],check=True)

subprocess.run(["python","-u","/study/verdict.py"],check=True)
