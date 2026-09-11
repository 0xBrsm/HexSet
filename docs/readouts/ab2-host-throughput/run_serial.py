"""Fresh-process AB2 host/mode matrix; run in the pinned one-CPU container."""
import json,os,subprocess,sys
from pathlib import Path
rows=[]
profiles=[('hexset','off'),('catanatron','off'),('hexset','fast'),('catanatron','fast')]
for i in range(8):
    order=profiles[i%4:]+profiles[:i%4]
    for host,mode in order:
        path=Path('/out')/f'{680100000+i}-{host}-{mode}.json'
        subprocess.run([sys.executable,'/study/worker.py','--host',host,'--mode',mode,
                        '--seed',str(680100000+i),'--out',str(path)],check=True)
        row=json.loads(path.read_text());rows.append(row)
        Path('/out/serial.json').write_text(json.dumps({'complete':len(rows)==32,'rows':rows},indent=2)+'\n')
        print(f'{len(rows)}/32 games complete',flush=True)
    for host in ('hexset','catanatron'):
        pair=[r for r in rows if r['seed']==680100000+i and r['host']==host]
        assert len(pair)==2
        assert all(pair[0][k]==pair[1][k] for k in ('actions','action_sha256','winner','points','turns')), pair
