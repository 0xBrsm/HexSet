import subprocess,sys
for seed in range(680200000,680200003):
    for host in ('catanatron','hexset'):
        subprocess.run([sys.executable,'/study/profile_host.py','--host',host,
                        '--seed',str(seed),'--out',f'/out/{host}-{seed}-profile.json'],check=True)
