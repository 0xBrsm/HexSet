"""Practical 30-worker AB2 throughput; each host uses the same board seeds."""
import json
from multiprocessing import Pool
from pathlib import Path
import time
from worker import play


def job(args):
    return play(*args)


if __name__ == '__main__':
    rows=[]
    for host in ('catanatron','hexset'):
        start=time.perf_counter()
        games=[]
        with Pool(30) as pool:
            tasks=[(host,'fast',680200000+i) for i in range(120)]
            for game in pool.imap_unordered(job,tasks,chunksize=1):
                games.append(game)
                if len(games)%20==0:
                    print(f'{host}: {len(games)}/120 done in {time.perf_counter()-start:.1f}s',flush=True)
        seconds=time.perf_counter()-start
        games.sort(key=lambda g:g['seed'])
        row={'host':host,'mode':'fast','workers':30,'seconds':seconds,
             'games_per_second':len(games)/seconds,'games':games}
        rows.append(row)
        Path('/out/pool.json').write_text(json.dumps({'complete':len(rows)==2,'rows':rows},indent=2)+'\n')
        print(f'{host}: {len(games)/seconds:.3f} games/s',flush=True)
