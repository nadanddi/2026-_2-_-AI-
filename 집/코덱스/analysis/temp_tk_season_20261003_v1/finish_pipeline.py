from pathlib import Path
import sys,time,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=ROOT/'집/코덱스/local/temp_tk_season_20261003_v1'
while not all((OUT/p).exists() for p in ['pfn_done.json','members_done.json']):time.sleep(10)
for script in ['analyze.py','verify.py']:
    print('Running',script,flush=True)
    subprocess.run([sys.executable,'-u',str(HERE/script)],cwd=ROOT,check=True)
print('ALL TK1 TK2 TK3 NUMERICAL AUDITS DONE',flush=True)
