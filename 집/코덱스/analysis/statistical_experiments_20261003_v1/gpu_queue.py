"""Restore the earlier checkpointed TK2 job even if the new GPU task fails."""
from pathlib import Path
import sys,subprocess
HERE=Path(__file__).resolve().parent
old=HERE.parent/'temp_tk_season_20261003_v1'
try:
    with (HERE/'gpu.log').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-u',str(HERE/'gpu.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
finally:
    with (old/'tk2_pfn_resume3.log').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-u',str(old/'tk2_pfn.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
