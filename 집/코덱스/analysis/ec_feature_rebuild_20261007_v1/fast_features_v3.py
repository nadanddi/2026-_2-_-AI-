"""Bootstrap fix for standalone v2 execution; preserve the failed v2 source."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import runpy
runpy.run_path(str(HERE/'fast_features_v2.py'),run_name='__main__')
