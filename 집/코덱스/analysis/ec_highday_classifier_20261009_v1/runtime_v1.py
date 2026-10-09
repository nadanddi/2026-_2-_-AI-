"""Bundled numeric runtime; no target/data loading."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'))
DLL=[]
if hasattr(os,'add_dll_directory'):
 for p in (ROOT/'.analysis-tools/python').glob('**/.libs'):DLL.append(os.add_dll_directory(str(p)))
