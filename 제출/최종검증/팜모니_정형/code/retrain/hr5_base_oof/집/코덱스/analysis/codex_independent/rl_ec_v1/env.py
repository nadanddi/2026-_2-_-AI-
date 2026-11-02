"""Read existing dependencies; never import the original project's env.py."""
import os
import sys
from pathlib import Path

SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
PACKAGES = SOURCE / '.analysis-tools' / 'python'
sys.path.insert(0, str(PACKAGES))
DLL_HANDLES = []
if hasattr(os, 'add_dll_directory'):
    for path in list(PACKAGES.glob('*.libs')) + list(PACKAGES.glob('*/.libs')):
        DLL_HANDLES.append(os.add_dll_directory(str(path)))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
sys.dont_write_bytecode = True
DATA = SOURCE / '온라인대회자료' / '정형데이터' / '참가자_배포'
