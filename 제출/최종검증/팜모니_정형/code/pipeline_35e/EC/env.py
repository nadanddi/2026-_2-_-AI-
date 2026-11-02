from pathlib import Path
import os,sys,glob
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
_handles=[]
core=os.environ.get('EC_CORE_DEPENDENCY_PATH')
if core and core not in sys.path:sys.path.insert(0,core)
extra=os.environ.get('EC_DEPENDENCY_PATH')
if extra and extra not in sys.path:sys.path.append(extra)
dirs=[str(ROOT/'dlls')]
for p in sys.path:
    if p:
        dirs.extend(glob.glob(str(Path(p)/'*'/'*.libs')))
        dirs.extend(glob.glob(str(Path(p)/'*.libs')))
        dirs.extend(glob.glob(str(Path(p)/'torch'/'lib')))
if hasattr(os,'add_dll_directory'):
    for p in dirs:
        if Path(p).is_dir():
            try:_handles.append(os.add_dll_directory(p))
            except OSError:pass
