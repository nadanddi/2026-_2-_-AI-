from pathlib import Path
import sys,runpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import env
import env_extra
assert Path(env.__file__).resolve().parent==HERE
for option,path in [('--data',HERE/'data'),('--checkpoint',HERE/'tabpfn-v2-regressor.ckpt')]:
    if option not in sys.argv:sys.argv.extend([option,str(path)])
runpy.run_path(str(HERE/'model.py'),run_name='__main__')
