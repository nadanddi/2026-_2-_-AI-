from pathlib import Path
import sys,importlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
core=importlib.import_module(sys.argv[1])
for threshold,cap in [(.8,float('nan')),(.8,float('inf')),(float('nan'),.1),(float('inf'),.1)]:
    try:core.downward([1.],[.5],[.9],[True],threshold=threshold,cap=cap)
    except ValueError:pass
    else:raise AssertionError('Nonfinite correction parameter accepted: '+str((threshold,cap)))
print('PASS_FINITE_CORRECTION_PARAMETERS',flush=True)
