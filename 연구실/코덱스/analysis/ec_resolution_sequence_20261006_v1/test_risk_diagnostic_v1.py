from pathlib import Path
import sys,importlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import pandas as pd
core=importlib.import_module(sys.argv[1])
q=pd.DataFrame(dict(A=[1.,1.],sub_ec=[.7,1.1],candidate=[1.,1.],risk=[.8,.2]))
assert core.risk_metrics(q)['risk_flag']['selected']==1,'Diagnostic >= threshold must include equality'
print('PASS_DIAGNOSTIC_THRESHOLD_EQUALITY',flush=True)
