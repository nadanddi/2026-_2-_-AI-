"""Mixed synthetic loss bootstrap against separately implemented manual quantiles."""
from pathlib import Path
import sys,importlib.util,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V=load('family24whole',H/'verify_full_v2.py');V.bootstrap(False)
M=load('family22math',H.parent/'ec_final_output_loss_20261004_v1/verification_math_v4.py')
np,pd=M.np,M.pd
records=[]
for farm in ['F13','F47']:
 for d in range(180):
  for h in range(24):
   y=.5+.001*d+.0002*h;b=y+.1+.003*np.sin(d);c=b+(.01 if d%3==0 else -.007)+.0001*np.cos(h)
   records.append((farm,d,h,y,b,c))
f=pd.DataFrame(records,columns=['farm','day','hour','y','baseline','candidate']);results=[]
for seed in [7,101,2024]:
 a=V.boot(f,seed);b=M.bootstrap(f,seed)
 M.near(a['ci_adjusted'],b['ci_adjusted']);M.near(a['ci95'],b['ci95'])
 assert a['p_worse']==b['p_worse']
 assert (a['p_worse']<V.ALPHA and a['ci_adjusted'][1]<0)==b['pass_gate']
 results.append(dict(seed=seed,p_worse=a['p_worse'],ci_adjusted=a['ci_adjusted'],manual_quantile_gate=b['pass_gate']))
r=dict(status='PASS_SYNTHETIC_ONLY',seeds=results,actual_data_reads=0,actual_fit=0,actual_predict=0,actual_score=0,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),H/'verify_full_v2.py',Path(M.__file__)]})
with (H/'independent_bootstrap_critic_result_v1.json').open('x',encoding='utf-8') as out:json.dump(r,out,indent=2)
print('PASS_SYNTHETIC_ONLY_MIXED_BOOTSTRAP',len(results))
