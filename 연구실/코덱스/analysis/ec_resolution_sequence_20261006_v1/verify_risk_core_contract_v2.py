from pathlib import Path
import sys,json,math,itertools,datetime,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
sys.path.insert(0,str(H));from risk_core_v2 import downward
count=0
for p,r,s,ok,threshold,cap in itertools.product([0.,.03,.5,1.,7.],[-.1,0.,.3,1.2],[0.,.2,.8,.9,1.],[False,True],[0.,.2,.8,.999999],[0.,.03,.1,2.]):
    expected=p-(min(max(p-r,0),cap,p)*max(0,min(1,(s-threshold)/(1-threshold))) if ok else 0)
    actual=float(downward(p,r,s,ok,threshold=threshold,cap=cap));assert math.isfinite(actual) and math.isclose(actual,expected,abs_tol=2e-15,rel_tol=1e-15);assert 0<=actual<=p and p-actual<=cap+2e-15;count+=1
bad=[float('nan'),float('inf'),-float('inf'),[.8],np.array([.8]),np.array(.8)]
rejected=0
for parameter,value in itertools.product(['threshold','cap'],bad):
    args=dict(threshold=.8,cap=.1);args[parameter]=value
    try:downward(1.,.5,.9,True,**args)
    except ValueError:rejected+=1
    else:raise AssertionError((parameter,value))
for threshold,cap in [(-.1,.1),(1.,.1),(1.1,.1),(.8,-.1)]:
    try:downward(1.,.5,.9,True,threshold=threshold,cap=cap)
    except ValueError:rejected+=1
    else:raise AssertionError((threshold,cap))
scalar_numpy=float(downward(1.,.5,.9,True,threshold=np.float64(.8),cap=np.float64(.1)));assert math.isclose(scalar_numpy,.95,abs_tol=1e-15)
out=dict(status='PASS_FINITE_BOUNDED_HELPER_CONTRACT',core_source_sha256=hashlib.sha256((H/'risk_core_v2.py').read_bytes()).hexdigest(),finite_grid_cases=count,rejected_parameter_cases=rejected,numpy_scalar='PASS',v1_nan_cap_defect='FIX_CONFIRMED',risk_fit=0,candidate_selected=False,limitations=['generic helper only; caller feature whitelist/candidate protocol still required','stage1 complete audit not run'])
path=H/('verify_risk_core_contract_v2_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
