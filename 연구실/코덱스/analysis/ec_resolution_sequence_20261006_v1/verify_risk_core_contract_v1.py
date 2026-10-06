from pathlib import Path
import sys,json,math,itertools,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
sys.path.insert(0,str(H));from risk_core_v1 import design,downward,FORBIDDEN
count=0
for p,r,s,ok in itertools.product([0.,.03,.5,1.],[0.,.3,1.2],[0.,.8,.9,1.],[False,True]):
    expected=p-(min(max(p-r,0),.1,p)*max(0,min(1,(s-.8)/.2)) if ok else 0)
    actual=float(downward(p,r,s,ok,threshold=.8,cap=.1));assert math.isclose(actual,expected,abs_tol=1e-15,rel_tol=0);assert 0<=actual<=p and p-actual<=.1+1e-15;count+=1
frame=pd.DataFrame(dict(farm=['F13','F13','F47'],day=[1,1,1],hour=[0,1,0],sensor=[10.,11.,12.],A=[.8,1.,.5],smooth_et=[1.,1.2,.6],smooth_lgb=[.6,.7,.4],smooth_mlp=[.8,1.,.5],smooth_pfn=[.8,1.,.5],sub_ec=[.7,.8,2.],inner_j=[0,1,2]))
base=design(frame,['sensor']);assert base.prefix_A.tolist()==[.8,.9,.5]
for label in FORBIDDEN:
    try:design(frame,[label])
    except ValueError:pass
    else:raise AssertionError(label)
changed=frame.copy();changed.sub_ec=[99.,98.,97.];changed.inner_j=[7,8,9];pd.testing.assert_frame_equal(base,design(changed,['sensor']))
future=frame.copy();future.loc[1,['A','smooth_et','smooth_lgb','smooth_mlp','smooth_pfn','sensor']]=99.;pd.testing.assert_series_equal(base.iloc[0],design(future,['sensor']).iloc[0])
other=frame.copy();other.loc[2,['A','smooth_et','smooth_lgb','smooth_mlp','smooth_pfn','sensor']]=99.;pd.testing.assert_series_equal(base.iloc[0],design(other,['sensor']).iloc[0])
pd.testing.assert_frame_equal(base,design(frame.iloc[[2,1,0]],['sensor']).loc[frame.index])
try:
    result=float(downward(1.,.5,1.,True,threshold=.8,cap=float('nan')));nan_cap_rejected=False;assert math.isnan(result)
except ValueError:nan_cap_rejected=True
out=dict(status='PASS_FINITE_PARAMETER_CONTRACT_WITH_CAP_NAN_DEFECT' if not nan_cap_rejected else 'PASS_WITH_CAP_NAN_REJECTED',finite_grid_cases=count,forbidden_column_names=len(FORBIDDEN),label_future_otherfarm_order_checks='PASS',cap_nan_rejected=nan_cap_rejected,scope='generic helper contract; no candidate thresholds selected or models fitted; explicit caller feature whitelist still required')
path=H/('verify_risk_core_contract_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
