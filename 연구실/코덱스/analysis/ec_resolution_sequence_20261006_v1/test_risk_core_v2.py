from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env,env_extra
import numpy as np,pandas as pd
from risk_core_v2 import design,downward
f=pd.DataFrame(dict(row_id=['F13_001_0','F13_001_1','F47_001_0'],farm=['F13','F13','F47'],day=[1]*3,hour=[0,1,0],sensor=[10.,11.,12.],A=[.8,1.,.5],smooth_et=[1.,1.2,.6],smooth_lgb=[.6,.7,.4],smooth_mlp=[.8,1.,.5],smooth_pfn=[.8,1.,.5],sub_ec=[.7,.8,2.],inner_j=[0,1,2]))
a=design(f,['sensor'])
changed=f.copy(); changed['sub_ec']=[99.,98.,97.]; changed['inner_j']=[7,8,9]; changed['row_id']=['secret']*3;changed['day']=[17]*3
pd.testing.assert_frame_equal(a,design(changed,['sensor']))
assert not {'sub_ec','inner_j','day','hour','row_id','farm'}&set(a.columns)
assert a.prefix_A.tolist()==[.8,.9,.5]
later=f.copy();later.loc[1,'A']=99.
pd.testing.assert_series_equal(a.iloc[0],design(later,['sensor']).iloc[0])
other=f.copy();other.loc[2,['A','smooth_et']]=99.
pd.testing.assert_series_equal(a.iloc[0],design(other,['sensor']).iloc[0])
order=f.iloc[[2,1,0]].copy();b=design(order,['sensor'])
np.testing.assert_array_equal(a.to_numpy(),b.loc[f.index].to_numpy())
for forbidden in ['sub_ec','inner_j','row_id','day']:
    try:design(f,[forbidden])
    except ValueError:pass
    else:raise AssertionError('Forbidden feature accepted: '+forbidden)
p=np.array([1.,1.,1.,.03,1.]);ref=np.array([.5,.5,1.2,0.,.5]);risk=np.array([.8,.9,1.,1.,1.]);ok=np.array([True,True,True,True,False])
c=downward(p,ref,risk,ok,threshold=.8,cap=.1)
np.testing.assert_allclose(c,[1.,.95,1.,0.,1.],rtol=0,atol=1e-15)
assert np.all(c>=0) and np.all(c<=p) and np.all(p-c<=.1+1e-15)
print('PASS_RISK_CORE_CONTRACT: labels excluded, prefix causal/order invariant, bounded correction',flush=True)

