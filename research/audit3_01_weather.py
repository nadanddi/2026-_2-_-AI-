import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX]); A=A[A.farm.isin(['F13','F47'])]
W=['out_temp','out_hum','out_rad','out_wspd']
def dayvec(df,cols):
    p=df.pivot_table(index=['farm','day'],columns='hour',values=cols)
    return p
P=dayvec(A,W)
print('days', P.shape, 'complete', P.dropna().shape)
Pc=P.dropna()
keys=Pc.round(2).apply(lambda r: hash(tuple(r.values)),axis=1)
g=keys.groupby(keys).apply(lambda s: list(s.index))
multi=[v for v in g if len(v)>1]
print('exact groups >1:', len(multi), 'total groups', len(g))
from collections import Counter
comp=Counter(tuple(sorted(Counter(f for f,d in v).items())) for v in multi)
print(comp.most_common(8))
# within-farm pair gaps
gaps={'F13':[],'F47':[]}
for v in multi:
    for f in ['F13','F47']:
        ds=sorted(d for ff,d in v if ff==f)
        if len(ds)>=2: gaps[f]+= [ds[i+1]-ds[i] for i in range(len(ds)-1)]
for f in gaps: print(f, 'pairs',len(gaps[f]), 'gap med', np.median(gaps[f]), Counter(gaps[f]).most_common(6))
# near duplicates: out_temp 24h vector distance
X=P['out_temp'].dropna()
idx=list(X.index); M=X.values
D=np.sqrt(((M[:,None,:]-M[None,:,:])**2).mean(-1))
np.fill_diagonal(D,np.inf)
nn=D.min(1)
print('nearest out_temp rmse quantiles', np.quantile(nn,[.1,.25,.5,.75,.9]).round(3))
print('frac days with a near twin (<0.3):', (nn<0.3).mean().round(3))
# other farms: do they also contain within-record weather duplicates?
O=tX[~tX.farm.isin(['F13','F47'])]
res=[]
for f,df in O.groupby('farm'):
    q=df.pivot_table(index='day',columns='hour',values='out_temp').dropna()
    if len(q)<20: continue
    m=q.values; d=np.sqrt(((m[:,None]-m[None])**2).mean(-1)); np.fill_diagonal(d,np.inf)
    res.append((f,len(q),(d.min(1)<0.05).mean().round(3)))
print('other farms frac days with exact out_temp twin within same farm:', res)
# for F13/F47
for f in ['F13','F47']:
    q=A[A.farm==f].pivot_table(index='day',columns='hour',values='out_temp').dropna(); m=q.values
    d=np.sqrt(((m[:,None]-m[None])**2).mean(-1)); np.fill_diagonal(d,np.inf)
    print(f,len(q),'frac exact twin within farm',(d.min(1)<0.05).mean().round(3))
