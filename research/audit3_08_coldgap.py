import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
A=tX.merge(ty[['row_id','sub_temp']],on='row_id')
A=A[A.farm.isin(['F13','F47'])].sort_values(['farm','t'])
A['sm3']=A.groupby('farm').in_temp.transform(lambda s:s.ewm(halflife=3).mean())
A['trend']=A.in_temp-A.sm3
A['g_raw']=A.sub_temp-A.in_temp; A['g_sm']=A.sub_temp-A.sm3
b=pd.cut(A.in_temp,[-9,6,8,10,12,15,99])
print(A.groupby(b,observed=True).agg(n=('g_raw','size'),g_raw=('g_raw','mean'),g_sm=('g_sm','mean'),trend=('trend','mean'),ndays=('day',lambda s: s.nunique())).round(3))
# regression gap_sm ~ trend + cold hinge, night only
import numpy.linalg as la
N=A[(A.hour<=6)|(A.hour>=20)].dropna(subset=["g_sm","trend","sm3"])
X=np.c_[np.ones(len(N)),N.trend,np.maximum(0,8-N.sm3),np.maximum(0,10-N.sm3)]
co=la.lstsq(X,N.g_sm.values,rcond=None)[0]; print('night gap_sm = %.3f + %.3f*trend + %.3f*max(0,8-sm3) + %.3f*max(0,10-sm3)'%tuple(co))
# per-day block bootstrap of the hinge coef
rng=np.random.default_rng(0); days=N[['farm','day']].drop_duplicates().values
bs=[]
g=dict(list(N.groupby(['farm','day'])))
for _ in range(300):
    pick=days[rng.integers(0,len(days),len(days))]
    S=pd.concat([g[tuple(p)] for p in pick])
    X=np.c_[np.ones(len(S)),S.trend,np.maximum(0,8-S.sm3),np.maximum(0,10-S.sm3)]
    bs.append(la.lstsq(X,S.g_sm.values,rcond=None)[0])
bs=np.array(bs); print('boot 90% CI hinge8',np.quantile(bs[:,2],[.05,.95]).round(3),'hinge10',np.quantile(bs[:,3],[.05,.95]).round(3))
