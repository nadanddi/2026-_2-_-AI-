"""Q1f: cross-farm day identity: does F13 day d equal/resemble F47 day d+k?"""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
cols=['in_temp','in_hum','in_co2','out_temp','out_hum','out_rad','act_heating','act_thermal','act_vent','act_circfan','sub_temp','sub_ec']
P = {(f,c): X[X.farm==f].pivot_table(index='day',columns='hour',values=c) for f in ['F13','F47'] for c in cols}
for c in cols:
    A=P[('F13',c)]; B=P[('F47',c)]
    res={}
    for k in range(-4,5):
        common_days=[d for d in A.index if d+k in B.index]
        diff=np.array([np.nanmean(np.abs(A.loc[d].values-B.loc[d+k].values)) for d in common_days])
        eq=np.array([np.allclose(A.loc[d].values,B.loc[d+k].values,equal_nan=True) for d in common_days])
        res[k]=(round(np.nanmean(diff),3), round(eq.mean(),3))
    print(f"{c:12s}", res)
# correlation of daily means across farms at offsets
for c in ['in_temp','sub_temp','sub_ec','in_co2']:
    a=P[('F13',c)].mean(1); b=P[('F47',c)].mean(1)
    print(c, {k: round(a.corr(b.shift(-k).reindex(a.index)),3) for k in range(-4,5)})
