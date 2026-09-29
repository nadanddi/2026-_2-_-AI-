"""Q1e: alternating (period-2) structure in daily means."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
for f in ['F13','F47']:
    d = X[X.farm==f]
    M = d.groupby('day')[['in_temp','in_hum','in_co2','out_temp','sub_temp','sub_ec','act_heating','act_thermal','act_vent','act_co2','act_circfan']].mean()
    M = M.reindex(range(M.index.min(),M.index.max()+1))
    # alternating component: x_d - (x_{d-1}+x_{d+1})/2
    alt = M - (M.shift(1)+M.shift(-1))/2
    sgn = np.where(M.index%2==0,1,-1)
    print(f, 'mean of alt * (+1 even,-1 odd) [>0 means even days higher]')
    print((alt.mul(sgn,axis=0)).mean().round(3).to_dict())
    print(' std of alt', alt.std().round(3).to_dict())
    # lag-1 autocorr of alt sign
    print(' AC1 of alt', {c: round(alt[c].autocorr(1),3) for c in alt})
    print(M['sub_ec'].dropna().round(2).head(60).tolist())
