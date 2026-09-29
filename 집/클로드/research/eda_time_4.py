"""Q1d: profile distance vs day lag; hour-level cross-lag between day d and d+1."""
import env, common
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
for f in ['F13','F47','F32','F05']:
    d = X[X.farm==f]
    for c in ['in_temp','in_hum','in_co2','out_temp','sub_temp','sub_ec']:
        if d[c].notna().mean()<0.3: continue
        P = d.pivot_table(index='day',columns='hour',values=c)
        P = P.reindex(range(P.index.min(),P.index.max()+1))
        prof = {k: np.nanmean(np.sqrt(((P.values[k:]-P.values[:-k])**2).mean(1))) for k in range(1,8)}
        mean = P.mean(1)
        md = {k: np.nanmean(np.abs(mean.values[k:]-mean.values[:-k])) for k in range(1,8)}
        # demeaned shape distance
        Q = P.sub(mean,axis=0)
        sh = {k: np.nanmean(np.sqrt(((Q.values[k:]-Q.values[:-k])**2).mean(1))) for k in range(1,8)}
        print(f, f"{c:9s}", 'daymean|diff|', ' '.join(f"{md[k]:.3f}" for k in md), '| shape dist', ' '.join(f"{sh[k]:.3f}" for k in sh))
