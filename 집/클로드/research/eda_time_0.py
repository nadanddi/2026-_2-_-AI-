import env, common
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
print(tX.shape, ty.shape, sX.shape)
print(tX.columns.tolist()); print(ty.columns.tolist())
print(tX.farm.value_counts())
for f in ['F13','F47']:
    a=tX[tX.farm==f]; b=ty[ty.farm==f]; c=sX[sX.farm==f]
    print(f,'trainX days',a.day.min(),a.day.max(),a.day.nunique(),'labels temp',b.dropna(subset=['sub_temp']).day.nunique(), 'ec',b.dropna(subset=['sub_ec']).day.nunique(),'test',c.day.min(),c.day.max())
    print(tX[tX.farm==f].isna().mean().round(2).to_dict())
print(tX.describe().T)
