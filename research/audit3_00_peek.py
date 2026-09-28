import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
print(tX.shape, ty.shape, sX.shape)
print(tX.columns.tolist()); print(ty.columns.tolist()); print(sX.columns.tolist())
print(tX.head(3)); print(ty.head(3))
for f in ['F13','F47']:
    a=tX[tX.farm==f]; b=sX[sX.farm==f]
    print(f, a.day.min(), a.day.max(), a.day.nunique(), 'test', b.day.min(), b.day.max(), sorted(b.day.unique()))
