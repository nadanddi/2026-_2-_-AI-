import env, common
import numpy as np, pandas as pd
from collections import Counter
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX])
# other farms coverage
cov=tX.groupby('farm').agg(n=('day','size'),d0=('day','min'),d1=('day','max'),nd=('day','nunique'),ot_na=('out_temp',lambda s:s.isna().mean()))
print(cov.to_string())
