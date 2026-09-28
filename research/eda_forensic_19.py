# -*- coding: utf-8 -*-
"""Forensic 19: copied indoor segments. Exact k-hour matches of (in_temp,in_hum,in_co2) at different
times within F13, F47, and across F13<->F47 (train+test). Also single-variable in_temp 6h matches."""
import env
import numpy as np, pandas as pd, common
from collections import defaultdict
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X=pd.concat([tX[tX.farm.isin(["F13","F47"])],sX]).sort_values(["farm","t"]).reset_index(drop=True)
def keys(g,cols,k):
    v=g[cols].values; t=g.t.values; out=[]
    for i in range(len(g)-k+1):
        if t[i+k-1]-t[i]!=k-1: continue
        w=v[i:i+k]
        if np.isnan(w).any(): continue
        if cols==["in_temp"] and np.ptp(w)==0: continue
        out.append((w.tobytes(),i))
    return out
for cols,k in [(["in_temp","in_hum","in_co2"],4),(["in_temp","in_hum","in_co2"],6),(["in_temp"],6),(["in_temp"],8)]:
    idx=defaultdict(list)
    for f in ["F13","F47"]:
        g=X[X.farm==f].reset_index(drop=True)
        for key,i in keys(g,cols,k): idx[key].append((f,int(g.t[i]),g.set[i]))
    dups=[v for v in idx.values() if len(v)>1]
    nontriv=[v for v in dups if len(set((a,b) for a,b,_ in v))>1]
    print(cols,k,"duplicate windows:",len(nontriv), "examples:",nontriv[:5])
