# -*- coding: utf-8 -*-
"""Forensic 26: how well do neighbouring days' EC levels (at test-like distances) predict a day's EC level?
Diagnostic of regime persistence only (label-neighbour interpolation is NOT a legal feature)."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
for f in ["F13","F47"]:
    y=ty[ty.farm==f].groupby("day").sub_ec.mean()
    y=y.reindex(range(y.index.min(),y.index.max()+1))
    for lo,hi in [(1,1),(2,2),(1,3),(3,5),(5,8),(8,12)]:
        pr=[];tt=[]
        for d in y.dropna().index:
            nb=[y.get(d+s*k) for k in range(lo,hi+1) for s in (-1,1)]
            nb=[v for v in nb if v is not None and not np.isnan(v)]
            if len(nb)>=2: pr.append(np.mean(nb)); tt.append(y[d])
        pr=np.array(pr);tt=np.array(tt)
        hi_t=tt>1
        print(f"{f} nbr dist {lo}-{hi}: n={len(tt)} rmse {np.sqrt(np.mean((pr-tt)**2)):.3f} corr {np.corrcoef(pr,tt)[0,1]:.2f}  on EC>1 days: bias {np.mean(pr[hi_t]-tt[hi_t]):+.2f} (n={hi_t.sum()})")
    print(f,"std of daily EC",round(y.std(),3))
