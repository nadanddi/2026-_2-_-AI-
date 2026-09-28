# -*- coding: utf-8 -*-
"""Forensic 15: persistence of the 2-day EC phase across gaps; local parity effect.
Detrend daily EC by centered 9-day rolling median (diagnostic only), then autocorr by lag.
Also: parity model -- local high-parity estimated from labelled days within +-W days,
evaluated leave-block-out on train days (how well can neighbours' parity predict a day's zigzag sign?)."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
for f in ["F13","F47"]:
    y=ty[ty.farm==f].groupby("day").sub_ec.mean()
    full=y.reindex(range(y.index.min(),y.index.max()+1))
    tr=full.rolling(9,center=True,min_periods=5).median()
    r=(full-tr)
    ac={}
    for L in range(1,15):
        a=r.values[:-L]; b=r.values[L:]; m=~np.isnan(a)&~np.isnan(b)
        ac[L]=round(np.corrcoef(a[m],b[m])[0,1],2)
    print(f,"detrended daily EC autocorr by lag:",ac)
    # log-ratio version
    lr=np.log(full)-np.log(full).rolling(9,center=True,min_periods=5).median()
    ac2={}
    for L in range(1,15):
        a=lr.values[:-L]; b=lr.values[L:]; m=~np.isnan(a)&~np.isnan(b)
        ac2[L]=round(np.corrcoef(a[m],b[m])[0,1],2)
    print(f,"detrended log EC autocorr:",ac2)
    # parity-from-neighbours: for each day d with labelled neighbours, sign s_d of residual;
    # predict with parity estimated from labelled days in [d-W,d-g]∪[d+g,d+W] (excluding a gap g like test)
    s=np.sign(r)
    for g in [2,4,6]:
        for W in [6,10]:
            hit=[];tot=0
            for d in r.dropna().index:
                cand=[k for k in range(d-W,d+W+1) if abs(k-d)>=g and k in r.index and not np.isnan(r[k])]
                if len(cand)<3: continue
                vote=sum(s[k]*(1 if (k-d)%2==0 else -1)*abs(r[k]) for k in cand)
                if vote==0: continue
                hit.append(np.sign(vote)==s[d])
            print(f"  {f} gap>={g} window {W}: parity-vote accuracy {np.mean(hit):.2f} (n={len(hit)})")
