# -*- coding: utf-8 -*-
"""Forensic 14: EC 2-day zigzag phase and high-EC regime timeline; layout vs test blocks."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
for f in ["F13","F47"]:
    y=ty[ty.farm==f].groupby("day").sub_ec.mean()
    tdays=set(sX[sX.farm==f].day.unique()); xdays=set(tX[tX.farm==f].day.unique())
    dmin,dmax=min(xdays|tdays),max(xdays|tdays)
    line=[];vals=[]
    for d in range(dmin,dmax+1):
        if d in y.index:
            e=y[d]; ch = "H" if e>1 else ("h" if e>0.7 else ("m" if e>0.4 else "l"))
        elif d in tdays: ch="E"
        else: ch="."
        line.append(ch)
    print(f,"day",dmin,"..",dmax)
    s="".join(line)
    for k in range(0,len(s),60): print(f"  {dmin+k:4d} {s[k:k+60]}")
    # zigzag sign: s_d = sign(E_d - mean(E_{d-1},E_{d+1})) when neighbors exist
    z={}
    for d in y.index:
        if d-1 in y.index and d+1 in y.index:
            z[d]=np.sign(y[d]-(y[d-1]+y[d+1])/2)
    zz=pd.Series(z)
    # phase = +1 if (high on even day); per day: ph = s_d * (+1 if d even else -1)
    ph = zz*np.where(zz.index%2==0,1,-1)
    # print per 10-day window mean phase and fraction alternating
    alt = [(d, zz[d]!=zz[d+1]) for d in zz.index if d+1 in zz.index]
    A=pd.Series(dict(alt))
    print(f,"alternation rate overall %.2f (random ~0.5)"%A.mean())
    W=pd.DataFrame({"ph":ph,"win":(ph.index//10)*10})
    print(f,"phase (high-on-even=+1) by 10-day window:", W.groupby("win").ph.agg(lambda s: round(s.mean(),2)).to_dict())
    # amplitude by window: mean |E_d - mean(nbrs)|
    amp=pd.Series({d:abs(y[d]-(y[d-1]+y[d+1])/2) for d in zz.index})
    print(f,"zigzag amplitude by 20-day window:", amp.groupby((amp.index//20)*20).mean().round(3).to_dict())
    # phase runs: consecutive days with same ph
    runs=[];cur=None;st=None;prev=None
    for d in ph.index:
        if cur is None or ph[d]!=cur or d!=prev+1:
            if cur is not None: runs.append((st,prev,cur))
            cur=ph[d];st=d
        prev=d
    runs.append((st,prev,cur))
    print(f,"longest phase-consistent runs:", sorted(runs,key=lambda r:r[0]-r[1])[:6])
    # high regime runs (daily mean>1)
    hi=[d for d in y.index if y[d]>1]
    print(f,"days with mean EC>1:",hi)
