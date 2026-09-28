# -*- coding: utf-8 -*-
"""Forensic 13: label interpolation-consistent segments (sub_ec, sub_temp), incl. across rounding."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
Y = ty[ty.farm.isin(["F13","F47"])].sort_values(["farm","t"]).reset_index(drop=True)
def segs(t,x,tol,Lmin):
    out=[];i=0;n=len(x)
    while i<n-1:
        best=i;j=i+2
        while j<n and t[j]-t[i]==j-i:
            line=x[i]+(x[j]-x[i])*np.arange(j-i+1)/(j-i)
            if np.all(np.abs(x[i:j+1]-line)<=tol): best=j; j+=1
            else: break
        if best-i+1>=Lmin: out.append((i,best)); i=best
        else: i+=1
    return out
for c,tol,minslope in [("sub_ec",0.0006,0.003),("sub_temp",0.006,0.05)]:
    for f in ["F13","F47"]:
        y=Y[Y.farm==f].reset_index(drop=True)
        ss=segs(y.t.values,y[c].values,tol,6)
        rows=[]
        for a,b in ss:
            sl=(y[c][b]-y[c][a])/(b-a)
            rows.append(dict(day=y.day[a],h0=y.hour[a],L=b-a+1,slope=round(sl,4),cross_mid=int(y.day[b]!=y.day[a])))
        R=pd.DataFrame(rows)
        if len(R)==0: print(c,f,"none"); continue
        st=R[R.slope.abs()>=minslope]
        print(c,f,"interp segs L>=6:",len(R),"with |slope|>=",minslope,":",len(st),"L max",R.L.max(),"cross midnight",R.cross_mid.sum())
        print(st.sort_values("L",ascending=False).head(10).to_string(index=False))
