# -*- coding: utf-8 -*-
"""Forensic 5: interpolation-consistent segments. A segment [i,j] is 'interp' if every
value lies within rounding tolerance of the straight line joining x_i and x_j.
Mark maximal segments with length >= L. Compare train vs test and link to label residuals."""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_3_G.pkl").sort_values(["farm","t"]).reset_index(drop=True)
TOL = {"in_temp":0.051,"in_hum":0.51,"in_co2":0.51}
def interp_segments(t, x, tol, Lmin):
    n=len(x); flag=np.zeros(n,bool); segs=[]
    i=0
    while i<n-1:
        best=i
        j=i+2
        while j<n and t[j]-t[i]==j-i and not np.isnan(x[i:j+1]).any():
            line = x[i] + (x[j]-x[i])*(np.arange(j-i+1))/(j-i)
            if np.all(np.abs(x[i:j+1]-line)<=tol): best=j; j+=1
            else: break
        if best-i+1>=Lmin:
            flag[i:best+1]=True; segs.append((i,best))
            i=best
        else: i+=1
    return flag, segs
rows=[]
for c,tol in TOL.items():
    for L in [5,6,8]:
        G[f"{c}_ip{L}"]=False
        for f in ["F13","F47"]:
            idx = G.index[G.farm==f]
            g = G.loc[idx]
            fl, segs = interp_segments(g.t.values, g[c].values.astype(float), tol, L)
            G.loc[idx, f"{c}_ip{L}"]=fl
            for s in ["train","test"]:
                m = (g.set=="train").values if s=="train" else (g.set=="test").values
                rows.append(dict(col=c,L=L,farm=f,set=s,pct=round(fl[m].mean()*100,2)))
r = pd.DataFrame(rows).pivot_table(index=["col","L"],columns=["farm","set"],values="pct")
print(r)
# any-var flag with L=6
G["ip_any6"] = G[[f"{c}_ip6" for c in TOL]].any(axis=1)
G["ip_temp6"] = G["in_temp_ip6"]
tr = G[G.set=="train"]
thr = tr.res.abs().quantile(.98)
for c in ["ip_any6","ip_temp6","in_hum_ip6","in_co2_ip6","in_temp_ip8"]:
    m = tr[c]
    print(c,"n=%d  |res| mean %.2f vs %.2f   P(big) %.3f vs %.3f"%(m.sum(), tr[m].res.abs().mean(), tr[~m].res.abs().mean(), (tr[m].res.abs()>thr).mean(), (tr[~m].res.abs()>thr).mean()))
# nonzero slope vs constant interp segments for temp
G.to_pickle(env.LOCAL+"/eda_forensic_5_G.pkl")
