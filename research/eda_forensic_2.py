# -*- coding: utf-8 -*-
"""Forensic 2: shape statistics of indoor sensors, train vs test (F13/F47).
runs of constants, linear runs (2nd diff 0), spikes, low CO2 context."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X = pd.concat([tX[tX.farm.isin(["F13","F47"])], sX], ignore_index=True).sort_values(["farm","t"]).reset_index(drop=True)
S = ["in_temp","in_hum","in_co2"]
# contiguity: previous row is t-1 of same farm
X["prev_ok"] = (X.farm.eq(X.farm.shift())) & (X.t.eq(X.t.shift()+1))
X["next_ok"] = (X.farm.eq(X.farm.shift(-1))) & (X.t.eq(X.t.shift(-1)-1))
res = []
for c in S:
    v = X[c]
    d1 = v - v.shift(); d1n = v.shift(-1) - v
    ok = X.prev_ok & X.next_ok
    lin = ok & np.isclose(d1, d1n, atol=1e-6) & (d1.abs()>1e-9)   # nonzero constant slope
    const = ok & np.isclose(d1,0) & np.isclose(d1n,0)
    spike = ok & (np.sign(d1) != np.sign(d1n)) & (np.minimum(d1.abs(), d1n.abs()) > {"in_temp":2,"in_hum":10,"in_co2":150}[c])
    X[c+"_lin"]=lin; X[c+"_const"]=const; X[c+"_spike"]=spike
    for (f,s),g in X.groupby(["farm","set"]):
        m = ok[g.index]
        res.append(dict(col=c,farm=f,set=s,n=int(m.sum()),
            lin=round(lin[g.index].sum()/m.sum()*100,2),
            const=round(const[g.index].sum()/m.sum()*100,2),
            spike=round(spike[g.index].sum()/m.sum()*100,3),
            absd1_mean=round(d1[g.index][m].abs().mean(),3),
            absd1_p99=round(d1[g.index][m].abs().quantile(.99),2)))
r = pd.DataFrame(res); print(r.to_string())
# linear run lengths (>=3 consecutive lin flags)
def runs(b):
    out=[];k=0
    for x in b:
        if x: k+=1
        else:
            if k: out.append(k)
            k=0
    if k: out.append(k)
    return np.array(out)
print("\nlinear-run length hist (consecutive rows flagged lin)")
for c in S:
    for (f,s),g in X.groupby(["farm","set"]):
        rr = runs(g[c+"_lin"].values)
        cr = runs(g[c+"_const"].values)
        print(c,f,s,"lin runs>=2:",(rr>=2).sum(),">=3:",(rr>=3).sum(),"max",rr.max() if len(rr) else 0,
              "| const runs>=3:",(cr>=3).sum(),"max",cr.max() if len(cr) else 0, "per1000rows lin>=2: %.2f"%((rr>=2).sum()/len(g)*1000))
# low co2 context
lo = X[X.in_co2<300]
print("\nlow co2 rows",lo.groupby(["farm","set"]).size().to_dict())
print("hour dist train low co2", lo[lo.set=="train"].hour.value_counts().sort_index().to_dict())
print("hour dist all train co2 mean by hour", X[X.set=="train"].groupby("hour").in_co2.mean().round(0).to_dict())
# runs of low co2
for f in ["F13","F47"]:
    g = X[(X.farm==f)&(X.set=="train")]
    rr = runs((g.in_co2<300).values); print(f,"low co2 run lengths",np.bincount(rr))
    # which days
    print(f,"days with low co2", g[g.in_co2<300].groupby("day").size().to_dict())
X.to_pickle(env.LOCAL+"/eda_forensic_2_X.pkl")
