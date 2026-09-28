# -*- coding: utf-8 -*-
"""Forensic 3: X-y consistency. sub_temp vs causal EWMA of in_temp; find hours/days
where inputs disagree with (observed) labels -> candidate restored input rows."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
X = pd.read_pickle(env.LOCAL+"/eda_forensic_2_X.pkl")
X = X.merge(ty[["row_id","sub_temp","sub_ec"]], on="row_id", how="left")
out=[]
for f in ["F13","F47"]:
    g = X[X.farm==f].copy()
    # hourly grid
    full = pd.DataFrame({"t":np.arange(g.t.min(), g.t.max()+1)})
    g = full.merge(g, on="t", how="left")
    x = g.in_temp.values; o = g.out_temp.values
    best=None
    for a in [0.2,0.3,0.4,0.5,0.6,0.7]:
        e = np.full(len(x),np.nan); prev=np.nan
        for i in range(len(x)):
            if np.isnan(x[i]): prev = prev  # hold
            else: prev = x[i] if np.isnan(prev) else a*x[i]+(1-a)*prev
            e[i]=prev
        m = ~np.isnan(g.sub_temp.values)&~np.isnan(e)&(g.set.values=="train")
        A = np.c_[e[m], np.ones(m.sum()), g.hour.values[m]==0]
        # simple fit with hour dummies
        H = pd.get_dummies(g.hour[m]).values.astype(float)
        A = np.c_[e[m], o[m], H]
        coef,*_ = np.linalg.lstsq(A, g.sub_temp.values[m], rcond=None)
        r = g.sub_temp.values[m]-A@coef
        rm = np.sqrt((r**2).mean())
        if best is None or rm<best[0]: best=(rm,a,coef,e)
    rm,a,coef,e = best
    print(f,"best alpha",a,"rmse",round(rm,3))
    H = pd.get_dummies(g.hour).values.astype(float)
    A = np.c_[e, o, H]
    g["pred_lin"] = A@coef
    g["res"] = g.sub_temp - g.pred_lin
    g["farm"]=f
    out.append(g)
G = pd.concat(out)
G = G[G.row_id.notna()]
tr = G[G.set=="train"].copy()
# day-level residual stats
d = tr.groupby(["farm","day"]).agg(res_mean=("res","mean"), res_sd=("res","std"),
     r_in=("in_temp", lambda s: np.nan), n=("res","size")).reset_index()
# within day corr of sub_temp and lagged ewma
def dcorr(gg):
    return np.corrcoef(gg.pred_lin, gg.sub_temp)[0,1] if gg.sub_temp.notna().sum()>5 else np.nan
d["corr"] = tr.groupby(["farm","day"]).apply(dcorr).values
d["abs_mean"] = d.res_mean.abs()
for f in ["F13","F47"]:
    dd = d[d.farm==f]
    print(f, "day res_mean sd %.3f, day res_sd median %.3f"%(dd.res_mean.std(), dd.res_sd.median()))
    print(" worst days by |mean|:", dd.nlargest(8,"abs_mean")[["day","res_mean","res_sd","corr"]].round(2).values.tolist())
    print(" worst days by corr:", dd.nsmallest(8,"corr")[["day","res_mean","res_sd","corr"]].round(2).values.tolist())
# hour level: large residual rows vs flags
tr["bigres"] = tr.res.abs() > tr.res.abs().quantile(.98)
for c in ["in_temp_lin","in_temp_const","in_temp_spike","in_co2_lin","in_hum_const"]:
    print(c, "P(bigres|flag)=%.3f  P(bigres|~flag)=%.3f  nflag=%d"%(tr[tr[c]==True].bigres.mean(), tr[tr[c]!=True].bigres.mean(), (tr[c]==True).sum()))
# neighbour of missing
tr["near_na"] = False
for f in ["F13","F47"]:
    g = tr[tr.farm==f]
    na_t = set(g.t[g.in_temp.isna()])
    near = g.t.apply(lambda t: any((t+k) in na_t for k in range(-3,4)))
    tr.loc[g.index,"near_na"] = near
print("near missing (+-3h): P(big)=%.3f vs %.3f n=%d"%(tr[tr.near_na].bigres.mean(), tr[~tr.near_na].bigres.mean(), tr.near_na.sum()))
tr["lowco2"]=tr.in_co2<300
print("low co2: P(big)=%.3f n=%d"%(tr[tr.lowco2].bigres.mean(), tr.lowco2.sum()))
G.to_pickle(env.LOCAL+"/eda_forensic_3_G.pkl"); d.to_csv(env.LOCAL+"/eda_forensic_3_days.csv",index=False)
