# -*- coding: utf-8 -*-
"""Forensic 7: per-day input signatures, train vs test (clean reference), and link to label residual."""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_6_G.pkl").sort_values(["farm","t"]).reset_index(drop=True)
def ah(T,RH):  # g/m3
    return 6.112*np.exp(17.67*T/(T+243.5))*RH*2.1674/(273.15+T)
G["ah"]=ah(G.in_temp,G.in_hum)
G["ok1"]=G.farm.eq(G.farm.shift())&G.t.eq(G.t.shift()+1)
G["ok2"]=G.ok1&G.ok1.shift(-1).fillna(False)
for c in ["in_temp","in_hum","in_co2","ah"]:
    d1 = G[c]-G[c].shift(); d1=d1.where(G.ok1)
    G["d1_"+c]=d1
    d2n = G[c].shift(-1)-2*G[c]+G[c].shift()
    G["d2_"+c]=d2n.where(G.ok1 & G.farm.eq(G.farm.shift(-1)) & G.t.eq(G.t.shift(-1)-1))
G["night"]=G.hour.isin([0,1,2,3,4,5,20,21,22,23])
# explained jump: residual of d1 in_temp against d1 out_temp and out_rad change
G["d1_out"]=(G.out_temp-G.out_temp.shift()).where(G.ok1)
G["d1_rad"]=(G.out_rad-G.out_rad.shift()).where(G.ok1)
m=G.d1_in_temp.notna()&(G.set=="test")
A=np.c_[G.d1_out[m],G.d1_rad[m]/100,np.ones(m.sum())]
coef,*_=np.linalg.lstsq(A,G.d1_in_temp[m],rcond=None)
G["jump_res"]=G.d1_in_temp-(np.c_[G.d1_out,G.d1_rad/100,np.ones(len(G))]@coef)
def agg(g):
    n=g[g.night]
    return pd.Series(dict(
        rough_T_night=n.d2_in_temp.abs().mean(),
        rough_H_night=n.d2_in_hum.abs().mean(),
        rough_C_night=n.d2_in_co2.abs().mean(),
        rough_AH=g.d2_ah.abs().mean(),
        maxjump_res=g.jump_res.abs().max(),
        min_in_minus_out_night=(n.in_temp-n.out_temp).min(),
        max_in_minus_out=(g.in_temp-g.out_temp).max(),
        corr_in_out=g[["in_temp","out_temp"]].corr().iloc[0,1],
        na=g.in_temp.isna().sum(),
        res_abs=g.res.abs().mean(), n=len(g)))
D=G.groupby(["farm","set","day"]).apply(agg).reset_index()
D=D[D.n==24]
feats=["rough_T_night","rough_H_night","rough_C_night","rough_AH","maxjump_res","min_in_minus_out_night","max_in_minus_out","corr_in_out"]
print("quantiles train vs test (complete days)")
for c in feats:
    for s in ["train","test"]:
        q=D[D.set==s][c].quantile([.01,.05,.5,.95,.99]).round(2).tolist()
        print(f"{c:24s} {s:5s} {q}")
tr=D[D.set=="train"].copy()
print("\nspearman with day |res| (train):")
for c in feats: print(c, round(tr[[c,"res_abs"]].corr("spearman").iloc[0,1],3))
# exceedance vs test max
te=D[D.set=="test"]
print("\ntrain days exceeding test extreme:")
for c,side in [("rough_T_night","hi"),("rough_H_night","hi"),("rough_C_night","hi"),("rough_AH","hi"),("maxjump_res","hi"),("min_in_minus_out_night","lo"),("corr_in_out","lo")]:
    lim = te[c].max() if side=="hi" else te[c].min()
    ex = tr[tr[c]>lim] if side=="hi" else tr[tr[c]<lim]
    print(f"{c:24s} lim {lim:.2f} n_ex {len(ex)}/{len(tr)}  res_abs ex {ex.res_abs.mean():.2f} vs rest {tr.drop(ex.index).res_abs.mean():.2f}  days {ex[['farm','day']].values.tolist()[:14]}")
D.to_csv(env.LOCAL+"/eda_forensic_7_days.csv",index=False)
G.to_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
