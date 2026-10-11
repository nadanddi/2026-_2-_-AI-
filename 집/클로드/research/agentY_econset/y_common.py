# -*- coding: utf-8 -*-
"""agentY_econset 공용: 기록별 입력 배열과 하루 요약/0~6시 특징."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, os, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
R = pickle.load(open(r"C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl", "rb"))
r = R["recs"].reset_index(drop=True)
r["rid"] = r.farm + "_" + r.day.map(lambda d: f"{d:03d}")
T = pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
CH = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd",
      "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
ACT = [c for c in CH if c.startswith("act_")]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X["rid"] = X.row_id.str[:7]; X["h"] = X.row_id.str[-2:].astype(int); Y["rid"] = Y.row_id.str[:7]
rid2i = dict(zip(r.rid, r.index)); tr = r.index[r.split == "train"].values
A = np.full((len(r), 24, len(CH)), np.nan)
for rid, g in X.sort_values(["rid", "h"]).groupby("rid"):
    if rid in rid2i and len(g) == 24: A[rid2i[rid]] = g[CH].values
r["ec"] = Y.groupby("rid").sub_ec.mean().reindex(r.rid).values
nm = np.nanmean

def dayfeat(i):
    f = {}
    if i is None or np.all(np.isnan(A[i])): return f
    for k, c in enumerate(CH):
        v = A[i, :, k]
        if np.all(np.isnan(v)): continue
        night = np.r_[v[0:6], v[20:24]]; day = v[9:17]
        f[c + "_mean"] = nm(v); f[c + "_rng"] = np.nanmax(v) - np.nanmin(v)
        f[c + "_night"] = nm(night); f[c + "_day"] = nm(day); f[c + "_dn"] = nm(day) - nm(night)
        f[c + "_eve"] = nm(v[18:24])
        if c in ACT:
            on = v > 0
            f[c + "_duty"] = np.nanmean(on); f[c + "_nchg"] = np.nansum(np.abs(np.diff(v)) > 0)
            f[c + "_first"] = int(np.argmax(on)) if on.any() else 24
            f[c + "_full"] = np.nanmean(v >= 99)
    return f

def early(i):
    f = {}
    for k, c in enumerate(CH):
        v = A[i, 0:7, k]
        if np.all(np.isnan(v)): continue
        f[c + "_m06"] = nm(v); f[c + "_d06"] = v[6] - v[0]; f[c + "_x06"] = np.nanmax(v) - np.nanmin(v)
        if c in ACT: f[c + "_on06"] = np.nanmean(v > 0)
    return f

def links(cmax=1.0):
    t = T[T.c1 <= cmax].sort_values("c1").drop_duplicates("a1")
    return dict(zip(t.b, t.a1)), dict(zip(t.b, t.fa != t.fb))

def build(prev, xf):
    rows = []
    for i in tr:
        p1 = prev.get(i); p2 = prev.get(p1) if p1 is not None else None
        d = {"i": i, "rid": r.rid[i], "farm": r.farm[i], "num": r.day[i], "ec": r.ec[i], "p1": p1, "xf": xf.get(i, False),
             "ec_p1": r.ec[p1] if p1 is not None else np.nan, "ec_p2": r.ec[p2] if p2 is not None else np.nan}
        for k, v in early(i).items(): d["o_" + k] = v
        F1 = dayfeat(p1); F2 = dayfeat(p2)
        for k, v in F1.items(): d["p1_" + k] = v
        for k in F1:
            if k in F2: d["dp_" + k] = F1[k] - F2[k]
        if p1 is not None:
            e1 = early(p1); e0 = early(i)
            for k in e0:
                if k.endswith("_m06") and k in e1: d["do_" + k] = e0[k] - e1[k]
        rows.append(d)
    D = pd.DataFrame(rows).set_index("i")
    par = {i: i for i in tr}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for b, a in prev.items(): par[fd(b)] = fd(a)
    D["grp"] = [fd(i) for i in D.index]
    return D

def load_oof():
    O = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
    O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
    P = O.groupby("row_id").agg(sg=("sg", "mean"), y=("sub_ec", "first")).reset_index()
    P["rid"] = P.row_id.str[:7]; P["h"] = P.row_id.str[-2:].astype(int)
    return P
