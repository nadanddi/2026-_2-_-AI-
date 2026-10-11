# -*- coding: utf-8 -*-
"""MX4: does the shade 'schedule mode' fingerprint add CV-predictable EC day residual beyond 15 day-mean controls?
Ridge / ExtraTrees grouped by EC DIAG10 fold; controls-only vs controls+shade meta (stepmin, mode, full/zero shares);
also rank-based chain AUC for shade_mode, and seasonal placement. Descriptive (not an adoption test)."""
import os, sys, pickle
R0 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R0)
import env  # noqa
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
OUT = os.path.join(R0, "local", "agentX_meta")
Z = pd.read_pickle(os.path.join(OUT, "mx3_Z.pkl")); M1 = pd.read_pickle(os.path.join(OUT, "mx1_Z.pkl"))
Z = Z.merge(M1[["farm", "day", "fold_e", "fold", "act_shade_full", "act_shade_z", "act_shade_nuniq"]], on=["farm", "day"])
CTL = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog", "vent0", "day"]
SH = ["shade_stepmin", "shade_mode", "act_shade_full", "act_shade_z", "act_shade_nuniq"]
rng = np.random.default_rng(1)
for tg, fo in (("re", "fold_e"), ("rle", "fold_e"), ("rt", "fold")):
    S = Z[Z[tg].notna()].reset_index(drop=True); y = S[tg].values; g = S[fo].values
    for mdl in ("ridge", "et"):
        out = {}
        for nm, cols in (("ctl", CTL), ("ctl+shade", CTL + SH), ("shade", SH)):
            Xm = S[cols].fillna(0).values; pr = np.zeros_like(y)
            for k in np.unique(g):
                tr = g != k
                m = RidgeCV(alphas=np.logspace(-2, 3, 12)) if mdl == "ridge" else ExtraTreesRegressor(300, min_samples_leaf=8, max_features=.5, random_state=0, n_jobs=-1)
                mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
                m.fit((Xm[tr] - mu) / sd, y[tr]); pr[~tr] = m.predict((Xm[~tr] - mu) / sd)
            out[nm] = pr
        p2 = S.day.values >= 179
        def r2(pr, m): return 1 - np.sum((y[m] - pr[m]) ** 2) / np.sum((y[m] - 0) ** 2)  # vs zero correction (residual already OOF)
        print("%s %-5s R2-vs-0 (all/p1/p2): " % (tg, mdl) + " | ".join("%s %.3f/%.3f/%.3f" % (k, r2(v, S.day.values >= 0), r2(v, ~p2), r2(v, p2)) for k, v in out.items()))
# chain AUC of shade_mode (exact match rate truth vs control)
R = pickle.load(open(os.path.join(R0, "local", "linker_v3", "base.pkl"), "rb")); rec = R["recs"].reset_index(drop=True)
T = pd.read_pickle(os.path.join(R0, "local", "linker_v3", "truth.pkl")); key = {i: (rec.farm[i], int(rec.day[i])) for i in range(len(rec))}
FI = pd.read_pickle(os.path.join(OUT, "mx3_Z.pkl")).set_index(["farm", "day"]).shade_mode
same_t, same_c = [], []
for _, t in T.iterrows():
    b, a = key[int(t.b)], key[int(t.a1)]
    if b in FI and a in FI:
        same_t.append(FI[b] == FI[a]); ctr = [k for k in FI.index if k[0] == a[0] and abs(k[1] - a[1]) <= 4 and k not in (a, b)]
        same_c.append(np.mean([FI[k] == FI[b] for k in ctr]))
print("shade_mode agreement: truth prev-same-house %.3f vs same-farm +-4d controls %.3f (n %d)" % (np.mean(same_t), np.mean(same_c), len(same_t)))
# month-ish profile
Z["blk"] = Z.day // 20
print(Z.groupby("blk").agg(n=("re", "size"), mode=("shade_mode", "mean"), re_mode1=("re", lambda v: v[Z.loc[v.index, "shade_mode"] == 1].mean()), re_mode0=("re", lambda v: v[Z.loc[v.index, "shade_mode"] == 0].mean())).round(3).T)
