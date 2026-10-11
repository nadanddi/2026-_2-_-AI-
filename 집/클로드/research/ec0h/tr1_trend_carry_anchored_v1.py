# -*- coding: utf-8 -*-
"""TR1 (2026-10-12 집 클로드; EC-expert lead: same-house daily change (23h - 0h) persists to the next day, rho +.29).
Fixed before running.  Targets: EV val records WITH an EA2 anchor (confident labelled previous record a, v7m p >= .8).
  dprev   = y_a(23) - y_a(0)   (training LABELS of the anchor record; train data unrestricted, 6.630)
  dmodel  = EA2_t(23) - EA2_t(0)  (today's change already predicted)
  pred_h  = EA2_t,h + k * (h / 23) * (dprev - dmodel)
  k       = LS slope over targets of the OTHER EV folds with number < 179 of (resid_t,h) on (h/23)(dprev - dmodel), clipped [0, 1].
PASS iff pass-1 targets: all 3 seeds better AND chunk-cluster P(worse) < .025 AND top-record share <= 50%; pass-2 reported.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Fv = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yall = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yall = Yall[Yall.row_id.str[:3].isin(["F13", "F47"])]
Yall["rec"] = Yall.row_id.str[:7]; Yall["h"] = Yall.row_id.str[8:].astype(int); YL = Yall.pivot_table(index="rec", columns="h", values="sub_ec")
anch = R.groupby("rec").anchored.first()
T = [dict(t=t, a=L.at[t, "prev"], fold=V.at[t, "fold"], chunk=V.at[t, "chunk"]) for t in V.index
     if bool(anch.get(t, False)) and t in L.index and L.at[t, "prev"] in YL.index]
T = pd.DataFrame(T); T["dprev"] = [YL.at[a, 23] - YL.at[a, 0] for a in T.a]
print("anchored targets", len(T), "| pass-2", int((T.t.str[4:7].astype(int) >= 179).sum()))
piv = {sd: G.pivot_table(index="rec", columns="hour", values="EA2") for sd, G in R.groupby("seed")}
Y = R.groupby(["rec", "hour"]).y.first().unstack(); hh = np.arange(24) / 23.
out = {}
for sd, P in piv.items():
    new = P.copy(); T["dm"] = [P.at[t, 23] - P.at[t, 0] for t in T.t]
    for k in T.fold.unique():
        oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
        X = np.concatenate([hh * (r.dprev - r.dm) for _, r in oth.iterrows()]); Yv = np.concatenate([(Y.loc[r.t] - P.loc[r.t]).values for _, r in oth.iterrows()])
        kk = float(np.clip((X * Yv).sum() / (X * X).sum(), 0, 1))
        for _, r in T[T.fold == k].iterrows(): new.loc[r.t] = P.loc[r.t].values + kk * hh * (r.dprev - r.dm)
    out[sd] = new
print("k example %.3f" % kk)
rng = np.random.default_rng(20261222)
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = [r for r in T.t if cond(r)]
    if not recs: continue
    cells = [rm((out[sd].loc[recs] - Y.loc[recs]).values) / rm((piv[sd].loc[recs] - Y.loc[recs]).values) - 1 for sd in piv]
    per = pd.Series(np.mean([(((out[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((piv[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for sd in piv], 0), index=recs)
    cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); idx = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[idx].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    if part == "pass-1": print("VERDICT TR1:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
