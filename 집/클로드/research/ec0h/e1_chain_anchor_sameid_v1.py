# -*- coding: utf-8 -*-
"""E1 (2026-10-12 집 클로드; rule reading 6.631: same-ID earlier evaluation record's computed prediction is an allowed feature).
EXPLORATORY (pass-2 sample tiny).  Fixed before running:
  targets : EV val records t WITHOUT an EA2 anchor whose v7m previous record p is UNLABELLED, same ID (xf False), p >= .8,
            p is an EV val record of the SAME fold with a smaller number (so its inputs are allowed and its chain prediction exists).
  arms    : E1  = p itself has an EA2 anchor (label information reaches t through p);  SA = p has no anchor (plain soft anchor).
  estimate: E = final-chain prediction of p at 23 h (EA2 column, per seed).
  pred_h  = EA2_t,h + c_h * w * (E - EA2_t,0); c_h = 0 h residual carry (DIAG10 pass-1, as EC0-S); w = LS slope of
            (y_t,0 - EA2_t,0) on (E - EA2_t,0) over targets of the OTHER folds (same arm), clipped [0, 1].
Report per arm: 0 h RMSE change, final-chain RMSE change (pass 1 / pass 2), records improved, chunk-cluster P(worse), top-record share.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = F[F.role == "val"].set_index("rec")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
anch = R.groupby("rec").anchored.first()
T = []
for t in V.index:
    if t not in L.index or bool(anch.get(t, True)): continue
    pr = L.loc[t]
    if pr.prev_is_labelled or pr.xf or pr.p < .8: continue
    p = pr.prev
    if p not in V.index or V.at[p, "fold"] != V.at[t, "fold"] or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]): continue
    T.append(dict(t=t, p=p, fold=V.at[t, "fold"], arm="E1" if bool(anch.get(p, False)) else "SA", chunk=V.at[t, "chunk"]))
T = pd.DataFrame(T); print("targets:", T.groupby("arm").size().to_dict(), "| pass-2:", T[T.t.str[4:7].astype(int) >= 179].groupby("arm").t.apply(list).to_dict())
piv = {sd: G.pivot_table(index="rec", columns="hour", values="EA2") for sd, G in R.groupby("seed")}
Y = R.groupby(["rec", "hour"]).y.first().unstack()
rng = np.random.default_rng(20261218)
for arm in ("E1", "SA"):
    A = T[T.arm == arm]
    if A.empty: print(arm, "none"); continue
    out = {}
    for sd, P in piv.items():
        new = P.copy()
        for k in A.fold.unique():
            oth = A[A.fold != k]
            if len(oth) < 3: w = .5
            else:
                x = P.loc[oth.p, 23].values - P.loc[oth.t, 0].values; yv = Y.loc[oth.t, 0].values - P.loc[oth.t, 0].values
                w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1))
            for _, r in A[A.fold == k].iterrows():
                d = w * (P.at[r.p, 23] - P.at[r.t, 0]); new.loc[r.t] = P.loc[r.t].values + c * d
        out[sd] = new
    for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        recs = [r for r in A.t if cond(r)]
        if not recs: print(f"  {arm} {part}: none"); continue
        h0 = np.mean([rm(out[sd].loc[recs, 0] - Y.loc[recs, 0]) / rm(piv[sd].loc[recs, 0] - Y.loc[recs, 0]) - 1 for sd in piv])
        cells = [rm((out[sd].loc[recs] - Y.loc[recs]).values) / rm((piv[sd].loc[recs] - Y.loc[recs]).values) - 1 for sd in piv]
        per = pd.Series(np.mean([(((out[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((piv[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for sd in piv], 0), index=recs)
        cl = per.groupby(A.set_index("t").chunk.reindex(recs).values).sum(); idx = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[idx].sum(1) >= 0).mean())
        print(f"  {arm} {part}: records {len(recs)} | 0h {100*h0:+.1f}% | final-chain cells {' '.join('%+.1f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P(worse) {pw:.3f} | top-record share {100*per.min()/per.sum():.0f}%" if per.sum() < 0 else
              f"  {arm} {part}: records {len(recs)} | 0h {100*h0:+.1f}% | final-chain cells {' '.join('%+.1f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P(worse) {pw:.3f}")
