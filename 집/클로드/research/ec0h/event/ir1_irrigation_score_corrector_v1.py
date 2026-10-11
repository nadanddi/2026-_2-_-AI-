# -*- coding: utf-8 -*-
"""IR1 (2026-10-12 집 클로드; user: study how other data move with nutrient-solution actions).  Fixed before running.
Signal (EV4/EV5): input-only irrigation detector -> previous same-house day's DROP score correlates with next-day 0 h EC residual
(partial -.22..-.25 after controlling next q0 and next own scores; same-house link needed).
Rule:
  detector : ExtraTrees on hours 8-13 (EC25 DROP events dEC25 <= -.05), inputs hours h-3..h (EV3 features); RETRAINED per EV fold
             on records that are not 'val'/'del' of that fold; score = sum of hourly probabilities over 8-13 h of a record.
  targets  : EV val records WITHOUT an EA2 anchor whose v7m prev p is same ID (xf False), p >= .8, smaller number, and p is not a
             'del' record (p may be training or a same-fold val record: only p's INPUTS are used).
  r0_hat   = beta * (score_p - mu), mu = mean score of training records of the fold; beta = LS slope of (y_t0 - EA2_t0) on
             (score_p - mu) over targets of the OTHER folds with number < 179 (pass 1 only), clipped to <= 0 (sign from EV4) .
  pred_h   = EA2_t,h + c_h * r0_hat,  c_h = DIAG10 pass-1 0 h residual carry (as EC0-S).
PASS iff pass-1 targets: all 3 seeds better AND chunk-cluster P(worse) < .025 AND top-record share <= 50%; pass-2 direction
reported (decidable only if >= 10 records).
"""
import os, sys, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
src = io.open(os.path.join(H, "ec0h", "event", "ev3_irrigation_detector_v1.py"), encoding="utf-8").read(); src = src[:src.index("for kind,hours,sel in")]
g = {"__name__": "x"}; exec(compile(src, "ev3", "exec"), g); D, Fe = g["D"], g["F"]
Fv = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
Z = D[D.h.isin(range(8, 14))].copy(); Z["y"] = (D.dEC <= -.05).reindex(Z.index).fillna(False).astype(int)
anch = R.groupby("rec").anchored.first(); score = {}; mu = {}
for k in sorted(Fv.fold.unique()):
    hide = set(Fv[Fv.fold == k].rec)
    tr = ~Z.rec.isin(hide)
    m = ExtraTreesClassifier(400, min_samples_leaf=10, max_features=.4, random_state=1, n_jobs=2).fit(Z.loc[tr, Fe + ["h"]].fillna(-99), Z.y[tr])
    Z["pr"] = m.predict_proba(Z[Fe + ["h"]].fillna(-99))[:, 1]
    s = Z.groupby("rec").pr.sum(); score[k] = s; mu[k] = float(s[~s.index.isin(hide)].mean())
T = []
for t in V.index:
    if t not in L.index or bool(anch.get(t, True)): continue
    pr = L.loc[t]; p = pr.prev; k = V.at[t, "fold"]
    if pr.xf or pr.p < .8 or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]): continue
    if p in set(Fv[(Fv.fold == k) & (Fv.role == "del")].rec) or p not in score[k].index: continue
    T.append(dict(t=t, p=p, fold=k, chunk=V.at[t, "chunk"], x=score[k][p] - mu[k]))
T = pd.DataFrame(T); print("targets", len(T), "| pass-2:", T[T.t.str[4:7].astype(int) >= 179].t.tolist())
piv = {sd: G.pivot_table(index="rec", columns="hour", values="EA2") for sd, G in R.groupby("seed")}
Y = R.groupby(["rec", "hour"]).y.first().unstack()
out = {}
for sd, P in piv.items():
    new = P.copy()
    for k in T.fold.unique():
        oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
        yv = Y.loc[oth.t, 0].values - P.loc[oth.t, 0].values
        beta = min(0.0, float((oth.x * yv).sum() / (oth.x ** 2).sum()))
        for _, r in T[T.fold == k].iterrows(): new.loc[r.t] = P.loc[r.t].values + c * beta * r.x
    out[sd] = new
print("beta (example, last fold, last seed): %.4f" % beta)
rng = np.random.default_rng(20261220)
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = [r for r in T.t if cond(r)]
    if not recs: print(part, "none"); continue
    h0 = np.mean([rm(out[sd].loc[recs, 0] - Y.loc[recs, 0]) / rm(piv[sd].loc[recs, 0] - Y.loc[recs, 0]) - 1 for sd in piv])
    cells = [rm((out[sd].loc[recs] - Y.loc[recs]).values) / rm((piv[sd].loc[recs] - Y.loc[recs]).values) - 1 for sd in piv]
    per = pd.Series(np.mean([(((out[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((piv[sd].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for sd in piv], 0), index=recs)
    cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); idx = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[idx].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | 0h {100*h0:+.1f}% | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    if part == "pass-1": print("VERDICT IR1:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
