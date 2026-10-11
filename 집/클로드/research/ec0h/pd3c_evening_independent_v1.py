# -*- coding: utf-8 -*-
"""PD3c (2026-10-12 집 클로드; independent confirmation of PD3 EVE-SA).  Fixed before running.
Coefficient: ONE constant w (and the 2.5/97.5% winsor bounds of x_eve and of fitted r0_hat) fitted ONCE on EV pass-1 EVE-SA targets
(PD2b/PD3 target rule: no EA2 anchor, v7m same ID, p >= .8, number gap 1..2, p = same-fold val), pooled over EV seeds; Ridge(1) on
standardised x_eve = mean(EA2_p, 18..23) - EA2_t(0), target r0 = y_t0 - EA2_t0.  Never refitted.
Independent data: DIAG10 EC OOF (tag CUR, sg) seeds 47 / 1414 / 6464 (different model pipeline and folds).
  targets: training records t with v7m prev p (same ID, p >= .8, number gap 1..2), p's LABEL ignored (only p's OOF prediction).
  x_eve = mean(q_p, 18..23) - q_t(0), winsorised with the EV bounds; r0_hat = clip(model(x_eve)); pred_h = q_t,h + c_h * r0_hat,
  c_h = DIAG10 pass-1 0 h residual carry excluding targets and prevs.
PASS iff DIAG10 pass-1: all 3 seeds better AND record bootstrap P(worse) < .025 AND top-record share <= 50%.  Pass-2 descriptive.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
anch = R.groupby("rec").anchored.first()
ok = lambda t, p, pp, xf: (not xf) and pp >= .8 and p[:3] == t[:3] and 1 <= int(t[4:7]) - int(p[4:7]) <= 2
# --- fit ONCE on EV pass-1 ---
TE = [(t, L.at[t, "prev"]) for t in V.index if t in L.index and not bool(anch.get(t, True)) and int(t[4:7]) < 179
      and ok(t, L.at[t, "prev"], L.at[t, "p"], L.at[t, "xf"]) and L.at[t, "prev"] in V.index and V.at[L.at[t, "prev"], "fold"] == V.at[t, "fold"]]
xs, ys = [], []
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); Yv = G.groupby(["rec", "hour"]).y.first().unstack()
    for t, p in TE: xs.append(P.loc[p, 18:23].mean() - P.at[t, 0]); ys.append(Yv.at[t, 0] - P.at[t, 0])
xs, ys = np.array(xs), np.array(ys); lo, hi = np.quantile(xs, [.025, .975]); xc = np.clip(xs, lo, hi)
mu, sdv = xc.mean(), xc.std() + 1e-9; M = Ridge(1.0).fit(((xc - mu) / sdv)[:, None], ys); fit = M.predict(((xc - mu) / sdv)[:, None]); flo, fhi = np.quantile(fit, [.025, .975])
print("EV pass-1 fit: pairs %d, slope per unit x %.3f, intercept %.4f, x bounds [%.3f, %.3f]" % (len(TE), M.coef_[0] / sdv, M.intercept_, lo, hi))
# --- independent DIAG10 ---
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; Q = {sd: G.groupby(["rec", "hour"]).sg.mean().unstack() for sd, G in O.groupby("seed")}
Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
T = [(t, L.at[t, "prev"]) for t in L.index if ok(t, L.at[t, "prev"], L.at[t, "p"], L.at[t, "xf"]) and t in Y.index and L.at[t, "prev"] in Y.index]
excl = {x for pr in T for x in pr}; Qm = sum(Q.values()) / len(Q)
p1 = [r for r in Y.index if int(r[4:7]) < 179 and r not in excl]; e = (Y.loc[p1] - Qm.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
print("DIAG10 targets: pass-1 %d, pass-2 %d" % (sum(int(t[4:7]) < 179 for t, _ in T), sum(int(t[4:7]) >= 179 for t, _ in T)))
rng = np.random.default_rng(20261228)
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    TT = [(t, p) for t, p in T if cond(t)]; recs = [t for t, _ in TT]
    cells, h0s, per = [], [], []
    for sd, Qs in Q.items():
        new = Qs.copy()
        for t, p in TT:
            x = np.clip(Qs.loc[p, 18:23].mean() - Qs.at[t, 0], lo, hi); hat = np.clip(M.predict([[(x - mu) / sdv]])[0], flo, fhi)
            new.loc[t] = Qs.loc[t].values + c * hat
        cells.append(rm((new.loc[recs] - Y.loc[recs]).values) / rm((Qs.loc[recs] - Y.loc[recs]).values) - 1)
        h0s.append(rm(new.loc[recs, 0] - Y.loc[recs, 0]) / rm(Qs.loc[recs, 0] - Y.loc[recs, 0]) - 1)
        per.append(((new.loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((Qs.loc[recs] - Y.loc[recs]) ** 2).sum(1))
    Pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(Pr), (20000, len(Pr))); pw = float((Pr.values[ii].sum(1) >= 0).mean())
    share = float(Pr.min() / Pr.sum()) if Pr.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | 0h {100*np.mean(h0s):+.1f}% | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((Pr<0).sum())}/{len(Pr)} | P {pw:.3f} | top share {share:.2f}")
    print("   largest:", ", ".join("%s %+.3f" % (r, v) for r, v in list(Pr.sort_values().items())[:4] + list(Pr.sort_values().items())[-3:]))
    if part == "pass-1": print("VERDICT PD3c:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
