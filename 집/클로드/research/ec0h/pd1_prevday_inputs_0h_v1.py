# -*- coding: utf-8 -*-
"""PD1 (2026-10-12 집 클로드; '0시 차근차근' step 2).  Fixed before running.
Question: do the PREVIOUS same-house day's INPUT-derived signals predict today's EC 0 h residual (r0 = y0 - q0, q = DIAG10 CUR sg,
seed mean)?  Link t <- p: linker v7m, same ID, p >= .8 (p's label NEVER used).
Features of p (inputs / model only): S_drop, S_rise = input-only irrigation detector scores of p (EV3 detector, block folds as EV5),
  dq_p = q_p(23) - q_p(0) (p's predicted daily change), x_sa = q_p(23) - q_t(0) (soft-anchor gap).
Model: Ridge (alpha 1, standardised) fitted on PASS-1 targets only, once; leave-chunk-out CV on pass 1 for stability; then applied
to PASS-2 targets (judgement set, never used for fitting).  Correction spread with c_h (DIAG10 pass-1 0 h residual carry).
Report: pass-1 CV and pass-2: 0 h RMSE and all-hour RMSE change, records improved, record bootstrap P(worse), top share;
coefficients.  Also the soft-anchor-only model (x_sa) for comparison.
"""
import os, sys, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
src = io.open(os.path.join(H_, "ec0h", "event", "ev3_irrigation_detector_v1.py"), encoding="utf-8").read(); src = src[:src.index("for kind,hours,sel in")]
g = {"__name__": "x"}; exec(compile(src, "ev3", "exec"), g); D, Fe = g["D"], g["F"]
D["blk"] = D.rec.str[:3] + (D.rec.str[4:7].astype(int) // 10).astype(str); ub = D.blk.unique(); rng = np.random.default_rng(7)
D["fold"] = D.blk.map(dict(zip(ub, rng.permutation(len(ub)) % 5)))
S = pd.DataFrame(index=D.rec.unique())
for kind, hours, sel in (("drop", range(8, 14), D.dEC <= -.05), ("rise", range(13, 19), D.dEC >= .05)):
    Z = D[D.h.isin(hours)].copy(); Z["y"] = sel.reindex(Z.index).fillna(False).astype(int); pr = np.zeros(len(Z))
    for k in range(5):
        te = (Z.fold == k).values; m = ExtraTreesClassifier(400, min_samples_leaf=10, max_features=.4, random_state=1, n_jobs=2).fit(Z.loc[~te, Fe + ["h"]].fillna(-99), Z.y[~te]); pr[te] = m.predict_proba(Z.loc[te, Fe + ["h"]].fillna(-99))[:, 1]
    Z["pr"] = pr; S[kind] = Z.groupby("rec").pr.sum()
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; Q = O.groupby(["rec", "hour"]).sg.mean().unstack(); Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p1r = [r for r in Q.index if int(r[4:7]) < 179]; e = (Y.loc[p1r] - Q.loc[p1r]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
rows = []
for t in L.index:
    p = L.at[t, "prev"]
    if L.at[t, "xf"] or L.at[t, "p"] < .8 or p[:3] != t[:3] or t not in Q.index or p not in Q.index or p not in S.index: continue
    rows.append(dict(t=t, p=p, p2=int(t[4:7]) >= 179, chunk=t[:3] + str(int(t[4:7]) // 10), S_drop=S.at[p, "drop"], S_rise=S.at[p, "rise"],
                     dq_p=Q.at[p, 23] - Q.at[p, 0], x_sa=Q.at[p, 23] - Q.at[t, 0], r0=Y.at[t, 0] - Q.at[t, 0]))
T = pd.DataFrame(rows); print("targets pass-1 %d, pass-2 %d" % ((~T.p2).sum(), T.p2.sum()))
rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
def evaluate(nm, feats):
    P1 = T[~T.p2]; mu, sd = P1[feats].mean(), P1[feats].std() + 1e-9
    cv = np.zeros(len(P1))
    for ch in P1.chunk.unique():
        te = (P1.chunk == ch).values; m = Ridge(1.0).fit(((P1[feats] - mu) / sd)[~te], P1.r0[~te]); cv[te] = m.predict(((P1[feats] - mu) / sd)[te])
    m = Ridge(1.0).fit((P1[feats] - mu) / sd, P1.r0); P2 = T[T.p2]; p2hat = m.predict((P2[feats] - mu) / sd)
    for part, G, hat in (("pass-1 CV", P1, cv), ("pass-2", P2, p2hat)):
        recs = list(G.t); E0 = (Y.loc[recs] - Q.loc[recs]).values; E1 = E0 - c[None, :] * hat[:, None]
        per = (E1 ** 2).sum(1) - (E0 ** 2).sum(1); ii = np.random.default_rng(1).integers(0, len(per), (20000, len(per))); pw = float((per[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        print(f"  {nm:12s} {part:9s} n {len(G):3d} | 0h {rm(E0[:,0]):.3f}->{rm(E1[:,0]):.3f} ({100*(rm(E1[:,0])/rm(E0[:,0])-1):+.1f}%) | day {rm(E0):.3f}->{rm(E1):.3f} ({100*(rm(E1)/rm(E0)-1):+.1f}%) | improved {int((per<0).sum())}/{len(per)} | P {pw:.3f} | top share {share:.2f}")
    print("  coefs:", dict(zip(feats, np.round(m.coef_ / sd.values, 3))))
evaluate("all signals", ["S_drop", "S_rise", "dq_p", "x_sa"])
evaluate("soft anchor", ["x_sa"])
evaluate("irrigation", ["S_drop", "S_rise"])
