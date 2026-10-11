# -*- coding: utf-8 -*-
"""PD3 (user idea: use several evening hours of the previous same-house day instead of the single 23 h point).  Fixed before running: same as PD2b except features x_eve = mean(EA2_p, 18..23) - EA2_t(0) and dq_eve = mean(EA2_p, 18..23) - mean(EA2_p, 0..5); arms EVE (x_eve, dq_eve, S_drop, S_rise), EVE-SA (x_eve only), SA-only (x_sa, reference).  Base PD2b (plan-critic fixes applied BEFORE running, ec0h/critic_plan_pd2.md): A) same-ID number gap 1..2 only; B) per fold, features winsorised to the 2.5/97.5% quantiles of the fitting targets (fit and apply) and r0_hat clipped to the 2.5/97.5% quantiles of the fitted values; C) c_h excludes target and prev records; D) x_sa-only arm reported under the same rule; dropped-by-dropna count printed.  Base: PD2 (2026-10-12 집 클로드; '0시 차근차근' step 3: PD1 on the EV FINAL CHAIN).  Fixed before running.
Targets: EV val records t WITHOUT an EA2 anchor whose v7m prev p is same ID, p >= .8, smaller number, and p is a val record of the
SAME fold (evaluation-like: previous day is an unlabelled evaluation record; its inputs and chain predictions are allowed, 6.631).
Features (p's label never used): S_drop, S_rise = p's input-only irrigation scores from the EV3 detector RETRAINED per EV fold
without that fold's val/del records; dq_p = EA2_p(23) - EA2_p(0); x_sa = EA2_p(23) - EA2_t(0).
Model: Ridge(alpha 1) on standardised features, fitted on targets of the OTHER folds with number < 179 (per seed), target
r0 = y_t(0) - EA2_t(0).  Correction: pred_h = EA2_t,h + c_h * r0_hat (c_h = DIAG10 pass-1 0 h residual carry).
PASS iff pass-1: all 3 seeds better (all-hour RMSE of targets) AND chunk-cluster P(worse) < .025 AND top-record share <= 50%;
pass-2 direction reported; 0 h RMSE change reported.
"""
import os, sys, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier
rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
src = io.open(os.path.join(H_, "ec0h", "event", "ev3_irrigation_detector_v1.py"), encoding="utf-8").read(); src = src[:src.index("for kind,hours,sel in")]
g = {"__name__": "x"}; exec(compile(src, "ev3", "exec"), g); D, Fe = g["D"], g["F"]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
anch = R.groupby("rec").anchored.first()
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
# per-fold irrigation scores
SC = {}
for k in sorted(Fv.fold.unique()):
    hid = set(Fv[Fv.fold == k].rec); SC[k] = {}
    for kind, hours, sel in (("drop", range(8, 14), D.dEC <= -.05), ("rise", range(13, 19), D.dEC >= .05)):
        Z = D[D.h.isin(hours)].copy(); Z["y"] = sel.reindex(Z.index).fillna(False).astype(int); tr = ~Z.rec.isin(hid)
        m = ExtraTreesClassifier(400, min_samples_leaf=10, max_features=.4, random_state=1, n_jobs=2).fit(Z.loc[tr, Fe + ["h"]].fillna(-99), Z.y[tr])
        Z["pr"] = m.predict_proba(Z[Fe + ["h"]].fillna(-99))[:, 1]; SC[k][kind] = Z.groupby("rec").pr.sum()
T = []
for t in V.index:
    if t not in L.index or bool(anch.get(t, True)): continue
    pr = L.loc[t]; p = pr.prev; k = V.at[t, "fold"]
    if pr.xf or pr.p < .8 or p[:3] != t[:3] or not (1 <= int(t[4:7]) - int(p[4:7]) <= 2) or p not in V.index or V.at[p, "fold"] != k: continue
    T.append(dict(t=t, p=p, fold=k, chunk=V.at[t, "chunk"], S_drop=SC[k]["drop"].get(p, np.nan), S_rise=SC[k]["rise"].get(p, np.nan)))
T0n = len(T); T = pd.DataFrame(T).dropna(); print("dropped by dropna:", T0n - len(T)); print("targets", len(T), "| pass-2", T[T.t.str[4:7].astype(int) >= 179].t.tolist())
excl = set(T.t) | set(T.p)
p1r = [r for r in QD.index if int(r[4:7]) < 179 and r not in excl]; e = (YD.loc[p1r] - QD.loc[p1r]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
Y = R.groupby(["rec", "hour"]).y.first().unstack(); rng = np.random.default_rng(20261227)
for ARM, FEATS in (("EVE", ["S_drop", "S_rise", "dq_eve", "x_eve"]), ("EVE-SA", ["x_eve"]), ("SA-only", ["x_sa"])):
    out = {}; base = {}
    for sd, G in R.groupby("seed"):
        P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
        T["dq_p"] = [P.at[p, 23] - P.at[p, 0] for p in T.p]; T["x_sa"] = [P.at[p, 23] - P.at[t, 0] for t, p in zip(T.t, T.p)]
        T["x_eve"] = [P.loc[p, 18:23].mean() - P.at[t, 0] for t, p in zip(T.t, T.p)]; T["dq_eve"] = [P.loc[p, 18:23].mean() - P.loc[p, 0:5].mean() for p in T.p]
        T["r0"] = [Y.at[t, 0] - P.at[t, 0] for t in T.t]
        for k in T.fold.unique():
            oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
            lo, hi = oth[FEATS].quantile(.025), oth[FEATS].quantile(.975)
            Xo = oth[FEATS].clip(lo, hi, axis=1); mu, s_ = Xo.mean(), Xo.std() + 1e-9
            m = Ridge(1.0).fit((Xo - mu) / s_, oth.r0); fit = m.predict((Xo - mu) / s_); flo, fhi = np.quantile(fit, [.025, .975])
            te = T[T.fold == k]; hat = np.clip(m.predict((te[FEATS].clip(lo, hi, axis=1) - mu) / s_), flo, fhi)
            for t, hh in zip(te.t, hat): new.loc[t] = P.loc[t].values + c * hh
        out[sd] = new
    print("== ARM", ARM)
    for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        recs = [r for r in T.t if cond(r)]
        if not recs: print(part, "none"); continue
        h0 = np.mean([rm(out[s].loc[recs, 0] - Y.loc[recs, 0]) / rm(base[s].loc[recs, 0] - Y.loc[recs, 0]) - 1 for s in out])
        cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
        per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
        cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        print(f"{part}: records {len(recs)} | 0h {100*h0:+.1f}% | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
        print("   per-record dSSE:", ", ".join("%s %+.3f" % (r, v) for r, v in per.sort_values().items() if abs(v) > .05))
        if part == "pass-1": print("VERDICT PD3", ARM, ":", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
